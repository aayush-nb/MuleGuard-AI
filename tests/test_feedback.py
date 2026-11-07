"""The self-learning loop: the toggle, the label rules, the gate, and the full cycle."""
import hashlib
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from mule import config as cfg
from mule import feedback as fb
from mule.data import load_clean
from mule.features import build_features
from mule.graph_layer import PastSum
from mule.model import fit, select_features


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(cfg, "SELF_LEARNING_ENABLED", True)
    monkeypatch.setattr(cfg, "BANK_DATA_PERMISSION_REF", "TEST-PERMISSION-1")


# --- The toggle --------------------------------------------------------------

def test_the_system_ships_with_self_learning_off():
    assert cfg.SELF_LEARNING_ENABLED is False
    assert cfg.BANK_DATA_PERMISSION_REF == ""


def test_the_static_pipeline_never_loads_the_feedback_loop():
    code = "import sys, run_pipeline; sys.exit('mule.feedback' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], cwd=cfg.ROOT).returncode == 0


@pytest.mark.parametrize("operation", [
    lambda w: fb.ingest(w, pd.DataFrame()),
    fb.score_new_transactions,
    lambda w: fb.record_reviews(w, pd.DataFrame()),
    fb.retrain,
    fb.rollback,
])
def test_every_operation_refuses_to_run_when_off(tmp_path, operation):
    with pytest.raises(fb.SelfLearningOff, match="off"):
        operation(fb.Warehouse(tmp_path))


def test_switching_on_without_bank_permission_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "SELF_LEARNING_ENABLED", True)
    with pytest.raises(fb.SelfLearningOff, match="permission"):
        fb.retrain(fb.Warehouse(tmp_path))


# --- Label rules -------------------------------------------------------------

def rows(*step_account):
    return pd.DataFrame(step_account, columns=["step", "account"]).assign(is_fraud=0)


def test_only_confirmed_mules_become_positive(monkeypatch):
    monkeypatch.setattr(cfg, "LABEL_MATURITY_DAYS", 1)
    labelled = fb.apply_labels(rows((10, 1), (10, 2)), [1], [], now_step=100)
    assert labelled.set_index("account")["is_fraud"].to_dict() == {1: 1, 2: 0}


def test_recent_rows_stay_out_because_a_mule_may_not_be_caught_yet(monkeypatch):
    monkeypatch.setattr(cfg, "LABEL_MATURITY_DAYS", 1)
    labelled = fb.apply_labels(rows((76, 1), (77, 2)), [], [], now_step=100)
    assert labelled["account"].tolist() == [1]


def test_accounts_awaiting_review_stay_out(monkeypatch):
    monkeypatch.setattr(cfg, "LABEL_MATURITY_DAYS", 1)
    labelled = fb.apply_labels(rows((10, 1), (10, 2)), [], [2], now_step=100)
    assert labelled["account"].tolist() == [1]


def test_windows_run_oldest_to_newest_and_never_touch_immature_rows(monkeypatch):
    for name in ("LABEL_MATURITY_DAYS", "VAL_DAYS", "GATE_DAYS"):
        monkeypatch.setattr(cfg, name, 1)
    table = pd.DataFrame({"step": np.arange(1, 121)})
    train, val, gate = fb.time_windows(table, now_step=120)
    assert train["step"].max() < val["step"].min()
    assert val["step"].max() < gate["step"].min()
    assert gate["step"].max() == 96


def test_a_later_decision_corrects_an_earlier_one(tmp_path):
    warehouse = fb.Warehouse(tmp_path)
    warehouse.add_labels(pd.DataFrame({
        "account_name": ["C1", "C1", "C2"], "verdict": ["mule", "legit", "mule"],
        "reviewed_step": [10, 20, 30], "source": "alert"}))
    assert fb.review_state(warehouse)[0] == {"C2"}


def test_an_alert_stays_open_until_reviewed_and_can_reopen(tmp_path):
    warehouse = fb.Warehouse(tmp_path)
    alert = {"risk_score": 0.9, "n_transactions": 1, "reason": "flagged", "model_version": "v0"}
    warehouse.add_alerts(pd.DataFrame(
        {"account_name": ["C1", "C2"], "alerted_step": [10, 10], **alert}))
    assert fb.review_state(warehouse)[1] == {"C1", "C2"}

    warehouse.add_labels(pd.DataFrame({
        "account_name": ["C1"], "verdict": ["legit"], "reviewed_step": [12], "source": "alert"}))
    assert fb.review_state(warehouse)[1] == {"C2"}

    warehouse.add_alerts(pd.DataFrame({"account_name": ["C1"], "alerted_step": [50], **alert}))
    assert fb.review_state(warehouse)[1] == {"C1", "C2"}


def test_reviews_reject_an_unknown_verdict(enabled, tmp_path):
    reviews = pd.DataFrame({"account_name": ["C1"], "verdict": ["maybe"],
                            "reviewed_step": [1], "source": "alert"})
    with pytest.raises(ValueError, match="maybe"):
        fb.record_reviews(fb.Warehouse(tmp_path), reviews)


def test_audit_comes_from_the_riskiest_unflagged_accounts(monkeypatch):
    monkeypatch.setattr(cfg, "AUDIT_POOL_SHARE", 0.1)
    monkeypatch.setattr(cfg, "AUDIT_RATE", 0.01)
    unflagged = pd.DataFrame({"risk_score": np.linspace(0, 1, 1000)})
    sample = fb.audit_sample(unflagged, n_accounts=1000, seed=1)
    assert len(sample) == 10
    assert sample["risk_score"].min() >= 0.9


def test_the_audit_keeps_running_when_nothing_is_flagged():
    unflagged = pd.DataFrame({"risk_score": np.linspace(0, 1, 50_000)})
    assert len(fb.audit_sample(unflagged, 50_000, seed=1)) > 0


# --- The gate ----------------------------------------------------------------

def comparison(new_hits, new_fp, old_hits=200, old_fp=30, legit=40_000, mules=250):
    def side(fp):
        return {"fp": fp, "tn": legit - fp, "fpr": fp / legit}
    return {"mules": mules, "alerts": old_hits + old_fp, "champion_hits": old_hits,
            "challenger_hits": new_hits, "champion": side(old_fp), "challenger": side(new_fp)}


@pytest.mark.parametrize("case, expected", [
    (comparison(200, 30), True),
    (comparison(197, 30), True),   # 1.5% fewer mules: inside the tolerance.
    (comparison(180, 30), False),  # 10% fewer mules for the same workload.
    (comparison(230, 8), True),    # More mules, far fewer false alarms.
    (comparison(230, 100), False),  # FPR clearly above both the cap and the live model.
    (comparison(230, 34), True),   # FPR a little higher, but not clearly worse.
    (comparison(0, 0, old_hits=0, mules=0), False),  # Nothing to judge by.
])
def test_gate(case, expected):
    assert fb.passes_gate(case) is expected


def test_gate_compares_at_the_same_workload_not_at_each_threshold():
    """A stricter threshold must not make a better ranker look like a worse model."""
    gate = pd.DataFrame({"account": range(6), "is_fraud": [1, 1, 1, 0, 0, 0], "x": range(6)})

    class Scores:
        def __init__(self, values):
            self.values = np.array(values, float)

        def predict_proba(self, frame):
            return np.c_[1 - self.values, self.values]

    champion = {"model": Scores([.9, .8, .1, .7, .2, .2]), "features": ["x"], "threshold": .5}
    challenger = {"model": Scores([.9, .8, .7, .1, .2, .2]), "features": ["x"], "threshold": .85}
    result = fb.compare_on_gate(challenger, champion, gate)
    assert result["challenger"]["tp"] < result["champion"]["tp"]  # Its own threshold is strict.
    assert result["challenger_hits"] == 3 > result["champion_hits"] == 2  # It ranks better.


def test_thin_data_skips_training_instead_of_guessing():
    table = pd.DataFrame({"step": [1, 2, 3], "account": [1, 2, 3], "is_fraud": [0, 0, 0]})
    challenger, result = fb.train_challenger(table, None, now_step=3)
    assert challenger is None
    assert "train window" in result["skipped"]


# --- Running for a long time -------------------------------------------------

def test_history_stays_separate_per_account_past_the_paysim_timespan():
    """A bank's clock runs far past PaySim's 743 hours."""
    late = 100_000
    counter = PastSum(np.array([1, 2]), np.array([late, 5]))
    assert counter.before(np.array([2]), np.array([late + 1]))[0] == 1
    assert counter.before(np.array([1]), np.array([late]))[0] == 0


# --- The full cycle ----------------------------------------------------------

def transactions(rng, first_step, last_step, per_hour, tag):
    """Legitimate traffic: existing recipients, and amounts of a few thousand."""
    n = (last_step - first_step + 1) * per_hour
    transfer = rng.random(n) < 0.5
    balance = rng.lognormal(9, 1, n)
    return pd.DataFrame({
        "step": np.repeat(np.arange(first_step, last_step + 1), per_hour),
        "type": np.where(transfer, "TRANSFER", "CASH_OUT"),
        "amount": rng.lognormal(7, 1, n),
        "nameOrig": [f"C{tag}{i}" for i in range(n)],
        "oldbalanceOrg": balance, "newbalanceOrig": 0.0,
        "nameDest": rng.integers(0, 3000, n).astype(str),
        "oldbalanceDest": rng.lognormal(9, 1, n), "newbalanceDest": 0.0,
    }).assign(nameDest=lambda d: "D" + d["nameDest"])


def mules(rng, first_step, last_step, per_hour, tag, full_drain):
    """Mule transfers. The old pattern drains accounts, and the new one does not."""
    n = (last_step - first_step + 1) * per_hour
    amount = rng.uniform(2e5, 5e5, n)
    return pd.DataFrame({
        "step": np.repeat(np.arange(first_step, last_step + 1), per_hour),
        "type": "TRANSFER", "amount": amount,
        "nameOrig": [f"V{tag}{i}" for i in range(n)],
        "oldbalanceOrg": amount if full_drain else amount * 1.5, "newbalanceOrig": 0.0,
        "nameDest": [f"M{tag}{i}" for i in range(n)],
        "oldbalanceDest": 0.0 if full_drain else rng.lognormal(9, 1, n), "newbalanceDest": 0.0,
    })


def caught(alerts, prefix):
    return alerts["account_name"].str.startswith(prefix).sum()


def test_the_loop_learns_a_new_pattern_the_live_model_misses(enabled, monkeypatch, tmp_path):
    """Mules change tactics. The live model misses them, and a retrain catches them."""
    rng = np.random.default_rng(3)
    monkeypatch.setattr(cfg, "ARTIFACTS", tmp_path / "static")
    monkeypatch.setattr(cfg, "XGB_PARAMS", {**cfg.XGB_PARAMS, "n_estimators": 80})
    for name, value in (("LABEL_MATURITY_DAYS", 1), ("VAL_DAYS", 2), ("GATE_DAYS", 2),
                        ("RETRAIN_EVERY_DAYS", 1), ("MIN_NEW_CONFIRMED_MULES", 5)):
        monkeypatch.setattr(cfg, name, value)
    (tmp_path / "static").mkdir()

    # Week 1: the old pattern only. Train the live model on it, as the static system did.
    old = pd.concat([transactions(rng, 1, 96, 120, "a"), mules(rng, 1, 96, 1, "a", True)])
    warehouse = fb.Warehouse(tmp_path / "warehouse")
    fb.ingest(warehouse, old)
    df, names = load_clean(warehouse.transactions)
    table = build_features(df, fb.scorable_rows(df), len(names))
    table["is_fraud"] = pd.Index(names).take(table["account"]).str.startswith("M").astype(np.int8)
    features = cfg.TRANSACTION + cfg.BALANCE
    model = fit(table[table["step"] <= 72], table[table["step"] > 72], features)
    static_path = cfg.ARTIFACTS / f"model_{cfg.PRODUCTION_VARIANT}.joblib"
    joblib.dump({"model": model, "features": features, "threshold": 0.5}, static_path)
    fingerprint = hashlib.sha256(static_path.read_bytes()).hexdigest()
    warehouse.save_registry({**warehouse.registry(), "last_scored_step": 96})

    # The pattern changes. The live model scores it, and misses it.
    new = pd.concat([transactions(rng, 97, 336, 120, "b"), mules(rng, 97, 336, 1, "b", False)])
    fb.ingest(warehouse, new)
    alerts = fb.score_new_transactions(warehouse)
    missed_before = 240 - caught(alerts[alerts["reason"] == "flagged"], "Mb")
    assert missed_before > 200

    # Analysts clear the false alarms, and victims report the mules it missed.
    alerted = alerts["account_name"]
    reviews = pd.concat([
        pd.DataFrame({"account_name": alerted, "source": "alert"}),
        pd.DataFrame({"account_name": [f"Mb{i}" for i in range(240)], "source": "customer_report"}),
    ]).drop_duplicates("account_name")
    reviews["verdict"] = np.where(reviews["account_name"].str.startswith("M"), "mule", "legit")
    reviews["reviewed_step"] = 336
    fb.record_reviews(warehouse, reviews)

    result = fb.retrain(warehouse)
    assert result["promoted"], result
    assert warehouse.registry()["live"] == "v1"
    assert warehouse.registry()["versions"][1]["permission_ref"] == "TEST-PERMISSION-1"

    # The same tactic now gets caught, and the static model file was never touched.
    fb.ingest(warehouse, pd.concat(
        [transactions(rng, 337, 360, 120, "c"), mules(rng, 337, 360, 1, "c", False)]))
    flagged = fb.score_new_transactions(warehouse).query("reason == 'flagged'")
    assert caught(flagged, "Mc") >= 20
    assert hashlib.sha256(static_path.read_bytes()).hexdigest() == fingerprint

    # A bad model can be taken out of service.
    assert fb.rollback(warehouse) == "v0"
    assert warehouse.registry()["live"] == "v0"
    with pytest.raises(ValueError, match="nothing to roll back"):
        fb.rollback(warehouse)
