"""Self-learning: analysts' decisions flow back into the model.

    ingest -> score -> analysts review -> labels -> retrain -> gate -> promote

Off by default (see the toggle in config.py). run_pipeline.py never imports this
module, so the static system behaves the same whether or not this exists.

Labels arrive late, and only for accounts someone looked at. That sets the rules:

  * Recent rows stay out of training, because a mule may not have been caught yet.
  * Accounts awaiting review stay out of training, because their truth is unknown.
  * Older unreviewed rows are assumed legitimate. A few will be missed mules.
  * Reviewing only flagged accounts would teach the model nothing about what it
    misses, so audits and customer reports feed in through the same label path.
  * A new model goes live only if it holds up on recent data it never trained on.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as cfg
from .data import EXPECTED_COLUMNS, TYPE_CODE, load_clean
from .evaluate import account_rollup, decision_metrics, pick_threshold, wilson_lower
from .features import build_features, mule_side
from .model import fit, score, select_features

LIVE_COLUMNS = [c for c in EXPECTED_COLUMNS if c not in ("isFraud", "isFlaggedFraud")]
ALERT_COLUMNS = ["account_name", "risk_score", "n_transactions", "reason", "alerted_step",
                 "model_version"]
LABEL_COLUMNS = ["account_name", "verdict", "reviewed_step", "source"]
VERDICTS = {"mule", "legit"}


class SelfLearningOff(RuntimeError):
    """Raised when the loop is asked to run without being switched on."""


def require_enabled():
    if not cfg.SELF_LEARNING_ENABLED:
        raise SelfLearningOff(
            "Self-learning is off, so the static system is in use. Set "
            "SELF_LEARNING_ENABLED in mule/config.py once deployed with a bank.")
    if not cfg.BANK_DATA_PERMISSION_REF:
        raise SelfLearningOff(
            "Self-learning needs permission to read live customer transactions. "
            "Set BANK_DATA_PERMISSION_REF in mule/config.py to the signed permission.")


# --- The rules: plain functions, shared with the dry run ----------------------

def apply_labels(table, mules, open_accounts, now_step):
    """Label rows with what is known today, and drop rows whose truth is unknown."""
    cutoff = now_step - cfg.LABEL_MATURITY_DAYS * cfg.HOURS_PER_DAY
    known = (table["step"] <= cutoff) & ~table["account"].isin(open_accounts)
    labelled = table[known].copy()
    labelled["is_fraud"] = labelled["account"].isin(mules).astype(np.int8)
    return labelled


def time_windows(table, now_step):
    """Split matured rows into train, val and gate, oldest to newest."""
    hours = cfg.HOURS_PER_DAY
    cutoff = now_step - cfg.LABEL_MATURITY_DAYS * hours
    gate_start = cutoff - cfg.GATE_DAYS * hours
    val_start = gate_start - cfg.VAL_DAYS * hours
    step = table["step"]
    return (table[step <= val_start],
            table[(step > val_start) & (step <= gate_start)],
            table[(step > gate_start) & (step <= cutoff)])


def has_enough_data(train, val, gate):
    """Check each window can do its job, or say why not."""
    for name, window in (("train", train), ("val", val), ("gate", gate)):
        mules = window.loc[window["is_fraud"] == 1, "account"].nunique()
        if mules < cfg.MIN_WINDOW_MULES:
            return f"only {mules} confirmed mules in the {name} window"
    # Below this count, even zero false positives cannot show the cap is met.
    needed = int(np.ceil(cfg.FPR_CONFIDENCE_Z**2 / cfg.MAX_FPR))
    legit = val.loc[val["is_fraud"] == 0, "account"].nunique()
    if legit < needed:
        return f"only {legit} legitimate accounts in the val window, {needed} needed"
    return None


def compare_on_gate(challenger, champion, gate):
    """Score both models on the gate window, at the same workload and at their own thresholds.

    Comparing only at their own thresholds would mistake a stricter threshold for a
    worse model, so the main test gives both the live model's number of alerts.
    """
    new = account_rollup(gate, score(challenger["model"], gate, challenger["features"]))
    old = account_rollup(gate, score(champion["model"], gate, champion["features"]))
    alerts = int((old["risk_score"] >= champion["threshold"]).sum())
    return {
        "mules": int(old["is_mule"].sum()),
        "alerts": alerts,
        "champion_hits": int(old.nlargest(alerts, "risk_score")["is_mule"].sum()),
        "challenger_hits": int(new.nlargest(alerts, "risk_score")["is_mule"].sum()),
        "champion": decision_metrics(old["is_mule"], old["risk_score"] >= champion["threshold"]),
        "challenger": decision_metrics(new["is_mule"], new["risk_score"] >= challenger["threshold"]),
    }


def passes_gate(comparison):
    """Decide if a new model may replace the live one.

    Labels mostly come from the live model's own alerts, and the new model's extra
    alerts were never reviewed. So the live model has a home advantage: a mule only
    the new model finds counts against it. That is why the new model only has to be
    not worse, with some tolerance, rather than clearly better.
    """
    catches_as_many = (comparison["challenger_hits"]
                       >= (1 - cfg.MAX_RECALL_DROP) * comparison["champion_hits"])
    new = comparison["challenger"]
    lowest_plausible_fpr = wilson_lower(new["fpr"], max(new["fp"] + new["tn"], 1))
    fpr_not_clearly_worse = lowest_plausible_fpr <= max(cfg.MAX_FPR, comparison["champion"]["fpr"])
    return bool(comparison["mules"] > 0 and catches_as_many and fpr_not_clearly_worse)


def train_challenger(table, champion, now_step):
    """Train a new model on labelled rows and compare it with the live one.

    Returns (challenger, result). The challenger is None if the data was too thin.
    """
    train, val, gate = time_windows(table, now_step)
    problem = has_enough_data(train, val, gate)
    if problem:
        return None, {"promoted": False, "skipped": problem}

    features = select_features(train, cfg.VARIANTS[cfg.PRODUCTION_VARIANT])
    model = fit(train, val, features)
    accounts = account_rollup(val, score(model, val, features))
    threshold = pick_threshold(accounts["is_mule"], accounts["risk_score"])
    challenger = {"model": model, "features": features, "threshold": threshold}

    result = compare_on_gate(challenger, champion, gate)
    result["promoted"] = passes_gate(result)
    return challenger, result


def audit_sample(unflagged, n_accounts, seed):
    """Pick unflagged accounts for review, at random from the riskiest of them."""
    pool_size = max(1, int(len(unflagged) * cfg.AUDIT_POOL_SHARE))
    pool = unflagged.nlargest(pool_size, "risk_score")
    size = min(len(pool), int(np.ceil(n_accounts * cfg.AUDIT_RATE)))
    return pool.sample(size, random_state=seed)


# --- Storage -----------------------------------------------------------------

class Warehouse:
    """Local stand-in for the bank's data warehouse.

    To connect a real one, replace these methods and keep what they return.
    """

    def __init__(self, root=None):
        self.root = Path(root or cfg.FEEDBACK_DIR)
        self.transactions = self.root / "transactions"
        self.models = self.root / "models"
        self.alerts_path = self.root / "alerts.csv"
        self.labels_path = self.root / "labels.csv"
        self.registry_path = self.root / "registry.json"
        self.transactions.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(exist_ok=True)

    def add_transactions(self, batch):
        missing = set(LIVE_COLUMNS) - set(batch.columns)
        if missing:
            raise ValueError(f"Batch is missing columns: {sorted(missing)}")
        # A live feed has no labels. These columns only keep the PaySim schema.
        batch = batch[LIVE_COLUMNS].assign(isFraud=0, isFlaggedFraud=0)[EXPECTED_COLUMNS]
        numeric = {c: "float64" for c in LIVE_COLUMNS if "alance" in c or c == "amount"}
        number = len(list(self.transactions.glob("*.parquet")))
        batch.astype(numeric | {"step": "int64"}).to_parquet(
            self.transactions / f"batch_{number:08d}.parquet", index=False)

    def alerts(self):
        return self._read(self.alerts_path, ALERT_COLUMNS)

    def labels(self):
        return self._read(self.labels_path, LABEL_COLUMNS)

    def add_alerts(self, alerts):
        self._append(self.alerts_path, alerts[ALERT_COLUMNS])

    def add_labels(self, labels):
        self._append(self.labels_path, labels[LABEL_COLUMNS])

    def registry(self):
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text())
        static_model = cfg.ARTIFACTS / f"model_{cfg.PRODUCTION_VARIANT}.joblib"
        return {"live": "v0", "last_scored_step": 0, "last_retrain_step": 0, "versions": [
            {"version": "v0", "path": str(static_model), "promoted": True}]}

    def save_registry(self, registry):
        self.registry_path.write_text(json.dumps(registry, indent=2))

    @staticmethod
    def _read(path, columns):
        return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)

    @staticmethod
    def _append(path, frame):
        frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def load_live_model(registry):
    entry = next(v for v in registry["versions"] if v["version"] == registry["live"])
    return joblib.load(entry["path"])


def review_state(warehouse):
    """Return (confirmed mules, accounts awaiting review), as sets of account names."""
    labels, alerts = warehouse.labels(), warehouse.alerts()
    latest = labels.sort_values("reviewed_step", kind="stable").drop_duplicates(
        "account_name", keep="last")  # A later decision corrects an earlier one.
    mules = set(latest.loc[latest["verdict"] == "mule", "account_name"])

    last_alert = alerts.groupby("account_name")["alerted_step"].max()
    last_review = latest.set_index("account_name")["reviewed_step"].reindex(last_alert.index)
    awaiting = set(last_alert[~(last_review >= last_alert)].index)
    return mules, awaiting


def scorable_rows(df, first_step=0):
    types = [TYPE_CODE[name] for name in cfg.SCORED_TYPES]
    return mule_side(df[df["type"].isin(types) & (df["step"] >= first_step)])


# --- Operations: each one needs the toggle on --------------------------------

def ingest(warehouse, batch):
    """Store new transactions. Feed whole hours, because `step` is the clock."""
    require_enabled()
    warehouse.add_transactions(batch)


def score_new_transactions(warehouse):
    """Score transactions not seen yet, and queue accounts for review."""
    require_enabled()
    registry = warehouse.registry()
    df, names = load_clean(warehouse.transactions)
    names = names.to_numpy()
    scored = scorable_rows(df, registry["last_scored_step"] + 1)
    if scored.empty:
        return warehouse.alerts().iloc[0:0]

    table = build_features(df, scored, len(names))
    bundle = load_live_model(registry)
    accounts = account_rollup(table, score(bundle["model"], table, bundle["features"]))
    accounts["account_name"] = names[accounts["account"].to_numpy()]

    # Confirmed mules are already dealt with, and queued accounts are already queued.
    mules, awaiting = review_state(warehouse)
    n_scored = len(accounts)
    accounts = accounts[~accounts["account_name"].isin(mules | awaiting)]

    now_step = int(df["step"].max())
    over = accounts["risk_score"] >= bundle["threshold"]
    alerts = pd.concat([
        accounts[over].assign(reason="flagged"),
        audit_sample(accounts[~over], n_scored, cfg.SEED + now_step).assign(reason="audit"),
    ]).assign(alerted_step=now_step, model_version=registry["live"])

    warehouse.add_alerts(alerts)
    registry["last_scored_step"] = now_step
    warehouse.save_registry(registry)
    return alerts[ALERT_COLUMNS]


def record_reviews(warehouse, reviews):
    """Store analysts' decisions. This is the feedback.

    Reports from customers or other teams go through here too, with their own
    `source`. They are how the system hears about mules it never flagged.
    """
    require_enabled()
    missing = set(LABEL_COLUMNS) - set(reviews.columns)
    if missing:
        raise ValueError(f"Reviews are missing columns: {sorted(missing)}")
    unknown = set(reviews["verdict"]) - VERDICTS
    if unknown:
        raise ValueError(f"Unknown verdicts: {sorted(unknown)}")
    warehouse.add_labels(reviews)


def retrain(warehouse):
    """Train a new model on confirmed labels, and promote it if it passes the gate.

    Run this on a schedule. It skips itself when it is too soon or nothing is new.
    """
    require_enabled()
    registry = warehouse.registry()
    df, names = load_clean(warehouse.transactions)
    now_step = int(df["step"].max())

    if now_step - registry["last_retrain_step"] < cfg.RETRAIN_EVERY_DAYS * cfg.HOURS_PER_DAY:
        return {"promoted": False, "skipped": "too soon since the last retrain"}
    labels = warehouse.labels()
    new_mules = labels[(labels["verdict"] == "mule")
                       & (labels["reviewed_step"] > registry["last_retrain_step"])]
    if new_mules["account_name"].nunique() < cfg.MIN_NEW_CONFIRMED_MULES:
        return {"promoted": False, "skipped": "too few newly confirmed mules"}

    index = pd.Index(names)
    mules, awaiting = review_state(warehouse)
    table = apply_labels(
        build_features(df, scorable_rows(df), len(names)),
        index.get_indexer(list(mules)), index.get_indexer(list(awaiting)), now_step)

    challenger, result = train_challenger(table, load_live_model(registry), now_step)
    if challenger is None:
        return result

    version = f"v{len(registry['versions'])}"
    path = warehouse.models / f"{version}.joblib"
    joblib.dump(challenger, path)
    registry["versions"].append({
        "version": version, "path": str(path), "promoted": result["promoted"],
        "trained_at_step": now_step, "permission_ref": cfg.BANK_DATA_PERMISSION_REF,
        "gate": {k: v for k, v in result.items() if k != "promoted"},
    })
    registry["last_retrain_step"] = now_step
    if result["promoted"]:
        registry["live"] = version
    warehouse.save_registry(registry)
    return {**result, "version": version}


def rollback(warehouse):
    """Put the previously promoted model back in service."""
    require_enabled()
    registry = warehouse.registry()
    promoted = [v["version"] for v in registry["versions"] if v["promoted"]]
    position = promoted.index(registry["live"])
    if position == 0:
        raise ValueError("Already on the first model, so there is nothing to roll back to.")
    registry["live"] = promoted[position - 1]
    warehouse.save_registry(registry)
    return registry["live"]
