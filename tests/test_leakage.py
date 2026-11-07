"""Leakage guards. Each test fails if the matching kind of leak is introduced."""
import numpy as np
import pandas as pd
import pytest

from mule import config as cfg
from mule.behavior_layer import behavior_features
from mule.data import TYPE_CODE, assign_split, split_steps
from mule.features import scored_rows
from mule.graph_layer import PastSum, graph_features, money_edges

N_ACCOUNTS = 300


def log(rows):
    """Build a transaction log from (step, type, orig, dest, amount) tuples."""
    df = pd.DataFrame(rows, columns=["step", "type", "orig", "dest", "amount"])
    df["type"] = df["type"].map(TYPE_CODE).astype(np.int8)
    df["orig"] = df["orig"].astype(np.int32)
    df["dest"] = df["dest"].astype(np.int32)
    df["amount"] = df["amount"].astype(float)
    df["step"] = df["step"].astype(np.int16)
    df["old_orig"] = 100.0
    df["new_orig"] = 0.0
    df["old_dest"] = 0.0
    df["new_dest"] = 0.0
    df["is_fraud"] = np.int8(0)
    df["is_flagged"] = np.int8(0)
    return df.sort_values("step", kind="stable").reset_index(drop=True)


@pytest.fixture
def random_log():
    rng = np.random.default_rng(0)
    n = 6000
    return log(zip(
        np.sort(rng.integers(1, 121, n)),
        rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN"], n),
        rng.integers(0, N_ACCOUNTS, n),
        rng.integers(0, N_ACCOUNTS, n),
        rng.uniform(1, 1000, n),
    ))


def scored(df):
    df = df.copy()
    df.loc[df.index[:5], "is_fraud"] = 1
    df.loc[df.index[:5], "type"] = TYPE_CODE["TRANSFER"]
    df.loc[df.index[5:10], "is_fraud"] = 1
    df.loc[df.index[5:10], "type"] = TYPE_CODE["CASH_OUT"]
    return df, scored_rows(df, (120, 120, 120))


def test_future_rows_cannot_change_past_features(random_log):
    """The core check: deleting the future must not move any past feature."""
    df, rows = scored(random_log)
    full = graph_features(df, rows, N_ACCOUNTS)

    cutoff = 70
    past_df = df[df["step"] <= cutoff]
    past_rows = rows[rows["step"] <= cutoff]
    truncated = graph_features(past_df, past_rows, N_ACCOUNTS)

    pd.testing.assert_frame_equal(full.loc[past_rows.index], truncated)


def test_changing_the_future_cannot_change_past_features(random_log):
    df, rows = scored(random_log)
    full = graph_features(df, rows, N_ACCOUNTS)

    altered = df.copy()
    future = altered["step"] > 70
    altered.loc[future, "amount"] *= 1000
    altered.loc[future, "dest"] = np.int32(0)
    changed = graph_features(altered, rows, N_ACCOUNTS)

    past = rows["step"] <= 70
    pd.testing.assert_frame_equal(full[past], changed[past])


def test_labels_do_not_reach_graph_features(random_log):
    df, rows = scored(random_log)
    flipped = df.assign(is_fraud=1 - df["is_fraud"], is_flagged=1 - df["is_flagged"])
    pd.testing.assert_frame_equal(
        graph_features(df, rows, N_ACCOUNTS), graph_features(flipped, rows, N_ACCOUNTS))


def test_rows_in_the_same_hour_cannot_see_each_other():
    df = log([(5, "TRANSFER", 1, 9, 10), (5, "TRANSFER", 2, 9, 10), (6, "TRANSFER", 3, 9, 10)])
    rows = scored_rows(df.assign(is_fraud=np.int8(1)), (6, 6, 6))
    features = graph_features(df, rows, 20)
    assert features["acct_in_degree"].tolist() == [0, 0, 2]
    assert features["acct_age_hours"].tolist() == [-1, -1, 1]


def test_snapshot_uses_finished_days_only():
    df = log([(24, "TRANSFER", 1, 9, 10), (25, "TRANSFER", 2, 9, 10),
              (30, "TRANSFER", 3, 9, 10), (49, "TRANSFER", 4, 9, 10)])
    rows = scored_rows(df.assign(is_fraud=np.int8(1)), (49, 49, 49))
    size = graph_features(df, rows, 20)["acct_component_size"].tolist()
    # Step 24 is day 0, with no finished day. Steps 25 and 30 see only step 24
    # (accounts 1 and 9). Step 49 sees steps 24, 25 and 30 (accounts 1, 2, 3 and 9).
    assert size == [0, 2, 2, 4]


def test_past_sum_window():
    counter = PastSum(np.array([1, 1, 1, 2]), np.array([1, 5, 9, 5]))
    assert counter.before(np.array([1]), np.array([9]))[0] == 2
    assert counter.before(np.array([1]), np.array([10]))[0] == 3
    assert counter.before(np.array([1]), np.array([10]), window=5)[0] == 2  # Steps 5 to 9.
    assert counter.before(np.array([1]), np.array([10]), window=4)[0] == 1  # Steps 6 to 9.
    assert counter.before(np.array([3]), np.array([10]))[0] == 0


def test_cash_in_edges_follow_the_money():
    source, target = money_edges(log([(1, "CASH_IN", 1, 2, 10), (1, "TRANSFER", 3, 4, 10)]))
    assert source.tolist() == [2, 3]
    assert target.tolist() == [1, 4]


def test_behavior_features_ignore_post_transaction_columns(random_log):
    blanked = random_log.assign(new_orig=np.nan, new_dest=np.nan)
    pd.testing.assert_frame_equal(behavior_features(random_log), behavior_features(blanked))


def test_production_models_exclude_forbidden_features():
    for name, features in cfg.VARIANTS.items():
        if name != "leaky_demo":
            assert not set(features) & set(cfg.POST_TX_FORBIDDEN), name
    assert "step" not in cfg.VARIANTS[cfg.PRODUCTION_VARIANT]


def test_drop_ratio_hides_the_exact_drain_rule():
    """A legit overdraft and a fraud full drain must look the same to this feature."""
    df = log([(1, "TRANSFER", 1, 2, 100), (1, "TRANSFER", 3, 4, 500)])
    assert behavior_features(df)["balance_drop_ratio"].tolist() == [1.0, 1.0]


def test_mule_side_is_never_the_victim():
    df = log([(1, "TRANSFER", 1, 2, 10), (1, "CASH_OUT", 3, 4, 10)]).assign(is_fraud=np.int8(1))
    rows = scored_rows(df, (1, 1, 1))
    assert rows["account"].tolist() == [2, 3]  # Mule that receives, mule that cashes out.
    assert rows["counterparty"].tolist() == [1, 4]  # Victim, then cash-out agent.


def test_scored_types_come_from_train_only():
    """A fraud type seen only after training cannot widen the scored population."""
    df = log([(1, "TRANSFER", 1, 2, 10), (9, "PAYMENT", 3, 4, 10)]).assign(is_fraud=np.int8(1))
    rows = scored_rows(df, (5, 7, 9))
    assert rows["type"].unique().tolist() == [TYPE_CODE["TRANSFER"]]


def test_split_is_chronological_and_excludes_the_collapsed_period():
    busy = np.repeat(np.arange(1, 241), 100)  # Ten full-volume days.
    quiet = np.repeat(np.arange(241, 481), 5)  # Volume collapses after that.
    step = np.concatenate([busy, quiet])

    train_end, val_end, test_end = split_steps(step)
    assert train_end < val_end < test_end == 240

    split = assign_split(step, (train_end, val_end, test_end))
    assert step[split == "train"].max() < step[split == "val"].min()
    assert step[split == "val"].max() < step[split == "test"].min()
    assert step[split == "test"].max() < step[split == "stress"].min()
    for name in ("train", "val", "test", "stress"):
        assert (split == name).any()
    # A step never straddles two splits.
    assert pd.DataFrame({"s": step, "p": split}).groupby("s")["p"].nunique().max() == 1
