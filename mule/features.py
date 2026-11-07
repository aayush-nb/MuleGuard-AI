"""Pass 1: run both feature generators and join them into one table."""
import numpy as np
import pandas as pd

from . import config as cfg
from .behavior_layer import behavior_features, post_transaction_features
from .data import TYPE_CODE, assign_split, load_clean, split_steps
from .graph_layer import graph_features

FEATURE_TABLE = cfg.PROCESSED / "features.parquet"
ACCOUNT_NAMES = cfg.PROCESSED / "account_names.parquet"


def mule_side(rows):
    """Name the mule-side account of each row, and its counterparty."""
    rows = rows.copy()
    transfer = rows["type"].to_numpy() == TYPE_CODE["TRANSFER"]
    orig, dest = rows["orig"].to_numpy(), rows["dest"].to_numpy()
    rows["account"] = np.where(transfer, dest, orig)
    rows["counterparty"] = np.where(transfer, orig, dest)
    return rows


def scored_rows(df, cuts):
    """Keep the transaction types that carry fraud in train, and name the mule side.

    The type list comes from the training window only.
    """
    train = df[df["step"] <= cuts[0]]
    fraud_types = np.unique(train.loc[train["is_fraud"] == 1, "type"])
    return mule_side(df[df["type"].isin(fraud_types)])


def build_features(df, scored, n_accounts):
    """Run both Pass 1 layers on `scored`, reading history from the full log `df`."""
    return pd.concat([
        scored[["step", "account", "is_fraud", "is_flagged"]],
        # Baseline only: the simulator's own fraud rule, never a model feature.
        pd.Series(np.isclose(scored["amount"], scored["old_orig"]), index=scored.index,
                  name="drain_rule"),
        behavior_features(scored),
        post_transaction_features(scored),
        graph_features(df, scored, n_accounts),
    ], axis=1)


def build_feature_table():
    df, account_names = load_clean()
    cuts = split_steps(df["step"].to_numpy())
    print(f"split cuts (last step of train, val, test): {cuts}")

    table = build_features(df, scored_rows(df, cuts), len(account_names))
    table["split"] = assign_split(table["step"].to_numpy(), cuts)

    cfg.PROCESSED.mkdir(parents=True, exist_ok=True)
    table.to_parquet(FEATURE_TABLE, index=False)
    account_names.to_frame("name").to_parquet(ACCOUNT_NAMES)
    print(f"feature table: {table.shape[0]:,} rows, {table.shape[1]} columns")
    return table
