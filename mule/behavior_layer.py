"""Pass 1 (behavior): the math of a single transaction.

Balance drops are derived from the balance before the transaction plus the amount,
so they never touch the post-transaction columns.
"""
import numpy as np
import pandas as pd

from .data import TYPE_CODE


def behavior_features(df):
    amount = df["amount"].to_numpy()
    orig_balance = df["old_orig"].to_numpy()
    dest_balance = df["old_dest"].to_numpy()
    step = df["step"].to_numpy()

    # Capped at 1: a balance cannot drop by more than it holds. An uncapped ratio
    # would expose the simulator's exact amount == balance fraud rule.
    drop = np.minimum(amount, orig_balance)
    drop_ratio = np.divide(drop, orig_balance, out=np.zeros_like(amount), where=orig_balance > 0)

    return pd.DataFrame({
        "log_amount": np.log1p(amount),
        "hour_of_day": step % 24,
        "is_transfer": df["type"].to_numpy() == TYPE_CODE["TRANSFER"],
        "log_orig_balance": np.log1p(orig_balance),
        "log_dest_balance": np.log1p(dest_balance),
        "balance_drop_ratio": drop_ratio,
        "inflow_share_of_dest": amount / (dest_balance + amount),
    }, index=df.index).astype(np.float32)


def post_transaction_features(df):
    """Forbidden features, built only for the leaky_demo variant."""
    amount = df["amount"].to_numpy()
    return pd.DataFrame({
        "log_orig_balance_after": np.log1p(df["new_orig"].to_numpy()),
        "log_dest_balance_after": np.log1p(df["new_dest"].to_numpy()),
        "orig_ledger_error": df["old_orig"].to_numpy() - amount - df["new_orig"].to_numpy(),
        "dest_ledger_error": df["old_dest"].to_numpy() + amount - df["new_dest"].to_numpy(),
    }, index=df.index).astype(np.float32)
