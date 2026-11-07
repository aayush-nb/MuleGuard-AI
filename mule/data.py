"""Load, clean and split the PaySim transaction log."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from . import config as cfg

EXPECTED_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]
TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
TYPE_CODE = {name: code for code, name in enumerate(TYPES)}


def load_clean(source=cfg.RAW_CSV):
    """Return (transactions, account_names), with accounts encoded as integers.

    `source` is a CSV file or a folder of parquet batches.
    Every cleaning rule depends only on the data, never on the label.
    """
    table = pq.read_table(source) if Path(source).is_dir() else pacsv.read_csv(source)
    if table.column_names != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected schema: {table.column_names}")
    nulls = {c: table[c].null_count for c in EXPECTED_COLUMNS if table[c].null_count}
    if nulls:
        raise ValueError(f"Unexpected nulls: {nulls}")

    # One shared dictionary, so an account keeps its id in either role.
    names = pa.chunked_array(table["nameOrig"].chunks + table["nameDest"].chunks)
    encoded = names.combine_chunks().dictionary_encode()
    ids = encoded.indices.to_numpy(zero_copy_only=False).astype(np.int32)
    n = table.num_rows

    df = pd.DataFrame({
        "step": table["step"].to_numpy().astype(np.int32),
        "type": table["type"].to_pandas().map(TYPE_CODE).to_numpy(np.int8),
        "amount": table["amount"].to_numpy(),
        "orig": ids[:n],
        "dest": ids[n:],
        "old_orig": table["oldbalanceOrg"].to_numpy(),
        "new_orig": table["newbalanceOrig"].to_numpy(),
        "old_dest": table["oldbalanceDest"].to_numpy(),
        "new_dest": table["newbalanceDest"].to_numpy(),
        "is_fraud": table["isFraud"].to_numpy().astype(np.int8),
        "is_flagged": table["isFlaggedFraud"].to_numpy().astype(np.int8),
    })
    account_names = encoded.dictionary.to_pandas()
    del table, names, encoded, ids

    before = len(df)
    df = df[df["amount"] > 0]  # A zero-amount row moved no money.
    df = df.drop_duplicates()
    print(f"cleaned: {before:,} -> {len(df):,} rows")
    return df.sort_values("step", kind="stable").reset_index(drop=True), account_names


def split_steps(step):
    """Return (train_end, val_end, test_end): the last step of each split.

    The split is chronological, so the model only ever predicts the future. Cuts
    fall on whole steps, because rows within one hour have no reliable order.

    Legitimate volume collapses late in the simulated month while fraud stays
    flat, so the daily fraud rate jumps by more than 10x. Cutting across that
    break would give each split a different fraud ratio, so train/val/test all
    sit inside the normal-volume period and the rest is a separate stress test.
    The break is detected from row volume alone, without looking at labels.
    """
    per_step = np.bincount(step)
    n_days = -(-(len(per_step) - 1) // cfg.HOURS_PER_DAY)
    padded = np.zeros(n_days * cfg.HOURS_PER_DAY)
    padded[: len(per_step) - 1] = per_step[1:]
    per_day = padded.reshape(n_days, cfg.HOURS_PER_DAY).sum(axis=1)
    normal_days = np.flatnonzero(per_day >= cfg.NORMAL_VOLUME_SHARE * per_day.max())
    test_end = int((normal_days[-1] + 1) * cfg.HOURS_PER_DAY)

    cumulative = per_step[: test_end + 1].cumsum() / per_step[: test_end + 1].sum()
    train_end = int(np.searchsorted(cumulative, cfg.TRAIN_FRACTION))
    val_end = int(np.searchsorted(cumulative, cfg.TRAIN_FRACTION + cfg.VAL_FRACTION))
    return train_end, val_end, test_end


def assign_split(step, cuts):
    """Label each row train, val, test or stress from its step."""
    train_end, val_end, test_end = cuts
    return np.select(
        [step <= train_end, step <= val_end, step <= test_end],
        ["train", "val", "test"],
        default="stress",
    )
