"""Every tunable decision in the system lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

SEED = 42

# --- Self-learning toggle ----------------------------------------------------
# False: the static system only. Nothing below in "Self-learning" takes effect.
# True: the feedback loop in run_feedback_loop.py runs. Only switch this on once
# deployed with a bank that has given permission to read live transaction data.
SELF_LEARNING_ENABLED = False
# Reference of that signed permission, for example "ACME-DPA-2027-014". The loop
# refuses to run without one, and stamps it on every model trained from live data.
BANK_DATA_PERMISSION_REF = ""

# --- Split -------------------------------------------------------------------
# Chronological 70/15/15 inside the normal-volume period (see data.split_steps).
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
# A day counts as normal volume if it reaches this share of the busiest day.
NORMAL_VOLUME_SHARE = 0.50

# --- Graph layer -------------------------------------------------------------
HOURS_PER_DAY = 24
VELOCITY_WINDOW_HOURS = 24
PAGERANK_DAMPING = 0.85
PAGERANK_ITERATIONS = 20

# --- Operating point ---------------------------------------------------------
# Business rule: maximise recall while the false positive rate stays under 0.1%.
MAX_FPR = 0.001
# One-sided 95% confidence on validation, so the cap is not met by luck alone.
FPR_CONFIDENCE_Z = 1.645

# --- Self-learning -----------------------------------------------------------
FEEDBACK_DIR = ROOT / "feedback"  # Local stand-in for the bank's data warehouse.
SCORED_TYPES = ["TRANSFER", "CASH_OUT"]
RETRAIN_EVERY_DAYS = 7  # Use 30 for monthly.
MIN_NEW_CONFIRMED_MULES = 50  # Below this, a retrain has nothing new to learn from.
# Rows younger than this are left out of training, because a mule may not have
# been caught yet. The gate and validation windows sit directly before that.
LABEL_MATURITY_DAYS = 7
VAL_DAYS = 7
GATE_DAYS = 7
MIN_WINDOW_MULES = 10  # A window with fewer mules cannot judge a model.
# Given as many alerts as the live model, a new model may catch at most this share
# fewer confirmed mules on the gate window.
MAX_RECALL_DROP = 0.02
# Accounts sent for review without being flagged, as a share of accounts scored.
# They come from the riskiest unflagged accounts, so the model hears about misses.
AUDIT_RATE = 0.0001
AUDIT_POOL_SHARE = 0.01

# --- XGBoost -----------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.08,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    tree_method="hist",
    eval_metric="aucpr",
    early_stopping_rounds=40,
    random_state=SEED,
    n_jobs=8,
)

# --- Feature groups ----------------------------------------------------------
# "Account" is the mule-side account of a row: nameDest on a TRANSFER (it receives
# the stolen money) and nameOrig on a CASH_OUT (it withdraws it). "cpty" is the
# other party. On a fraud TRANSFER, nameOrig is the victim and must never be flagged.
TRANSACTION = ["log_amount", "hour_of_day", "is_transfer"]

# Pre-transaction values only: known before the money moves.
BALANCE = [
    "log_orig_balance",
    "log_dest_balance",
    "balance_drop_ratio",
    "inflow_share_of_dest",
]

_PER_ACCOUNT = [
    "in_degree",
    "out_degree",
    "unique_counterparties",
    "log_in_flow",
    "log_out_flow",
    "pass_through_ratio",
    "age_hours",
    "in_degree_24h",
    "pagerank",
    "component_size",
]
GRAPH = [f"acct_{f}" for f in _PER_ACCOUNT] + [f"cpty_{f}" for f in _PER_ACCOUNT] + [
    "amount_vs_acct_avg_inflow"
]

# Post-transaction values. Banned from every production model: the dataset notes
# that detected fraud is cancelled, so these columns carry the outcome. They exist
# only so the "leaky_demo" variant can measure how much damage they would do.
POST_TX_FORBIDDEN = [
    "log_orig_balance_after",
    "log_dest_balance_after",
    "orig_ledger_error",
    "dest_ledger_error",
]

# Model variants compared on validation. "combined" is the production model.
VARIANTS = {
    "transaction_only": TRANSACTION,
    "behavior_only": TRANSACTION + BALANCE,
    "no_balance": TRANSACTION + GRAPH,
    "combined": TRANSACTION + BALANCE + GRAPH,
    "leaky_demo": TRANSACTION + BALANCE + GRAPH + POST_TX_FORBIDDEN,
}
PRODUCTION_VARIANT = "combined"
