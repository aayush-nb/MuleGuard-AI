"""Run the self-learning loop. Off by default: see the toggle in mule/config.py.

    python run_feedback_loop.py status
    python run_feedback_loop.py ingest transactions.csv   Store new transactions
    python run_feedback_loop.py score                     Queue accounts for review
    python run_feedback_loop.py review decisions.csv      Store analysts' decisions
    python run_feedback_loop.py retrain                   Run this weekly or monthly
    python run_feedback_loop.py rollback                  Restore the previous model

transactions.csv uses the PaySim columns, without the two label columns.
decisions.csv has the columns account_name, verdict, reviewed_step and source,
where verdict is mule or legit.
"""
import json
import sys

import pandas as pd

from mule import config as cfg
from mule import feedback


def status():
    state = "ON" if cfg.SELF_LEARNING_ENABLED else "OFF"
    permission = cfg.BANK_DATA_PERMISSION_REF or "none recorded"
    print(f"Self-learning: {state}\nBank data permission: {permission}")
    if cfg.SELF_LEARNING_ENABLED and cfg.BANK_DATA_PERMISSION_REF:
        registry = feedback.Warehouse().registry()
        print(f"Live model: {registry['live']} of {len(registry['versions'])} versions")
    else:
        print("In use: the static system (run_pipeline.py)")


def main(command, argument=None):
    if command == "status":
        return status()
    feedback.require_enabled()
    warehouse = feedback.Warehouse()
    if command == "ingest":
        feedback.ingest(warehouse, pd.read_csv(argument))
    elif command == "score":
        alerts = feedback.score_new_transactions(warehouse)
        print(f"{len(alerts):,} accounts queued for review -> {warehouse.alerts_path}")
    elif command == "review":
        feedback.record_reviews(warehouse, pd.read_csv(argument))
    elif command == "retrain":
        print(json.dumps(feedback.retrain(warehouse), indent=2))
    elif command == "rollback":
        print(f"Live model is now {feedback.rollback(warehouse)}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    try:
        main(*sys.argv[1:3])
    except feedback.SelfLearningOff as off:
        sys.exit(str(off))
