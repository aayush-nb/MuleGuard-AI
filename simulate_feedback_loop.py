"""Dry run of the self-learning loop on historical PaySim data.

Replays the hours after validation as if they were live, with simulated analysts
and victims. It reads no live data and changes no live state, so it needs no toggle.

  control  PaySim as it is. Its fraud never changes, so this can only show that
           retraining does no harm.
  evasive  From the first replayed hour, mules stop emptying accounts and leave
           5-40% behind. This change of tactics is invented, and injected to see
           what happens to the one-line rule, the static model and the loop.

PaySim covers one month, so the schedule is compressed from weeks into days.
"""
import json

import joblib
import numpy as np
import pandas as pd

from mule import config as cfg
from mule import report
from mule.evaluate import account_rollup, decision_metrics
from mule.feedback import apply_labels, audit_sample, train_challenger
from mule.features import FEATURE_TABLE
from mule.model import score

BATCH_HOURS = 24
RETRAIN_EVERY_HOURS = 96
REVIEW_DELAY_HOURS = 24
VICTIM_REPORT_RATE = 0.5  # Share of missed mules that a victim reports.
VICTIM_REPORT_DELAY_HOURS = 48
EVASIVE_DRAIN = (0.60, 0.95)  # Share of the balance an evasive mule takes.

RESULTS = cfg.ARTIFACTS / "feedback_simulation.json"
COUNTS = ["tp", "fp", "fn", "tn"]


def load_table():
    columns = cfg.VARIANTS[cfg.PRODUCTION_VARIANT] + ["step", "account", "is_fraud", "drain_rule",
                                                      "split"]
    return pd.read_parquet(FEATURE_TABLE, columns=columns)


def make_evasive(table, first_step):
    """Rewrite fraud rows so mules leave money behind. Graph features do not change."""
    table = table.copy()
    rows = (table["is_fraud"] == 1) & (table["step"] >= first_step)
    drain = np.random.default_rng(cfg.SEED).uniform(*EVASIVE_DRAIN, rows.sum()).astype(np.float32)
    amount = np.expm1(table.loc[rows, "log_amount"].to_numpy())
    table.loc[rows, "log_orig_balance"] = np.log1p(amount / drain)
    table.loc[rows, "balance_drop_ratio"] = drain
    table.loc[rows, "drain_rule"] = False
    return table


def replay(table, first_step):
    """Run the loop hour by hour. Return per-batch results and each retrain's outcome."""
    rng = np.random.default_rng(cfg.SEED)
    static = joblib.load(cfg.ARTIFACTS / f"model_{cfg.PRODUCTION_VARIANT}.joblib")
    live = static

    known = table[(table["step"] < first_step) & (table["is_fraud"] == 1)]
    confirmed_at = dict(zip(known["account"], known["step"]))  # Cases the bank already has.
    review_done_at = {}
    results, retrains = [], []
    last_retrain, last_step = first_step, int(table["step"].max())

    for start in range(first_step, last_step + 1, BATCH_HOURS):
        end = min(start + BATCH_HOURS - 1, last_step)
        batch = table[(table["step"] >= start) & (table["step"] <= end)]

        accounts = account_rollup(batch, score(live["model"], batch, live["features"]))
        flagged = accounts["risk_score"] >= live["threshold"]
        fixed = account_rollup(batch, score(static["model"], batch, static["features"]))
        rule = account_rollup(batch, batch["drain_rule"].to_numpy(float))
        for system, y, decision in (
            ("self_learning", accounts["is_mule"], flagged),
            ("static_model", fixed["is_mule"], fixed["risk_score"] >= static["threshold"]),
            ("one_line_rule", rule["is_mule"], rule["risk_score"] > 0),
        ):
            results.append({"end": end, "system": system, **decision_metrics(y, decision)})

        # Analysts review the flags and audits, and victims report some of the rest.
        reviewed = pd.concat([accounts[flagged],
                              audit_sample(accounts[~flagged], len(accounts), seed=end)])
        done = end + REVIEW_DELAY_HOURS
        review_done_at.update(dict.fromkeys(reviewed["account"].tolist(), done))
        confirmed_at.update(dict.fromkeys(
            reviewed.loc[reviewed["is_mule"] == 1, "account"].tolist(), done))
        missed = accounts[(accounts["is_mule"] == 1)
                          & ~accounts["account"].isin(reviewed["account"])]
        reported = missed[rng.random(len(missed)) < VICTIM_REPORT_RATE]
        confirmed_at.update(dict.fromkeys(
            reported["account"].tolist(), end + VICTIM_REPORT_DELAY_HOURS))

        if end - last_retrain >= RETRAIN_EVERY_HOURS:
            last_retrain = end
            mules = [a for a, step in confirmed_at.items() if step <= end]
            awaiting = [a for a, step in review_done_at.items() if step > end]
            labelled = apply_labels(table[table["step"] <= end], mules, awaiting, end)
            challenger, outcome = train_challenger(labelled, live, end)
            retrains.append({"step": end, **outcome})
            print(f"  retrain at step {end}: {json.dumps(outcome)}", flush=True)
            if outcome["promoted"]:
                live = challenger

    return pd.DataFrame(results), retrains


def rates(frame):
    """Turn summed counts into recall and FPR, per system."""
    totals = frame.groupby("system")[COUNTS].sum()
    return {system: {
        "recall": row["tp"] / max(row["tp"] + row["fn"], 1),
        "precision": row["tp"] / max(row["tp"] + row["fp"], 1),
        "fpr": row["fp"] / max(row["fp"] + row["tn"], 1),
        **{c: int(row[c]) for c in COUNTS},
    } for system, row in totals.iterrows()}


def summarise(results, retrains):
    promoted = [r["step"] for r in retrains if r["promoted"]]
    summary = {"retrains": retrains, "whole_replay": rates(results)}
    if promoted:
        summary["first_promotion_step"] = promoted[0]
        summary["before_first_promotion"] = rates(results[results["end"] <= promoted[0]])
        summary["after_first_promotion"] = rates(results[results["end"] > promoted[0]])
    return summary


def main():
    # Compress the schedule: the bank defaults need more history than one month holds.
    cfg.LABEL_MATURITY_DAYS, cfg.VAL_DAYS, cfg.GATE_DAYS = 1, 2, 2

    table = load_table()
    first_step = int(table.loc[table["split"] == "test", "step"].min())
    scenarios = {"control": table, "evasive": make_evasive(table, first_step)}

    summaries, frames = {}, {}
    for name, scenario in scenarios.items():
        print(f"scenario: {name}", flush=True)
        frames[name], retrains = replay(scenario, first_step)
        summaries[name] = summarise(frames[name], retrains)
        print(json.dumps({k: v for k, v in summaries[name].items() if k != "retrains"}, indent=1),
              flush=True)

    RESULTS.write_text(json.dumps({
        "first_step": first_step, "retrain_every_hours": RETRAIN_EVERY_HOURS,
        "victim_report_rate": VICTIM_REPORT_RATE, "scenarios": summaries,
    }, indent=2))
    report.feedback_chart(frames, summaries, first_step, RETRAIN_EVERY_HOURS,
                          cfg.ARTIFACTS / "feedback_simulation.png")


if __name__ == "__main__":
    main()
