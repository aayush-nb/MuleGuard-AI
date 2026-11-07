"""Run the mule detection system.

    python run_pipeline.py features   Pass 1: build the point-in-time feature table
    python run_pipeline.py train      Pass 2: fit models, set the threshold (train + val only)
    python run_pipeline.py test       Final evaluation on the locked test window
    python run_pipeline.py all        All three, in order

`train` never reads test rows, so features and models can be iterated without
touching the test window.
"""
import json
import sys

import joblib
import pandas as pd

from mule import config as cfg
from mule import report
from mule.evaluate import account_rollup, baseline_metrics, evaluate_split, pick_threshold
from mule.features import ACCOUNT_NAMES, FEATURE_TABLE, build_feature_table
from mule.model import fit, score, select_features

VAL_REPORT = cfg.ARTIFACTS / "validation_report.json"
METRICS = cfg.ARTIFACTS / "metrics.json"
FLAGGED = cfg.ARTIFACTS / "flagged_accounts.csv"


def model_path(variant):
    return cfg.ARTIFACTS / f"model_{variant}.joblib"


def split_summary(table):
    grouped = table.groupby("split")
    summary = pd.DataFrame({
        "first_step": grouped["step"].min(),
        "last_step": grouped["step"].max(),
        "rows": grouped.size(),
        "fraud": grouped["is_fraud"].sum(),
        "fraud_rate": grouped["is_fraud"].mean(),
    }).loc[["train", "val", "test", "stress"]]
    return summary


def train_stage():
    table = pd.read_parquet(FEATURE_TABLE, filters=[("split", "in", ["train", "val"])])
    train, val = table[table["split"] == "train"], table[table["split"] == "val"]
    prevalence = float(account_rollup(train, train["is_fraud"].to_numpy(float))["is_mule"].mean())

    results = {}
    for variant, candidates in cfg.VARIANTS.items():
        features = select_features(train, candidates)
        model = fit(train, val, features)
        scores = score(model, val, features)
        accounts = account_rollup(val, scores)
        threshold = pick_threshold(accounts["is_mule"], accounts["risk_score"])
        joblib.dump({"model": model, "features": features, "threshold": threshold},
                    model_path(variant))
        results[variant] = {
            "threshold": threshold,
            "best_iteration": int(model.best_iteration),
            "features": features,
            "val": evaluate_split(val, scores, threshold, prevalence),
        }
        m = results[variant]["val"]
        print(f"{variant:18s} features={len(features):2d} trees={model.best_iteration:3d} "
              f"PR-AUC={m['pr_auc_account']:.4f} recall={m['account']['recall']:.4f} "
              f"precision={m['account']['precision']:.4f} FPR={m['account']['fpr']:.5%} "
              f"transfer={m['recall_transfer_mules']:.3f} cashout={m['recall_cashout_mules']:.3f}")

    cfg.ARTIFACTS.mkdir(exist_ok=True)
    VAL_REPORT.write_text(json.dumps({"reference_prevalence": prevalence, "variants": results},
                                     indent=2))


def test_stage():
    table = pd.read_parquet(FEATURE_TABLE)
    summary = split_summary(table)
    print(summary.to_string())
    val_report = json.loads(VAL_REPORT.read_text())
    prevalence = val_report["reference_prevalence"]
    frames = {name: table[table["split"] == name] for name in ("val", "test", "stress")}

    results = {}
    for variant in cfg.VARIANTS:
        saved = joblib.load(model_path(variant))
        results[variant] = {"threshold": saved["threshold"], "features": saved["features"]}
        for name, frame in frames.items():
            scores = score(saved["model"], frame, saved["features"])
            results[variant][name] = evaluate_split(frame, scores, saved["threshold"], prevalence)

    baselines = {
        rule: {name: baseline_metrics(frame, column, prevalence) for name, frame in frames.items()}
        for rule, column in {"isFlaggedFraud": "is_flagged", "exact_drain_rule": "drain_rule"}.items()
    }

    production = joblib.load(model_path(cfg.PRODUCTION_VARIANT))
    model, features, threshold = production["model"], production["features"], production["threshold"]
    rollups = {name: account_rollup(frames[name], score(model, frames[name], features))
               for name in ("val", "test")}

    flagged = rollups["test"][rollups["test"]["risk_score"] >= threshold]
    flagged = flagged.sort_values("risk_score", ascending=False)
    names = pd.read_parquet(ACCOUNT_NAMES)["name"].to_numpy()
    flagged.insert(0, "account_name", names[flagged["account"].to_numpy()])
    flagged.drop(columns="account").to_csv(FLAGGED, index=False)

    METRICS.write_text(json.dumps({
        "splits": json.loads(summary.to_json(orient="index")),
        "reference_prevalence": prevalence,
        "max_fpr": cfg.MAX_FPR,
        "production_variant": cfg.PRODUCTION_VARIANT,
        "variants": results,
        "baselines": baselines,
    }, indent=2))

    report.pr_curve_chart({"Validation": rollups["val"], "Test": rollups["test"]}, threshold,
                          cfg.ARTIFACTS / "pr_curve.png")
    report.fraud_rate_chart(table[["step", "is_fraud", "split"]], cfg.ARTIFACTS / "fraud_rate.png")
    report.ranked_bars(
        list(cfg.VARIANTS), [results[v]["test"]["account"]["recall"] for v in cfg.VARIANTS],
        "Mules caught on the test window, threshold set on validation", "Recall, per account",
        cfg.ARTIFACTS / "ablation.png", xlim=(0, 1.12))
    gain = model.get_booster().get_score(importance_type="total_gain")
    top = sorted(gain, key=gain.get, reverse=True)[:12]
    total = sum(gain.values())
    report.ranked_bars(top, [gain[f] / total for f in top], "What the production model relies on",
                       "Share of total gain", cfg.ARTIFACTS / "feature_importance.png")

    for variant in cfg.VARIANTS:
        for name in ("val", "test", "stress"):
            m = results[variant][name]
            a = m["account"]
            print(f"{variant:18s} {name:7s} PR-AUC={m['pr_auc_account']:.4f} "
                  f"recall={a['recall']:.4f} precision={a['precision']:.4f} "
                  f"FPR={a['fpr']:.5%} (upper {a['fpr_upper_95']:.5%}) "
                  f"transfer={m['recall_transfer_mules']:.3f} cashout={m['recall_cashout_mules']:.3f}")
    print(json.dumps(baselines, indent=1))
    print(f"flagged accounts on test: {len(flagged):,} -> {FLAGGED}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage not in {"features", "train", "test", "all"}:
        sys.exit(__doc__)
    if stage in {"features", "all"}:
        build_feature_table()
    if stage in {"train", "all"}:
        train_stage()
    if stage in {"test", "all"}:
        test_stage()
