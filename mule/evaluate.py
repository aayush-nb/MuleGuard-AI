"""Turn transaction scores into account decisions, and measure them."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve

from . import config as cfg


def account_rollup(frame, scores):
    """Collapse scored rows into one row per mule-side account.

    An account's risk is its single riskiest transaction, and it counts as a mule
    if any of its mule-side transactions was fraud.
    """
    rolled = pd.DataFrame({
        "account": frame["account"].to_numpy(),
        "risk_score": scores,
        "is_mule": frame["is_fraud"].to_numpy(),
    }).groupby("account", sort=False).agg(
        risk_score=("risk_score", "max"),
        is_mule=("is_mule", "max"),
        n_transactions=("risk_score", "size"),
    )
    return rolled.reset_index()


def wilson_upper(rate, n, z=cfg.FPR_CONFIDENCE_Z):
    """Upper confidence bound on a rate observed over n trials."""
    denominator = 1 + z**2 / n
    centre = rate + z**2 / (2 * n)
    margin = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2))
    return (centre + margin) / denominator


def wilson_lower(rate, n, z=cfg.FPR_CONFIDENCE_Z):
    """Lower confidence bound on a rate observed over n trials."""
    denominator = 1 + z**2 / n
    centre = rate + z**2 / (2 * n)
    margin = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2))
    return (centre - margin) / denominator


def pick_threshold(y, scores, max_fpr=cfg.MAX_FPR):
    """Return the lowest threshold whose FPR stays under the cap with confidence.

    The lowest such threshold is the one with the highest recall. Using the upper
    bound rather than the observed FPR leaves room for sampling noise, since the
    cap has to hold on data the threshold has never seen.
    """
    fpr, _, thresholds = roc_curve(y, scores, drop_intermediate=False)
    allowed = wilson_upper(fpr, int((y == 0).sum())) < max_fpr
    return float(thresholds[np.flatnonzero(allowed)[-1]])


def decision_metrics(y, flagged, reference_prevalence=None):
    """Confusion-matrix metrics for a set of binary decisions."""
    y, flagged = np.asarray(y).astype(bool), np.asarray(flagged).astype(bool)
    tp, fp = int((y & flagged).sum()), int((~y & flagged).sum())
    fn, tn = int((y & ~flagged).sum()), int((~y & ~flagged).sum())
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    metrics = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": recall,
        "precision": tp / max(tp + fp, 1),
        "fpr": fpr,
        "fpr_upper_95": float(wilson_upper(fpr, max(fp + tn, 1))),
    }
    if reference_prevalence is not None:
        # Precision depends on the fraud rate, so report it at a common rate too.
        p = reference_prevalence
        hits, alarms = recall * p, recall * p + fpr * (1 - p)
        metrics["precision_at_reference_prevalence"] = hits / alarms if alarms else 0.0
    return metrics


def evaluate_split(frame, scores, threshold, reference_prevalence):
    """Report account-level and transaction-level results for one split."""
    accounts = account_rollup(frame, scores)
    y_acct, s_acct = accounts["is_mule"], accounts["risk_score"]
    y_txn = frame["is_fraud"].to_numpy()
    is_transfer = frame["is_transfer"].to_numpy() == 1
    fraud = y_txn == 1
    return {
        "accounts": int(len(accounts)),
        "mules": int(y_acct.sum()),
        "mule_rate": float(y_acct.mean()),
        "pr_auc_account": float(average_precision_score(y_acct, s_acct)),
        "pr_auc_transaction": float(average_precision_score(y_txn, scores)),
        "account": decision_metrics(y_acct, s_acct >= threshold, reference_prevalence),
        "transaction": decision_metrics(y_txn, scores >= threshold),
        "recall_transfer_mules": float((scores[fraud & is_transfer] >= threshold).mean()),
        "recall_cashout_mules": float((scores[fraud & ~is_transfer] >= threshold).mean()),
    }


def baseline_metrics(frame, column, reference_prevalence):
    """Measure a yes/no rule the same way as the model, at account level."""
    accounts = account_rollup(frame, frame[column].to_numpy().astype(float))
    return decision_metrics(accounts["is_mule"], accounts["risk_score"] > 0, reference_prevalence)
