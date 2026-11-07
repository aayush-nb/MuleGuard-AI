"""The threshold and account roll-up decide who gets flagged, so test them directly."""
import numpy as np
import pandas as pd

from mule.evaluate import account_rollup, decision_metrics, pick_threshold, wilson_upper


def test_threshold_keeps_fpr_under_the_cap_with_confidence():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(200_000, int), np.ones(500, int)]
    scores = np.r_[rng.beta(1, 20, 200_000), rng.beta(8, 2, 500)]

    threshold = pick_threshold(y, scores, max_fpr=0.001)
    result = decision_metrics(y, scores >= threshold)

    assert result["fpr_upper_95"] < 0.001
    assert result["recall"] > 0.5


def test_threshold_gives_up_recall_rather_than_break_the_cap():
    """If frauds hide among the legit scores, the cap wins."""
    y = np.r_[np.zeros(10_000, int), np.ones(100, int)]
    scores = np.linspace(0, 1, 10_100)  # Frauds sit at the top, but legit rows overlap.
    scores[-100:] = np.linspace(0, 1, 100)

    threshold = pick_threshold(y, scores, max_fpr=0.001)
    assert decision_metrics(y, scores >= threshold)["fpr"] < 0.001


def test_upper_bound_sits_above_the_observed_rate_and_tightens_with_data():
    assert wilson_upper(0.001, 10_000) > 0.001
    assert wilson_upper(0.001, 1_000_000) < wilson_upper(0.001, 10_000)


def test_account_takes_its_riskiest_transaction_and_any_fraud_makes_it_a_mule():
    frame = pd.DataFrame({"account": [7, 7, 8], "is_fraud": [0, 1, 0]})
    rolled = account_rollup(frame, np.array([0.2, 0.9, 0.4])).set_index("account")
    assert rolled.loc[7, "risk_score"] == 0.9
    assert rolled.loc[7, "is_mule"] == 1
    assert rolled.loc[7, "n_transactions"] == 2
    assert rolled.loc[8, "is_mule"] == 0


def test_precision_is_rescaled_to_the_reference_fraud_rate():
    y = np.r_[np.ones(100, int), np.zeros(100, int)]
    flagged = np.r_[np.ones(80, bool), np.zeros(20, bool), np.ones(10, bool), np.zeros(90, bool)]
    result = decision_metrics(y, flagged, reference_prevalence=0.01)
    assert result["precision"] == 80 / 90
    # At a 1% fraud rate the same 80% recall and 10% FPR give far lower precision.
    assert abs(result["precision_at_reference_prevalence"] - 0.008 / (0.008 + 0.099)) < 1e-9
