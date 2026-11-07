"""Pass 2: select features on train, then fit the XGBoost decision maker."""
import numpy as np
from xgboost import XGBClassifier

from . import config as cfg

MAX_CORRELATION = 0.95
MIN_GAIN_SHARE = 0.0005
CORRELATION_SAMPLE = 300_000


def select_features(train, candidates):
    """Drop constant, redundant and unused features, reading train rows only."""
    X = train[candidates]
    kept = [c for c in candidates if X[c].nunique(dropna=True) > 1]

    sample = X[kept].sample(min(CORRELATION_SAMPLE, len(X)), random_state=cfg.SEED)
    correlation = sample.corr(method="spearman").abs()
    redundant = set()
    for i, first in enumerate(kept):
        if first not in redundant:
            redundant.update(c for c in kept[i + 1:] if correlation.loc[first, c] > MAX_CORRELATION)
    kept = [c for c in kept if c not in redundant]

    probe = XGBClassifier(**{**cfg.XGB_PARAMS, "n_estimators": 150, "early_stopping_rounds": None})
    probe.fit(train[kept], train["is_fraud"])
    gain = probe.get_booster().get_score(importance_type="total_gain")
    total = sum(gain.values())
    return [c for c in kept if gain.get(c, 0) / total >= MIN_GAIN_SHARE]


def fit(train, val, features):
    """Train on train. Validation only decides when to stop adding trees."""
    model = XGBClassifier(**cfg.XGB_PARAMS)
    model.fit(train[features], train["is_fraud"],
              eval_set=[(val[features], val["is_fraud"])], verbose=False)
    return model


def score(model, frame, features):
    return model.predict_proba(frame[features])[:, 1].astype(np.float64)
