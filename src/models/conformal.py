"""Conformal prediction and three-band alert routing."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from mapie.classification import SplitConformalClassifier
from mapie.metrics.classification import classification_coverage_score

try:
    import lightgbm as lgb
    _LGB_BOOSTER_TYPE = lgb.Booster
except ImportError:  # pragma: no cover
    lgb = None  # type: ignore[assignment]
    _LGB_BOOSTER_TYPE = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class _BoosterWrapper:
    """Sklearn-compatible wrapper around lgb.Booster.

    SplitConformalClassifier requires predict_proba; lgb.Booster only exposes
    predict() which returns class-1 probabilities.
    """

    def __init__(self, booster: Any) -> None:
        self._booster = booster
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n, 2) probability array: [P(class=0), P(class=1)]."""
        p = self._booster.predict(X)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions at threshold 0.5."""
        return (self._booster.predict(X) >= 0.5).astype(int)


def fit_conformal(
    model: Any,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    alpha: float = 0.05,
) -> SplitConformalClassifier:
    """Fit a SplitConformalClassifier on calibration data.

    Args:
        model: Fitted lgb.Booster or sklearn estimator with predict_proba.
        X_cal: Feature matrix for calibration (must be held out from training).
        y_cal: Binary labels for calibration data.
        alpha: Miscoverage rate; alpha=0.05 gives a 95% coverage guarantee.

    Returns:
        Fitted SplitConformalClassifier ready for predict_bands().
    """
    if _LGB_BOOSTER_TYPE is not None and isinstance(model, _LGB_BOOSTER_TYPE):
        estimator: Any = _BoosterWrapper(model)
    else:
        estimator = model

    clf = SplitConformalClassifier(
        estimator=estimator,
        confidence_level=1.0 - alpha,
        prefit=True,
    )
    clf.conformalize(X_cal, y_cal)
    logger.info(
        "Fitted conformal predictor on %d calibration samples (alpha=%.3f).",
        len(X_cal),
        alpha,
    )
    return clf


def compute_coverage(
    conformal: SplitConformalClassifier,
    X: pd.DataFrame,
    y: np.ndarray,
) -> float:
    """Return empirical marginal coverage: fraction of labels inside prediction set.

    The conformal guarantee is coverage >= 1 - alpha with high probability when
    calibration and test distributions are exchangeable.

    Args:
        conformal: Fitted SplitConformalClassifier from fit_conformal().
        X: Feature matrix.
        y: True binary labels.

    Returns:
        Float in [0, 1].
    """
    _, y_pset = conformal.predict_set(X)
    coverage = float(classification_coverage_score(y, y_pset)[0])
    logger.info("Conformal coverage: %.4f.", coverage)
    return coverage


def predict_bands(
    conformal: SplitConformalClassifier,
    X: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.Series:
    """Assign each alert to one of three routing bands using prediction sets.

    Band logic derived from the conformal prediction set for each sample:
    - auto_fp: prediction set = {BENIGN only}. Model is confident the alert is
      not an attack. False negative rate bounded by alpha (conformal guarantee).
    - auto_tp: prediction set = {attack only}. Model is confident it is an attack.
    - uncertain: prediction set = {BENIGN, attack}. Route to Stage 2 LLM.

    Args:
        conformal: Fitted SplitConformalClassifier from fit_conformal().
        X: Feature matrix (index preserved in returned Series).
        thresholds: Dict with "auto_fp_threshold" and "auto_tp_threshold" keys
            (used only for logging context; routing logic uses prediction sets).

    Returns:
        pd.Series with values in {"auto_fp", "uncertain", "auto_tp"} and the
        same index as X.
    """
    _, y_pset = conformal.predict_set(X)
    benign_in_set: np.ndarray = y_pset[:, 0, 0]
    attack_in_set: np.ndarray = y_pset[:, 1, 0]

    bands = np.where(
        ~attack_in_set & benign_in_set,
        "auto_fp",
        np.where(
            attack_in_set & ~benign_in_set,
            "auto_tp",
            "uncertain",
        ),
    )
    series = pd.Series(bands, index=X.index, name="band")

    dist = series.value_counts().to_dict()
    logger.info(
        "Band distribution (fp_thresh=%.3f, tp_thresh=%.3f): %s",
        thresholds.get("auto_fp_threshold", float("nan")),
        thresholds.get("auto_tp_threshold", float("nan")),
        dist,
    )
    return series
