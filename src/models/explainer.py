"""SHAP TreeExplainer for LightGBM model explanations."""

import logging
import warnings
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def build_explainer(model: lgb.Booster) -> shap.TreeExplainer:
    """Create a SHAP TreeExplainer for a LightGBM booster.

    Args:
        model: Trained lgb.Booster.

    Returns:
        shap.TreeExplainer instance.
    """
    return shap.TreeExplainer(model)


def explain_batch(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
) -> np.ndarray:
    """Compute SHAP values for a batch of samples.

    For binary LightGBM classifiers, returns the SHAP values for the positive
    class with shape (n_samples, n_features).

    Args:
        explainer: SHAP TreeExplainer.
        X: Feature matrix (n_samples, n_features).

    Returns:
        ndarray of shape (n_samples, n_features).
    """
    # SHAP 0.46+ emits a cosmetic warning about output format changes for LightGBM
    # binary classifiers. The code handles both list and ndarray outputs correctly.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="LightGBM binary classifier with TreeExplainer",
            category=UserWarning,
        )
        raw = explainer.shap_values(X)
    # LightGBM binary classification may return a list [neg_class, pos_class]
    # in some SHAP versions; take the positive class values.
    if isinstance(raw, list):
        return np.array(raw[1])
    return np.array(raw)


def top_k_features(
    shap_values_row: np.ndarray,
    feature_names: list[str],
    feature_values: np.ndarray,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Return the k features with the largest absolute SHAP values.

    Args:
        shap_values_row: 1-D SHAP values array for a single sample.
        feature_names: List of feature name strings.
        feature_values: 1-D array of the raw feature values for this sample.
        k: Number of top features to return.

    Returns:
        List of k dicts sorted by abs(shap_value) descending, each containing:
        - feature (str)
        - shap_value (float)
        - feature_value (float)
    """
    if len(shap_values_row) != len(feature_names):
        raise ValueError(
            f"shap_values_row length {len(shap_values_row)} does not match "
            f"feature_names length {len(feature_names)}."
        )
    indices = np.argsort(np.abs(shap_values_row))[::-1][:k]
    return [
        {
            "feature": feature_names[i],
            "shap_value": float(shap_values_row[i]),
            "feature_value": float(feature_values[i]),
        }
        for i in indices
    ]
