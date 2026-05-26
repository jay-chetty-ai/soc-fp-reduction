"""LightGBM classifier training, Optuna hyperparameter tuning, and evaluation."""

import logging
import pickle
import random
from pathlib import Path
from typing import Any, Optional

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.models.integrity import save_hash, verify_hash

logger = logging.getLogger(__name__)

# Suppress verbose Optuna logs; caller controls INFO/WARNING.
optuna.logging.set_verbosity(optuna.logging.WARNING)

_CHECKSUMS_FILENAME = "checksums.json"


def split_for_calibration(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Hold out a calibration split from the training data for conformal prediction.

    The calibration split is excluded from Optuna CV to prevent leakage into
    hyperparameter selection.

    Args:
        X: Feature matrix.
        y: Binary label series.
        config: Parsed config.yaml.

    Returns:
        (X_cv, y_cv, X_cal, y_cal) where X_cv is used for Optuna CV and
        X_cal is held out for conformal calibration.
    """
    frac = config["tuning"]["calibration_split"]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=42)
    cv_idx, cal_idx = next(sss.split(X, y))
    return (
        X.iloc[cv_idx].reset_index(drop=True),
        y.iloc[cv_idx].reset_index(drop=True),
        X.iloc[cal_idx].reset_index(drop=True),
        y.iloc[cal_idx].reset_index(drop=True),
    )


def _build_lgb_params(trial: optuna.Trial, config: dict[str, Any]) -> dict[str, Any]:
    """Sample LightGBM hyperparameters from Optuna trial."""
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "is_unbalance": config["stage1"]["is_unbalance"],
        "verbose": -1,
        "n_jobs": -1,
        "num_leaves": trial.suggest_int("num_leaves", 31, 512),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 5,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
    }


def _make_convergence_callback(config: dict[str, Any]):
    """Return an Optuna callback that halts the study on a PR-AUC plateau."""
    patience: int = config["tuning"]["convergence_patience"]
    delta: float = config["tuning"]["convergence_delta"]

    def callback(study: optuna.Study, trial: optuna.Trial) -> None:
        """Stop the study when best PR-AUC has not improved within patience window."""
        completed = [t for t in study.trials if t.value is not None]
        if len(completed) < patience:
            return
        recent = completed[-patience:]
        best_in_window = max(t.value for t in recent)
        if study.best_value - best_in_window < delta:
            logger.info(
                "Convergence callback: no improvement > %.4f in last %d trials. "
                "Stopping study.",
                delta,
                patience,
            )
            study.stop()

    return callback


def tune(
    X_cv: pd.DataFrame,
    y_cv: pd.Series,
    config: dict[str, Any],
) -> tuple[dict[str, Any], int, optuna.Study]:
    """Tune LightGBM hyperparameters using Optuna TPE with stratified CV.

    Runs n_trials Optuna trials; each trial performs cv_folds-fold stratified
    cross-validation with early stopping per fold. Halts early when the best
    PR-AUC has not improved by convergence_delta in convergence_patience trials.

    Args:
        X_cv: Feature matrix for cross-validation (calibration rows excluded).
        y_cv: Binary labels for cross-validation.
        config: Parsed config.yaml.

    Returns:
        (best_params, best_n_estimators, study) where best_n_estimators is
        the mean best_iteration across CV folds of the best trial.
    """
    n_trials: int = config["tuning"]["n_trials"]
    n_folds: int = config["tuning"]["cv_folds"]
    early_stop: int = config["stage1"]["early_stopping_rounds"]
    n_est_ceiling: int = config["tuning"]["n_estimators_ceiling"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    best_iterations_by_trial: dict[int, list[int]] = {}

    def objective(trial: optuna.Trial) -> float:
        """Run one Optuna trial: k-fold CV with early stopping, return mean PR-AUC."""
        params = _build_lgb_params(trial, config)
        fold_scores: list[float] = []
        fold_iterations: list[int] = []
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
            X_tr, X_val = X_cv.iloc[tr_idx], X_cv.iloc[val_idx]
            y_tr, y_val = y_cv.iloc[tr_idx], y_cv.iloc[val_idx]
            ds_tr = lgb.Dataset(X_tr, label=y_tr)
            ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr)
            booster = lgb.train(
                params,
                ds_tr,
                num_boost_round=n_est_ceiling,
                valid_sets=[ds_val],
                callbacks=[
                    lgb.early_stopping(early_stop, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            proba = booster.predict(X_val)
            score = average_precision_score(y_val, proba)
            fold_scores.append(score)
            fold_iterations.append(booster.best_iteration)
        best_iterations_by_trial[trial.number] = fold_iterations
        mean_score = float(np.mean(fold_scores))
        logger.debug(
            "Trial %d: PR-AUC=%.4f (folds=%s, iters=%s)",
            trial.number,
            mean_score,
            [f"{s:.4f}" for s in fold_scores],
            fold_iterations,
        )
        return mean_score

    study_name = config["tuning"].get("optuna_study_name", "stage1_lgbm_tuning")
    storage = config["tuning"].get("optuna_storage") or None
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    convergence_cb = _make_convergence_callback(config)
    study.optimize(objective, n_trials=n_trials, callbacks=[convergence_cb])

    best_params = study.best_params.copy()
    best_params.update({
        "objective": "binary",
        "metric": "binary_logloss",
        "is_unbalance": config["stage1"]["is_unbalance"],
        "verbose": -1,
        "n_jobs": -1,
        "subsample_freq": 5,
    })
    best_n_estimators = int(
        np.mean(best_iterations_by_trial.get(study.best_trial.number, [100]))
    )
    logger.info(
        "Tuning complete. Best trial=%d, PR-AUC=%.4f, n_estimators=%d, params=%s",
        study.best_trial.number,
        study.best_value,
        best_n_estimators,
        study.best_params,
    )
    return best_params, best_n_estimators, study


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
    best_params: dict[str, Any],
    n_estimators: int,
) -> lgb.Booster:
    """Train the final LightGBM model on the full training dataset.

    Args:
        X_train: Feature matrix (includes CV + calibration data).
        y_train: Binary labels.
        config: Parsed config.yaml.
        best_params: Hyperparameters from tune().
        n_estimators: Fixed number of boosting rounds (no early stopping here).

    Returns:
        Trained lgb.Booster.
    """
    ds = lgb.Dataset(X_train, label=y_train)
    booster = lgb.train(
        best_params,
        ds,
        num_boost_round=max(n_estimators, 1),
        callbacks=[lgb.log_evaluation(period=-1)],
    )
    logger.info("Final model trained with %d estimators.", n_estimators)
    return booster


def evaluate(
    model: lgb.Booster,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate the model on a test set.

    Args:
        model: Trained lgb.Booster.
        X_test: Test feature matrix.
        y_test: True binary labels.
        threshold: Decision threshold for precision/recall/F1 (default 0.5).

    Returns:
        Dict with keys: pr_auc, precision, recall, f1, confusion_matrix.
    """
    proba = model.predict(X_test)
    preds = (proba >= threshold).astype(int)
    pr_auc = float(average_precision_score(y_test, proba))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    cm = confusion_matrix(y_test, preds).tolist()
    results = {
        "pr_auc": pr_auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }
    logger.info(
        "Evaluation: PR-AUC=%.4f precision=%.4f recall=%.4f F1=%.4f",
        pr_auc, prec, rec, f1,
    )
    return results


def predict_proba(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Return P(true_positive) for each row in X.

    Args:
        model: Trained lgb.Booster.
        X: Feature matrix.

    Returns:
        1-D ndarray of probabilities in [0, 1].
    """
    return model.predict(X)


def save_model(
    model: lgb.Booster,
    path: Path,
    checksums_path: Optional[Path] = None,
) -> None:
    """Pickle the model and save its SHA-256 hash for integrity verification.

    Args:
        model: Trained lgb.Booster.
        path: Output path for the pickled model.
        checksums_path: Path for checksums.json (defaults to path.parent/checksums.json).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    cs_path = Path(checksums_path) if checksums_path else path.parent / _CHECKSUMS_FILENAME
    save_hash(path, cs_path)
    logger.info("Model saved to %s.", path)


def load_model(
    path: Path,
    checksums_path: Optional[Path] = None,
) -> lgb.Booster:
    """Load and integrity-verify a pickled LightGBM model.

    Args:
        path: Path to the pickled model file.
        checksums_path: Path for checksums.json (defaults to path.parent/checksums.json).

    Returns:
        Loaded lgb.Booster.

    Raises:
        ModelIntegrityError: If the file's hash does not match the stored hash.
    """
    path = Path(path)
    cs_path = Path(checksums_path) if checksums_path else path.parent / _CHECKSUMS_FILENAME
    verify_hash(path, cs_path)
    with open(path, "rb") as f:
        model: lgb.Booster = pickle.load(f)
    logger.info("Model loaded from %s (integrity verified).", path)
    return model
