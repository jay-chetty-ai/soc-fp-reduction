"""CLI entry point for Stage 1 LightGBM training with Optuna hyperparameter tuning.

Usage:
    python scripts/train_stage1.py [--config config.yaml] [--skip-tuning]
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.features import (
    add_temporal_features,
    clean_features,
    encode_labels,
    get_feature_columns,
    per_day_stratified_split,
)
from src.data.loader import load_dataset, validate_schema
from src.models.classifier import evaluate, predict_proba, save_model, train, tune
from src.models.conformal import fit_conformal, save_conformal, compute_coverage
from src.models.explainer import build_explainer, explain_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _print_summary(results: dict, split_label: str = "per-label stratified test") -> None:
    width = 52
    print("\n" + "=" * width)
    print(" Stage 1 Evaluation Summary")
    print(f" Evaluation split: {split_label}")
    print("=" * width)
    print(f"  PR-AUC (target >= 0.85)  : {results['pr_auc']:.4f}")
    print(f"  Precision                : {results['precision']:.4f}")
    print(f"  Recall (target >= 0.95)  : {results['recall']:.4f}")
    print(f"  F1                       : {results['f1']:.4f}")
    cm = results["confusion_matrix"]
    print(f"  Confusion matrix: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
    print("=" * width + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 1 LightGBM classifier.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip Optuna and use LightGBM defaults (for quick smoke tests).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info("Loading dataset...")
    df = load_dataset(config)
    validate_schema(df)

    logger.info("Engineering features...")
    df = clean_features(df)
    df = add_temporal_features(df)

    logger.info("Splitting dataset using per-label stratified split (70/15/15)...")
    train_df, val_df, test_df = per_day_stratified_split(
        df, train_ratio=0.70, val_ratio=0.15, random_state=42
    )

    feat_cols = get_feature_columns(train_df)
    X_train = train_df[feat_cols]
    y_train = encode_labels(train_df)
    X_val = val_df[feat_cols]
    y_val = encode_labels(val_df)
    X_test = test_df[feat_cols]
    y_test = encode_labels(test_df)

    logger.info(
        "Dataset: train=%d val=%d test=%d features=%d attack_rate_train=%.2f%%",
        len(X_train),
        len(X_val),
        len(X_test),
        len(feat_cols),
        100.0 * y_train.mean(),
    )

    if args.skip_tuning:
        logger.info("Skipping Optuna tuning; using default parameters.")
        best_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "is_unbalance": config["stage1"]["is_unbalance"],
            "verbose": -1,
            "n_jobs": -1,
            "num_leaves": 63,
            "learning_rate": 0.05,
        }
        best_n_estimators = 200
        study = None
    else:
        logger.info("Running Optuna hyperparameter tuning (n_trials=%d)...", config["tuning"]["n_trials"])
        best_params, best_n_estimators, study = tune(X_train, y_train, config)
        logger.info(
            "Best Optuna trial: PR-AUC=%.4f  n_estimators=%d",
            study.best_value,
            best_n_estimators,
        )
        logger.info("Best params: %s", study.best_params)

    logger.info("Training final model on 70%% training split (n_estimators=%d)...", best_n_estimators)
    model = train(X_train, y_train, config, best_params, best_n_estimators)

    logger.info("Evaluating on per-label stratified test split...")
    results = evaluate(model, X_test, y_test)
    _print_summary(results)

    model_path = Path(config["stage1"]["model_artifact_path"])
    logger.info("Saving model to %s...", model_path)
    save_model(model, model_path)

    # Use the validation split for conformal calibration -- it is genuinely held out
    # from training and covers all attack families (from per_day_stratified_split).
    logger.info("Fitting conformal predictor on validation split (%d rows)...", len(X_val))
    conformal = fit_conformal(model, X_val, y_val, alpha=config["conformal"]["alpha"])
    coverage = compute_coverage(conformal, X_test, y_test)
    logger.info(
        "Conformal empirical coverage on test set: %.4f (target >= %.2f)",
        coverage,
        1 - config["conformal"]["alpha"],
    )

    conformal_path = Path(config["conformal"]["artifact_path"])
    logger.info("Saving conformal predictor to %s...", conformal_path)
    save_conformal(conformal, conformal_path)

    logger.info("Generating SHAP explanations on test set sample (100 rows)...")
    explainer = build_explainer(model)
    sample = X_test.head(100)
    shap_values = explain_batch(explainer, sample)
    logger.info("SHAP values computed: shape=%s", shap_values.shape)

    if results["pr_auc"] < config["targets"]["pr_auc"]:
        logger.error(
            "PR-AUC %.4f is below target %.2f. Review features and tuning.",
            results["pr_auc"],
            config["targets"]["pr_auc"],
        )
        sys.exit(1)
    if results["recall"] < config["targets"]["tp_recall"]:
        logger.error(
            "Recall %.4f is below target %.2f.",
            results["recall"],
            config["targets"]["tp_recall"],
        )
        sys.exit(1)
    logger.info("All metric targets met. Training complete.")


if __name__ == "__main__":
    main()
