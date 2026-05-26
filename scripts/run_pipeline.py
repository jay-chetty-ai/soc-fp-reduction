"""Run the full SOC alert triage pipeline against real data.

Usage examples:

    # Process day-5 temporal hold-out (requires full dataset in data/raw/):
    python scripts/run_pipeline.py

    # Process a specific CSV file (e.g. the 10K demo fixture):
    python scripts/run_pipeline.py --input data/fixtures/fixture_10k.csv

    # Skip Stage 2 LLM calls (dry run; uncertain alerts get needs_review):
    python scripts/run_pipeline.py --no-llm

    # Save full results (parquet) and metrics JSON to custom output:
    python scripts/run_pipeline.py --output results/run_$(date +%Y%m%d_%H%M%S).parquet

Prerequisites (run once before this script):
    python scripts/train_stage1.py        # trains model + conformal predictor
    python scripts/build_rag_index.py     # builds FAISS index + saves training_df
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _check_artifacts(config: dict) -> list[str]:
    """Return a list of missing artifact paths so callers can fail fast."""
    required = {
        "Stage 1 model": config["stage1"]["model_artifact_path"],
        "Conformal predictor": config["conformal"]["artifact_path"],
        "FAISS index": config["rag"]["faiss_index_path"],
        "Training DataFrame": config["rag"]["training_df_path"],
    }
    missing = []
    for label, path in required.items():
        if not Path(path).exists():
            missing.append(f"  {label}: {path}")
    return missing


def _compute_metrics(records: list, elapsed: float, llm_enabled: bool) -> dict:
    """Compute evaluation metrics from DispositionRecord list.

    Args:
        records: List of DispositionRecord objects from run_batch().
        elapsed: Wall-clock seconds for the pipeline run.
        llm_enabled: Whether Stage 2 LLM calls were enabled.

    Returns:
        Metrics dict suitable for JSON serialisation and dashboard consumption.
    """
    total = len(records)
    band_counts: dict[str, int] = {}
    verdict_counts: dict[str, int] = {}
    y_true: list[int] = []
    y_score: list[float] = []
    y_pred: list[int] = []

    for r in records:
        band_counts[r.band] = band_counts.get(r.band, 0) + 1
        verdict_counts[r.final_verdict] = verdict_counts.get(r.final_verdict, 0) + 1
        if r.true_label is not None:
            y_true.append(r.true_label)
            y_score.append(r.ml_score)
            y_pred.append(1 if r.final_verdict == "true_positive" else 0)

    auto_fp_n = band_counts.get("auto_fp", 0)
    auto_tp_n = band_counts.get("auto_tp", 0)
    volume_reduction = (auto_fp_n + auto_tp_n) / total if total > 0 else 0.0

    metrics: dict = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_alerts": total,
        "elapsed_seconds": round(elapsed, 2),
        "throughput_per_second": round(total / elapsed, 2) if elapsed > 0 else 0.0,
        "band_counts": band_counts,
        "band_pct": {k: round(100.0 * v / total, 1) for k, v in band_counts.items()},
        "verdict_counts": verdict_counts,
        "volume_reduction": round(volume_reduction, 4),
        # 7 min median analyst triage time per auto-closed FP alert
        "analyst_time_saved_hours": round(auto_fp_n * 7 / 60, 1),
        "llm_enabled": llm_enabled,
    }

    if y_true:
        from sklearn.metrics import (
            average_precision_score,
            confusion_matrix,
            f1_score,
            precision_recall_curve,
            precision_score,
            recall_score,
        )

        y_true_arr = np.array(y_true)
        y_score_arr = np.array(y_score)
        y_pred_arr = np.array(y_pred)

        metrics["pr_auc"] = round(
            float(average_precision_score(y_true_arr, y_score_arr)), 4
        )
        metrics["precision"] = round(
            float(precision_score(y_true_arr, y_pred_arr, zero_division=0)), 4
        )
        metrics["recall"] = round(
            float(recall_score(y_true_arr, y_pred_arr, zero_division=0)), 4
        )
        metrics["f1"] = round(
            float(f1_score(y_true_arr, y_pred_arr, zero_division=0)), 4
        )
        metrics["confusion_matrix"] = confusion_matrix(
            y_true_arr, y_pred_arr
        ).tolist()

        prec_pts, rec_pts, _ = precision_recall_curve(y_true_arr, y_score_arr)
        # Decimate to at most 200 points to keep JSON size manageable
        step = max(1, len(prec_pts) // 200)
        metrics["pr_curve"] = {
            "precision": [round(float(v), 4) for v in prec_pts[::step]],
            "recall": [round(float(v), 4) for v in rec_pts[::step]],
        }

    return metrics


def _print_results(records: list, elapsed: float) -> None:
    """Print a formatted summary table to stdout."""
    total = len(records)
    verdict_counts: dict[str, int] = {}
    band_counts: dict[str, int] = {}
    for r in records:
        verdict_counts[r.final_verdict] = verdict_counts.get(r.final_verdict, 0) + 1
        band_counts[r.band] = band_counts.get(r.band, 0) + 1

    auto_fp = band_counts.get("auto_fp", 0)
    auto_tp = band_counts.get("auto_tp", 0)
    uncertain = band_counts.get("uncertain", 0)
    volume_reduction = (auto_fp + auto_tp) / total if total > 0 else 0.0

    width = 52
    print("\n" + "=" * width)
    print(" Pipeline Results")
    print("=" * width)
    print(f"  Total alerts processed : {total}")
    print(f"  Elapsed time           : {elapsed:.1f}s")
    print(f"  Throughput             : {total / elapsed:.1f} alerts/s")
    print()
    print("  Band routing:")
    print(f"    auto_fp              : {auto_fp:>6}  ({100*auto_fp/total:.1f}%)")
    print(f"    auto_tp              : {auto_tp:>6}  ({100*auto_tp/total:.1f}%)")
    print(f"    uncertain (-> Stage2): {uncertain:>6}  ({100*uncertain/total:.1f}%)")
    print()
    print("  Final verdicts:")
    for v, n in sorted(verdict_counts.items()):
        print(f"    {v:<24} : {n:>6}  ({100*n/total:.1f}%)")
    print()
    print(f"  Volume reduction       : {100*volume_reduction:.1f}%  (target >= 70%)")
    print("=" * width + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SOC alert triage pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--input",
        default=None,
        metavar="PATH",
        help=(
            "Path to input CSV (feature-engineered or raw CICIDS2017 format). "
            "If omitted, loads the day-5 temporal hold-out from data/raw/."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Write per-alert results to this path. Parquet format recommended "
            "(preserves list fields). CSV also supported (list fields JSON-encoded)."
        ),
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help=(
            "Skip Stage 2 LLM calls. Uncertain alerts receive needs_review. "
            "Useful for smoke tests and throughput benchmarking."
        ),
    )
    parser.add_argument(
        "--max-alerts",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N alerts (useful for quick demos).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Fail fast if artifacts are missing
    missing = _check_artifacts(config)
    if missing:
        logger.error(
            "Missing artifacts. Run train_stage1.py and build_rag_index.py first:\n%s",
            "\n".join(missing),
        )
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Load artifacts
    # -------------------------------------------------------------------------
    from src.models.classifier import load_model
    from src.models.conformal import load_conformal
    from src.models.explainer import build_explainer
    from src.llm.embeddings import load_embedding_model
    from src.llm.retrieval import load_index

    logger.info("Loading Stage 1 model...")
    model = load_model(Path(config["stage1"]["model_artifact_path"]))

    logger.info("Loading conformal predictor...")
    conformal = load_conformal(Path(config["conformal"]["artifact_path"]))

    logger.info("Loading FAISS index...")
    faiss_index = load_index(Path(config["rag"]["faiss_index_path"]))

    logger.info("Loading training DataFrame for RAG label lookups...")
    training_df = pd.read_parquet(Path(config["rag"]["training_df_path"]))
    logger.info("Training DataFrame: %d rows.", len(training_df))

    device = config["rag"].get("device", "auto")
    logger.info("Loading embedding model (device=%s)...", device)
    embedding_model = load_embedding_model(config["rag"]["embedding_model"], device=device)

    logger.info("Building SHAP explainer...")
    explainer = build_explainer(model)

    # -------------------------------------------------------------------------
    # Anthropic client
    # -------------------------------------------------------------------------
    anthropic_client = None
    if not args.no_llm:
        try:
            from src.utils.secrets import load_api_key
            import anthropic
            api_key = load_api_key()
            anthropic_client = anthropic.Anthropic(api_key=api_key)
            logger.info("Anthropic client initialised (Stage 2 LLM enabled).")
        except Exception as exc:
            logger.warning(
                "Could not initialise Anthropic client (%s). "
                "Uncertain alerts will be marked needs_review. "
                "Set ANTHROPIC_API_KEY to enable Stage 2.",
                exc,
            )
    else:
        logger.info("--no-llm flag set: Stage 2 LLM calls disabled.")

    # -------------------------------------------------------------------------
    # Load input data
    # -------------------------------------------------------------------------
    from src.data.features import add_temporal_features, clean_features, get_feature_columns
    from src.data.loader import load_dataset, validate_schema
    from src.data.features import temporal_train_test_split

    if args.input is not None:
        logger.info("Loading input from %s...", args.input)
        df = pd.read_csv(args.input, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        if "Timestamp" not in df.columns:
            from src.data.loader import _infer_timestamps
            rng = np.random.default_rng(42)
            df["Timestamp"] = _infer_timestamps(Path(args.input).name, len(df), rng)
        df = clean_features(df)
        df = add_temporal_features(df)
    else:
        test_day = config["data"]["test_day"]
        logger.info("No --input given; loading full dataset and using day-%d hold-out...", test_day)
        raw_df = load_dataset(config)
        validate_schema(raw_df)
        raw_df = clean_features(raw_df)
        raw_df = add_temporal_features(raw_df)
        _, df = temporal_train_test_split(raw_df, test_day=test_day)
        logger.info("Day-5 hold-out: %d rows.", len(df))

    if args.max_alerts is not None and args.max_alerts < len(df):
        logger.info("Capping to %d alerts (--max-alerts).", args.max_alerts)
        df = df.head(args.max_alerts).reset_index(drop=True)

    logger.info("Processing %d alerts through the pipeline...", len(df))

    # -------------------------------------------------------------------------
    # Tripwire store (persistent)
    # -------------------------------------------------------------------------
    from src.pipeline.tripwire import TripwireStore, record_auto_fp
    tripwire_path = Path("models/tripwire.jsonl")
    tripwire_store = TripwireStore(path=tripwire_path)

    # -------------------------------------------------------------------------
    # Build PipelineComponents and run
    # -------------------------------------------------------------------------
    from src.pipeline.orchestrator import PipelineComponents, run_batch

    components = PipelineComponents(
        classifier=model,
        conformal=conformal,
        explainer=explainer,
        embedding_model=embedding_model,
        faiss_index=faiss_index,
        training_df=training_df,
        anthropic_client=anthropic_client,
        config=config,
    )

    import time
    t0 = time.perf_counter()
    records = run_batch(df, config, components)
    elapsed = time.perf_counter() - t0

    # Record auto-FP alerts in the tripwire store
    feat_cols = get_feature_columns(df)
    for i, rec in enumerate(records):
        if rec.band == "auto_fp":
            alert_fields = df.iloc[i][feat_cols].to_dict()
            record_auto_fp(rec.alert_id, alert_fields, tripwire_store)

    _print_results(records, elapsed)

    # -------------------------------------------------------------------------
    # Compute and save metrics
    # -------------------------------------------------------------------------
    metrics = _compute_metrics(records, elapsed, llm_enabled=anthropic_client is not None)

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    metrics_path = metrics_dir / f"evaluation_{run_ts}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics written to %s.", metrics_path)

    # -------------------------------------------------------------------------
    # Save pipeline results (always written as parquet to results/)
    # -------------------------------------------------------------------------
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_parquet = results_dir / f"evaluation_{run_ts}.parquet"
    results_df = pd.DataFrame([r.model_dump() for r in records])
    results_df.to_parquet(results_parquet, index=False)
    logger.info("Results parquet written to %s (%d rows).", results_parquet, len(results_df))

    # Optional additional output path (--output flag)
    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".parquet":
            results_df.to_parquet(out_path, index=False)
        else:
            # CSV: JSON-encode list fields so they survive round-trips
            csv_df = results_df.copy()
            for col in ("shap_top5", "similar_alerts", "recommended_actions"):
                if col in csv_df.columns:
                    csv_df[col] = csv_df[col].apply(json.dumps)
            csv_df.to_csv(out_path, index=False)
        logger.info("Results also written to %s.", out_path)


if __name__ == "__main__":
    main()
