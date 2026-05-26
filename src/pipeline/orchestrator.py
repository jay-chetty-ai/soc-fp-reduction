"""End-to-end alert triage pipeline: Stage 1 -> conformal routing -> Stage 2."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_VALID_BANDS = {"auto_fp", "uncertain", "auto_tp"}
_VALID_VERDICTS = {"true_positive", "false_positive", "needs_review", "auto_fp", "auto_tp"}


class DispositionRecord(BaseModel):
    """Full triage record for a single alert."""

    alert_id: str
    band: str
    ml_score: float
    final_verdict: str
    stage2_verdict: str | None = None
    stage2_confidence: float | None = None
    stage2_rationale: str | None = None
    adversarial_verdict: str | None = None
    final_confidence: float | None = None


class PipelineComponents(BaseModel):
    """Container for all initialised pipeline components."""

    model_config = {"arbitrary_types_allowed": True}

    classifier: Any
    conformal: Any
    explainer: Any
    embedding_model: Any
    faiss_index: Any
    training_df: pd.DataFrame
    anthropic_client: Any = None
    config: dict


def run_batch(
    df: pd.DataFrame,
    config: dict,
    components: PipelineComponents,
) -> list[DispositionRecord]:
    """Process a batch of alerts through the full pipeline.

    Stages:
    1. Stage 1: LightGBM scoring.
    2. Conformal routing: auto-FP / uncertain / auto-TP.
    3. Stage 2: LLM adjudication for uncertain-band alerts (with adversarial).

    Args:
        df: Feature-engineered DataFrame (output of add_temporal_features).
        config: Parsed config.yaml dict.
        components: Initialised pipeline components.

    Returns:
        List of DispositionRecord, one per input row.
    """
    from src.data.features import encode_labels, get_feature_columns
    from src.models.classifier import predict_proba
    from src.models.conformal import predict_bands
    from src.models.explainer import explain_batch, top_k_features
    from src.llm.adjudicator import adjudicate, build_prompt, get_system_prompt
    from src.llm.adversarial import challenge, get_adversarial_system_prompt, reconcile
    from src.llm.embeddings import alert_to_text, embed_alerts
    from src.llm.retrieval import retrieve_similar

    feat_cols = get_feature_columns(df)
    X = df[feat_cols]

    # Stage 1: score all alerts
    ml_scores = predict_proba(components.classifier, X)
    logger.info("Stage 1 scored %d alerts.", len(df))

    # Conformal routing
    thresholds = {
        "auto_fp_threshold": config["conformal"]["auto_fp_threshold"],
        "auto_tp_threshold": config["conformal"]["auto_tp_threshold"],
    }
    bands = predict_bands(components.conformal, X, thresholds)
    logger.info("Band distribution: %s", bands.value_counts().to_dict())

    # SHAP for all alerts (used in Stage 2 prompts)
    shap_values = explain_batch(components.explainer, X)

    # Build training texts for RAG
    train_feat_cols = get_feature_columns(components.training_df)
    records: list[DispositionRecord] = []

    for i, (idx, row) in enumerate(df.iterrows()):
        alert_id = str(uuid.uuid4())[:8]
        band = str(bands.iloc[i])
        score = float(ml_scores[i])

        if band == "auto_fp":
            records.append(DispositionRecord(
                alert_id=alert_id,
                band=band,
                ml_score=score,
                final_verdict="false_positive",
            ))
            continue

        if band == "auto_tp":
            records.append(DispositionRecord(
                alert_id=alert_id,
                band=band,
                ml_score=score,
                final_verdict="true_positive",
            ))
            continue

        # Uncertain band: Stage 2 LLM adjudication
        shap_row = shap_values[i]
        shap_top_k = config.get("stage1", {}).get("shap_top_k", 5)
        top5 = top_k_features(shap_row, feat_cols, row[feat_cols].values, k=shap_top_k)

        # RAG: retrieve similar historical alerts
        alert_text = alert_to_text(row)
        query_emb = embed_alerts(components.embedding_model, [alert_text])
        sims, sim_idx = retrieve_similar(components.faiss_index, query_emb[0], k=config["rag"]["top_k"])
        similar = [
            {
                "alert_id": f"hist_{si}",
                "label": str(components.training_df.iloc[si]["Label"]) if si < len(components.training_df) else "unknown",
                "similarity": float(sims[j]),
            }
            for j, si in enumerate(sim_idx)
        ]

        if components.anthropic_client is None:
            records.append(DispositionRecord(
                alert_id=alert_id,
                band=band,
                ml_score=score,
                final_verdict="needs_review",
            ))
            continue

        # Stage 2 call
        user_prompt = build_prompt(row, top5, similar)
        stage2 = adjudicate(
            components.anthropic_client,
            get_system_prompt(),
            user_prompt,
            config,
        )

        # Adversarial challenge
        from src.llm.adversarial import build_adversarial_prompt
        adv_prompt = build_adversarial_prompt(
            stage2,
            alert_text,
            "\n".join(f"{e['feature']}: {e['shap_value']:.4f}" for e in top5),
        )
        adv = challenge(
            components.anthropic_client,
            get_adversarial_system_prompt(),
            adv_prompt,
            config,
        )

        confidence_threshold = config.get("adversarial", {}).get("confidence_threshold_high", 0.80)
        final = reconcile(stage2, adv, confidence_threshold=confidence_threshold)

        records.append(DispositionRecord(
            alert_id=alert_id,
            band=band,
            ml_score=score,
            final_verdict=final.verdict,
            stage2_verdict=stage2.verdict,
            stage2_confidence=stage2.confidence,
            stage2_rationale=stage2.rationale,
            adversarial_verdict=adv.counter_verdict if adv else None,
            final_confidence=final.confidence,
        ))

    logger.info(
        "Pipeline complete: %d records. Verdicts: %s",
        len(records),
        {v: sum(1 for r in records if r.final_verdict == v) for v in _VALID_VERDICTS},
    )
    return records
