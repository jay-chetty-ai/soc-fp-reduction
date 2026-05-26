"""End-to-end alert triage pipeline: Stage 1 -> conformal routing -> Stage 2."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_VALID_BANDS = {"auto_fp", "uncertain", "auto_tp"}
_VALID_VERDICTS = {"true_positive", "false_positive", "needs_review", "auto_fp", "auto_tp"}


class DispositionRecord(BaseModel):
    """Full triage record for a single alert.

    All fields populated by run_batch(). List fields default to empty so
    auto_fp / auto_tp records don't require LLM data.
    """

    alert_id: str
    band: str
    ml_score: float
    final_verdict: str
    # Stage 2 fields (uncertain band only)
    stage2_verdict: str | None = None
    stage2_confidence: float | None = None
    stage2_rationale: str | None = None
    adversarial_verdict: str | None = None
    adversarial_rationale: str | None = None
    final_confidence: float | None = None
    reconciliation_note: str | None = None
    recommended_actions: list[str] = Field(default_factory=list)
    # Explanation fields (all bands)
    shap_top5: list[dict] = Field(default_factory=list)
    # RAG context (uncertain band only)
    similar_alerts: list[dict] = Field(default_factory=list)
    # Ground truth (present when input data has Label column)
    true_label: int | None = None


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
    1. Stage 1: LightGBM scoring and SHAP explanation for all alerts.
    2. Conformal routing: auto-FP / uncertain / auto-TP.
    3. Stage 2: LLM adjudication + adversarial challenge for uncertain-band alerts.

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
    from src.llm.adversarial import (
        build_adversarial_prompt,
        challenge,
        get_adversarial_system_prompt,
        reconcile,
    )
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

    # SHAP for all alerts (used in prompts and dashboard for all bands)
    shap_values = explain_batch(components.explainer, X)
    shap_top_k = config.get("stage1", {}).get("shap_top_k", 5)

    # Ground truth labels (present when input has Label column)
    has_labels = "Label" in df.columns
    if has_labels:
        true_labels = encode_labels(df).values
    else:
        true_labels = None

    records: list[DispositionRecord] = []

    for i, (idx, row) in enumerate(df.iterrows()):
        alert_id = str(uuid.uuid4())[:8]
        band = str(bands.iloc[i])
        score = float(ml_scores[i])
        true_label = int(true_labels[i]) if has_labels else None

        # SHAP top-k computed for every alert so the dashboard can display it
        top5 = top_k_features(shap_values[i], feat_cols, row[feat_cols].values, k=shap_top_k)

        if band == "auto_fp":
            records.append(DispositionRecord(
                alert_id=alert_id,
                band=band,
                ml_score=score,
                final_verdict="false_positive",
                shap_top5=top5,
                true_label=true_label,
            ))
            continue

        if band == "auto_tp":
            records.append(DispositionRecord(
                alert_id=alert_id,
                band=band,
                ml_score=score,
                final_verdict="true_positive",
                shap_top5=top5,
                true_label=true_label,
            ))
            continue

        # Uncertain band: RAG retrieval then Stage 2 LLM adjudication
        alert_text = alert_to_text(row)
        query_emb = embed_alerts(components.embedding_model, [alert_text])
        sims, sim_idx = retrieve_similar(
            components.faiss_index, query_emb[0], k=config["rag"]["top_k"]
        )
        similar: list[dict] = [
            {
                "alert_id": f"hist_{si}",
                "label": (
                    str(components.training_df.iloc[si]["Label"])
                    if si < len(components.training_df)
                    else "unknown"
                ),
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
                shap_top5=top5,
                similar_alerts=similar,
                true_label=true_label,
            ))
            continue

        # Stage 2 LLM call
        try:
            user_prompt = build_prompt(row, top5, similar)
            stage2 = adjudicate(
                components.anthropic_client,
                get_system_prompt(),
                user_prompt,
                config,
            )

            # Adversarial challenge
            shap_summary = "\n".join(
                f"{e['feature']}: {e['shap_value']:.4f}" for e in top5
            )
            adv_prompt = build_adversarial_prompt(stage2, alert_text, shap_summary)
            adv = challenge(
                components.anthropic_client,
                get_adversarial_system_prompt(),
                adv_prompt,
                config,
            )

            confidence_threshold = config.get("adversarial", {}).get(
                "confidence_threshold_high", 0.80
            )
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
                adversarial_rationale=adv.counter_rationale if adv else None,
                final_confidence=final.confidence,
                reconciliation_note=final.reconciliation_note or None,
                recommended_actions=stage2.recommended_actions,
                shap_top5=top5,
                similar_alerts=similar,
                true_label=true_label,
            ))

        except Exception as exc:
            logger.error(
                "Unhandled exception for alert %s; marking needs_review: %s",
                alert_id,
                exc,
            )
            records.append(DispositionRecord(
                alert_id=alert_id,
                band=band,
                ml_score=score,
                final_verdict="needs_review",
                shap_top5=top5,
                similar_alerts=similar,
                true_label=true_label,
            ))

    logger.info(
        "Pipeline complete: %d records. Verdicts: %s",
        len(records),
        {v: sum(1 for r in records if r.final_verdict == v) for v in _VALID_VERDICTS},
    )
    return records
