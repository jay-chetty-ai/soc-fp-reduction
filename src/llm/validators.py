"""Pydantic models for LLM output validation (Security Control S5).

Every LLM response is validated against these schemas. On any parse failure
the verdict is downgraded to "needs_review" rather than propagating an error
or auto-closing the alert.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Stage2Verdict(BaseModel):
    """Structured output from the Stage 2 LLM adjudication call."""

    verdict: Literal["true_positive", "false_positive", "needs_review"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    supporting_history: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)

    @field_validator("rationale")
    @classmethod
    def rationale_non_empty(cls, v: str) -> str:
        """Rationale must not be empty."""
        if not v.strip():
            raise ValueError("rationale must not be empty")
        return v


class AdversarialVerdict(BaseModel):
    """Output from the adversarial validation agent."""

    counter_verdict: Literal["true_positive", "false_positive", "needs_review"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    counter_rationale: str
    weakest_evidence: str
    agrees_with_initial: bool = False

    @field_validator("counter_rationale", "weakest_evidence")
    @classmethod
    def fields_non_empty(cls, v: str) -> str:
        """Counter rationale and weakest evidence must not be empty."""
        if not v.strip():
            raise ValueError("field must not be empty")
        return v


class FinalVerdict(BaseModel):
    """Reconciled verdict from Stage 2 + adversarial agent."""

    verdict: Literal["true_positive", "false_positive", "needs_review"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reconciliation_note: str = ""


class AdjudicatorTaskInput(BaseModel):
    """A2A task input schema for the adjudicator agent."""

    alert_id: str
    alert_fields: dict
    shap_top5: list[dict]
    similar_alerts: list[dict]
    ml_score: float = Field(..., ge=0.0, le=1.0)


class AdversarialTaskInput(BaseModel):
    """A2A task input schema for the adversarial agent."""

    alert_id: str
    initial_verdict: Stage2Verdict
    alert_fields: dict
    shap_top5: list[dict]
    similar_alerts: list[dict]
    ml_score: float = Field(..., ge=0.0, le=1.0)
