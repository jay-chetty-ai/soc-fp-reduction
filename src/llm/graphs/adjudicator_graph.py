"""LangGraph adjudicator graph: sanitize -> build_prompt -> call_llm -> validate."""

from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from src.llm.adjudicator import adjudicate, build_prompt, get_system_prompt
from src.llm.sanitizer import sanitize_alert_dict
from src.llm.validators import Stage2Verdict

logger = logging.getLogger(__name__)

_MAX_RETRIES_DEFAULT = 2


class AdjudicatorState(BaseModel):
    """State schema for the adjudicator LangGraph.

    Required fields (alert_id, raw_alert, shap_top5, similar_alerts, ml_score)
    must be supplied at invocation. All other fields have defaults and are
    populated by graph nodes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    alert_id: str
    raw_alert: dict
    shap_top5: list[dict]
    similar_alerts: list[dict]
    ml_score: float = Field(..., ge=0.0, le=1.0)
    client: Any = None
    config: dict = Field(default_factory=dict)
    max_retries: int = _MAX_RETRIES_DEFAULT
    verdict: Optional[Stage2Verdict] = None
    error: Optional[str] = None
    retry_count: int = 0
    user_prompt: Optional[str] = None
    system_prompt: Optional[str] = None


def _sanitize_node(state: AdjudicatorState) -> dict:
    """Sanitize raw_alert fields to neutralize prompt injection (S1)."""
    sanitized = sanitize_alert_dict(state.raw_alert)
    return {"raw_alert": sanitized}


def _build_prompt_node(state: AdjudicatorState) -> dict:
    """Build LLM prompts from sanitized alert, SHAP values, and similar alerts."""
    sys_prompt = get_system_prompt()
    user_prompt = build_prompt(state.raw_alert, state.shap_top5, state.similar_alerts)
    return {"system_prompt": sys_prompt, "user_prompt": user_prompt}


def _call_llm_node(state: AdjudicatorState) -> dict:
    """Call the Anthropic API.

    adjudicate() returns a needs_review/confidence=0.0 verdict on any failure.
    That sentinel is detected by _validate_node to trigger retries.
    """
    verdict = adjudicate(
        state.client,
        state.system_prompt,
        state.user_prompt,
        state.config,
    )
    return {"verdict": verdict}


def _validate_node(state: AdjudicatorState) -> dict:
    """Detect error-fallback verdicts and increment retry counter.

    A verdict of needs_review with confidence=0.0 is the sentinel that
    adjudicate() emits when the API call or JSON parse failed. Legitimate
    LLM-generated needs_review responses will have confidence > 0.
    """
    v = state.verdict
    if v is not None and not (v.verdict == "needs_review" and v.confidence == 0.0):
        return {"error": None}
    return {
        "retry_count": state.retry_count + 1,
        "error": "LLM returned an unparseable or error response.",
    }


def _route_after_validate(state: AdjudicatorState) -> str:
    """Routing decision after validation.

    - No error: proceed to END.
    - Error and retries remain: loop back to call_llm.
    - Error and retries exhausted: go to set_fallback.
    """
    if state.error is None:
        return "done"
    if state.retry_count < state.max_retries:
        logger.info(
            "Retry %d/%d for alert %s.",
            state.retry_count,
            state.max_retries,
            state.alert_id,
        )
        return "call_llm"
    return "set_fallback"


def _set_fallback_node(state: AdjudicatorState) -> dict:
    """Set a definitive needs_review verdict after retries are exhausted."""
    error_msg = (
        f"Max retries ({state.max_retries}) exhausted for alert {state.alert_id}."
    )
    logger.warning(error_msg)
    return {
        "verdict": Stage2Verdict(
            verdict="needs_review",
            confidence=0.0,
            rationale="Max retries exhausted; manual review required.",
            supporting_history=[],
            recommended_actions=["Route to Tier-2 analyst."],
        ),
        "error": error_msg,
    }


def _build_graph() -> StateGraph:
    g = StateGraph(AdjudicatorState)

    g.add_node("sanitize", _sanitize_node)
    g.add_node("build_prompt", _build_prompt_node)
    g.add_node("call_llm", _call_llm_node)
    g.add_node("validate", _validate_node)
    g.add_node("set_fallback", _set_fallback_node)

    g.add_edge(START, "sanitize")
    g.add_edge("sanitize", "build_prompt")
    g.add_edge("build_prompt", "call_llm")
    g.add_edge("call_llm", "validate")
    g.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"call_llm": "call_llm", "done": END, "set_fallback": "set_fallback"},
    )
    g.add_edge("set_fallback", END)

    return g


adjudicator_graph = _build_graph().compile()
