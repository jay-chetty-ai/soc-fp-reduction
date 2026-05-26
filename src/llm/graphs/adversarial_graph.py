"""LangGraph adversarial validation graph."""

from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from src.llm.adversarial import (
    AdversarialVerdict,
    build_adversarial_prompt,
    challenge,
    get_adversarial_system_prompt,
)
from src.llm.validators import Stage2Verdict

logger = logging.getLogger(__name__)


class AdversarialState(BaseModel):
    """State schema for the adversarial validation LangGraph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    alert_id: str
    initial_verdict: Stage2Verdict
    alert_summary: str = ""
    shap_summary: str = ""
    client: Any = None
    config: dict = Field(default_factory=dict)
    adversarial_verdict: Optional[AdversarialVerdict] = None
    error: Optional[str] = None
    user_prompt: Optional[str] = None
    system_prompt: Optional[str] = None


def _build_adversarial_prompt_node(state: AdversarialState) -> dict:
    """Build the adversarial challenge prompt."""
    sys_prompt = get_adversarial_system_prompt()
    user_prompt = build_adversarial_prompt(
        state.initial_verdict,
        state.alert_summary,
        state.shap_summary,
    )
    return {"system_prompt": sys_prompt, "user_prompt": user_prompt}


def _call_adversarial_node(state: AdversarialState) -> dict:
    """Call the adversarial LLM agent."""
    verdict = challenge(
        state.client,
        state.system_prompt,
        state.user_prompt,
        state.config,
    )
    if verdict is None:
        return {"error": "Adversarial agent call failed."}
    return {"adversarial_verdict": verdict, "error": None}


def _build_graph() -> StateGraph:
    g = StateGraph(AdversarialState)

    g.add_node("build_prompt", _build_adversarial_prompt_node)
    g.add_node("call_adversarial", _call_adversarial_node)

    g.add_edge(START, "build_prompt")
    g.add_edge("build_prompt", "call_adversarial")
    g.add_edge("call_adversarial", END)

    return g


adversarial_graph = _build_graph().compile()
