"""A2A (Agent-to-Agent) client with inprocess and http execution modes."""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import ValidationError

from src.llm.validators import (
    AdversarialTaskInput,
    AdversarialVerdict,
    AdjudicatorTaskInput,
    Stage2Verdict,
)

logger = logging.getLogger(__name__)


class A2ATaskError(Exception):
    """Raised when an A2A task fails validation or execution."""


class A2AClient:
    """Client for invoking adjudicator and adversarial agents.

    Supports two modes:
    - ``inprocess``: Directly invokes the LangGraph compiled graphs within
      the current process. No HTTP overhead; used for testing and local runs.
    - ``http``: Sends JSON POST requests to the agents' HTTP endpoints (not
      implemented in this POC; raises NotImplementedError).

    Args:
        config: Parsed config.yaml dict.
        mode: Execution mode; "inprocess" or "http".
        anthropic_client: Anthropic client instance (required for inprocess mode).
    """

    def __init__(
        self,
        config: dict,
        mode: Literal["inprocess", "http"] = "inprocess",
        anthropic_client: Any = None,
    ) -> None:
        self.config = config
        self.mode = mode
        self._client = anthropic_client

    def send_task(self, agent: Literal["adjudicator", "adversarial"], payload: dict) -> dict:
        """Invoke an agent task and return the result as a dict.

        Args:
            agent: Target agent name.
            payload: Task input payload (validated against the agent's schema).

        Returns:
            Dict representation of the agent's output verdict.

        Raises:
            A2ATaskError: If payload validation fails or the agent is unknown.
            NotImplementedError: If mode is "http".
        """
        if self.mode == "http":
            raise NotImplementedError("HTTP mode is not implemented in this POC.")

        if agent == "adjudicator":
            return self._run_adjudicator(payload)
        if agent == "adversarial":
            return self._run_adversarial(payload)
        raise A2ATaskError(f"Unknown agent: {agent!r}")

    def _run_adjudicator(self, payload: dict) -> dict:
        """Run the adjudicator graph inprocess."""
        try:
            task_input = AdjudicatorTaskInput.model_validate(payload)
        except ValidationError as exc:
            raise A2ATaskError(f"Invalid adjudicator payload: {exc}") from exc

        from src.llm.graphs.adjudicator_graph import AdjudicatorState, adjudicator_graph

        state = {
            "alert_id": task_input.alert_id,
            "raw_alert": task_input.alert_fields,
            "shap_top5": task_input.shap_top5,
            "similar_alerts": task_input.similar_alerts,
            "ml_score": task_input.ml_score,
            "client": self._client,
            "config": self.config,
        }
        result = adjudicator_graph.invoke(state)
        verdict: Stage2Verdict = result["verdict"]
        return verdict.model_dump()

    def _run_adversarial(self, payload: dict) -> dict:
        """Run the adversarial graph inprocess."""
        try:
            task_input = AdversarialTaskInput.model_validate(payload)
        except ValidationError as exc:
            raise A2ATaskError(f"Invalid adversarial payload: {exc}") from exc

        from src.llm.graphs.adversarial_graph import adversarial_graph

        state = {
            "alert_id": task_input.alert_id,
            "initial_verdict": task_input.initial_verdict,
            "alert_summary": str(task_input.alert_fields),
            "shap_summary": str(task_input.shap_top5),
            "client": self._client,
            "config": self.config,
        }
        result = adversarial_graph.invoke(state)
        verdict: AdversarialVerdict | None = result.get("adversarial_verdict")
        if verdict is None:
            raise A2ATaskError("Adversarial agent produced no verdict.")
        return verdict.model_dump()
