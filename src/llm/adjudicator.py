"""Stage 2 LLM adjudication: prompt building and API call."""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
from pydantic import ValidationError

from src.llm.sanitizer import sanitize_alert_dict
from src.llm.validators import Stage2Verdict

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a senior SOC analyst reviewing network security alerts. "
    "Your task is to determine whether each alert is a true positive (genuine attack) "
    "or a false positive (benign traffic incorrectly flagged). "
    "Be precise, analytical, and conservative: when uncertain, return needs_review."
)


def build_prompt(
    alert: pd.Series | dict,
    shap_top5: list[dict],
    similar_alerts: list[dict],
) -> str:
    """Build the Stage 2 adjudication prompt.

    The prompt uses XML-style delimiters to separate data from instructions,
    mitigating prompt injection (S1). Alert field values are sanitized before
    insertion.

    Args:
        alert: Single alert row as a pd.Series or dict.
        shap_top5: List of dicts with keys 'feature', 'shap_value', 'feature_value'.
        similar_alerts: List of dicts with keys 'alert_id' (and optionally 'label',
            'similarity').

    Returns:
        Complete user-turn prompt string.
    """
    if isinstance(alert, pd.Series):
        raw_fields = alert.to_dict()
    else:
        raw_fields = dict(alert)

    sanitized = sanitize_alert_dict(raw_fields)
    alert_block = "\n".join(f"  {k}: {v}" for k, v in sanitized.items())

    shap_lines = "\n".join(
        f"  {i + 1}. {entry['feature']}: SHAP={entry.get('shap_value', 0):.4f}, "
        f"value={entry.get('feature_value', 'N/A')}"
        for i, entry in enumerate(shap_top5)
    )

    similar_lines = "\n".join(
        f"  - {a.get('alert_id', f'alert_{i}')}: "
        f"label={a.get('label', 'unknown')}, "
        f"similarity={a.get('similarity', 0):.3f}"
        for i, a in enumerate(similar_alerts)
    )

    return f"""Analyze the following network security alert and determine the disposition.

<alert_data>
{alert_block}
</alert_data>

Top 5 features by SHAP importance:
{shap_lines}

Most similar historical alerts (from analyst-verified cases):
{similar_lines}

Reason step by step before reaching a conclusion. Then return ONLY a valid JSON object:
{{
  "verdict": "<true_positive|false_positive|needs_review>",
  "confidence": <float 0.0-1.0>,
  "rationale": "<your reasoning>",
  "supporting_history": ["<alert_id>", ...],
  "recommended_actions": ["<action>", ...]
}}"""


def adjudicate(
    client: Any,
    system_prompt: str,
    user_prompt: str,
    config: dict,
) -> Stage2Verdict:
    """Call the Anthropic API and parse the Stage 2 verdict.

    On any failure (timeout, malformed JSON, schema violation) returns a
    "needs_review" verdict rather than raising. Auto-close never happens on
    a malformed response (S5).

    Args:
        client: Anthropic client instance (real or mock).
        system_prompt: System-turn instructions for the model.
        user_prompt: User-turn prompt built by build_prompt().
        config: Parsed config.yaml dict.

    Returns:
        Validated Stage2Verdict.
    """
    try:
        import anthropic
        response = client.messages.create(
            model=config["stage2"]["model"],
            max_tokens=config["stage2"]["max_tokens"],
            temperature=config["stage2"].get("temperature", 0.1),
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = response.content[0].text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = "\n".join(
                line for line in text.splitlines()
                if not line.startswith("```")
            ).strip()
        data = json.loads(text)
        verdict = Stage2Verdict.model_validate(data)
        logger.info(
            "Stage 2 verdict: %s (confidence=%.2f).",
            verdict.verdict,
            verdict.confidence,
        )
        return verdict
    except anthropic.APITimeoutError as exc:
        logger.warning("Stage 2 API timeout: %s", exc)
    except anthropic.APIConnectionError as exc:
        logger.warning("Stage 2 API connection error: %s", exc)
    except (json.JSONDecodeError, ValidationError, KeyError, IndexError) as exc:
        logger.warning("Stage 2 response parse failure: %s", exc)
    return _fallback_verdict("Stage 2 adjudication failed; routing to needs_review.")


def _fallback_verdict(rationale: str) -> Stage2Verdict:
    return Stage2Verdict(
        verdict="needs_review",
        confidence=0.0,
        rationale=rationale,
        supporting_history=[],
        recommended_actions=["Route to Tier-2 analyst for manual review."],
    )


def get_system_prompt() -> str:
    """Return the default Stage 2 system prompt."""
    return _SYSTEM_PROMPT
