"""Adversarial validation agent and verdict reconciliation."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from src.llm.validators import AdversarialVerdict, FinalVerdict, Stage2Verdict

logger = logging.getLogger(__name__)

_ADVERSARIAL_SYSTEM_PROMPT = (
    "You are a skeptical SOC analyst tasked with challenging security alert verdicts. "
    "Your goal is to find weaknesses in the initial verdict and argue the opposing case. "
    "Be rigorous: identify the weakest evidence, propose alternative explanations, "
    "and provide a counter-verdict. Your role is to prevent false conclusions, "
    "not to agree with the initial analyst."
)

# When Stage 2 confidence exceeds this value and the adversarial agent disagrees,
# Stage 2 wins the tie-break.
_CONFIDENCE_THRESHOLD_HIGH = 0.80


def build_adversarial_prompt(
    initial_verdict: Stage2Verdict,
    alert_summary: str,
    shap_summary: str,
) -> str:
    """Build the adversarial agent's challenge prompt.

    Args:
        initial_verdict: The Stage 2 verdict to challenge.
        alert_summary: Plain-text alert description (sanitized by caller).
        shap_summary: Plain-text SHAP feature summary.

    Returns:
        User-turn prompt for the adversarial agent.
    """
    return f"""Challenge the following security alert verdict.

Initial verdict: {initial_verdict.verdict} (confidence: {initial_verdict.confidence:.2f})
Initial rationale: {initial_verdict.rationale}

Alert summary:
{alert_summary}

SHAP feature evidence:
{shap_summary}

Argue against this verdict. Identify the weakest piece of evidence and propose
an alternative explanation. Return ONLY a valid JSON object:
{{
  "counter_verdict": "<true_positive|false_positive|needs_review>",
  "confidence": <float 0.0-1.0>,
  "counter_rationale": "<your argument against the initial verdict>",
  "weakest_evidence": "<the weakest point in the initial rationale>"
}}"""


def challenge(
    client: Any,
    system_prompt: str,
    user_prompt: str,
    config: dict,
) -> AdversarialVerdict | None:
    """Call the adversarial agent API and parse the result.

    Returns None on any failure so that reconcile() can fall back gracefully.

    Args:
        client: Anthropic client instance.
        system_prompt: System-turn instructions.
        user_prompt: Challenge prompt built by build_adversarial_prompt().
        config: Parsed config.yaml dict.

    Returns:
        AdversarialVerdict if the call succeeds; None otherwise.
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
        if text.startswith("```"):
            text = "\n".join(
                line for line in text.splitlines()
                if not line.startswith("```")
            ).strip()
        data = json.loads(text)
        verdict = AdversarialVerdict.model_validate(data)
        logger.info(
            "Adversarial verdict: %s (confidence=%.2f).",
            verdict.counter_verdict,
            verdict.confidence,
        )
        return verdict
    except anthropic.APIConnectionError as exc:
        logger.warning("Adversarial agent connection error; skipping: %s", exc)
    except anthropic.APITimeoutError as exc:
        logger.warning("Adversarial agent timeout; skipping: %s", exc)
    except (json.JSONDecodeError, ValidationError, KeyError, IndexError) as exc:
        logger.warning("Adversarial response parse failure; skipping: %s", exc)
    return None


def reconcile(
    stage2: Stage2Verdict,
    adversarial: AdversarialVerdict | None,
) -> FinalVerdict:
    """Reconcile Stage 2 and adversarial verdicts into a final disposition.

    Reconciliation logic:
    - If adversarial is None (call failed), accept Stage 2 verdict unchanged.
    - If both agree: final verdict = agreed verdict; confidence = mean confidence.
    - If they disagree and Stage 2 confidence > threshold: Stage 2 wins.
    - If they disagree and Stage 2 confidence <= threshold: downgrade to needs_review.

    Args:
        stage2: Stage 2 LLM verdict.
        adversarial: Adversarial agent verdict; None if the call failed.

    Returns:
        FinalVerdict with reconciled disposition.
    """
    if adversarial is None:
        logger.info("No adversarial verdict; accepting Stage 2 result.")
        return FinalVerdict(
            verdict=stage2.verdict,
            confidence=stage2.confidence,
            reconciliation_note="Adversarial agent unavailable; Stage 2 verdict accepted.",
        )

    if stage2.verdict == adversarial.counter_verdict:
        avg_confidence = (stage2.confidence + adversarial.confidence) / 2.0
        logger.info(
            "Agreement: verdict=%s, confidence=%.3f.",
            stage2.verdict,
            avg_confidence,
        )
        return FinalVerdict(
            verdict=stage2.verdict,
            confidence=avg_confidence,
            reconciliation_note="Stage 2 and adversarial agent agree.",
        )

    # Disagreement case
    if stage2.confidence > _CONFIDENCE_THRESHOLD_HIGH:
        note = (
            f"Disagreement: Stage 2 says {stage2.verdict} "
            f"(confidence={stage2.confidence:.2f}) vs adversarial {adversarial.counter_verdict} "
            f"(confidence={adversarial.confidence:.2f}). "
            f"Stage 2 confidence exceeds {_CONFIDENCE_THRESHOLD_HIGH:.2f} threshold; Stage 2 wins."
        )
        logger.info(note)
        return FinalVerdict(
            verdict=stage2.verdict,
            confidence=stage2.confidence,
            reconciliation_note=note,
        )

    note = (
        f"Disagreement: Stage 2 says {stage2.verdict} "
        f"(confidence={stage2.confidence:.2f}) vs adversarial {adversarial.counter_verdict} "
        f"(confidence={adversarial.confidence:.2f}). "
        f"Stage 2 confidence below threshold; downgraded to needs_review."
    )
    logger.info(note)
    return FinalVerdict(
        verdict="needs_review",
        confidence=min(stage2.confidence, adversarial.confidence),
        reconciliation_note=note,
    )


def get_adversarial_system_prompt() -> str:
    """Return the default adversarial system prompt."""
    return _ADVERSARIAL_SYSTEM_PROMPT
