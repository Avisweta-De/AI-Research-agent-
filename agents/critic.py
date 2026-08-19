"""
Critic Agent
Evaluates the draft report for quality, factual grounding, and coherence.
Sends back to Writer if quality is below threshold (max 2 revision loops).
"""

from __future__ import annotations

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage

from config import get_llm, MIN_QUALITY_SCORE, MAX_REVISION_LOOPS
from state import ResearchState

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """You are a Research Report Critic. Evaluate the given research report \
on the following criteria and provide a structured assessment.

Evaluation criteria:
1. **Factual Grounding**: Are claims supported by the cited sources? Are there unsupported assertions?
2. **Citation Accuracy**: Are inline citations [1], [2] etc. used correctly and consistently? \
Does the References section match?
3. **Coherence**: Does the report flow logically? Are transitions smooth?
4. **Completeness**: Are all subtopics covered adequately? Is the executive summary accurate?
5. **Writing Quality**: Is the tone professional and academic? Is the language clear?

Output format (strict JSON — no markdown fences, no extra text):
{
  "quality_score": <integer 1-10>,
  "issues_found": [
    "specific issue 1",
    "specific issue 2",
    ...
  ],
  "revised_report": "<the full revised report with all issues fixed — if quality_score < 7, otherwise copy the original report>"
}

IMPORTANT: The "revised_report" field must contain the COMPLETE report text — either improved or unchanged."""


def _parse_critique(raw_text: str) -> dict:
    """Parse the Critic's JSON response with fallback handling."""
    text = raw_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        parsed = json.loads(text)
        return parsed
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Total fallback — return a passing score with the original text
    return {
        "quality_score": 7,
        "issues_found": ["Critic response could not be parsed."],
        "revised_report": "",
    }


def critic_node(state: ResearchState) -> dict:
    """
    LangGraph node: Critic Agent.
    Evaluates the draft report and decides whether to accept or revise.
    """
    draft = state.get("draft_report", "")
    revision_count = state.get("revision_count", 0)

    logger.info(
        "Critic Agent: evaluating report (revision #%d, %d chars)",
        revision_count,
        len(draft),
    )

    llm = get_llm(temperature=0.2)

    try:
        response = llm.invoke([
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Please evaluate the following research report:\n\n{draft}"
            )),
        ])

        critique = _parse_critique(response.content)

    except Exception as e:
        logger.error("Critic Agent failed: %s", e)
        # On failure, accept the draft as-is
        critique = {
            "quality_score": 7,
            "issues_found": [f"Critic evaluation failed: {e}"],
            "revised_report": draft,
        }

    quality_score = critique.get("quality_score", 7)
    revised_report = critique.get("revised_report", "") or draft

    logger.info(
        "Critic Agent: quality_score=%d, issues=%d",
        quality_score,
        len(critique.get("issues_found", [])),
    )

    # Decide: accept or send back for revision
    if quality_score >= MIN_QUALITY_SCORE or revision_count >= MAX_REVISION_LOOPS:
        # Accept — either quality is good enough or we've hit the revision cap
        if revision_count >= MAX_REVISION_LOOPS and quality_score < MIN_QUALITY_SCORE:
            logger.warning(
                "Critic Agent: accepting report despite score %d (revision cap reached)",
                quality_score,
            )
        return {
            "critique": critique,
            "final_report": revised_report,
            "revision_count": revision_count,
            "status": "complete",
        }
    else:
        # Send back for revision
        logger.info("Critic Agent: requesting revision (score %d < %d)", quality_score, MIN_QUALITY_SCORE)
        return {
            "critique": critique,
            "revision_count": revision_count + 1,
            "status": "revision_requested",
        }
