"""
Analyst Agent
Extracts structured intelligence from each search result using the LLM.
Processes sources in batches to stay within context-window and rate limits.
"""

from __future__ import annotations

import json
import logging
import time
from langchain_core.messages import SystemMessage, HumanMessage

from config import get_llm
from state import ResearchState

logger = logging.getLogger(__name__)

ANALYST_SYSTEM_PROMPT = """You are a Research Analyst. For each source provided, extract a \
structured analysis. Be precise and objective.

For EACH source, produce:
{
  "title": "...",
  "source": "url or identifier",
  "key_findings": ["finding 1", "finding 2", ...],
  "methodology": "brief description of methodology used, or 'N/A'",
  "relevance_score": <integer 1-10>
}

Return a JSON array of analysis objects — one per source.
Return ONLY valid JSON — no markdown fences, no extra commentary."""

BATCH_SIZE = 4  # Process N sources per LLM call


def _parse_analyses(raw_text: str) -> list[dict]:
    """Attempt to parse JSON array from LLM response, with fallback cleanup."""
    text = raw_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return []


def analyst_node(state: ResearchState) -> dict:
    """
    LangGraph node: Analyst Agent.
    Processes search results in batches, extracting structured analyses.
    """
    search_results = state.get("search_results", [])
    logger.info("Analyst Agent: analyzing %d sources", len(search_results))

    if not search_results:
        return {
            "analyzed_sources": [],
            "status": "analysis_complete",
            "errors": ["No search results to analyze."],
        }

    llm = get_llm(temperature=0.2)
    all_analyses: list[dict] = []
    errors: list[str] = []

    # Process in batches
    for batch_start in range(0, len(search_results), BATCH_SIZE):
        batch = search_results[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        logger.info("Analyst Agent: processing batch %d (%d sources)", batch_num, len(batch))

        # Build the source descriptions for the prompt
        source_descriptions = []
        for idx, src in enumerate(batch, 1):
            desc = (
                f"--- Source {idx} ---\n"
                f"Title: {src.get('title', 'Untitled')}\n"
                f"URL: {src.get('source_url', 'N/A')}\n"
                f"Type: {src.get('source_type', 'unknown')}\n"
                f"Content:\n{src.get('full_text', src.get('snippet', ''))[:2000]}\n"
            )
            source_descriptions.append(desc)

        try:
            response = llm.invoke([
                SystemMessage(content=ANALYST_SYSTEM_PROMPT),
                HumanMessage(content=(
                    "Analyze the following sources and return a JSON array "
                    "of structured analysis objects:\n\n"
                    + "\n".join(source_descriptions)
                )),
            ])

            analyses = _parse_analyses(response.content)
            all_analyses.extend(analyses)
            logger.info("Analyst Agent: batch %d → %d analyses", batch_num, len(analyses))

        except Exception as e:
            msg = f"Analyst batch {batch_num} failed: {e}"
            logger.error(msg)
            errors.append(msg)

        # Small delay between batches to respect rate limits
        if batch_start + BATCH_SIZE < len(search_results):
            time.sleep(1)

    logger.info("Analyst Agent: total %d analyses produced", len(all_analyses))

    result: dict = {
        "analyzed_sources": all_analyses,
        "status": "analysis_complete",
    }
    if errors:
        result["errors"] = errors
    return result
