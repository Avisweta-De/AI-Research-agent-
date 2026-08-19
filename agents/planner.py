"""
Planner Agent
Decomposes a research topic into 4–6 targeted search queries / subtopics.
"""

from __future__ import annotations

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage

from config import get_llm, SUBTOPIC_MIN, SUBTOPIC_MAX
from state import ResearchState

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = f"""You are a Research Planning Specialist. Your job is to take a broad \
research topic and decompose it into {SUBTOPIC_MIN}–{SUBTOPIC_MAX} focused, diverse subtopics \
that together provide comprehensive coverage of the subject.

Guidelines:
- Each subtopic should be a concrete, searchable query (not just a keyword).
- Cover different angles: technical, applications, challenges, recent advances, future directions.
- Avoid overlapping subtopics — maximize breadth of coverage.
- Return ONLY valid JSON — no markdown fences, no extra text.

Output format (strict JSON):
{{
  "subtopics": [
    "subtopic query 1",
    "subtopic query 2",
    ...
  ]
}}"""


def planner_node(state: ResearchState) -> dict:
    """
    LangGraph node: Planner Agent.
    Takes the raw topic and produces a list of subtopics / search queries.
    """
    topic = state.get("topic", "")
    logger.info("Planner Agent: decomposing topic '%s'", topic)

    try:
        llm = get_llm(temperature=0.4)

        response = llm.invoke([
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Research topic: {topic}\n\n"
                f"Generate {SUBTOPIC_MIN}–{SUBTOPIC_MAX} focused subtopics/search queries "
                f"that will give comprehensive coverage of this topic."
            )),
        ])

        # Parse the JSON response
        raw_text = response.content.strip()

        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()

        parsed = json.loads(raw_text)
        subtopics = parsed.get("subtopics", [])

        # Ensure within bounds
        subtopics = subtopics[:SUBTOPIC_MAX]

        logger.info("Planner Agent: generated %d subtopics", len(subtopics))
        return {
            "subtopics": subtopics,
            "status": "planning_complete",
        }

    except json.JSONDecodeError as e:
        logger.error("Planner Agent: JSON parse error — %s", e)
        # Fallback: use the topic itself as the only subtopic
        return {
            "subtopics": [topic],
            "status": "planning_complete",
            "errors": [f"Planner JSON parse error: {e}"],
        }
    except Exception as e:
        logger.error("Planner Agent failed: %s", e)
        return {
            "subtopics": [topic],
            "status": "planning_complete",
            "errors": [f"Planner error: {e}"],
        }
