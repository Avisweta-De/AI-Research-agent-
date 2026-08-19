"""
Writer Agent
Generates a structured, cited research report in markdown format.
Supports revision based on Critic feedback.
"""

from __future__ import annotations

import logging
from langchain_core.messages import SystemMessage, HumanMessage

from config import get_llm
from state import ResearchState

logger = logging.getLogger(__name__)

WRITER_SYSTEM_PROMPT = """You are an expert Research Report Writer. Produce a comprehensive, \
well-structured research report in markdown format.

Report structure (use these exact headings):
# Executive Summary
# Introduction
# Key Findings
## <Subtopic 1>
## <Subtopic 2>
... (one subsection per subtopic)
# Synthesis & Discussion
# Limitations & Future Directions
# References

Rules:
1. Every factual claim MUST have an inline citation like [1], [2], etc.
2. The References section must list every cited source with its number, title, and URL.
3. Write in a professional, academic tone — clear, precise, objective.
4. The Executive Summary should be 150–250 words.
5. Each Key Findings subsection should synthesize information from multiple sources when possible.
6. The Synthesis section should identify patterns, contradictions, and overarching themes.
7. Do NOT fabricate information. Only use what is provided in the source analyses.
8. Output ONLY the markdown report — no preamble, no closing remarks."""

REVISION_PROMPT = """You previously wrote a research report, but the Critic Agent found issues.

CRITIC FEEDBACK:
Quality Score: {quality_score}/10
Issues Found:
{issues}

PREVIOUS DRAFT:
{draft}

Please revise the report addressing ALL the issues above. Maintain the same structure \
and citation format. Output ONLY the revised markdown report."""


def writer_node(state: ResearchState) -> dict:
    """
    LangGraph node: Writer Agent.
    Generates or revises the research report based on analyst summaries.
    """
    topic = state.get("topic", "")
    analyzed_sources = state.get("analyzed_sources", [])
    critique = state.get("critique", {})
    revision_count = state.get("revision_count", 0)

    is_revision = bool(critique) and revision_count > 0
    mode = "revising" if is_revision else "writing"
    logger.info("Writer Agent: %s report for '%s' (revision #%d)", mode, topic, revision_count)

    llm = get_llm(temperature=0.3)

    if is_revision:
        # ── Revision mode: incorporate Critic feedback ──────────────────
        issues_text = "\n".join(
            f"- {issue}" for issue in critique.get("issues_found", [])
        )
        prompt = REVISION_PROMPT.format(
            quality_score=critique.get("quality_score", "N/A"),
            issues=issues_text or "- No specific issues listed.",
            draft=state.get("draft_report", ""),
        )

        response = llm.invoke([
            SystemMessage(content=WRITER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
    else:
        # ── First draft mode ────────────────────────────────────────────
        # Build source summary for the LLM
        source_summaries = []
        for idx, src in enumerate(analyzed_sources, 1):
            findings = "\n".join(
                f"  • {f}" for f in src.get("key_findings", [])
            )
            summary = (
                f"[{idx}] {src.get('title', 'Untitled')}\n"
                f"    Source: {src.get('source', 'N/A')}\n"
                f"    Relevance: {src.get('relevance_score', 'N/A')}/10\n"
                f"    Methodology: {src.get('methodology', 'N/A')}\n"
                f"    Key Findings:\n{findings}\n"
            )
            source_summaries.append(summary)

        subtopics = state.get("subtopics", [])
        subtopics_text = "\n".join(f"- {st}" for st in subtopics)

        prompt = (
            f"Research Topic: {topic}\n\n"
            f"Subtopics to cover:\n{subtopics_text}\n\n"
            f"Source Analyses ({len(analyzed_sources)} sources):\n\n"
            + "\n".join(source_summaries)
            + "\n\nWrite the full research report now."
        )

        response = llm.invoke([
            SystemMessage(content=WRITER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

    draft = response.content.strip()
    logger.info("Writer Agent: produced %d-character report", len(draft))

    return {
        "draft_report": draft,
        "status": "writing_complete",
    }
