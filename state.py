"""
Shared state definition for the LangGraph research pipeline.
All agents read from and write to this TypedDict.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class ResearchState(TypedDict):
    """Shared state passed through every node in the research graph."""

    # ── User input ──────────────────────────────────────────────────────
    topic: str                          # Original research topic

    # ── Planner output ──────────────────────────────────────────────────
    subtopics: list[str]                # 4-6 decomposed search queries

    # ── Search Agent output ─────────────────────────────────────────────
    search_results: list[dict]
    # Each dict: {title, source_url, snippet, full_text, source_type}

    # ── Analyst output ──────────────────────────────────────────────────
    analyzed_sources: list[dict]
    # Each dict: {title, source, key_findings, methodology, relevance_score}

    # ── Writer output ───────────────────────────────────────────────────
    draft_report: str                   # Markdown report with citations

    # ── Critic output ───────────────────────────────────────────────────
    critique: dict
    # {quality_score: int, issues_found: list[str], revised_report: str}

    # ── Final output ────────────────────────────────────────────────────
    final_report: str                   # Accepted final report

    # ── Control flow ────────────────────────────────────────────────────
    revision_count: int                 # Number of Writer↔Critic iterations
    status: str                         # Current pipeline stage

    # ── Error tracking (append-only via reducer) ────────────────────────
    errors: Annotated[list[str], operator.add]
