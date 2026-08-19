"""
LangGraph StateGraph assembly.
Wires all 5 agents together with conditional revision loops.

Flow:
  START → Planner → Search → Analyst → Writer → Critic
  Critic → Writer  (if quality < 7 and revisions < 2)
  Critic → END     (if quality ≥ 7 or revisions ≥ 2)
"""

from __future__ import annotations

import logging
from langgraph.graph import StateGraph, START, END

from state import ResearchState
from agents.planner import planner_node
from agents.search import search_node
from agents.analyst import analyst_node
from agents.writer import writer_node
from agents.critic import critic_node

logger = logging.getLogger(__name__)


def _should_revise(state: ResearchState) -> str:
    """
    Conditional edge function: decides whether the Critic routes
    back to the Writer for revision or finishes.
    """
    status = state.get("status", "")
    if status == "revision_requested":
        return "writer"
    return "end"


def build_graph() -> StateGraph:
    """
    Construct and compile the multi-agent research pipeline graph.
    Returns the compiled LangGraph application.
    """
    workflow = StateGraph(ResearchState)

    # ── Register agent nodes ────────────────────────────────────────────
    workflow.add_node("planner", planner_node)
    workflow.add_node("search", search_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)

    # ── Define linear edges ─────────────────────────────────────────────
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "search")
    workflow.add_edge("search", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "critic")

    # ── Conditional edge: Critic → Writer (revise) or → END (accept) ──
    workflow.add_conditional_edges(
        "critic",
        _should_revise,
        {
            "writer": "writer",
            "end": END,
        },
    )

    # ── Compile ─────────────────────────────────────────────────────────
    app = workflow.compile()
    logger.info("Research pipeline graph compiled successfully.")
    return app
