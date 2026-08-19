"""
Search Agent
Runs parallel web (Tavily) and academic (ArXiv) searches for each subtopic,
then deduplicates the results.
"""

from __future__ import annotations

import logging
from tools.tavily_search import search_web
from tools.arxiv_search import search_arxiv
from state import ResearchState

logger = logging.getLogger(__name__)


def _normalize_title(title: str) -> str:
    """Lowercase, strip whitespace for dedup comparison."""
    return " ".join(title.lower().split())


def _deduplicate(results: list[dict]) -> list[dict]:
    """Remove duplicates by URL or normalized title."""
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[dict] = []

    for item in results:
        url = item.get("source_url", "").strip()
        title_norm = _normalize_title(item.get("title", ""))

        if url and url in seen_urls:
            continue
        if title_norm and title_norm in seen_titles:
            continue

        if url:
            seen_urls.add(url)
        if title_norm:
            seen_titles.add(title_norm)
        unique.append(item)

    return unique


def search_node(state: ResearchState) -> dict:
    """
    LangGraph node: Search Agent.
    Fetches results from Tavily + ArXiv for every subtopic,
    deduplicates, and returns the combined list.
    """
    subtopics = state.get("subtopics", [])
    logger.info("Search Agent: searching %d subtopics", len(subtopics))

    all_results: list[dict] = []
    errors: list[str] = []

    for i, query in enumerate(subtopics, 1):
        logger.info("Search Agent: [%d/%d] Searching '%s'", i, len(subtopics), query)

        # --- Web search via Tavily ---
        try:
            web_results = search_web(query)
            all_results.extend(web_results)
        except Exception as e:
            msg = f"Web search failed for '{query}': {e}"
            logger.error(msg)
            errors.append(msg)

        # --- Academic search via ArXiv ---
        try:
            arxiv_results = search_arxiv(query)
            all_results.extend(arxiv_results)
        except Exception as e:
            msg = f"ArXiv search failed for '{query}': {e}"
            logger.error(msg)
            errors.append(msg)

    # Deduplicate
    unique_results = _deduplicate(all_results)

    logger.info(
        "Search Agent: %d raw → %d unique results",
        len(all_results),
        len(unique_results),
    )

    result: dict = {
        "search_results": unique_results,
        "status": "search_complete",
    }
    if errors:
        result["errors"] = errors
    return result
