"""
Tavily web search tool.
Wraps the langchain-tavily integration for use by the Search Agent.
"""

from __future__ import annotations

import os
import logging
from config import TAVILY_API_KEY, RESULTS_PER_QUERY

logger = logging.getLogger(__name__)


def search_web(query: str, max_results: int = RESULTS_PER_QUERY) -> list[dict]:
    """
    Search the web using Tavily and return structured results.

    Returns a list of dicts, each containing:
        - title: str
        - source_url: str
        - snippet: str
        - full_text: str
        - source_type: "web"
    """
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set — skipping web search.")
        return []

    try:
        # Import here to avoid import errors when key is missing
        from langchain_tavily import TavilySearch

        tool = TavilySearch(
            max_results=max_results,
            topic="general",
            include_raw_content=True,
            api_key=TAVILY_API_KEY,
        )

        raw = tool.invoke({"query": query})

        results: list[dict] = []

        # Handle both list and string responses
        if isinstance(raw, list):
            for item in raw:
                results.append({
                    "title": item.get("title", "Untitled"),
                    "source_url": item.get("url", ""),
                    "snippet": item.get("content", "")[:500],
                    "full_text": item.get("raw_content", item.get("content", "")),
                    "source_type": "web",
                })
        elif isinstance(raw, str):
            results.append({
                "title": f"Web result for: {query}",
                "source_url": "",
                "snippet": raw[:500],
                "full_text": raw,
                "source_type": "web",
            })

        logger.info("Tavily returned %d results for '%s'", len(results), query)
        return results

    except Exception as e:
        logger.error("Tavily search failed for '%s': %s", query, e)
        return []
