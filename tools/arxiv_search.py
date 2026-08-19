"""
ArXiv paper search tool.
Uses the `arxiv` Python library directly for paper discovery.
"""

from __future__ import annotations

import logging
import arxiv
from config import RESULTS_PER_QUERY

logger = logging.getLogger(__name__)


def search_arxiv(query: str, max_results: int = RESULTS_PER_QUERY) -> list[dict]:
    """
    Search ArXiv for academic papers matching the query.

    Returns a list of dicts, each containing:
        - title: str
        - source_url: str
        - snippet: str          (abstract)
        - full_text: str        (abstract — full PDF text is not fetched)
        - source_type: "arxiv"
        - authors: list[str]
        - published: str
    """
    try:
        client = arxiv.Client(
            page_size=max_results,
            delay_seconds=1.0,      # Be polite to the API
            num_retries=3,
        )

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results: list[dict] = []
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "source_url": paper.entry_id,
                "snippet": paper.summary[:500] if paper.summary else "",
                "full_text": paper.summary or "",
                "source_type": "arxiv",
                "authors": [a.name for a in paper.authors[:5]],
                "published": paper.published.strftime("%Y-%m-%d") if paper.published else "",
            })

        logger.info("ArXiv returned %d results for '%s'", len(results), query)
        return results

    except Exception as e:
        logger.error("ArXiv search failed for '%s': %s", query, e)
        return []
