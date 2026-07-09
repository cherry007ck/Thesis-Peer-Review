"""
mcp_servers/arxiv_search.py
arXiv API client for fetching recent academic preprints.

Free, no API key required. Best source for cutting-edge ML/AI preprints.
API docs: https://arxiv.org/help/api/user-manual

We use the public Atom feed endpoint — no authentication needed.
"""
from __future__ import annotations

import time
from typing import Optional

import feedparser
import httpx
from pydantic import BaseModel

from mcp_servers.semantic_scholar import FoundPaper   # reuse shared schema


# ---------------------------------------------------------------------------
# arXiv categories relevant to ML/AI/CS papers
# ---------------------------------------------------------------------------
_CS_CATS = "cs.LG cs.AI cs.CV cs.CL stat.ML"

_BASE_URL = "https://export.arxiv.org/api/query"
_TIMEOUT  = 20.0


class ArxivSearch:
    """
    Queries the arXiv API for recent preprints.
    Results are sorted by submission date (newest first).
    All methods are synchronous (called via asyncio.to_thread in orchestrator).
    """

    def search_recent(
        self,
        query: str,
        max_results: int = 5,
        after_year: Optional[int] = None,
    ) -> list[FoundPaper]:
        """
        Search arXiv for recent papers matching the query.

        Args:
            query:       Keyword / phrase search string.
            max_results: Maximum number of results.
            after_year:  Filter to only include papers submitted after this year.

        Returns:
            List of FoundPaper objects (source='arXiv'), newest first.
        """
        # Build arXiv search query — httpx handles URL encoding of params automatically
        search_query = f"all:{query}"

        params = {
            "search_query": search_query,
            "start":        0,
            "max_results":  max_results * 2,   # over-fetch; some may be filtered
            "sortBy":       "submittedDate",
            "sortOrder":    "descending",
        }

        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.get(_BASE_URL, params=params)
                resp.raise_for_status()
                raw_xml = resp.text
        except Exception as exc:
            print(f"  [ArxivSearch] Request failed: {exc}")
            return []

        feed = feedparser.parse(raw_xml)
        papers: list[FoundPaper] = []

        for entry in feed.entries:
            # Extract year from published date (e.g. "2024-03-15T...")
            published = entry.get("published", "")
            year: Optional[int] = None
            if published:
                try:
                    year = int(published[:4])
                except ValueError:
                    pass

            if after_year and year and year <= after_year:
                continue

            authors = [a.get("name", "") for a in entry.get("authors", [])]
            arxiv_id = entry.get("id", "").split("/abs/")[-1]
            url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else entry.get("id", "")

            # Clean up abstract (remove newlines)
            abstract = entry.get("summary", "").replace("\n", " ").strip()

            papers.append(FoundPaper(
                source="arXiv",
                title=entry.get("title", "Unknown Title").replace("\n", " ").strip(),
                authors=authors,
                year=year,
                abstract=abstract[:600] + ("..." if len(abstract) > 600 else ""),
                venue="arXiv preprint",
                citation_count=0,   # arXiv API doesn't provide citation counts
                url=url,
            ))

            if len(papers) >= max_results:
                break

        return papers
