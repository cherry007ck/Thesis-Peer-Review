"""
mcp_servers/semantic_scholar.py
Semantic Scholar Academic Graph API client.

Free, no API key required. Covers 200M+ papers with strong CS/ML/AI coverage.
API docs: https://api.semanticscholar.org/api-docs/

Rate limits (unauthenticated): ~100 requests per 5 minutes.
We use at most 2-3 requests per pipeline run, so this is never an issue.
"""
from __future__ import annotations

import time
from typing import Optional

import httpx
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class S2Author(BaseModel):
    name: str
    authorId: Optional[str] = None


class FoundPaper(BaseModel):
    """A paper retrieved from an academic search API."""
    source: str                          # "SemanticScholar" or "arXiv"
    title: str
    authors: list[str]
    year: Optional[int] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    citation_count: int = 0
    url: Optional[str] = None
    relevance_note: str = ""             # Why this paper is relevant


# ---------------------------------------------------------------------------
# Semantic Scholar Client
# ---------------------------------------------------------------------------

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_FIELDS   = "title,authors,year,abstract,venue,citationCount,externalIds,openAccessPdf"
_TIMEOUT  = 15.0   # seconds


class SemanticScholarSearch:
    """
    Queries the Semantic Scholar Graph API for academic papers.
    All methods are synchronous (called via asyncio.to_thread in orchestrator).
    """

    def search_papers(
        self,
        query: str,
        limit: int = 5,
        min_year: Optional[int] = None,
    ) -> list[FoundPaper]:
        """
        Search for papers matching the query string.

        Args:
            query:    Natural-language or keyword search string.
            limit:    Maximum number of results to return.
            min_year: If set, only return papers from this year onwards.

        Returns:
            A list of FoundPaper objects ordered by relevance.
        """
        params: dict = {
            "query": query,
            "fields": _FIELDS,
            "limit": min(limit * 2, 20),   # over-fetch then filter by year
        }

        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.get(f"{_BASE_URL}/paper/search", params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            print(f"  [SemanticScholar] Request failed: {exc}")
            return []

        papers: list[FoundPaper] = []
        for item in data.get("data", []):
            year = item.get("year")
            if min_year and year and year < min_year:
                continue

            authors = [a.get("name", "") for a in item.get("authors", [])]

            # Build URL: prefer open-access PDF, fall back to S2 page
            ext = item.get("externalIds", {})
            url = None
            if item.get("openAccessPdf"):
                url = item["openAccessPdf"].get("url")
            if not url and ext.get("DOI"):
                url = f"https://doi.org/{ext['DOI']}"
            if not url and ext.get("ArXiv"):
                url = f"https://arxiv.org/abs/{ext['ArXiv']}"

            papers.append(FoundPaper(
                source="SemanticScholar",
                title=item.get("title", "Unknown Title"),
                authors=authors,
                year=year,
                abstract=item.get("abstract") or "",
                venue=item.get("venue") or "",
                citation_count=item.get("citationCount") or 0,
                url=url or "",
            ))

            if len(papers) >= limit:
                break

        # Sort by citation count (most-cited = most established)
        papers.sort(key=lambda p: p.citation_count, reverse=True)
        return papers

    def search_sota(
        self,
        query: str,
        after_year: int,
        limit: int = 5,
    ) -> list[FoundPaper]:
        """
        Search specifically for recent SOTA papers published AFTER the
        manuscript's publication year — used to detect superseded claims.
        """
        return self.search_papers(query, limit=limit, min_year=after_year)
