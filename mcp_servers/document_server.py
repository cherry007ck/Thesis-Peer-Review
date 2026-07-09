"""
mcp_servers/document_server.py
Local Document Engine — loads, indexes, and queries the local paper corpus.
Exposes MCP-style tool methods with strict Pydantic typing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class Author(BaseModel):
    name: str
    affiliation: Optional[str] = None


class Section(BaseModel):
    title: str
    content: str


class Paper(BaseModel):
    id: str
    title: str
    abstract: str
    authors: list[Author]
    keywords: list[str]
    sections: list[Section]
    year: Optional[int] = None
    doi: Optional[str] = None


# ---------------------------------------------------------------------------
# DocumentServer
# ---------------------------------------------------------------------------

class DocumentServer:
    """
    Lightweight local document engine that loads JSON-encoded papers from
    `data/samples/` and exposes MCP-style tool methods for retrieval.
    """

    CORPUS_PATH: Path = Path(__file__).parent.parent / "data" / "samples"

    def __init__(self) -> None:
        self._corpus: dict[str, Paper] = {}
        self._load_corpus()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_corpus(self) -> None:
        """Scan CORPUS_PATH and load every *.json file as a Paper."""
        if not self.CORPUS_PATH.exists():
            return
        for fp in self.CORPUS_PATH.glob("*.json"):
            try:
                raw = json.loads(fp.read_text(encoding="utf-8"))
                paper = Paper(**raw)
                self._corpus[paper.id] = paper
            except Exception as exc:
                print(f"[DocumentServer] Warning: could not load {fp.name}: {exc}")

    # ------------------------------------------------------------------
    # MCP Tool Methods
    # ------------------------------------------------------------------

    def fetch_paper_by_id(self, paper_id: str) -> Optional[Paper]:
        """
        Retrieve a single paper from the local corpus by its unique ID.

        Args:
            paper_id: The unique identifier for the paper (e.g. 'paper_001').

        Returns:
            A Paper model if found, else None.
        """
        paper = self._corpus.get(paper_id)
        if paper is None:
            print(f"[DocumentServer] Paper '{paper_id}' not found in corpus.")
        return paper

    def query_local_corpus(self, keywords: list[str]) -> list[Paper]:
        """
        Search the corpus for papers whose title, abstract, or keyword list
        contains any of the supplied keywords (case-insensitive substring match).

        Args:
            keywords: A list of search terms.

        Returns:
            A list of matching Paper models (may be empty).
        """
        keywords_lower = [kw.lower() for kw in keywords]
        results: list[Paper] = []
        for paper in self._corpus.values():
            haystack = (
                paper.title.lower()
                + " "
                + paper.abstract.lower()
                + " "
                + " ".join(k.lower() for k in paper.keywords)
            )
            if any(kw in haystack for kw in keywords_lower):
                results.append(paper)
        return results

    def list_all_papers(self) -> list[Paper]:
        """Return every paper currently loaded in the corpus."""
        return list(self._corpus.values())
