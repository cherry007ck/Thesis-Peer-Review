"""
review_agents/sota_agent.py
SOTA (State-of-the-Art) Comparison Agent.

Receives the manuscript + a list of real papers retrieved from Semantic Scholar
and arXiv, then uses the local Ollama LLM to:
  - Identify which claims are already addressed by newer/better papers
  - Flag missing citations the authors should have included
  - Assess the manuscript's genuine novelty contribution
  - Score overall SOTA positioning
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from mcp_servers.document_server import Paper
from mcp_servers.semantic_scholar import FoundPaper
from review_agents._ollama_client import (
    DEFAULT_MODEL,
    call_ollama,
    flatten_str_list,
)


# ---------------------------------------------------------------------------
# Pydantic Response Schema
# ---------------------------------------------------------------------------

class SOTAReview(BaseModel):
    paper_id: str
    agent: str = "SOTAAgent"
    similar_papers_found: list[str]        # "Title — Authors (Year)" strings
    novelty_gaps: list[str]                # gaps in claimed novelty
    superseded_claims: list[str]           # claims already addressed in literature
    missing_citations: list[str]           # papers that should have been cited
    sota_score: int = Field(ge=0, le=10)   # how well-positioned vs. SOTA
    overall_score: int = Field(ge=0, le=10)
    summary: str
    recommendations: list[str]


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SOTA_SYSTEM_PROMPT = """
You are a senior academic reviewer specialising in research novelty assessment
and state-of-the-art (SOTA) literature analysis.

You will receive:
1. A manuscript to evaluate
2. A list of RELATED PAPERS found via live academic search (Semantic Scholar + arXiv)

Your task is to compare the manuscript against the provided related papers and assess:

1. NOVELTY GAPS — Are there aspects the manuscript claims as novel that the
   related papers have already addressed? List each gap clearly.
2. SUPERSEDED CLAIMS — Specific claims in the manuscript that a newer paper
   has already demonstrated, improved upon, or disproved.
3. MISSING CITATIONS — Papers from the provided list that the authors clearly
   should have cited in their Related Work section but did not.
4. SOTA SCORE — How well the manuscript is positioned relative to the current
   state of the art. Score 0-10 (10 = genuinely novel, well-situated).
5. OVERALL SCORE — Holistic novelty and contribution score, 0-10.
6. RECOMMENDATIONS — Specific, actionable suggestions referencing the found papers.

You MUST respond ONLY with a valid JSON object matching this exact schema:
{
  "paper_id": "<string>",
  "agent": "SOTAAgent",
  "similar_papers_found": ["<Title — Authors (Year)>"],
  "novelty_gaps": ["<gap description>"],
  "superseded_claims": ["<claim from manuscript that is already addressed in paper X>"],
  "missing_citations": ["<paper title + authors + year that should be cited>"],
  "sota_score": <int 0-10>,
  "overall_score": <int 0-10>,
  "summary": "<2-3 sentence SOTA assessment>",
  "recommendations": ["<actionable recommendation referencing specific papers>"]
}

Do NOT include any text, explanation, or markdown outside the JSON object.
Output raw JSON only.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SOTAAgent:
    """
    Compares the manuscript against real papers retrieved from Semantic Scholar
    and arXiv, using llama3 to perform novelty and gap analysis.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    def review(
        self,
        paper: Paper,
        retrieved_papers: list[FoundPaper] | None = None,
        rag_context: list[str] | None = None,
    ) -> SOTAReview:
        """
        Run a SOTA comparison review.

        Args:
            paper:             The manuscript to review.
            retrieved_papers:  Papers found via Semantic Scholar + arXiv APIs.
            rag_context:       Additional RAG context to consider.

        Returns:
            A validated SOTAReview instance.
        """
        if not retrieved_papers:
            retrieved_papers = []

        print(f"  [{self.__class__.__name__}] Sending to {self.model} with {len(retrieved_papers)} retrieved papers...")
        user_message = self._build_prompt(paper, retrieved_papers)
        data = call_ollama(
            system_prompt=SOTA_SYSTEM_PROMPT,
            user_message=user_message,
            model=self.model,
        )
        data["paper_id"] = paper.id
        data.setdefault("agent", "SOTAAgent")

        # Normalize all string-list fields
        data["similar_papers_found"] = flatten_str_list(data.get("similar_papers_found", []))
        data["novelty_gaps"]         = flatten_str_list(data.get("novelty_gaps", []))
        data["superseded_claims"]    = flatten_str_list(data.get("superseded_claims", []))
        data["missing_citations"]    = flatten_str_list(data.get("missing_citations", []))
        data["recommendations"]      = flatten_str_list(data.get("recommendations", []))

        review = SOTAReview(**data)
        print(f"  [SOTAAgent] Done. SOTA Score: {review.sota_score}/10")
        return review

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(self, paper: Paper, retrieved_papers: list[FoundPaper]) -> str:
        """Combine manuscript text with retrieved papers into a single prompt."""
        lines = [
            f"# MANUSCRIPT TO REVIEW",
            f"",
            f"**Title:** {paper.title}",
            f"**Abstract:** {paper.abstract}",
            f"",
        ]
        for section in paper.sections:
            lines.append(f"## {section.title}")
            lines.append(section.content[:800])   # truncate long sections
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("# RELATED PAPERS FOUND VIA LIVE ACADEMIC SEARCH")
        lines.append("(Use these to assess novelty and identify missing citations)")
        lines.append("")

        if not retrieved_papers:
            lines.append("_No related papers were retrieved from the APIs._")
        else:
            for i, p in enumerate(retrieved_papers, 1):
                author_str = ", ".join(p.authors[:3])
                if len(p.authors) > 3:
                    author_str += " et al."
                lines.append(f"### Paper {i}: {p.title}")
                lines.append(f"**Authors:** {author_str} | **Year:** {p.year} | **Source:** {p.source}")
                if p.venue:
                    lines.append(f"**Venue:** {p.venue}")
                if p.citation_count > 0:
                    lines.append(f"**Citations:** {p.citation_count}")
                if p.abstract:
                    lines.append(f"**Abstract:** {p.abstract[:500]}...")
                if p.url:
                    lines.append(f"**URL:** {p.url}")
                lines.append("")

        return "\n".join(lines)
