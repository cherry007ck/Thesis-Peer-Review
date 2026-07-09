"""
review_agents/theory_agent.py
Theory Review Agent — checks conceptual consistency, terminology stability,
and literature grounding across sections of a scientific manuscript.

Uses a local Ollama model for inference (default: llama3:8b-instruct-q4_0).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from mcp_servers.document_server import Paper
from mcp_servers.semantic_scholar import FoundPaper
from review_agents._ollama_client import (
    DEFAULT_MODEL,
    build_manuscript_text,
    call_ollama,
    flatten_str_list,
)


# ---------------------------------------------------------------------------
# Pydantic Response Schema
# ---------------------------------------------------------------------------

class ConceptDriftInstance(BaseModel):
    term: str
    section_a: str
    definition_a: str
    section_b: str
    definition_b: str
    severity: str = Field(pattern="^(low|medium|high|critical)$")


class LiteratureGap(BaseModel):
    area: str
    description: str
    suggested_references: list[str]


class TheoryReview(BaseModel):
    paper_id: str
    agent: str = "TheoryAgent"
    concept_drift: list[ConceptDriftInstance]
    literature_gaps: list[LiteratureGap]
    definition_conflicts: list[str]
    theoretical_grounding_score: int = Field(ge=0, le=10)
    overall_score: int = Field(ge=0, le=10)
    summary: str
    recommendations: list[str]


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

THEORY_SYSTEM_PROMPT = """
You are an expert academic reviewer specialising in theoretical foundations,
conceptual coherence, and literature review quality in scientific manuscripts.

Your task is to evaluate the THEORETICAL GROUNDING of the provided manuscript.

Focus exclusively on:
1. CONCEPT DRIFT — Identify terms or concepts that are defined differently
   across sections, or used inconsistently. Rate each instance low/medium/high/critical.
2. LITERATURE GAPS — Flag important related works, foundational papers, or
   competing theories that the manuscript fails to cite or discuss.
3. DEFINITION CONFLICTS — Note any formal definitions that contradict each other
   or contradict standard usage in the field.
4. THEORETICAL GROUNDING SCORE — Rate 0-10.
5. OVERALL SCORE — Holistic theory quality score 0-10.
6. RECOMMENDATIONS — Concrete, actionable suggestions.

You MUST respond ONLY with a valid JSON object matching this exact schema:
{
  "paper_id": "<string>",
  "agent": "TheoryAgent",
  "concept_drift": [
    {
      "term": "<term>",
      "section_a": "<section title>",
      "definition_a": "<how it is used in section A>",
      "section_b": "<section title>",
      "definition_b": "<how it is used in section B>",
      "severity": "low|medium|high|critical"
    }
  ],
  "literature_gaps": [
    {
      "area": "<topic area>",
      "description": "<what is missing>",
      "suggested_references": ["<author, year>"]
    }
  ],
  "definition_conflicts": ["<conflict description>"],
  "theoretical_grounding_score": <int 0-10>,
  "overall_score": <int 0-10>,
  "summary": "<2-3 sentence overall assessment>",
  "recommendations": ["<actionable recommendation>"]
}

Do NOT include any text, explanation, or markdown outside the JSON object.
Output raw JSON only.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TheoryAgent:
    """
    Evaluates conceptual consistency and literature grounding using a local
    Ollama LLM for inference.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    def review(self, paper: Paper, retrieved_papers: list[FoundPaper] | None = None, rag_context: list[str] | None = None) -> TheoryReview:
        """
        Run a full theory review on the supplied Paper.

        Args:
            paper:             A fully-loaded Paper model from DocumentServer.
            retrieved_papers:  Real papers from Semantic Scholar + arXiv (optional).
            rag_context:       External text context for RAG (optional).

        Returns:
            A validated TheoryReview instance.
        """
        print(f"  [{self.__class__.__name__}] Sending to {self.model}...")
        manuscript = build_manuscript_text(paper, retrieved_papers, rag_context)
        data = call_ollama(
            system_prompt=THEORY_SYSTEM_PROMPT,
            user_message=manuscript,
            model=self.model,
        )
        data["paper_id"] = paper.id
        data.setdefault("agent", "TheoryAgent")

        # Coerce invalid severity values
        for cd in data.get("concept_drift", []):
            if cd.get("severity") not in ("low", "medium", "high", "critical"):
                cd["severity"] = "medium"

        # Ensure literature_gaps have the right shape; coerce str lists
        for gap in data.get("literature_gaps", []):
            gap.setdefault("suggested_references", [])
            gap["suggested_references"] = flatten_str_list(gap["suggested_references"])

        # Coerce any list-of-dict responses into list-of-str
        data["definition_conflicts"] = flatten_str_list(data.get("definition_conflicts", []))
        data["recommendations"]      = flatten_str_list(data.get("recommendations", []))

        review = TheoryReview(**data)
        print(f"  [TheoryAgent] Done. Score: {review.overall_score}/10")
        return review
