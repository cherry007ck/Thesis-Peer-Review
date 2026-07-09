"""
review_agents/writing_agent.py
Writing Review Agent — evaluates abstract quality, narrative flow,
paragraph transitions, and overall structural storytelling of a manuscript.

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

class TransitionIssue(BaseModel):
    from_section: str
    to_section: str
    issue: str
    severity: str = Field(pattern="^(low|medium|high|critical)$")


class WritingReview(BaseModel):
    paper_id: str
    agent: str = "WritingAgent"
    abstract_clarity_score: int = Field(ge=0, le=10)
    abstract_issues: list[str]
    transition_issues: list[TransitionIssue]
    narrative_flow_score: int = Field(ge=0, le=10)
    structural_issues: list[str]
    overall_score: int = Field(ge=0, le=10)
    summary: str
    recommendations: list[str]


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

WRITING_SYSTEM_PROMPT = """
You are a professional scientific editor with expertise in academic writing,
argumentation structure, and narrative flow in research manuscripts.

Your task is to evaluate the WRITING QUALITY of the provided manuscript.

Focus exclusively on:
1. ABSTRACT CLARITY — Does the abstract clearly convey the problem, method,
   and contribution? Score 0-10 and list specific issues.
2. TRANSITION ISSUES — For each pair of adjacent sections, assess the quality
   of the logical bridge between them. Flag missing or weak transitions.
3. NARRATIVE FLOW — Does the paper tell a coherent story from motivation to
   conclusion? Score 0-10.
4. STRUCTURAL ISSUES — Identify misplaced content, missing sections,
   or logical ordering problems.
5. OVERALL SCORE — Holistic writing quality score 0-10.
6. RECOMMENDATIONS — Concrete, actionable writing improvements.

You MUST respond ONLY with a valid JSON object matching this exact schema:
{
  "paper_id": "<string>",
  "agent": "WritingAgent",
  "abstract_clarity_score": <int 0-10>,
  "abstract_issues": ["<issue>"],
  "transition_issues": [
    {
      "from_section": "<section title>",
      "to_section": "<section title>",
      "issue": "<description>",
      "severity": "low|medium|high|critical"
    }
  ],
  "narrative_flow_score": <int 0-10>,
  "structural_issues": ["<issue>"],
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

class WritingAgent:
    """
    Evaluates the writing quality and narrative structure of a manuscript
    using a local Ollama LLM for inference.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    def review(self, paper: Paper, retrieved_papers: list[FoundPaper] | None = None, rag_context: list[str] | None = None) -> WritingReview:
        """
        Run a full writing review on the supplied Paper.

        Args:
            paper:             A fully-loaded Paper model from DocumentServer.
            retrieved_papers:  Real papers from Semantic Scholar + arXiv (optional).
            rag_context:       Additional text context to supplement the manuscript.

        Returns:
            A validated WritingReview instance.
        """
        print(f"  [{self.__class__.__name__}] Sending to {self.model}...")
        manuscript_text = build_manuscript_text(paper, retrieved_papers, rag_context)
        data = call_ollama(
            system_prompt=WRITING_SYSTEM_PROMPT,
            user_message=manuscript_text,
            model=self.model,
        )
        data["paper_id"] = paper.id
        data.setdefault("agent", "WritingAgent")

        # Coerce invalid severity values
        for ti in data.get("transition_issues", []):
            if ti.get("severity") not in ("low", "medium", "high", "critical"):
                ti["severity"] = "medium"

        # Coerce any list-of-dict responses into list-of-str
        data["abstract_issues"]   = flatten_str_list(data.get("abstract_issues", []))
        data["structural_issues"] = flatten_str_list(data.get("structural_issues", []))
        data["recommendations"]   = flatten_str_list(data.get("recommendations", []))

        review = WritingReview(**data)
        print(f"  [WritingAgent] Done. Score: {review.overall_score}/10")
        return review
