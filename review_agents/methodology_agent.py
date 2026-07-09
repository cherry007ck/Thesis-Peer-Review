"""
review_agents/methodology_agent.py
Methodology Review Agent — evaluates causal claims, data validity,
statistical rigour, and empirical soundness of a scientific manuscript.

Uses a local Ollama model for inference (default: llama3:8b-instruct-q4_0).
Receives real related papers from Semantic Scholar + arXiv as grounding context.
"""
from __future__ import annotations

import json
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

class CausalClaimIssue(BaseModel):
    claim: str
    issue: str
    severity: str = Field(pattern="^(low|medium|high|critical)$")


class MethodologyReview(BaseModel):
    paper_id: str
    agent: str = "MethodologyAgent"
    causal_claims: list[CausalClaimIssue]
    data_validity: str
    formula_issues: list[str]
    missing_controls: list[str]
    statistical_soundness_score: int = Field(ge=0, le=10)
    overall_score: int = Field(ge=0, le=10)
    summary: str
    recommendations: list[str]


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

METHODOLOGY_SYSTEM_PROMPT = """
You are a senior scientific peer reviewer with deep expertise in research methodology,
causal inference, experimental design, and statistical analysis.

Your task is to evaluate the METHODOLOGY of the provided manuscript excerpt.

Focus exclusively on:
1. CAUSAL CLAIMS — Identify every statement implying causation. Flag unsupported or
   confounded causal assertions. Rate each as low/medium/high/critical severity.
2. DATA VALIDITY — Assess whether the dataset described is appropriate, unbiased,
   and sufficient for the claims made.
3. FORMULA / EQUATION ISSUES — Detect dimensional inconsistencies, undefined variables,
   or mathematically unsound expressions.
4. MISSING CONTROLS — List experimental controls that are absent but required.
5. STATISTICAL SOUNDNESS — Score the statistical rigour from 0-10.
6. OVERALL SCORE — Holistic methodology quality score 0-10.
7. RECOMMENDATIONS — Concrete, actionable suggestions for improvement.

You MUST respond ONLY with a valid JSON object matching this exact schema:
{
  "paper_id": "<string>",
  "agent": "MethodologyAgent",
  "causal_claims": [
    {"claim": "<text>", "issue": "<explanation>", "severity": "low|medium|high|critical"}
  ],
  "data_validity": "<paragraph assessment>",
  "formula_issues": ["<issue description>"],
  "missing_controls": ["<control name>"],
  "statistical_soundness_score": <int 0-10>,
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

class MethodologyAgent:
    """
    Evaluates the methodological rigour of a scientific manuscript
    using a local Ollama LLM for inference.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    def review(self, paper: Paper, retrieved_papers: list[FoundPaper] | None = None, rag_context: list[str] | None = None) -> MethodologyReview:
        """
        Run a full methodology review on the supplied Paper.

        Args:
            paper:             A fully-loaded Paper model from DocumentServer.
            retrieved_papers:  Real papers from Semantic Scholar + arXiv (optional).
            rag_context:       Additional retrieved context snippets (optional).

        Returns:
            A validated MethodologyReview instance.
        """
        print(f"  [{self.__class__.__name__}] Sending to {self.model}...")
        manuscript = build_manuscript_text(paper, retrieved_papers, rag_context)
        data = call_ollama(
            system_prompt=METHODOLOGY_SYSTEM_PROMPT,
            user_message=manuscript,
            model=self.model,
        )
        data["paper_id"] = paper.id
        data.setdefault("agent", "MethodologyAgent")

        # Ensure severity values are valid; coerce unknowns to "medium"
        for claim in data.get("causal_claims", []):
            if claim.get("severity") not in ("low", "medium", "high", "critical"):
                claim["severity"] = "medium"

        # Coerce any list-of-dict responses into list-of-str
        data["formula_issues"]   = flatten_str_list(data.get("formula_issues", []))
        data["missing_controls"] = flatten_str_list(data.get("missing_controls", []))
        data["recommendations"]  = flatten_str_list(data.get("recommendations", []))

        review = MethodologyReview(**data)
        print(f"  [MethodologyAgent] Done. Score: {review.overall_score}/10")
        return review
