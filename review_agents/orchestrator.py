"""
review_agents/orchestrator.py
Central Reviewer Agent — hierarchical routing state machine.

Phase 2 additions:
  - Pre-retrieval step: queries Semantic Scholar + arXiv before dispatching agents
  - Injects retrieved papers as context into all agent prompts
  - Adds a 4th SOTAAgent running in parallel with the other three
  - Renders Section 5 (Online Search Results) and Section 6 (SOTA Comparison)
    in the final Markdown report
"""
from __future__ import annotations

import asyncio
import datetime
from dataclasses import dataclass, field
from typing import Optional

from mcp_servers.document_server import DocumentServer, Paper
from mcp_servers.web_search_server import WebSearchServer, FoundPaper
from mcp_servers.rag_server import RAGServer
from review_agents._ollama_client import DEFAULT_MODEL
from review_agents.methodology_agent import MethodologyAgent, MethodologyReview
from review_agents.theory_agent import TheoryAgent, TheoryReview
from review_agents.writing_agent import WritingAgent, WritingReview
from review_agents.sota_agent import SOTAAgent, SOTAReview


# ---------------------------------------------------------------------------
# Consolidated Review Schema
# ---------------------------------------------------------------------------

@dataclass
class SystematicPeerReview:
    """Container for all agent outputs and the final arbitrated report."""
    paper_id: str
    paper_title: str
    methodology_review: MethodologyReview
    theory_review: TheoryReview
    writing_review: WritingReview
    sota_review: SOTAReview
    retrieved_papers: list[FoundPaper]
    conflicts_detected: list[str]
    arbitration_commentary: str
    final_recommendation: str   # ACCEPT / MINOR_REVISION / MAJOR_REVISION / REJECT
    overall_score: float
    report_markdown: str = field(default="", repr=False)
    generated_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Hierarchical routing state machine for the peer-review pipeline.

    States:
        IDLE -> RETRIEVING -> DISPATCHING -> AGGREGATING -> ARBITRATING -> REPORTING -> DONE
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.document_server = DocumentServer()
        self.web_search_server = WebSearchServer()
        self.rag_server = RAGServer()
        self.methodology_agent = MethodologyAgent(model=model)
        self.theory_agent = TheoryAgent(model=model)
        self.writing_agent = WritingAgent(model=model)
        self.sota_agent = SOTAAgent(model=model)
        self._state: str = "IDLE"
        self.model = model

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run_review(self, paper_id: str) -> SystematicPeerReview:
        """
        Execute the full hierarchical peer-review pipeline for the given paper.

        Args:
            paper_id: ID of the paper to review (must exist in DocumentServer).

        Returns:
            A fully-populated SystematicPeerReview with Markdown report.
        """
        # --- Step 1: Load paper ------------------------------------------
        self._set_state("RETRIEVING")
        paper = self._fetch_paper(paper_id)

        # Build a meaningful search query from the paper title
        # (keywords field may be arXiv category codes like 'cs.CL', not useful phrases)
        search_terms = self._build_search_query(paper)

        # Serialize web search calls (2 concurrent calls to same API causes 429)
        print(f"  [Orchestrator] Searching Semantic Scholar + arXiv for: '{search_terms[:50]}'...")
        retrieved = await asyncio.to_thread(
            self.web_search_server.search_external_citations,
            search_terms.split(),
        )
        await asyncio.sleep(3)   # rate-limit courtesy between S2 calls
        sota_papers = await asyncio.to_thread(
            self.web_search_server.search_sota_papers,
            search_terms.split(),
            paper.year or 2022,
            5,
        )
        all_retrieved: list[FoundPaper] = retrieved + [
            p for p in sota_papers if p.url not in {r.url for r in retrieved}
        ]
        print(f"  [Orchestrator] Total unique papers retrieved: {len(all_retrieved)}")
        
        print(f"  [Orchestrator] Querying local RAG corpus...")
        rag_context = await asyncio.to_thread(self.rag_server.query, paper.title, 3)
        if rag_context:
            print(f"  [Orchestrator] Found {len(rag_context)} chunks from local RAG corpus.")

        # --- Step 2: Dispatch simultaneously to all four agents ----------
        self._set_state("DISPATCHING")
        print(f"  [Orchestrator] Dispatching paper '{paper.title}' to all agents...")

        (method_review, theory_review, write_review, sota_review) = await asyncio.gather(
            asyncio.to_thread(self.methodology_agent.review, paper, retrieved, rag_context),
            asyncio.to_thread(self.theory_agent.review, paper, retrieved, rag_context),
            asyncio.to_thread(self.writing_agent.review, paper, retrieved, rag_context),
            asyncio.to_thread(self.sota_agent.review, paper, all_retrieved, rag_context),
        )
        print("  [Orchestrator] All agent reviews received.")

        # --- Step 3: Aggregate and conflict-check ------------------------
        self._set_state("AGGREGATING")
        conflicts = self._detect_conflicts(method_review, theory_review, write_review, sota_review)

        # --- Step 4: Arbitration -----------------------------------------
        self._set_state("ARBITRATING")
        arbitration = self._arbitrate(method_review, theory_review, write_review, sota_review, conflicts)
        final_rec = self._compute_recommendation(method_review, theory_review, write_review, sota_review)
        overall_score = self._compute_overall_score(method_review, theory_review, write_review, sota_review)

        # --- Step 5: Build report ----------------------------------------
        self._set_state("REPORTING")
        review = SystematicPeerReview(
            paper_id=paper.id,
            paper_title=paper.title,
            methodology_review=method_review,
            theory_review=theory_review,
            writing_review=write_review,
            sota_review=sota_review,
            retrieved_papers=all_retrieved,
            conflicts_detected=conflicts,
            arbitration_commentary=arbitration,
            final_recommendation=final_rec,
            overall_score=overall_score,
        )
        review.report_markdown = self._render_markdown(review, paper)

        self._set_state("DONE")
        return review

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_search_query(self, paper) -> str:
        """
        Build a meaningful search query from the paper title + real keywords.
        Filters out arXiv category codes (e.g. 'cs.LG', 'stat.ML') which are
        useless as search terms.
        """
        # Prefer real keywords if they're not arXiv category codes
        real_keywords = [
            k for k in (paper.keywords or [])
            if not (len(k) <= 7 and ("." in k or k.startswith("cs") or k.startswith("stat")))
        ]
        if real_keywords:
            return " ".join(real_keywords[:6])

        # Fall back to paper title (skip stop words and very short tokens)
        STOP = {"a","an","the","of","in","for","on","with","and","or","to","is","are",
                "by","from","we","our","its","this","that","as","at","be","via"}
        title_words = [
            w for w in paper.title.split()
            if w.lower() not in STOP and len(w) > 3
        ]
        return " ".join(title_words[:7])

    def _set_state(self, new_state: str) -> None:
        print(f"  [Orchestrator] State: {self._state} -> {new_state}")
        self._state = new_state

    def _fetch_paper(self, paper_id: str) -> Paper:
        paper = self.document_server.fetch_paper_by_id(paper_id)
        if paper is None:
            raise ValueError(
                f"Paper '{paper_id}' not found in corpus. "
                f"Available: {[p.id for p in self.document_server.list_all_papers()]}"
            )
        print(f"  [Orchestrator] Loaded paper: '{paper.title}' ({paper_id})")
        return paper

    def _detect_conflicts(
        self,
        m: MethodologyReview,
        t: TheoryReview,
        w: WritingReview,
        s: SOTAReview,
    ) -> list[str]:
        conflicts: list[str] = []
        scores = {
            "Methodology": m.overall_score,
            "Theory":      t.overall_score,
            "Writing":     w.overall_score,
            "SOTA":        s.overall_score,
        }
        avg = sum(scores.values()) / len(scores)

        for agent, score in scores.items():
            if abs(score - avg) > 3:
                conflicts.append(
                    f"Score divergence: {agent} score ({score}) deviates more than 3 points "
                    f"from the group average ({avg:.1f}). Manual arbitration recommended."
                )

        critical_causal = any(c.severity == "critical" for c in m.causal_claims)
        if critical_causal and w.overall_score >= 7:
            conflicts.append(
                "Tension: Methodology agent flagged critical causal issues, "
                "yet Writing agent scored high. A well-written but methodologically "
                "flawed paper is a publication risk."
            )
        return conflicts

    def _arbitrate(
        self,
        m: MethodologyReview,
        t: TheoryReview,
        w: WritingReview,
        s: SOTAReview,
        conflicts: list[str],
    ) -> str:
        if not conflicts:
            return (
                "All four reviewing agents are in substantial agreement. "
                "The manuscript exhibits consistent quality across methodology, "
                "theoretical grounding, writing, and SOTA positioning. "
                "No conflicting signals require resolution."
            )
        lines = [
            "The following conflicts between reviewer assessments were detected and resolved:",
            "",
        ]
        for i, conflict in enumerate(conflicts, 1):
            lines.append(f"{i}. {conflict}")
        lines += [
            "",
            "Arbitration decision: Greater weight is given to the Methodology review, "
            "as fundamental methodological flaws are the most critical barrier to publication. "
            "SOTA positioning is weighted second — a paper that duplicates existing work "
            "provides limited scientific value regardless of writing quality.",
        ]
        return "\n".join(lines)

    def _compute_recommendation(
        self,
        m: MethodologyReview,
        t: TheoryReview,
        w: WritingReview,
        s: SOTAReview,
    ) -> str:
        # Methodology 40%, SOTA 30%, Theory 20%, Writing 10%
        weighted = (
            m.overall_score * 0.40
            + s.overall_score * 0.30
            + t.overall_score * 0.20
            + w.overall_score * 0.10
        )
        if weighted >= 8.0:   return "ACCEPT"
        elif weighted >= 6.5: return "MINOR_REVISION"
        elif weighted >= 4.5: return "MAJOR_REVISION"
        else:                 return "REJECT"

    def _compute_overall_score(
        self,
        m: MethodologyReview,
        t: TheoryReview,
        w: WritingReview,
        s: SOTAReview,
    ) -> float:
        return round(
            m.overall_score * 0.40
            + s.overall_score * 0.30
            + t.overall_score * 0.20
            + w.overall_score * 0.10,
            2
        )

    # ------------------------------------------------------------------
    # Markdown Report Renderer
    # ------------------------------------------------------------------

    def _render_markdown(self, review: SystematicPeerReview, paper: Paper) -> str:
        m = review.methodology_review
        t = review.theory_review
        w = review.writing_review
        s = review.sota_review

        rec_badge = {
            "ACCEPT":          "[ACCEPT]",
            "MINOR_REVISION":  "[MINOR REVISION]",
            "MAJOR_REVISION":  "[MAJOR REVISION]",
            "REJECT":          "[REJECT]",
        }.get(review.final_recommendation, review.final_recommendation)

        lines: list[str] = [
            "# Systematic Peer Review Report",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| **Paper ID** | `{review.paper_id}` |",
            f"| **Title** | {review.paper_title} |",
            f"| **Generated** | {review.generated_at} |",
            f"| **Overall Score** | **{review.overall_score:.2f} / 10** |",
            f"| **Recommendation** | {rec_badge} |",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            "> This report was generated by the **Local Multi-Agent Peer Review System** (Phase 2).",
            "> Four specialist agents reviewed the manuscript — including a live SOTA Comparison Agent",
            "> backed by real Semantic Scholar and arXiv search results.",
            "",
        ]

        # Score table
        lines += [
            "### Score Summary",
            "",
            "| Dimension | Score | Weight |",
            "|-----------|-------|--------|",
            f"| Methodology | {m.overall_score}/10 | 40% |",
            f"| SOTA Positioning | {s.sota_score}/10 | 30% |",
            f"| Theory | {t.overall_score}/10 | 20% |",
            f"| Writing | {w.overall_score}/10 | 10% |",
            f"| **Weighted Overall** | **{review.overall_score:.2f}/10** | -- |",
            "",
            "---",
            "",
        ]

        # ---- Section 1: Methodology ----
        lines += [
            "## 1. Methodology Review",
            "",
            f"**Statistical Soundness Score:** {m.statistical_soundness_score}/10  ",
            f"**Overall Methodology Score:** {m.overall_score}/10",
            "",
            "### 1.1 Summary",
            "",
            m.summary,
            "",
            "### 1.2 Causal Claim Issues",
            "",
        ]
        for ci in m.causal_claims:
            lines.append(f"- **[{ci.severity.upper()}]** *\"{ci.claim}\"*")
            lines.append(f"  - _{ci.issue}_")
        lines += ["", "### 1.3 Data Validity", "", m.data_validity, ""]
        lines += ["### 1.4 Formula / Equation Issues", ""]
        for fi in m.formula_issues:
            lines.append(f"- {fi}")
        lines += ["", "### 1.5 Missing Controls", ""]
        for mc in m.missing_controls:
            lines.append(f"- {mc}")
        lines += ["", "### 1.6 Recommendations", ""]
        for i, rec in enumerate(m.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines += ["", "---", ""]

        # ---- Section 2: Theory ----
        lines += [
            "## 2. Theory & Literature Review",
            "",
            f"**Theoretical Grounding Score:** {t.theoretical_grounding_score}/10  ",
            f"**Overall Theory Score:** {t.overall_score}/10",
            "",
            "### 2.1 Summary",
            "",
            t.summary,
            "",
            "### 2.2 Concept Drift Instances",
            "",
        ]
        for cd in t.concept_drift:
            lines.append(f"- **Term:** `{cd.term}` [{cd.severity.upper()}]")
            lines.append(f"  - In *{cd.section_a}*: {cd.definition_a}")
            lines.append(f"  - In *{cd.section_b}*: {cd.definition_b}")
        lines += ["", "### 2.3 Literature Gaps", ""]
        for lg in t.literature_gaps:
            lines.append(f"#### {lg.area}")
            lines.append(lg.description)
            lines.append(f"**Suggested references:** {', '.join(lg.suggested_references)}")
            lines.append("")
        lines += ["### 2.4 Definition Conflicts", ""]
        for dc in t.definition_conflicts:
            lines.append(f"- {dc}")
        lines += ["", "### 2.5 Recommendations", ""]
        for i, rec in enumerate(t.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines += ["", "---", ""]

        # ---- Section 3: Writing ----
        lines += [
            "## 3. Writing & Narrative Review",
            "",
            f"**Abstract Clarity Score:** {w.abstract_clarity_score}/10  ",
            f"**Narrative Flow Score:** {w.narrative_flow_score}/10  ",
            f"**Overall Writing Score:** {w.overall_score}/10",
            "",
            "### 3.1 Summary",
            "",
            w.summary,
            "",
            "### 3.2 Abstract Issues",
            "",
        ]
        for ai in w.abstract_issues:
            lines.append(f"- {ai}")
        lines += ["", "### 3.3 Section Transition Analysis", ""]
        for ti in w.transition_issues:
            lines.append(f"- **{ti.from_section} -> {ti.to_section}** [{ti.severity.upper()}]")
            lines.append(f"  - {ti.issue}")
        lines += ["", "### 3.4 Structural Issues", ""]
        for si in w.structural_issues:
            lines.append(f"- {si}")
        lines += ["", "### 3.5 Recommendations", ""]
        for i, rec in enumerate(w.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines += ["", "---", ""]

        # ---- Section 4: Conflict & Arbitration ----
        lines += [
            "## 4. Conflict Detection & Arbitration",
            "",
            "### 4.1 Detected Conflicts",
            "",
        ]
        if review.conflicts_detected:
            for conf in review.conflicts_detected:
                lines.append(f"- [!] {conf}")
        else:
            lines.append("_No significant scoring conflicts detected between agents._")
        lines += [
            "",
            "### 4.2 Arbitration Commentary",
            "",
            review.arbitration_commentary,
            "",
            "---",
            "",
        ]

        # ---- Section 5: Online Search Results ----
        lines += [
            "## 5. Related Papers Found Online (Semantic Scholar + arXiv)",
            "",
            f"*{len(review.retrieved_papers)} papers retrieved via live API search.*",
            "",
        ]
        s2_papers = [p for p in review.retrieved_papers if p.source == "SemanticScholar"]
        ax_papers  = [p for p in review.retrieved_papers if p.source == "arXiv"]

        if s2_papers:
            lines += ["### 5.1 Semantic Scholar Results", ""]
            for p in s2_papers[:5]:
                author_str = ", ".join(p.authors[:2])
                if len(p.authors) > 2:
                    author_str += " et al."
                lines.append(f"- **{p.title}** -- {author_str} ({p.year})")
                if p.venue:
                    lines.append(f"  *{p.venue}*")
                if p.citation_count > 0:
                    lines.append(f"  Citations: {p.citation_count}")
                if p.url:
                    lines.append(f"  {p.url}")
                if p.abstract:
                    lines.append(f"  > {p.abstract[:250]}...")
                lines.append("")

        if ax_papers:
            lines += ["### 5.2 arXiv Recent Preprints", ""]
            for p in ax_papers[:5]:
                author_str = ", ".join(p.authors[:2])
                if len(p.authors) > 2:
                    author_str += " et al."
                lines.append(f"- **{p.title}** -- {author_str} ({p.year})")
                lines.append(f"  *arXiv preprint*")
                if p.url:
                    lines.append(f"  {p.url}")
                if p.abstract:
                    lines.append(f"  > {p.abstract[:250]}...")
                lines.append("")

        lines += ["---", ""]

        # ---- Section 6: SOTA Comparison ----
        lines += [
            "## 6. SOTA Comparison & Novelty Analysis",
            "",
            f"**SOTA Positioning Score:** {s.sota_score}/10  ",
            f"**Overall SOTA Score:** {s.overall_score}/10",
            "",
            "### 6.1 Summary",
            "",
            s.summary,
            "",
            "### 6.2 Novelty Gaps",
            "",
        ]
        for ng in s.novelty_gaps:
            lines.append(f"- {ng}")
        lines += ["", "### 6.3 Superseded Claims", ""]
        if s.superseded_claims:
            for sc in s.superseded_claims:
                lines.append(f"- {sc}")
        else:
            lines.append("_No superseded claims identified._")
        lines += ["", "### 6.4 Missing Citations", ""]
        for mc in s.missing_citations:
            lines.append(f"- {mc}")
        lines += ["", "### 6.5 Recommendations", ""]
        for i, rec in enumerate(s.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines += ["", "---", ""]

        # ---- Section 7: Final Decision ----
        lines += [
            "## 7. Final Editorial Decision",
            "",
            f"### {rec_badge}",
            "",
            f"**Weighted Score:** {review.overall_score:.2f} / 10.00  ",
            f"*(Methodology 40% + SOTA 30% + Theory 20% + Writing 10%)*",
            "",
            (
                "The manuscript requires **major revision** before it can be considered "
                "for publication. The fundamental methodological flaws, SOTA positioning gaps, "
                "and writing issues collectively indicate that significant rework is needed."
                if review.final_recommendation in ("MAJOR_REVISION", "REJECT")
                else
                "The manuscript is on track for publication pending the listed revisions. "
                "Please address all reviewer recommendations and resubmit a tracked-changes version."
            ),
            "",
            "---",
            "",
            "*Report generated by the **Local Multi-Agent Peer Review System** -- Phase 2*  ",
            "*Agents: MethodologyAgent | TheoryAgent | WritingAgent | SOTAAgent*  ",
            "*Web Search: Semantic Scholar API + arXiv API (live)*  ",
            f"*LLM: {self.model} (local Ollama)*  ",
            f"*Timestamp: {review.generated_at}*",
        ]

        return "\n".join(lines)
