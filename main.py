"""
main.py
Entry point for the Local Multi-Agent Peer Review System -- Phase 1.

Usage:
    python main.py [paper_id]

    paper_id defaults to 'paper_001' if not supplied.

The pipeline:
    1. Boots the DocumentServer and WebSearchServer (MCP stubs).
    2. Runs the Orchestrator which concurrently dispatches the manuscript
       to the Methodology, Theory, and Writing review agents.
    3. Aggregates results, arbitrates conflicts, and renders a Markdown report.
    4. Saves the report to `review_report_<paper_id>.md` in the project root.
"""
from __future__ import annotations

import asyncio
import io
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Terminal output helpers (safe for Windows cp1252 consoles)
# ---------------------------------------------------------------------------

# Force stdout to UTF-8 so Rich and print() can emit any character.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich import print as rprint
    # Create a Console that writes to our UTF-8 wrapped stdout
    _CONSOLE = Console(file=sys.stdout, highlight=False)
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    _CONSOLE = None  # type: ignore

from review_agents._ollama_client import DEFAULT_MODEL
from review_agents.orchestrator import Orchestrator


def _print(msg: str) -> None:
    if RICH_AVAILABLE:
        _CONSOLE.print(msg)  # type: ignore
    else:
        print(msg)


def _banner() -> None:
    if RICH_AVAILABLE:
        _CONSOLE.print(  # type: ignore
            Panel.fit(
                "[bold cyan]Local Multi-Agent Peer Review System[/bold cyan]\n"
                "[dim]Phase 1 - Master's Thesis Pipeline[/dim]",
                border_style="bright_blue",
            )
        )
    else:
        print("=" * 60)
        print("  Local Multi-Agent Peer Review System - Phase 1")
        print("=" * 60)


def _section(title: str) -> None:
    if RICH_AVAILABLE:
        _CONSOLE.rule(f"[bold yellow]{title}[/bold yellow]")  # type: ignore
    else:
        print(f"\n{'-' * 60}")
        print(f"  {title}")
        print(f"{'-' * 60}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def main(paper_id: str = "paper_001") -> None:
    _banner()

    _section("Pipeline Initialization")
    _print(f"[green]>[/green] Target paper: [bold]{paper_id}[/bold]")
    _print(f"[green]>[/green] Local model  : [bold]{DEFAULT_MODEL}[/bold]")
    _print("[green]>[/green] Booting MCP servers and agent pool...")

    orchestrator = Orchestrator(model=DEFAULT_MODEL)
    _print("[green]*[/green] Orchestrator ready. Agents will call Ollama for each review.\n")

    # --- Run the full review pipeline ------------------------------------
    _section("Executing Multi-Agent Review")
    t0 = time.perf_counter()

    review = await orchestrator.run_review(paper_id)

    elapsed = time.perf_counter() - t0
    _print(f"\n[green]*[/green] Review completed in [bold]{elapsed:.2f}s[/bold]")

    # --- Display score summary -------------------------------------------
    _section("Score Summary")
    m = review.methodology_review
    t = review.theory_review
    w = review.writing_review

    summary_lines = [
        f"  Methodology   :  {m.overall_score}/10",
        f"  SOTA          :  {review.sota_review.sota_score}/10",
        f"  Theory        :  {t.overall_score}/10",
        f"  Writing       :  {w.overall_score}/10",
        f"  ---------------------",
        f"  OVERALL       :  {review.overall_score:.2f}/10",
        f"  RECOMMENDATION:  {review.final_recommendation}",
    ]
    for line in summary_lines:
        _print(line)

    # --- Conflict report -------------------------------------------------
    if review.conflicts_detected:
        _section("Conflicts Detected")
        for conflict in review.conflicts_detected:
            _print(f"  [!]  {conflict}")

    # --- Save Markdown report --------------------------------------------
    _section("Saving Report")
    output_path = Path(__file__).parent / f"review_report_{paper_id}.md"
    output_path.write_text(review.report_markdown, encoding="utf-8")
    _print(f"[green]*[/green] Report saved to: [bold]{output_path}[/bold]")

    # --- Render first 60 lines of report in terminal ---------------------
    _section("Report Preview (first 60 lines)")
    preview_lines = review.report_markdown.splitlines()[:60]
    if RICH_AVAILABLE:
        _CONSOLE.print(Markdown("\n".join(preview_lines)))  # type: ignore
    else:
        print("\n".join(preview_lines))
        print("\n... (see full report in the saved file)")

    _print("\n[bold green]** Pipeline completed successfully. Zero runtime errors. **[/bold green]\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    paper_id = sys.argv[1] if len(sys.argv) > 1 else "paper_001"
    asyncio.run(main(paper_id))
