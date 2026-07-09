"""
review_agents/_ollama_client.py
Shared Ollama inference helper used by all review agents.

Handles:
- Sending a system + user message to a local Ollama model
- Extracting clean JSON from the model response (strips markdown fences)
- Retrying up to MAX_RETRIES times on JSON parse failures
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import ollama

DEFAULT_MODEL = "llama3:8b-instruct-q4_0"
MAX_RETRIES = 3
TEMPERATURE = 0.1        # Low temperature → more deterministic JSON output


def call_ollama(
    system_prompt: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """
    Call the local Ollama model and return a parsed JSON dict.

    Args:
        system_prompt: The agent's system instruction (defines output schema).
        user_message:  The manuscript text to review.
        model:         Ollama model tag to use.
        max_retries:   Number of parse-retry attempts on malformed JSON.

    Returns:
        Parsed dict from the model's JSON response.

    Raises:
        RuntimeError: If all retries are exhausted without valid JSON.
    """
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                options={
                    "temperature": TEMPERATURE,
                    "num_predict": 4096,
                },
            )

            raw: str = response["message"]["content"]

            # Strip markdown code fences if the model wrapped its JSON
            clean = re.sub(r"```(?:json)?", "", raw).strip()
            # Sometimes models add trailing commentary after the closing }
            # Find the outermost JSON object
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in model response.")
            # Escape any stray backslashes (e.g. from LaTeX \theta) to prevent JSONDecodeError
            json_str = match.group().replace("\\", "\\\\")
            return json.loads(json_str)

        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            last_error = exc
            print(
                f"  [OllamaClient] Attempt {attempt}/{max_retries} failed "
                f"({type(exc).__name__}: {exc}). Retrying..."
            )
            if attempt < max_retries:
                time.sleep(1)

    raise RuntimeError(
        f"[OllamaClient] All {max_retries} attempts failed. "
        f"Last error: {last_error}"
    )


def flatten_str_list(items: list) -> list[str]:
    """
    Coerce a list that may contain dicts into a list of plain strings.

    LLMs sometimes return list items as dicts (e.g. {"conflict": "..."})
    instead of plain strings. This function handles both cases:
      - str items  → kept as-is
      - dict items → joined "key: value" pairs into one string
      - other      → str() cast
    """
    result: list[str] = []
    for item in items:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            # Join all values (or "key: value" pairs) into a single string
            parts = [f"{v}" for v in item.values() if v]
            result.append(" | ".join(parts) if parts else str(item))
        else:
            result.append(str(item))
    return result


# Hard cap on manuscript body sent to the LLM.
# llama3:8b has ~8K token context; system prompt + retrieved papers consume ~2K,
# leaving ~6K tokens (~4800 chars) for the manuscript itself.
MAX_MANUSCRIPT_CHARS = 3500
MAX_RETRIEVED_CHARS  = 1200   # retrieved papers section
MAX_RAG_CHARS        = 1000   # internal RAG corpus section


def build_manuscript_text(paper, retrieved_papers=None, rag_context=None) -> str:
    """
    Format a Paper object into a manuscript string for the LLM prompt.
    Applies a total character budget to avoid overflowing the model context.

    Args:
        paper:             The Paper model to format.
        retrieved_papers:  Optional list of FoundPaper objects from web search.
        rag_context:       Optional list of strings (chunks) from local RAG corpus.
    """
    # Budget each section equally from the total char allowance
    sections = paper.sections
    if sections:
        chars_per_section = min(800, MAX_MANUSCRIPT_CHARS // len(sections))
    else:
        chars_per_section = 800

    lines = [
        f"Title: {paper.title}",
        f"",
        f"Abstract: {paper.abstract[:600]}",
        f"",
    ]
    for section in sections:
        lines.append(f"## {section.title}")
        content = section.content
        if len(content) > chars_per_section:
            content = content[:chars_per_section] + "...[truncated]"
        lines.append(content)
        lines.append("")

    # Inject retrieved papers as grounding context (capped)
    if retrieved_papers:
        lines.append("---")
        lines.append("## Related Papers Found via Live Academic Search")
        lines.append("(Use these to ground your review: flag gaps, missing citations, or superseded claims)")
        lines.append("")
        chars_used = 0
        for i, p in enumerate(retrieved_papers[:5], 1):
            author_str = ", ".join(p.authors[:2])
            if len(p.authors) > 2:
                author_str += " et al."
            entry_lines = [f"[{i}] {p.title} -- {author_str} ({p.year})"]
            if p.venue:
                entry_lines.append(f"    Venue: {p.venue}")
            if p.citation_count > 0:
                entry_lines.append(f"    Citations: {p.citation_count}")
            if p.abstract:
                entry_lines.append(f"    Abstract: {p.abstract[:200]}...")
            entry_lines.append("")
            entry_text = "\n".join(entry_lines)
            if chars_used + len(entry_text) > MAX_RETRIEVED_CHARS:
                break
            lines.append(entry_text)
            chars_used += len(entry_text)

    # Inject internal RAG context
    if rag_context:
        lines.append("---")
        lines.append("## Related Internal Corpus Context")
        lines.append("(Excerpts from the local database related to this paper)")
        lines.append("")
        chars_used = 0
        for i, chunk in enumerate(rag_context, 1):
            entry_text = f"[{i}] {chunk}\n"
            if chars_used + len(entry_text) > MAX_RAG_CHARS:
                break
            lines.append(entry_text)
            chars_used += len(entry_text)

    return "\n".join(lines)
