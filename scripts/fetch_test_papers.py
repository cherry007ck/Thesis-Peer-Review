"""
scripts/fetch_test_papers.py
Downloads real ML research papers from arXiv and converts them
to the Paper JSON schema used by DocumentServer.

What it does:
  1. Searches arXiv for a set of ML topics
  2. Picks recent, well-cited papers
  3. Fetches their HTML version (arxiv.org/html/<id>) and parses sections
  4. Falls back to abstract-only if HTML not available
  5. Saves each as data/samples/paper_<N>.json

Usage:
    python scripts/fetch_test_papers.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import httpx
import feedparser

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("[WARNING] beautifulsoup4 not installed; will use abstract-only sections.")


# ---------------------------------------------------------------------------
# Papers to fetch: (arxiv_id, intended_paper_slot)
# Chosen as real, impactful ML papers with good HTML versions on arXiv
# ---------------------------------------------------------------------------
TARGET_PAPERS = [
    # Slot 2: Attention Is All You Need — foundational transformer paper (2017)
    ("1706.03762", "paper_002"),
    # Slot 3: Economy Statistical Recurrent Units for Nonlinear Granger Causality (2019)
    # Directly competing method with paper_001
    ("1911.09879", "paper_003"),
    # Slot 4: General Identifiability for Causal Representation Learning (2023)
    # Recent SOTA in causal ML
    ("2310.15450", "paper_004"),
]

SAMPLES_DIR = Path(__file__).parent.parent / "data" / "samples"
ARXIV_API   = "https://export.arxiv.org/api/query"
ARXIV_HTML  = "https://arxiv.org/html/{arxiv_id}"
TIMEOUT     = 25.0


# ---------------------------------------------------------------------------
# arXiv metadata fetch
# ---------------------------------------------------------------------------

def fetch_arxiv_metadata(arxiv_id: str) -> dict | None:
    """Fetch paper metadata (title, abstract, authors, year) from arXiv API."""
    params = {
        "id_list": arxiv_id,
        "max_results": 1,
    }
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.get(ARXIV_API, params=params)
            resp.raise_for_status()
        feed = feedparser.parse(resp.text)
        if not feed.entries:
            print(f"  [!] No entry found for {arxiv_id}")
            return None
        entry = feed.entries[0]
        year = int(entry.get("published", "2020")[:4])
        authors = [a.get("name", "") for a in entry.get("authors", [])]
        # Extract keywords from categories
        tags = [t.get("term", "") for t in entry.get("tags", [])]
        return {
            "arxiv_id": arxiv_id,
            "title":    entry.get("title", "Unknown").replace("\n", " ").strip(),
            "abstract": entry.get("summary", "").replace("\n", " ").strip(),
            "authors":  authors,
            "year":     year,
            "keywords": tags[:8],
            "doi":      f"https://arxiv.org/abs/{arxiv_id}",
        }
    except Exception as exc:
        print(f"  [!] Metadata fetch failed for {arxiv_id}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Section extraction from arXiv HTML
# ---------------------------------------------------------------------------

def fetch_sections_from_html(arxiv_id: str, abstract: str) -> list[dict]:
    """
    Fetch the HTML version of the paper and extract named sections.
    Falls back to splitting the abstract into synthetic sections if HTML fails.
    """
    if not BS4_AVAILABLE:
        return make_fallback_sections(abstract)

    url = ARXIV_HTML.format(arxiv_id=arxiv_id)
    try:
        headers = {"User-Agent": "Mozilla/5.0 (academic research bot)"}
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code != 200:
                print(f"  [!] HTML version returned {resp.status_code} for {arxiv_id}; using fallback.")
                return make_fallback_sections(abstract)
    except Exception as exc:
        print(f"  [!] HTML fetch failed: {exc}; using fallback.")
        return make_fallback_sections(abstract)

    soup = BeautifulSoup(resp.text, "lxml")
    sections: list[dict] = []

    # arXiv HTML papers use <section> tags with <h2> or <h3> headings
    for section_tag in soup.find_all("section"):
        # Find heading
        heading_tag = section_tag.find(["h2", "h3", "h4"])
        if not heading_tag:
            continue
        title = heading_tag.get_text(strip=True)
        # Skip empty titles or reference/appendix sections
        if not title or title.lower() in ("references", "bibliography", "acknowledgements", "appendix"):
            continue

        # Extract text paragraphs, skip math-heavy or empty ones
        paragraphs = []
        for p in section_tag.find_all("p"):
            text = p.get_text(" ", strip=True)
            # Filter out very short or formula-only lines
            if len(text) > 40:
                paragraphs.append(text)

        content = " ".join(paragraphs)
        # Truncate very long sections to keep prompts manageable
        content = content[:3000] if len(content) > 3000 else content

        if content.strip():
            sections.append({"title": title, "content": content})

    if len(sections) < 2:
        print(f"  [!] Only {len(sections)} section(s) parsed from HTML; using fallback.")
        return make_fallback_sections(abstract)

    print(f"  [OK] Parsed {len(sections)} sections from HTML.")
    return sections[:8]   # cap at 8 sections


def make_fallback_sections(abstract: str) -> list[dict]:
    """
    Create plausible-looking sections from the abstract when HTML is unavailable.
    Splits the abstract into intro / method / result / conclusion quarters.
    """
    words = abstract.split()
    q = max(len(words) // 4, 30)
    chunks = [
        " ".join(words[:q]),
        " ".join(words[q:2*q]),
        " ".join(words[2*q:3*q]),
        " ".join(words[3*q:]),
    ]
    return [
        {"title": "Introduction",  "content": chunks[0]},
        {"title": "Methodology",   "content": chunks[1]},
        {"title": "Experiments",   "content": chunks[2]},
        {"title": "Conclusion",    "content": chunks[3]},
    ]


# ---------------------------------------------------------------------------
# Build Paper JSON
# ---------------------------------------------------------------------------

def build_paper_json(slot: str, meta: dict, sections: list[dict]) -> dict:
    return {
        "id":       slot,
        "title":    meta["title"],
        "abstract": meta["abstract"],
        "authors":  [{"name": a} for a in meta["authors"][:6]],
        "keywords": meta["keywords"],
        "year":     meta["year"],
        "doi":      meta["doi"],
        "sections": sections,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nFetching {len(TARGET_PAPERS)} real arXiv ML papers into {SAMPLES_DIR}\n")

    for arxiv_id, slot in TARGET_PAPERS:
        out_path = SAMPLES_DIR / f"{slot}.json"
        print(f"{'='*60}")
        print(f"Paper: arxiv:{arxiv_id}  ->  {slot}.json")

        if out_path.exists():
            data = json.loads(out_path.read_text(encoding="utf-8"))
            print(f"  [SKIP] Already downloaded: {data['title'][:60]}...")
            print(f"         ({len(data['sections'])} sections)\n")
            continue

        # 1. Metadata
        print(f"  Fetching metadata...")
        meta = fetch_arxiv_metadata(arxiv_id)
        if not meta:
            print(f"  [SKIP] Could not fetch metadata.\n")
            continue
        print(f"  Title : {meta['title'][:70]}...")
        print(f"  Year  : {meta['year']}  Authors: {len(meta['authors'])}")

        time.sleep(1)   # be polite to arXiv

        # 2. Sections from HTML
        print(f"  Fetching HTML sections from arxiv.org/html/{arxiv_id}...")
        sections = fetch_sections_from_html(arxiv_id, meta["abstract"])

        # 3. Assemble and save
        paper = build_paper_json(slot, meta, sections)
        out_path.write_text(json.dumps(paper, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Saved -> {out_path}  ({len(sections)} sections, "
              f"{sum(len(s['content']) for s in sections)} chars)\n")

        time.sleep(2)   # rate-limit courtesy pause

    print("Done! Papers saved:\n")
    for arxiv_id, slot in TARGET_PAPERS:
        p = SAMPLES_DIR / f"{slot}.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            print(f"  [{slot}] {data['title'][:65]}...")
    print()


if __name__ == "__main__":
    main()
