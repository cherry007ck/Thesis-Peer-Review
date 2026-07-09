# Local Multi-Agent Peer Review System

A hierarchical, multi-agent AI system designed to automate and augment the scientific peer review process. This pipeline uses local LLMs (via [Ollama](https://ollama.com/)) and Retrieval-Augmented Generation (RAG) to evaluate academic manuscripts on methodology, theoretical grounding, writing quality, and state-of-the-art (SOTA) positioning.

This project was built for a Master's Thesis.

## Features

- **Multi-Agent Architecture**: Four specialized reviewing agents (Methodology, Theory, Writing, and SOTA) driven by `llama3:8b-instruct`.
- **Live Internet Grounding**: Queries Semantic Scholar and arXiv APIs to retrieve real-time SOTA papers for comparison.
- **Local RAG Corpus**: Ingests and embeds a local corpus of PDFs/JSONs using `ChromaDB` and `nomic-embed-text` to check the manuscript against internal knowledge.
- **Hierarchical Orchestration**: An Orchestrator agent that detects conflicts between reviewer scores, arbitrates disagreements, and compiles a comprehensive Markdown report.
- **100% Local Inference**: Runs entirely on your local machine using Ollama, keeping proprietary research data private.

## Architecture

The system operates as a state machine (`IDLE -> RETRIEVING -> DISPATCHING -> AGGREGATING -> ARBITRATING -> REPORTING`):

1. **Document Server**: Loads the target manuscript.
2. **Web Search Server**: Retrieves live citations and abstracts.
3. **RAG Server**: Queries the local ChromaDB for semantically similar chunks.
4. **Agent Pool**: Dispatches all context (Manuscript + Web Papers + RAG Chunks) to the four agents concurrently.
5. **Arbitration**: Compiles scores and generates the final `review_report_*.md`.

For deeper technical details, see the documentation in `docs/`:
- [Architecture Overview](docs/architecture.md)
- [Agent Specifications](docs/agents.md)
- [MCP Server Setup](docs/mcp_servers.md)

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com/) (installed and running)
- Required Models: `ollama pull llama3:8b-instruct-q4_0` and `ollama pull nomic-embed-text`

## Setup & Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Index your local corpus:**
   To build the ChromaDB vector database, place your `.json` papers in `data/samples/` and run:
   ```bash
   python scripts/index_corpus.py
   ```

3. **Run a Review:**
   ```bash
   python main.py paper_002
   ```
   
   The Orchestrator will run the pipeline and output a full peer-review report to the project root (e.g., `review_report_paper_002.md`).
