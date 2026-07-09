# MCP Servers (Data Layer)

The system isolates data retrieval into modular "servers", inspired by the Model Context Protocol (MCP) design pattern. This ensures agents remain stateless and focused solely on inference.

## 1. Document Server (`mcp_servers/document_server.py`)
Acts as the local database for manuscripts.
- **Functionality**: Scans the `data/samples/` directory on boot and loads all `.json` files.
- **Schema Validation**: Validates all loaded JSON files against the strict `Paper` Pydantic model (Title, Abstract, Authors, Keywords, Sections).
- **Retrieval**: Provides `fetch_paper_by_id(paper_id)` to the Orchestrator.

## 2. Web Search Server (`mcp_servers/web_search_server.py`)
Acts as the bridge to the live internet, gathering grounding context to prevent LLM hallucinations. It wraps two specialized API clients:

### Semantic Scholar API (`semantic_scholar.py`)
- Searches the Semantic Scholar graph API using title keywords.
- Retrieves highly-cited, peer-reviewed papers.
- Extracts Title, Authors, Year, Venue, Abstract, and Citation Counts.

### arXiv API (`arxiv_search.py`)
- Searches the arXiv preprint server using the Atom XML feed (`feedparser`).
- Filters for recent preprints (SOTA) in Computer Science / Machine Learning categories.
- Extracts Title, Authors, Year, and Abstracts.

**Concurrency & Rate Limiting**:
The Orchestrator dispatches searches to these APIs via `asyncio.to_thread`. To avoid HTTP 429 Rate Limit errors from Semantic Scholar, requests are serialized with courteous time delays, and HTTP clients are configured with `follow_redirects=True` to handle arXiv's 301 redirects.
