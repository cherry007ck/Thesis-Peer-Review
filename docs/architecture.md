# System Architecture

The Local Multi-Agent Peer Review System is a robust, locally-hosted LLM pipeline designed to evaluate scientific manuscripts (specifically in Machine Learning and Causal Discovery) mimicking a real academic peer-review process.

## High-Level Workflow

The system operates as a state machine managed by the `Orchestrator`, transitioning through the following states:

1. **IDLE**: Awaiting a paper to review.
2. **RETRIEVING**: 
   - Loads the target paper from the local `DocumentServer`.
   - Cleans the paper's metadata to generate a natural language search query.
   - Queries the live internet via the `WebSearchServer` (Semantic Scholar + arXiv) to find related SOTA papers and citations.
3. **DISPATCHING**:
   - The manuscript and the retrieved internet context are packaged.
   - The payload is dispatched asynchronously to four independent specialist agents (`MethodologyAgent`, `TheoryAgent`, `WritingAgent`, `SOTAAgent`).
4. **AGGREGATING**:
   - Agents return their structured Pydantic evaluations.
   - The orchestrator detects any severe scoring conflicts (e.g., strong writing but fatally flawed methodology).
5. **ARBITRATING**:
   - Conflicts are resolved using a weighted scoring model where Methodology and SOTA positioning take precedence.
6. **REPORTING**:
   - The final scores, agent reviews, internet search results, and final editorial recommendation are rendered into a comprehensive Markdown report.
7. **DONE**: Pipeline complete.

## Core Components

### 1. Model Inference (`_ollama_client.py`)
All intelligence is powered by local LLMs via Ollama (default: `llama3:8b-instruct-q4_0`). 
- **Context Truncation**: Automatically chunks and truncates manuscript sections to fit safely within the 8K context window.
- **Robust JSON Parsing**: Uses regex to strip markdown fences and extract raw JSON.
- **Auto-Recovery**: If the LLM hallucinates schema keys or emits invalid JSON escape sequences, the client catches the `JSONDecodeError` and automatically retries the inference.
- **List Normalization**: Implements `flatten_str_list` to gracefully coerce dictionary-lists (a common LLM hallucination) back into standard string lists to satisfy Pydantic validators.

### 2. MCP Servers (Data Layer)
- **DocumentServer**: Loads local `.json` manuscript files (following the `Paper` schema) from the `data/samples/` directory.
- **WebSearchServer**: Interfaces with real-world academic APIs to ground the LLM's evaluations in actual literature.

### 3. Agent Pool
Four independent agents evaluate the paper concurrently using `asyncio.gather()`. See `agents.md` for detailed agent breakdowns.

## Technology Stack
- **Python 3.10+** (Asyncio for concurrency)
- **Ollama** (Local LLM inference)
- **Pydantic v2** (Strict schema validation for agent outputs)
- **HTTPX & Feedparser** (Async web requests and XML parsing)
- **BeautifulSoup4** (HTML section extraction for test data)
