You are an expert Principal AI Systems Engineer specializing in Agentic Frameworks and the Model Context Protocol (MCP). We are building Phase 1 of my Master's Thesis: A Local Multi-Agent System for the Automated Quality Assurance and Editorial Review of Scientific Manuscripts.

We need to build a modular, local Python pipeline using the `google-antigravity` SDK (or standard asynchronous multi-agent design patterns) that instantiates a hierarchical peer-review board.

### 1. TARGET DIRECTORY STRUCTURE
Create the following modular structure in our workspace:
├── mcp_servers/
│   ├── __init__.py
│   ├── document_server.py      # Local Document Engine (JSON/Vector schema for the 200 raw papers)
│   └── web_search_server.py    # Academic discovery mockup/API connector (Semantic Scholar/Brave)
├── review_agents/
│   ├── __init__.py
│   ├── orchestrator.py         # Central Reviewer Agent (Routes, aggregates, arbitrates consensus)
│   ├── methodology_agent.py    # Causal logic & empirical soundness checks
│   ├── theory_agent.py         # Conceptual consistency & literature grounding checks
│   └── writing_agent.py        # Storylining, narrative progression, and flow checks
├── data/
│   └── samples/                # To hold raw papers or text mocks for early pipeline testing
├── main.py                     # Entry point orchestrating the multi-agent execution loop
└── requirements.txt            # Project dependencies

### 2. CORE COMPONENT REQUIREMENTS

#### A. MCP Server Mocks/Stubs (mcp_servers/)
- In `document_server.py`, implement a class or lightweight fast-mcp interface exposing capabilities to load, read, and query local papers. Include tool methods: `fetch_paper_by_id(id)`, `query_local_corpus(keywords)`.
- In `web_search_server.py`, expose an academic web lookup capability: `search_external_citations(keywords)` using a fallback/mock array that mimics structural web metadata retrieval.

#### B. The Specialized Sub-Agents (review_agents/)
Develop highly specialized system prompts and response schemas (JSON format) for each agent:
1. Methodology Agent: Evaluates causal claims, data validity, and formula setups.
2. Theory Agent: Compares definitions across sections to flag naming or conceptual drift.
3. Writing Agent: Checks paragraph-to-paragraph transition transitions, clarity of abstract, and structural story flow.

#### C. The Central Reviewer Agent (orchestrator.py)
- Implement a hierarchical routing state machine. 
- Step 1: Receives a target manuscript from the corpus.
- Step 2: Dispatches the document simultaneously to the Methodology, Theory, and Writing agents.
- Step 3: Gathers their JSON review structures, checks for conflicting evaluations, resolves discrepancies via an arbitration prompt, and outputs a single finalized "Systematic Peer Review Report" markdown document.

### 3. EXECUTABLE STEP-BY-STEP AGENDA FOR ANTIGRAVITY
Please execute these steps sequentially, verifying your code via the terminal:
1. Create all files and directories specified in the tree.
2. Write a `requirements.txt` file including necessary libraries (e.g., `pydantic`, `fastmcp` or standard SDK setups). Install them using the terminal.
3. Implement the clean, modular Python source code for the MCP stubs and the individual agents. Use strict typing and Pydantic schemas for LLM tool outputs where appropriate.
4. Write a robust mock sample dataset in `data/samples/` representing a flawed scientific paper so we can verify our pipeline works.
5. Implement the main execution orchestration pipeline in `main.py` that hooks everything together and runs a test execution.
6. Run the code in the terminal to verify zero runtime errors and present the resulting Markdown peer review artifact.