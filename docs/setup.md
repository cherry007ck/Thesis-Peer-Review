# Setup and Usage

## Prerequisites
1. **Python 3.10+**
2. **Ollama**: Must be installed and running locally.
3. **Local LLM**: Pull the default inference model before running.
   ```bash
   ollama run llama3:8b-instruct-q4_0
   ```

## Installation
1. Clone the repository.
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Includes `pydantic`, `ollama`, `httpx`, `feedparser`, `beautifulsoup4`)*

## Fetching Real Test Data
To test the pipeline on real Machine Learning papers, run the fetch script. It uses BeautifulSoup to parse HTML versions of arXiv preprints and format them into the local JSON schema:
```bash
python scripts/fetch_test_papers.py
```
This will populate `data/samples/` with papers like "Attention Is All You Need" (`paper_002.json`).

## Running the Pipeline
Execute the main entrypoint and provide the ID of the paper you wish to review:
```bash
python -u main.py paper_002
```
*(Note: The `-u` flag ensures unbuffered output so you can see live progress in the terminal).*

**Outputs:**
- A detailed terminal summary of agent scores and arbitrary conflicts.
- A comprehensive Markdown report saved to the project root: `review_report_paper_002.md`.
