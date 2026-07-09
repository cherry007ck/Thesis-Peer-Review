# Peer Review Agents

The system utilizes four specialized LLM agents. Each agent is provided with the manuscript text and a list of real-world retrieved papers. They evaluate the manuscript from different perspectives and output strict JSON adhering to Pydantic schemas.

## 1. Methodology Agent (`MethodologyAgent`)
**Focus:** Evaluates the empirical soundness, experimental design, and statistical rigour of the paper.
**Weight in Final Score:** 40%
**Key Validations:**
- Detects unsupported causal claims (e.g., claiming "A causes B" when the method only proves correlation).
- Validates dataset diversity and appropriateness.
- Checks mathematical formulas and metrics for missing justifications.
- Identifies missing experimental controls.

## 2. SOTA Comparison Agent (`SOTAAgent`)
**Focus:** Evaluates the novelty and positioning of the paper against real, currently published literature.
**Weight in Final Score:** 30%
**Key Validations:**
- Compares the manuscript's claims against the live papers retrieved from Semantic Scholar and arXiv.
- Identifies **Novelty Gaps** (where the paper claims a "first" that has already been done).
- Flags **Superseded Claims** (where newer literature invalidates the paper's premises).
- Suggests **Missing Citations** based on the retrieved context.

## 3. Theory Agent (`TheoryAgent`)
**Focus:** Evaluates the literature review, theoretical grounding, and conceptual consistency.
**Weight in Final Score:** 20%
**Key Validations:**
- Detects **Concept Drift** (where the author defines a term in the Introduction, but uses it differently in the Methodology).
- Identifies structural gaps in the literature review.
- Flags contradictory definitions.

## 4. Writing Agent (`WritingAgent`)
**Focus:** Evaluates narrative flow, clarity, and structural coherence.
**Weight in Final Score:** 10%
**Key Validations:**
- Assesses the abstract for clarity and completeness.
- Analyzes transition issues between sections (e.g., abrupt jumps from Methodology to Results without connective tissue).
- Flags structural formatting issues.

## Final Recommendation Arbitration
The Orchestrator weights the agent scores to produce a final decision out of 10.0:
- **>= 8.0:** ACCEPT
- **>= 6.5:** MINOR_REVISION
- **>= 4.5:** MAJOR_REVISION
- **< 4.5:** REJECT
