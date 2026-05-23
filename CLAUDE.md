# Claude Code Project Kickoff: SOC False Positive Reduction POC

## Instructions for Claude Code

You are building a proof-of-concept AI system that reduces false positive alerts in an enterprise SOC (Security Operations Center) environment. This project follows spec-based Agile development. Before writing any code, you will produce the specification documents described below. After specs are approved, you will execute sprints with epics, stories, and tasks. Every story must have passing tests before moving to the next story.

-----

## Project Context

### What this system does

A two-stage hybrid alert triage pipeline:

- **Stage 1**: A calibrated LightGBM/XGBoost classifier scores every alert for P(true_positive). Conformal prediction splits alerts into three bands: auto-close FP, escalate TP, and uncertain.
- **Stage 2**: The uncertain band goes to an LLM (Claude API) with RAG over historical alert dispositions. The LLM reasons from precedent and outputs a structured verdict with natural language explanation.
- **Adversarial validation**: A second LLM call with a different prompt attempts to disprove Stage 2 findings (inspired by Cloudflare’s Project Glasswing multi-agent harness architecture, May 2026).
- **Analyst UI**: Streamlit dashboard showing alert details, SHAP explanations, LLM rationale, similar historical alerts, and feedback capture.

### Key design decisions (already made, do not revisit)

1. **Dataset**: CICIDS2017 as primary (2.8M flows, 78 features, 5-day capture). No OCSF mapping for this POC – use the dataset’s native schema.
1. **Stage 1 model**: LightGBM with `is_unbalance=True`. XGBoost as alternative for comparison. Not deep learning – XGBoost/LightGBM outperform DL on tabular security data (Shwartz-Ziv & Armon 2021, confirmed 2024 111-dataset benchmark).
1. **Class imbalance handling**: cost-sensitive learning via `is_unbalance=True` or `scale_pos_weight`. Skip CTGAN/SMOTE for this POC.
1. **Conformal prediction**: `mapie` library, alpha=0.05 for 95% coverage guarantee. Three bands: P(TP)<0.05 auto-FP, P(TP)>0.85 auto-TP, middle band routes to Stage 2.
1. **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim). CUDA when available, CPU fallback.
1. **Vector store**: `faiss-cpu` for the RAG retrieval layer.
1. **Stage 2 LLM**: Anthropic Claude API (claude-sonnet-4-20250514). Configurable to swap for local model later.
1. **Explainability**: SHAP TreeExplainer on the LightGBM output. LLM-generated natural language rationale for Stage 2.
1. **Evaluation**: PR-AUC as primary metric (not ROC-AUC). Temporal hold-out: train on days 1-4, test on day 5.
1. **No GPU required for core pipeline**. LightGBM and SHAP are CPU-only. Embeddings benefit from GPU but work on CPU. LLM inference is API-based.

### What is explicitly out of scope for this POC

- OCSF schema mapping layer
- CTGAN synthetic data generation
- GNN/graph-based features (Tier 4)
- Foundation-Sec-8B fine-tuning
- Analyst feedback loop and retraining
- Adversarial robustness testing with ART (FGSM/PGD)
- Multi-dataset compositing (UNSW-NB15, BETH, ToN-IoT)
- Kafka/streaming ingestion (use file-based replay)

-----

## Phase 1: Specification Documents

Before any code, produce three documents in a `/docs` directory:

### 1. Requirements Document (`docs/requirements.md`)

- Functional requirements (what the system must do)
- Non-functional requirements (latency targets, accuracy thresholds, API rate limits)
- Acceptance criteria per requirement (testable, measurable)
- Use the following targets from the research:
  - Stage 1 PR-AUC >= 0.85 on temporal hold-out
  - Auto-FP band false negative rate <= 1% (conformal guarantee)
  - Stage 2 LLM response time < 10 seconds per alert
  - Stage 1 scoring latency < 500ms per alert
  - End-to-end alert volume reduction >= 70%
  - True positive recall >= 95%
  - SHAP explanation generated for every Stage 1 decision
  - LLM rationale generated for every Stage 2 decision

### 2. Architecture & Design Document (`docs/architecture.md`)

- System architecture diagram (ASCII or mermaid)
- Component breakdown with interfaces
- Data flow from ingestion to final disposition
- The three-band conformal routing logic
- RAG retrieval pipeline design
- Adversarial validation agent design
- Stage 2 LLM prompt template (use the one from the research document Section 5.3)
- Technology stack with versions
- Directory/module structure

### 3. Test Plan (`docs/test_plan.md`)

- Unit test specifications per component
- Integration test specifications (use a 10K-row stratified subset as test fixture)
- End-to-end smoke test (5 alerts through full pipeline including real Claude API call)
- Performance benchmarks (latency, throughput)
- Metric validation tests (PR-AUC, precision, recall, F1 against thresholds)
- Mock strategy for Claude API in unit/integration tests (use fixture responses)

**Wait for approval of all three documents before proceeding to Phase 2.**

-----

## Phase 2: Sprint Planning

After specs are approved, create a sprint backlog (`docs/sprint_backlog.md`) organized as follows:

### Epic 1: Data Ingestion & Stage 1 Classifier

**Story 1.1**: Dataset acquisition and loading

- Download CICIDS2017 from canonical source
- Load into pandas/DuckDB
- Validate schema and row counts
- Create the 10K stratified test fixture subset
- **Tests**: data integrity checks, shape validation, class distribution verification

**Story 1.2**: Feature engineering (Tier 1)

- Extract/clean the 78 flow features
- Handle missing values and infinities (known CICIDS2017 issue)
- Create temporal features (time-of-day, day-of-week from flow timestamps)
- Train/test split: days 1-4 train, day 5 test (temporal hold-out)
- **Tests**: no NaN/inf in output, feature ranges validated, temporal split correctness

**Story 1.3**: LightGBM classifier training and evaluation

- Train LightGBM with `is_unbalance=True`
- Hyperparameter tuning via 5-fold stratified CV on training set only
- Evaluate on temporal hold-out: PR-AUC, precision, recall, F1, confusion matrix
- Generate SHAP values for all test predictions
- Save trained model artifact
- **Tests**: PR-AUC >= 0.85, SHAP values generated for every prediction, model artifact saved and loadable

### Epic 2: Conformal Calibration & Stage 2 LLM Adjudication

**Story 2.1**: Conformal prediction and three-band routing

- Apply `mapie` conformal prediction (alpha=0.05)
- Implement three-band routing logic with configurable thresholds
- Measure band distribution (what % falls in each band)
- Verify auto-FP band false negative rate <= 1%
- **Tests**: conformal coverage >= 95%, band assignment deterministic, FN rate in auto-FP band validated

**Story 2.2**: RAG retrieval layer

- Embed historical alerts (training set dispositions) with MiniLM-L6-v2
- Build FAISS index
- Implement top-5 similarity retrieval for a given alert
- **Tests**: embedding dimensions correct (384), FAISS index queryable, retrieval returns 5 results with cosine scores

**Story 2.3**: Stage 2 LLM adjudication

- Implement the structured prompt template (alert + enrichment + similar historicals + SHAP top-5)
- Call Claude API with structured JSON output
- Parse response: verdict, confidence, rationale, recommended_actions
- Implement the adversarial validation agent (second call, different prompt, tries to disprove)
- Reconciliation logic when Stage 2 and adversarial agent disagree
- **Tests**: prompt renders correctly, API response parses to expected schema, adversarial agent produces counter-arguments, disagreement handled gracefully. Mock Claude API in unit tests.

**Story 2.4**: End-to-end pipeline integration

- Wire Stage 1 -> conformal routing -> Stage 2 -> final disposition
- Implement tripwire logic (retroactive IOC check on auto-closed alerts within 7-day window)
- Measure end-to-end metrics on full test set
- **Tests**: full pipeline runs on 10K subset without errors, metrics computed and logged, tripwire triggers correctly on synthetic re-opened alert

### Epic 3: Analyst UI & Demo

**Story 3.1**: Streamlit dashboard

- Alert list view with disposition, confidence, band assignment
- Detail view: alert fields, SHAP force plot, LLM rationale, similar historical alerts
- Filter by band (auto-FP, uncertain, auto-TP)
- Feedback capture (analyst can override disposition with rationale)
- **Tests**: Streamlit app launches, all views render, feedback saves to DB/file

**Story 3.2**: Metrics dashboard

- PR-AUC curve visualization
- Confusion matrix heatmap
- Band distribution pie chart
- Alert volume reduction summary
- Analyst time savings estimate (7 min/alert median)
- **Tests**: all charts render with real data from the evaluation run

**Story 3.3**: Documentation and demo

- Update README with setup instructions, architecture diagram, results summary
- Record a walkthrough or produce screenshots for portfolio
- **Tests**: README renders correctly, all setup steps reproducible

-----

## Development Rules

1. **Test-gated progression**: Do not start the next story until all tests for the current story pass. Run the full test suite after each story completion.
1. **Test command**: `pytest tests/ -v --tb=short` for the full suite. Each epic should have its own test module (e.g., `tests/test_epic1_data.py`, `tests/test_epic2_llm.py`).
1. **No hardcoded paths**: Use a config file (`config.yaml` or `settings.py`) for data paths, API keys, model paths, thresholds.
1. **Logging**: Use Python `logging` module, not print statements. Log level configurable.
1. **Type hints**: All function signatures must have type hints.
1. **Docstrings**: All public functions must have docstrings.
1. **Git commits**: Commit after each passing story with a descriptive message referencing the story number.

-----

## Project Structure

```
soc-fp-reduction/
├── docs/
│   ├── requirements.md
│   ├── architecture.md
│   ├── test_plan.md
│   └── sprint_backlog.md
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # Dataset download and loading
│   │   └── features.py        # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py      # LightGBM/XGBoost training and inference
│   │   ├── conformal.py       # Conformal prediction and band routing
│   │   └── explainer.py       # SHAP explanations
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── embeddings.py      # Sentence-transformer embeddings
│   │   ├── retrieval.py       # FAISS RAG retrieval
│   │   ├── adjudicator.py     # Stage 2 LLM adjudication
│   │   └── adversarial.py     # Adversarial validation agent
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── orchestrator.py    # End-to-end pipeline
│   │   └── tripwire.py        # Retroactive IOC check
│   └── ui/
│       ├── __init__.py
│       └── dashboard.py       # Streamlit app
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Shared fixtures (10K subset, mock API)
│   ├── test_epic1_data.py
│   ├── test_epic2_llm.py
│   └── test_epic3_ui.py
├── config.yaml
├── requirements.txt
├── CLAUDE.md                  # Project conventions for Claude Code
└── README.md
```

-----

## Tech Stack (pin versions)

```
python>=3.11
pandas>=2.2
numpy>=1.26
scikit-learn>=1.5
lightgbm>=4.5
xgboost>=2.1
shap>=0.46
mapie>=0.9
imbalanced-learn>=0.12
sentence-transformers>=3.2
torch>=2.4
faiss-cpu>=1.8
anthropic>=0.40
duckdb>=1.1
streamlit>=1.39
pydantic>=2.9
pytest>=8.0
pyyaml>=6.0
```

-----

## Reference Material

The attached research document (`AI-Driven_False_Positive_Reduction_for_Enterprise_SOC_Alert_Triage_-_A_Build-Ready_Research_Synthesis__May_2026__-_POC.md`) contains:

- Section 1: Problem definition with quantitative baselines
- Section 2: Dataset details, limitations, and synthetic generation options
- Section 3: Feature engineering tiers with SHAP-ranked predictive power
- Section 4: Model architecture comparison and the two-stage cascade recommendation
- Section 5: Explainability (SHAP, LLM rationale prompt template, conformal calibration)
- Section 6: System architecture diagram and tech stack
- Section 7: Evaluation framework (metrics, temporal hold-out, statistical significance)
- Section 8: Risks, adversarial considerations, concept drift, data poisoning

Use this document as the authoritative reference for all design decisions. Do not contradict its findings or recommendations.

-----

## Writing Style Rules

- Never use em dashes in any document or code comment.
- Follow anti-AI writing style: avoid words like “delve”, “crucial”, “comprehensive”, “robust”, “streamline”, “leverage”, “cutting-edge”, “paradigm”, “synergy”, “holistic”, “transformative”. Write plainly and directly.
- No promotional language. State what the system does, not how impressive it is.
- Technical documents should be precise and concrete, not aspirational.

-----

## Start Here

1. Read this entire prompt and the attached research document.
1. Produce the three specification documents (requirements, architecture, test plan).
1. Present them for review.
1. After approval, produce the sprint backlog.
1. After sprint backlog approval, begin Epic 1, Story 1.1.
1. After each story, run the test suite and confirm all tests pass before proceeding.