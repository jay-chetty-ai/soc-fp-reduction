# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [3.0.0] -- Epic 3 Complete -- 2026-05-26

### Added

**Analyst dashboard (Story 3.1)**
- `src/ui/dashboard.py`: full Streamlit analyst UI with role-based authentication via `streamlit-authenticator`
- Alert list view with sortable table, band color coding (auto-FP: green, uncertain: amber, auto-TP: red), and sidebar band filter
- Alert detail view: feature table, SHAP horizontal bar chart (red = push toward TP, blue = push toward FP), LLM rationale, adversarial counter-rationale, top-5 similar historical alerts with cosine similarity scores, recommended actions
- Feedback capture (analyst role only): verdict override dropdown, free-text rationale, writes to `data/processed/feedback.jsonl` and audit log
- Dark/light mode toggle with CSS injection; theme persists in session state
- Session timeout enforced from `config.yaml auth.session_timeout_minutes`
- Viewer role: feedback section hidden; all read-only views available

**Metrics page (Story 3.2)**
- PR-AUC precision-recall curve (matplotlib)
- Confusion matrix heatmap (seaborn)
- Band distribution pie chart (matplotlib)
- Summary table: total alerts, band counts, volume reduction %, analyst hours saved
- Reads from latest `metrics/evaluation_*.json` written by `scripts/run_pipeline.py`

**Pipeline metrics output (Story 3.2)**
- `scripts/run_pipeline.py`: `_compute_metrics()` computes PR-AUC, precision, recall, F1, confusion matrix, decimated PR curve (200 pts), band counts, verdict counts, volume reduction, analyst time saved; writes to `metrics/evaluation_<timestamp>.json` and `results/evaluation_<timestamp>.parquet`
- `src/pipeline/orchestrator.py`: `DispositionRecord` extended with `stage2_verdict`, `stage2_confidence`, `stage2_rationale`, `adversarial_verdict`, `adversarial_rationale`, `final_confidence`, `reconciliation_note`, `recommended_actions`, `similar_alerts`; SHAP computed for all three bands

**Documentation (Story 3.3)**
- `docs/setup.md`: full setup guide with 3-command pipeline workflow, CICIDS2017 download instructions, config reference, troubleshooting
- `docs/screenshots/`: placeholder directory for dashboard screenshots (captured after training run)
- `README.md`: Epic status table updated to Complete, test count 148, default dashboard credentials, dashboard launch instructions

**Auth config (S8)**
- `config.yaml auth:` section: bcrypt-hashed credentials for `analyst` and `viewer` accounts, cookie settings, session timeout

**Tests (Story 3.1-3.3)**
- `tests/test_epic3_ui.py`: 45 tests covering DispositionRecord schema, data loading, band filtering, user role resolution, SHAP chart rendering, feedback writing, metrics chart rendering, metrics correctness, and module imports

### Changed
- `config.yaml`: added `auth` and `dashboard` sections
- `src/pipeline/orchestrator.py`: `DispositionRecord` now carries full Stage 2 and adversarial fields; `shap_top5` populated for all bands (not uncertain only)

---

## [2.0.0] -- Post-Epic-2 Production Hardening -- 2026-05-25

All production code paths now use real implementations. No stubs or placeholders remain in `src/` or `scripts/`.

### Added
- `scripts/build_rag_index.py`: embeds the full training set (or a stratified sample via `--sample-size N`), builds a FAISS flat inner-product index, and saves both `models/faiss_index.bin` and `models/training_df.parquet`
- `scripts/run_pipeline.py`: production entry point; loads all saved artifacts, creates a real Anthropic client, processes a batch of alerts end-to-end, writes output CSV, prints band distribution and volume reduction summary
- `src/models/conformal.py`: `save_conformal()` and `load_conformal()` with SHA-256 integrity verification; shares `models/checksums.json` with the LightGBM model artifact
- `src/pipeline/tripwire.py`: `TripwireStore(path=...)` for JSON Lines file persistence; loads existing records from disk on startup and appends new records in real time; `TripwireStore(path=None)` remains in-memory for tests

### Changed
- `config.yaml`: added `conformal.artifact_path`, `stage1.shap_top_k`, `rag.embedding_batch_size`, `rag.training_df_path`, `adversarial.confidence_threshold_high`, `data.test_day` -- all were previously hardcoded literals in source files
- `src/llm/adversarial.py`: `reconcile()` now accepts `confidence_threshold` as a parameter (default from config); `_CONFIDENCE_THRESHOLD_HIGH = 0.80` kept as module-level default
- `src/llm/embeddings.py`: `embed_alerts()` accepts `batch_size` parameter; callers pass `config["rag"]["embedding_batch_size"]`
- `src/llm/a2a/client.py`: inprocess mode passes `max_retries` from config into adjudicator graph state
- `src/pipeline/orchestrator.py`: reads `shap_top_k` and `confidence_threshold_high` from config; passes threshold to `reconcile()`
- `scripts/train_stage1.py`: fits and saves conformal predictor after model training; reads `test_day` from config
- All three scripts now read `test_day` from `config["data"]["test_day"]` instead of using a hardcoded `5`

---

## [1.0.0] -- Epic 2 Complete -- 2026-05-25

### Added

**Conformal prediction (Story 2.1)**
- `src/models/conformal.py`: `SplitConformalClassifier` (MAPIE 1.4.0 API) with `_BoosterWrapper` to bridge `lgb.Booster` to sklearn's `predict_proba` interface; `fit_conformal()`, `predict_bands()`, `compute_coverage()`; three-band routing (`auto_fp`, `uncertain`, `auto_tp`)

**RAG retrieval layer (Story 2.2)**
- `src/llm/embeddings.py`: `load_model()`, `embed_alerts()`, `alert_to_text()`, `resolve_device()` with CUDA auto-detect
- `src/llm/retrieval.py`: `build_index()` (FAISS `IndexFlatIP`), `save_index()`, `load_index()`, `retrieve_similar()` with L2-normalized cosine similarity

**LLM adjudication (Story 2.3)**
- `src/llm/sanitizer.py` (S1): strips control characters, null bytes, injection phrases, escapes XML delimiters; wraps content in `<alert_data>` tags
- `src/llm/redactor.py` (S6): field allowlist strips IPs and non-network-feature fields before any data crosses the trust boundary
- `src/llm/validators.py` (S5): Pydantic schemas for `Stage2Verdict`, `AdversarialVerdict`, `FinalVerdict`; `parse_llm_response()` falls back to `needs_review` on any parse failure
- `src/llm/rate_limiter.py` (S7): token-bucket rate limiter, circuit breaker, exponential backoff with jitter
- `src/llm/adjudicator.py`: prompt assembly using Section 5.3 template with SHAP top-k and RAG context
- `src/llm/adversarial.py`: adversarial counter-prompt, `reconcile()` with configurable confidence threshold
- `src/llm/graphs/adjudicator_graph.py`: LangGraph `StateGraph` with sanitize -> build_prompt -> call_llm -> validate nodes; retry up to `max_retries`; fallback to `needs_review`
- `src/llm/graphs/adversarial_graph.py`: same graph structure with counter-prompt and adversarial state
- `src/llm/a2a/client.py`: inprocess A2A client invoking LangGraph graphs directly; HTTP mode not implemented
- `src/llm/a2a/agent_cards/adjudicator.json`, `adversarial.json`: A2A Agent Cards

**Pipeline integration (Story 2.4)**
- `src/pipeline/orchestrator.py`: end-to-end wiring; `PipelineComponents` dataclass; `run_batch()` routes by band, calls A2A agents, writes audit entries per alert, catches per-alert exceptions
- `src/pipeline/tripwire.py`: `TripwireStore` with `AutoFPRecord` Pydantic model; `record()` and `check_ioc()` with 7-day lookback window

**Tests**
- `tests/test_epic2_llm.py`: 60+ tests covering conformal, RAG, adjudication, adversarial, reconciliation, LangGraph graph execution, A2A protocol, pipeline integration
- `tests/test_security.py`: 30+ tests for all security controls S1-S7

**Configuration**
- Added `stage2`, `adversarial`, `rag`, `agents`, `tripwire`, `a2a` sections to `config.yaml`

---

## [0.1.0] -- Epic 1 Complete -- 2026-05-25

### Added

**Data ingestion (Story 1.1)**
- `src/data/loader.py`: `load_dataset()`, `validate_schema()`, `create_fixture_subset()`; infers timestamps from filenames (CICIDS2017 ML release strips original timestamps)
- `src/utils/secrets.py` (S2): `load_api_key()` with format validation, `redact_secrets()`, `RedactionFilter` logging handler
- `src/utils/audit.py` (S3): `AuditLogger` with SHA-256 hash chain; `AuditEntry` and `FeedbackEntry` Pydantic models; `logs/audit.jsonl` append-only log
- `data/fixtures/fixture_10k.csv`: stratified 10K-row subset of real CICIDS2017 data; committed for CI

**Feature engineering (Story 1.2)**
- `src/data/features.py`: `clean_features()` (removes NaN/inf, clips negatives from CICFlowMeter timer bug), `add_temporal_features()`, `temporal_train_test_split()`, `get_feature_columns()`, `encode_labels()`

**Stage 1 classifier (Story 1.3)**
- `src/models/classifier.py`: `train()`, `tune()` (Optuna TPE, 5-fold stratified CV, convergence callback), `evaluate()`, `predict_proba()`, `save_model()`, `load_model()`
- `src/models/explainer.py`: `build_explainer()`, `explain_batch()`, `top_k_features()`; handles SHAP 0.51 list/ndarray output variation
- `src/models/integrity.py` (S4): `save_hash()`, `verify_hash()`, `ModelIntegrityError`
- `scripts/train_stage1.py`: CLI entry point with `--skip-tuning` flag and `--config` override

**Tests**
- `tests/test_epic1_data.py`: 40+ tests covering data loading, feature engineering, classifier training, SHAP values, model integrity
- `tests/test_security.py`: initial tests for S2, S3, S4

**Configuration**
- `config.yaml`: `data`, `stage1`, `tuning` sections
- `.env.example`: API key template

---

## [0.0.1] -- Initial scaffold -- 2026-05-25

### Added
- Project directory structure
- `CLAUDE.md` project specification
- `docs/requirements.md`, `docs/architecture.md`, `docs/test_plan.md`, `docs/sprint_backlog.md`, `docs/threat_model.md`, `docs/setup.md`
- `requirements.txt` with pinned dependencies
- `config.yaml` skeleton
- `.env.example`
