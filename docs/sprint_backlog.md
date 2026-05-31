# Sprint Backlog: SOC False Positive Reduction POC

**Version**: 2.1  
**Date**: 2026-05-28  
**Status**: Approved and complete -- v1.1 Story 1.2b done, PR open on feature/stratified-split-evaluation

## Status Summary

| Epic | Description | Status |
|------|-------------|--------|
| 1 | Data Ingestion and Stage 1 Classifier | **Complete** (157 tests passing); Story 1.2b complete |
| 2 | Conformal Calibration and Stage 2 LLM | **Complete** (157 tests passing) |
| Post-Epic-2 | Production hardening (real scripts, config cleanup) | **Complete** |
| 3 | Analyst UI and Demo | **Complete** (157 tests passing) |

---

## Ground Rules

- No story starts until all tests for the prior story pass: `pytest tests/ -v --tb=short`
- Every story ends with a git commit referencing the story number
- Security controls S1-S8 are implemented as part of the stories below, not after
- `config.yaml` is the single source of truth for all thresholds, paths, and parameters
- No hardcoded values, no print statements, type hints and docstrings on all public functions

---

## Epic 1: Data Ingestion and Stage 1 Classifier -- **COMPLETE**

---

### Story 1.1: Dataset Acquisition and Loading

**Goal**: Download CICIDS2017, load it into pandas, validate the schema, create the 10K stratified fixture, and stand up the two foundational security utilities (secrets management and audit logging) that every subsequent story depends on.

**Why secrets and audit first**: `src/utils/secrets.py` (S2) must exist before any code that touches the API key, and `src/utils/audit.py` (S3) must exist before the pipeline records any decision. Both are simpler to build correctly from the start than to retrofit.

#### Tasks

| # | Task | File(s) | Notes |
|---|------|---------|-------|
| 1.1.1 | Download CICIDS2017 CSVs | `scripts/download_data.py` | Source: Canadian Institute for Cybersecurity. Files: Monday-WorkingHours.pcap_ISCX.csv through Friday. Store in `data/raw/`. Log file sizes and row counts. |
| 1.1.2 | Implement `secrets.py` (S2) | `src/utils/secrets.py` | `load_api_key()`: reads `ANTHROPIC_API_KEY` from env, validates `sk-ant-` prefix, raises `ValueError` with descriptive message on failure. `redact_secrets(msg)`: regex replace of `sk-ant-[^\s]+` with `[REDACTED]`. `RedactionFilter`: `logging.Filter` subclass that redacts every log record message before emission. Wire `RedactionFilter` into the root logger at app startup. |
| 1.1.3 | Implement `audit.py` (S3) | `src/utils/audit.py` | `AuditEntry` and `FeedbackEntry` Pydantic models per architecture Section 2.11. `AuditLogger.log_decision()` and `log_feedback()` write JSON lines to `logs/audit.jsonl`. SHA-256 hash chain: each entry carries `previous_entry_hash`; first entry uses `sha256("GENESIS")`. |
| 1.1.4 | Implement `loader.py` | `src/data/loader.py` | `load_dataset(config)`: glob CSVs from `config.data.raw_dir`, concatenate, strip whitespace from column names. `validate_schema(df)`: assert 79+ columns, `Label` present, no duplicate names. `create_fixture_subset(df, n, random_state)`: stratified sample via `sklearn.model_selection.StratifiedShuffleSplit`. Write fixture to `data/fixtures/fixture_10k.csv`. |
| 1.1.5 | Write Story 1.1 tests | `tests/test_epic1_data.py` | TC-1.1.1 through TC-1.1.5 (schema validation, fixture stratification, row count, column whitespace). |
| 1.1.6 | Write security tests for S2 and S3 | `tests/test_security.py` | TC-S.15 through TC-S.18 (hash chain), TC-S.22 through TC-S.24 (secrets redaction, fail-fast). |

**Definition of Done**:
- `pytest tests/test_epic1_data.py::TestDataLoader tests/test_security.py -v` passes with 0 failures
- `data/fixtures/fixture_10k.csv` committed to the repository
- `logs/audit.jsonl` created on first audit log write (empty on fresh checkout is fine)

**Commit message**: `story-1.1: dataset loading, fixture creation, secrets management (S2), audit logging (S3)`

---

### Story 1.2: Feature Engineering

**Goal**: Clean the 78 raw CICIDS2017 features, create temporal features, and produce the deterministic temporal train/test split.

#### Tasks

| # | Task | File(s) | Notes |
|---|------|---------|-------|
| 1.2.1 | Implement `features.py` | `src/data/features.py` | `clean_features(df)`: replace `inf`/`-inf` with `NaN`, drop rows with any NaN in feature columns, log dropped count at WARNING. `add_temporal_features(df)`: parse `Timestamp` column (format `DD/MM/YYYY HH:MM:SS`), add `hour_of_day` (int, 0-23) and `day_of_week` (int, 0-6). `temporal_train_test_split(df, test_day=5)`: split on day number within dataset's date range; day 5 = test, days 1-4 = train. `get_feature_columns()`: returns list of numeric feature column names, excluding `Label`, `Timestamp`, and any object-dtype columns. `encode_labels(df)`: BENIGN -> 0, all others -> 1. |
| 1.2.2 | Write Story 1.2 tests | `tests/test_epic1_data.py` | TC-1.2.1 through TC-1.2.8 (no NaN/inf, temporal ranges, feature count, split correctness, no overlap, physical bounds). |

**Definition of Done**:
- `pytest tests/test_epic1_data.py::TestFeatureEngineering -v` passes with 0 failures
- `clean_features` on the full CICIDS2017 dataset logs the count of dropped rows

**Commit message**: `story-1.2: feature engineering, temporal features, train/test split`

---

### Story 1.2b: Per-Label Stratified Split (v1.1)

**Goal**: Add `per_day_stratified_split()` to `src/data/features.py` as the primary evaluation method. This eliminates the distribution shift caused by any single-day temporal hold-out (CICIDS2017 attack types are partitioned one per day, so a day-5 hold-out never sees DDoS, PortScan, or Bot during training).

**Background**: The v1.0 temporal hold-out evaluated the model on attack types that appear only in the Friday data and never in the Monday-Thursday training data. This is a CICIDS2017 dataset artifact, not a model quality problem. The per-label split groups by specific attack class and allocates 70/15/15 of each group to train/val/test, guaranteeing every attack family appears in all three splits.

**What stays**: `temporal_train_test_split` is not removed. Its tests continue to pass. Downstream code that currently reads `train_df, test_df = temporal_train_test_split(...)` in `scripts/train_stage1.py` is updated to use the new split.

#### Tasks

| # | Task | File(s) | Notes |
|---|------|---------|-------|
| 1.2b.1 | Add `per_day_stratified_split()` | `src/data/features.py` | Groups df by `Label` column. For each group, shuffles rows with `random_state`, then slices train/val/test at `[0:n_train]`, `[n_train:n_train+n_val]`, `[n_train+n_val:]` where sizes are `ceil(len(g)*train_ratio)` / `ceil(len(g)*val_ratio)` with remainder to test. Concatenates across all groups. Returns `(train_df, val_df, test_df)`. |
| 1.2b.2 | Update `scripts/train_stage1.py` | `scripts/train_stage1.py` | Replace `temporal_train_test_split` call with `per_day_stratified_split`. Pass `val_df` to conformal calibration instead of carving 20% off training data. Pass combined `train_df + val_df` to `build_rag_index.py` for FAISS indexing. Log split sizes per attack class at INFO level. |
| 1.2b.3 | Write Story 1.2b tests | `tests/test_epic1_data.py` | TC-1.2b.1 through TC-1.2b.6 (see test plan). |

**Definition of Done**:
- `pytest tests/test_epic1_data.py::TestPerLabelSplit -v` passes with 0 failures ✓
- `pytest tests/test_epic1_data.py -v` still passes (existing Story 1.2 tests unaffected) ✓
- `scripts/train_stage1.py` uses `per_day_stratified_split` and logs split sizes per label class ✓

**Status**: **Complete** -- model trained (PR-AUC=1.0000, recall=0.9998 on test split), full 10K clean pipeline run complete (recall=0.9929, volume_reduction=95.6%, 0 attacks silently missed). See `results/analysis_v1.1_10k.md`.

**Commit message**: `story-1.2b: per-label stratified split, update training script to use new split`

---

### Story 1.3: LightGBM Classifier Training and Evaluation

**Goal**: Train LightGBM with Optuna hyperparameter tuning, evaluate on the day-5 hold-out, generate SHAP explanations for every prediction, and save the model artifact with SHA-256 integrity verification.

#### Tasks

| # | Task | File(s) | Notes |
|---|------|---------|-------|
| 1.3.1 | Implement `integrity.py` (S4) | `src/models/integrity.py` | `save_hash(artifact_path, checksums_path)`: SHA-256 of file in 1MB chunks, write to `models/checksums.json`. `verify_hash(artifact_path, checksums_path)`: recompute and compare; raise `ModelIntegrityError(expected, actual)` on mismatch. |
| 1.3.2 | Implement `classifier.py` | `src/models/classifier.py` | `train(X_train, y_train, config, best_params)`: train final LightGBM on full training split using best Optuna params. `tune(X_train, y_train, config)`: Optuna study with TPE sampler, 5-fold stratified CV per trial, `early_stopping_rounds=50` per fold, convergence callback (see architecture Section 2.3). Returns `(best_params, best_n_estimators, study)`. `evaluate(model, X_test, y_test)`: returns dict with `pr_auc`, `precision`, `recall`, `f1`, `confusion_matrix`. `predict_proba(model, X)`: returns `np.ndarray` of P(TP). `save_model(model, path)`: pickle model, call `integrity.save_hash`. `load_model(path)`: call `integrity.verify_hash`, unpickle. |
| 1.3.3 | Implement `explainer.py` | `src/models/explainer.py` | `build_explainer(model)`: `shap.TreeExplainer(model, model_output="probability")`. `explain_batch(explainer, X)`: returns array of shape `(n_samples, n_features)`. `top_k_features(shap_values, feature_names, k)`: returns list of k dicts sorted by abs SHAP value descending. |
| 1.3.4 | Wire tuning into a training script | `scripts/train_stage1.py` | CLI entry point: loads config, runs `tune()`, logs best params and CV PR-AUC, calls `train()` with best params, calls `evaluate()`, saves model. Prints summary table to stdout. |
| 1.3.5 | Write Story 1.3 tests | `tests/test_epic1_data.py` | TC-1.3.1 through TC-1.3.7 (trains without error, PR-AUC >= 0.85, recall >= 0.95, save/load, SHAP shape and values, top_k structure, XGBoost baseline). |
| 1.3.6 | Write S4 security tests | `tests/test_security.py` | TC-S.19 through TC-S.21 (hash saved at save time, correct model loads, tampered model raises `ModelIntegrityError`). |

**Definition of Done**:
- `pytest tests/test_epic1_data.py::TestClassifier tests/test_security.py::TestModelIntegrity -v` passes with 0 failures
- `scripts/train_stage1.py` runs end-to-end on the full dataset and produces a model at `models/stage1_model.pkl`
- `models/checksums.json` exists and contains the model hash
- Best Optuna trial logged to stdout and to `logs/app.log`

**Commit message**: `story-1.3: LightGBM training, Optuna tuning, SHAP explainer, model integrity (S4)`

---

## Epic 2: Conformal Calibration and Stage 2 LLM Adjudication -- **COMPLETE**

---

### Story 2.1: Conformal Prediction and Three-Band Routing

**Goal**: Calibrate Stage 1 probabilities with MAPIE conformal prediction and implement the three-band routing logic with a verified false-negative rate guarantee.

#### Tasks

| # | Task | File(s) | Notes |
|---|------|---------|-------|
| 2.1.1 | Implement `conformal.py` | `src/models/conformal.py` | `fit_conformal(model, X_cal, y_cal, alpha)`: wraps `lgb.Booster` in `_BoosterWrapper` (adds `predict_proba`), creates `SplitConformalClassifier(estimator=wrapper, confidence_level=1-alpha, prefit=True)`, calls `clf.conformalize(X_cal, y_cal)`. `predict_bands(conformal, X, thresholds)`: `predict_set()` returns `(y_pred, y_pset)` where `y_pset[:, 1, 0]` = TP in set; band logic per architecture Section 4. `compute_coverage`: `classification_coverage_score` returns array; use `float(result[0])`. `save_conformal` / `load_conformal` with SHA-256 integrity. |
| 2.1.2 | Write Story 2.1 tests | `tests/test_epic2_llm.py` | TC-2.1.1 through TC-2.1.6 (fits, coverage >= 95%, valid band values, determinism, FN rate <= 1%, no overlaps). |

**Definition of Done**:
- `pytest tests/test_epic2_llm.py::TestConformal -v` passes with 0 failures
- Band distribution logged when run against the 10K fixture (printed as INFO log)
- Auto-FP false negative rate <= 1% validated on the fixture test set

**Commit message**: `story-2.1: conformal prediction, three-band routing, FN rate validation`

---

### Story 2.2: RAG Retrieval Layer

**Goal**: Embed historical training alerts with MiniLM-L6-v2 and build a FAISS index that supports real-time top-5 similarity retrieval for uncertain-band alerts.

#### Tasks

| # | Task | File(s) | Notes |
|---|------|---------|-------|
| 2.2.1 | Implement `embeddings.py` | `src/llm/embeddings.py` | `load_model(device)`: `SentenceTransformer("all-MiniLM-L6-v2", device=resolved_device)`. `embed_alerts(model, alerts, batch_size)`: `model.encode(alerts, batch_size=batch_size, normalize_embeddings=True)`. `alert_to_text(alert)`: serializes allowlisted fields to a readable string (`"Protocol: 6, Dst Port: 443, Flow Duration: 98234, ..."`). `resolve_device(config)`: returns `"cuda"` if `torch.cuda.is_available()` and config `rag.device == "auto"`, else `"cpu"`. |
| 2.2.2 | Implement `retrieval.py` | `src/llm/retrieval.py` | `build_index(embeddings)`: `faiss.IndexFlatIP(384)`; add L2-normalized embeddings. `save_index` / `load_index`: `faiss.write_index` / `faiss.read_index`. `retrieve_similar(index, query, k)`: L2-normalize query, `index.search(query, k)`, return `(distances, indices)`. |
| 2.2.3 | Write Story 2.2 tests | `tests/test_epic2_llm.py` | TC-2.2.1 through TC-2.2.6 (384-dim output, non-empty text, index build/save, load parity, k results, scores in [0,1]). |

**Definition of Done**:
- `pytest tests/test_epic2_llm.py::TestRAG -v` passes with 0 failures
- FAISS index built from training set, saved to `models/faiss_index.bin`
- GPU acceleration used when available (logged at startup)

**Commit message**: `story-2.2: MiniLM-L6-v2 embeddings, FAISS index build and retrieval`

---

### Story 2.3: Stage 2 LLM Adjudication (LangGraph + A2A)

**Goal**: Implement the adjudicator and adversarial agents as LangGraph StateGraphs, expose them via A2A protocol, and wire in all prompt-layer security controls (S1, S5, S6, S7).

This is the most complex story. The security modules must be implemented before the graphs -- they are called inside graph nodes, not around them.

#### Tasks

| # | Task | File(s) | Notes |
|---|------|---------|-------|
| 2.3.1 | Implement `sanitizer.py` (S1) | `src/llm/sanitizer.py` | `sanitize_field(value)`: strip null bytes, ANSI escapes, Unicode control chars (U+0000-U+001F except tab/newline); case-insensitive replace of `ignore.*instructions?` patterns; escape `</alert_data>` and `<system>` tags. `sanitize_alert(alert, allowed_fields)`: apply `sanitize_field` to each value; wrap result in `<alert_data>...</alert_data>`. |
| 2.3.2 | Implement `redactor.py` (S6) | `src/llm/redactor.py` | `redact(alert)`: returns dict containing only `ALLOWED_FIELDS` keys (see architecture Section 2.11). Strips `Source IP`, `Destination IP`, and any field not on the allowlist silently. |
| 2.3.3 | Implement `validators.py` (S5) | `src/llm/validators.py` | Pydantic models: `Stage2Verdict`, `AdversarialVerdict`, `FinalVerdict` (see architecture Section 2.11). All with `model_config = ConfigDict(extra="forbid")` to reject unexpected fields. `parse_llm_response(raw_text, model_class)`: `json.loads` then `model_class.model_validate`; on any exception returns `None`. |
| 2.3.4 | Implement `rate_limiter.py` (S7) | `src/llm/rate_limiter.py` | `RateLimiter(max_per_hour, max_per_day)`: sliding window counters. `acquire()`: returns `True` if under limits, `False` if exceeded (never blocks). `CircuitBreaker(threshold)`: `check(uncertain_count, total_count)` returns `True` (halt) if `uncertain_count/total_count > threshold`. Exponential backoff: `min(base * 2**attempt + random.uniform(0,1), max_wait)`, `base=1`, `max_wait=30`, max 3 retries. |
| 2.3.5 | Implement graph state schemas | `src/llm/graphs/state_schemas.py` | `AdjudicatorState` and `AdversarialState` TypedDicts per architecture Sections 2.8 and 2.9. |
| 2.3.6 | Implement adjudicator graph | `src/llm/graphs/adjudicator_graph.py` | Five nodes: `sanitize_node`, `build_prompt_node`, `call_llm_node`, `validate_node`, `fallback_node`. Conditional edge from `validate_node`: route to END (success), back to `call_llm_node` (retry), or to `fallback_node` (max retries exceeded). `call_llm_node` catches all exceptions and routes to `fallback_node`. Prompt assembly uses Section 5.3 template from architecture Section 7. |
| 2.3.7 | Implement adversarial graph | `src/llm/graphs/adversarial_graph.py` | Same node structure; `build_counter_prompt_node` replaces `build_prompt_node`; receives `initial_verdict` from state. Shares `sanitize_node` and `fallback_node` implementations via import. |
| 2.3.8 | Implement `reconcile.py` | `src/llm/graphs/reconcile.py` | Plain function `reconcile(stage2, adversarial) -> FinalVerdict`. Decision table from architecture Section 6. |
| 2.3.9 | Implement A2A schemas | `src/llm/a2a/schemas.py` | Pydantic models for A2A task input (`AdjudicatorTaskInput`, `AdversarialTaskInput`) and output (`AdjudicatorTaskOutput`, `AdversarialTaskOutput`). These wrap the LangGraph state inputs/outputs for the A2A transport layer. |
| 2.3.10 | Write Agent Cards | `src/llm/a2a/agent_cards/adjudicator.json`, `adversarial.json` | Static JSON files per architecture Section 10.2. Valid JSON, contains `name`, `url`, `version`, `capabilities`, `skills`. |
| 2.3.11 | Implement A2A servers | `src/llm/a2a/adjudicator_server.py`, `adversarial_server.py` | FastAPI + `a2a-sdk` servers. Each exposes `GET /.well-known/agent.json` (serves agent card) and `POST /` (`tasks/send` handler). Handler: deserialize payload to task input schema, invoke the compiled LangGraph, serialize state output to A2A artifact. |
| 2.3.12 | Implement A2A client | `src/llm/a2a/client.py` | `A2AClient(config)`: `inprocess` mode calls graph directly (no HTTP); `http` mode uses `httpx.AsyncClient`. `async send_task(agent, payload) -> dict`. `async get_agent_card(agent) -> dict`. Both modes expose identical async interface. |
| 2.3.13 | Write Story 2.3 tests | `tests/test_epic2_llm.py` | TC-2.3.1 through TC-2.3.23 (prompt rendering, valid parse, malformed JSON fallback, timeout fallback, adversarial output, reconciliation cases, LangGraph compilation and execution, retry logic, A2A agent cards, A2A task send/receive, error handling, inprocess/http parity). |
| 2.3.14 | Write S1, S5, S6, S7 security tests | `tests/test_security.py` | TC-S.1 through TC-S.14 (injection neutralization, XML escape, control char strip, validator accept/reject, field allowlist). |

**Definition of Done**:
- `pytest tests/test_epic2_llm.py::TestAdjudication tests/test_security.py -v` passes with 0 failures
- Both LangGraph graphs compile and run against the mock API without errors
- A2A `inprocess` mode works end-to-end (adjudicator → adversarial → reconcile)
- Agent Cards are valid JSON and pass TC-2.3.18 and TC-2.3.19

**Commit message**: `story-2.3: LangGraph adjudicator + adversarial agents, A2A protocol, security controls S1/S5/S6/S7`

---

### Story 2.4: End-to-End Pipeline Integration

**Goal**: Wire all components into a single orchestrator, implement the tripwire, measure end-to-end metrics on the full test set, and verify the complete audit trail.

#### Tasks

| # | Task | File(s) | Notes |
|---|------|---------|-------|
| 2.4.1 | Implement `orchestrator.py` | `src/pipeline/orchestrator.py` | `PipelineComponents` dataclass: holds references to model, conformal, explainer, embedding model, FAISS index, A2A client, rate limiter, circuit breaker, audit logger. `run_batch(df, config, components)`: feature engineering → Stage 1 scoring → SHAP → conformal bands → (uncertain: RAG + A2A adjudication + A2A adversarial + reconcile) → audit entry per alert → return `list[DispositionRecord]`. Per-alert exceptions caught; assign `needs_review`, log error, continue batch. |
| 2.4.2 | Implement `tripwire.py` | `src/pipeline/tripwire.py` | `record_auto_fp(alert_id, alert_record, store_path)`: append JSON line to `data/processed/tripwire_store.jsonl`. `check_ioc(ioc, store_path, lookback_days)`: scan entries within window, match on `Destination Port` + `Protocol` + IP prefix if present, return list of `alert_id` strings. |
| 2.4.3 | Compute and log end-to-end metrics | `src/pipeline/orchestrator.py` | After `run_batch`: log PR-AUC, recall, band distribution counts and percentages, alert volume reduction %, estimated analyst time saved (band_uncertain_count × 7 min). All to `logs/app.log` at INFO level. |
| 2.4.4 | Write Story 2.4 tests | `tests/test_epic2_llm.py` | TC-2.4.1 through TC-2.4.6 (full pipeline 10K, non-null verdicts, PR-AUC parity, tripwire record + match, tripwire no-match, tripwire lookback window). |
| 2.4.5 | Write S3 integration test | `tests/test_security.py` | TC-S.25 (10-alert batch produces complete audit trail, hash chain valid, no API key in log). |
| 2.4.6 | Write integration tests | `tests/test_epic2_llm.py` | IT-01 through IT-05 (Stage 1 on 10K fixture, Stage 2 with mocked API, full pipeline mocked, rate limiter halts at limit, circuit breaker triggers). |

**Definition of Done**:
- `pytest tests/test_epic2_llm.py tests/test_security.py -v` passes with 0 failures (full Epic 2 suite)
- Full pipeline runs on the 10K fixture without errors
- Audit log hash chain validates after a 10-alert run
- Metrics summary printed to stdout after each `run_batch` call

**Commit message**: `story-2.4: end-to-end pipeline, tripwire, metrics logging, full audit trail`

---

## Post-Epic-2 Production Hardening -- **COMPLETE**

These items were identified after Epic 2 tests passed during a production-readiness audit. They do not add new features; they replace stubs and placeholder implementations with real code.

| Item | File(s) | What changed |
|------|---------|--------------|
| `save_conformal` / `load_conformal` | `src/models/conformal.py` | Implemented with pickle + SHA-256 integrity check; was missing entirely |
| FAISS build script | `scripts/build_rag_index.py` | New script; embeds full training set, saves FAISS index + training_df.parquet |
| Pipeline run entry point | `scripts/run_pipeline.py` | New script; loads all artifacts, creates real Anthropic client, processes batch, prints summary |
| TripwireStore file persistence | `src/pipeline/tripwire.py` | `TripwireStore(path=...)` appends to JSON Lines file and reloads on startup |
| `adversarial.confidence_threshold_high` | `config.yaml` + `adversarial.py` | Moved `0.80` literal to config; `reconcile()` accepts it as a parameter |
| `stage1.shap_top_k` | `config.yaml` + `orchestrator.py` | Moved `k=5` literal to config |
| `rag.embedding_batch_size` | `config.yaml` + `embeddings.py` | Moved `batch_size=64` literal to config |
| `data.test_day` | `config.yaml` + all three scripts | Moved `test_day=5` literal to config |
| `agents.max_retries` | `config.yaml` + `a2a/client.py` | Wired config value (3) into adjudicator graph state; was defaulting to 2 |
| `conformal.artifact_path` | `config.yaml` | Added missing config key; was hardcoded in `train_stage1.py` |

---

## Epic 3: Analyst UI and Demo -- **COMPLETE**

---

### Story 3.1: Streamlit Dashboard

**Goal**: Build the analyst-facing dashboard with alert list, detail view (SHAP + LLM rationale + similar alerts), band filters, feedback capture, dark/light mode, and role-based authentication (S8).

#### Tasks

| # | Task | File(s) | Notes |
|---|------|---------|-------|
| 3.1.1 | Set up auth config | `config.yaml` `auth:` section | Add `auth.users` list with hashed passwords (bcrypt via `streamlit_authenticator.Hasher`). Two test accounts: `analyst_user` (role: analyst) and `viewer_user` (role: viewer). Hashed passwords committed; plaintext never committed. |
| 3.1.2 | Implement dashboard core + auth (S8) | `src/ui/dashboard.py` | `streamlit_authenticator.Authenticate` on startup. Session timeout from `config.yaml auth.session_timeout_minutes`. Role stored in `st.session_state`. Route unauthenticated users to login form only. |
| 3.1.3 | Implement alert list view | `src/ui/dashboard.py` | Load latest `DispositionRecord` batch from `data/processed/`. Sortable table: alert ID, timestamp, final verdict, confidence, band. Band filter sidebar. Color-code rows by band (auto-FP: green, uncertain: amber, auto-TP: red). |
| 3.1.4 | Implement detail view | `src/ui/dashboard.py` | On row click: show full feature table, SHAP force plot (via `shap.plots.force` + `matplotlib`, embedded as PNG), LLM rationale + recommended actions, adversarial counter-rationale, top-5 similar historical alerts with similarity scores. |
| 3.1.5 | Implement feedback capture | `src/ui/dashboard.py` | Analyst role only: dropdown (true_positive / false_positive / needs_review), free-text rationale, submit button. On submit: write to `data/processed/feedback.jsonl`, call `audit_logger.log_feedback`. Viewer role: feedback section hidden. |
| 3.1.6 | Implement dark/light mode | `src/ui/dashboard.py` | Toggle button in sidebar. Inject custom CSS for each theme. Dark theme: dark grey background, light text. Light theme: white background, dark text. Persist choice in `st.session_state`. |
| 3.1.7 | Write Story 3.1 tests | `tests/test_epic3_ui.py` | TC-3.1.1 through TC-3.1.7 (launches, row count, filter, detail view, feedback save, viewer no feedback, unauthenticated login redirect). |

**Definition of Done**:
- `pytest tests/test_epic3_ui.py::TestDashboard -v` passes with 0 failures
- Dashboard launches with `streamlit run src/ui/dashboard.py` and all views render against the 10K fixture results

**Commit message**: `story-3.1: Streamlit dashboard, alert list, detail view, feedback, dark/light mode, auth (S8)`

---

### Story 3.2: Metrics Dashboard

**Goal**: Add a metrics page to the dashboard showing the PR-AUC curve, confusion matrix, band distribution, and volume reduction summary.

#### Tasks

| # | Task | File(s) | Notes |
|---|------|---------|-------|
| 3.2.1 | Implement metrics page | `src/ui/dashboard.py` | Separate Streamlit page (or tab): PR-AUC precision-recall curve (matplotlib), confusion matrix heatmap (seaborn), band distribution pie chart (Plotly or matplotlib), summary table (total alerts, auto-FP count, auto-TP count, uncertain count, volume reduction %, time saved in hours). All charts generated from the evaluation results JSON in `metrics/`. |
| 3.2.2 | Save evaluation results | `src/pipeline/orchestrator.py` | After `run_batch` on test set, write a `metrics/evaluation_<timestamp>.json` with all metric values and band counts. |
| 3.2.3 | Write Story 3.2 tests | `tests/test_epic3_ui.py` | TC-3.2.1 (PR-AUC chart renders), TC-3.2.2 (volume reduction numerically correct). |

**Definition of Done**:
- `pytest tests/test_epic3_ui.py::TestMetricsDashboard -v` passes with 0 failures
- Metrics page renders with data from the evaluation run

**Commit message**: `story-3.2: metrics dashboard, PR-AUC curve, confusion matrix, band distribution`

---

### Story 3.3: Documentation and Demo

**Goal**: Update the README with setup instructions, architecture summary, and results. Produce screenshots of the dashboard for portfolio use.

#### Tasks

| # | Task | File(s) | Notes |
|---|------|---------|-------|
| 3.3.1 | Update README | `README.md` | Sections: Project Summary, Quick Start (3 commands to run), Architecture (link to `docs/architecture.md` + inline ASCII diagram), Evaluation Results (PR-AUC, recall, volume reduction from the actual run), Security Controls Summary, Tech Stack. |
| 3.3.2 | Produce dashboard screenshots | `docs/screenshots/` | At minimum: alert list view (light mode), detail view with SHAP plot (dark mode), metrics page. Committed to `docs/screenshots/`. |
| 3.3.3 | Verify README setup steps | -- | Follow the README Quick Start from a clean venv and confirm all steps reproduce without error. Fix any discrepancies. |

**Definition of Done**:
- README renders correctly on GitHub
- All setup steps in the README reproduce from a clean venv
- Screenshots committed and linked from README

**Commit message**: `story-3.3: README, architecture summary, evaluation results, demo screenshots`

---

## Dependency Map

```
1.1 (loader, secrets S2, audit S3)
  └── 1.2 (features)
        └── 1.3 (classifier, Optuna, SHAP, integrity S4)
              └── 2.1 (conformal)
                    └── 2.2 (embeddings, FAISS)
                          └── 2.3 (LangGraph agents, A2A, sanitizer S1, validators S5, redactor S6, rate_limiter S7)
                                └── 2.4 (orchestrator, tripwire, metrics)
                                      └── 3.1 (dashboard, auth S8)
                                            └── 3.2 (metrics page)
                                                  └── 3.3 (README, screenshots)
```

Each story is a strict prerequisite for the next. No parallelism within the critical path.

---

## Security Control Delivery Schedule

| Control | Story | Rationale |
|---------|-------|-----------|
| S2: Secrets management | 1.1 | Required before any code that references the API key |
| S3: Audit logging | 1.1 | Required before the pipeline records any decision |
| S4: Model artifact integrity | 1.3 | Required when the first model is saved |
| S1: Prompt injection mitigation | 2.3 | Required before first LLM call is assembled |
| S5: LLM output validation | 2.3 | Required before first LLM response is parsed |
| S6: Data minimization | 2.3 | Required before first data crosses the trust boundary |
| S7: Rate limiting | 2.3 | Required before first API call is dispatched |
| S8: Dashboard authentication | 3.1 | Required before dashboard is usable |

---

## Estimated Effort

| Story | Estimate | Primary complexity |
|-------|----------|-------------------|
| 1.1 | 2 days | Data download + two security utilities from scratch |
| 1.2 | 0.5 days | Straightforward pandas transforms |
| 1.3 | 3 days | Optuna integration, SHAP, convergence callback |
| 2.1 | 0.5 days | Thin wrapper around MAPIE |
| 2.2 | 1 day | FAISS index, embedding pipeline |
| 2.3 | 4 days | LangGraph graphs, A2A servers + client, 4 security modules |
| 2.4 | 1.5 days | Orchestrator wiring, tripwire, metrics |
| 3.1 | 2 days | Streamlit layout, SHAP rendering, auth |
| 3.2 | 0.5 days | Charts from existing data |
| 3.3 | 0.5 days | README, screenshots |
| **Total** | **~16 days** | |
