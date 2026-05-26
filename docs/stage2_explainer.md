# Stage 2 Pipeline: Claude API, RAG, and Adversarial Validation

## Why Stage 2 exists

Stage 1 (LightGBM + conformal prediction) sorts every alert into three bands:

- **auto_fp** (P(TP) < 0.05): closed automatically as false positive
- **auto_tp** (P(TP) > 0.85): escalated automatically as true positive
- **uncertain** (everything in between): cannot be decided with statistical confidence

The uncertain band is the problem. The ML model knows it does not know. These alerts are too risky to auto-close and too numerous to hand to a human without any triage. Stage 2 is the triage layer: it uses a language model with injected context to reason about each uncertain alert and produce a structured verdict.

In our run on the 10K fixture: 936 out of 9,996 alerts (9.4%) reached Stage 2.

---

## What the LLM actually sees

This is the core question. Claude is stateless -- each API call is a blank slate with no memory of prior alerts, no access to the training data, and no awareness that it is part of a larger pipeline. Everything the model knows about an alert must be placed in the prompt.

The Stage 2 prompt (`src/llm/adjudicator.py: build_prompt()`) contains four things:

### 1. The alert's network features

The raw alert row (a pandas Series of ~80 numeric columns) passes through two filters before reaching the prompt.

First, `redact_alert()` (`src/llm/redactor.py`) applies a field allowlist of ~65 network-feature columns, stripping anything not on it -- including IP addresses and flow identifiers that could identify internal hosts (Security Control S6).

Second, `sanitize_alert_dict()` (`src/llm/sanitizer.py`) processes each remaining value: it strips ASCII control characters, applies `html.escape()` to neutralize XML delimiter injection, then checks for known prompt injection phrases (e.g. "IGNORE ALL PREVIOUS INSTRUCTIONS") and replaces them with `[REDACTED_INJECTION]`.

The result is inserted into the prompt inside `<alert_data>` XML delimiters (added by `build_prompt()`, not the sanitizer) so the model can distinguish data from instructions:

```
<alert_data>
  Flow Duration: 1823441
  Total Fwd Packets: 6
  Total Backward Packets: 2
  Fwd Packet Length Max: 1460
  ...
</alert_data>
```

### 2. SHAP top-5 features

The SHAP TreeExplainer runs on every alert (all bands, not just uncertain). For Stage 2 prompts, the five features with the largest absolute SHAP values are included:

```
Top 5 features by SHAP importance:
  1. Flow Duration: SHAP=+0.8342, value=1823441
  2. Fwd Packet Length Max: SHAP=+0.6109, value=1460
  3. Bwd Packets/s: SHAP=-0.4821, value=0.0
  4. Flow Bytes/s: SHAP=+0.3977, value=4801.2
  5. Init_Win_bytes_forward: SHAP=-0.2103, value=65535
```

A positive SHAP value means that feature pushed the model's score toward true positive. Negative means it pushed toward false positive. This tells the LLM *why* the ML model scored the alert the way it did, giving it a starting point for reasoning rather than asking it to re-derive everything from raw numbers.

### 3. RAG: top-5 similar historical alerts

This is where the RAG retrieval feeds in. The five most similar alerts from the training set are listed with their analyst-verified labels and similarity scores:

```
Most similar historical alerts (from analyst-verified cases):
  - hist_441823: label=DoS Hulk, similarity=0.941
  - hist_109034: label=BENIGN, similarity=0.887
  - hist_782211: label=DoS Hulk, similarity=0.881
  - hist_290847: label=BENIGN, similarity=0.874
  - hist_553109: label=DoS Hulk, similarity=0.863
```

This gives the model a precedent-based frame: "alerts that look like this one were classified as X in the past." That is the primary source of domain knowledge. Without RAG, the model has only the feature values and SHAP scores to work from; with RAG it has verified historical outcomes for similar cases.

### 4. The instruction

The system prompt defines the role: senior SOC analyst, task is TP vs FP vs needs_review, be conservative. The user prompt ends with: reason step by step, then return a specific JSON schema.

---

## How the RAG index is built

Source: `scripts/build_rag_index.py`, `src/llm/embeddings.py`, `src/llm/retrieval.py`.

**Step 1: Convert alerts to text.** Each training-set row is serialized to a plain-text string by `alert_to_text()`. Rather than including all 80 features, the function uses 12 discriminative features from a fixed list (`_TEXT_FEATURES`): Destination Port, Flow Duration, Total Fwd/Backward Packets, Flow Bytes/s, Flow Packets/s, SYN/ACK/FIN/RST flag counts, and Init_Win_bytes. The output format is:

```
Network flow alert: Destination Port=80, Flow Duration=1.823e+06, Total Fwd Packets=6, ...
```

This gives the sentence encoder something readable -- it is not trained on raw numeric vectors, but it can encode the semantic content of labeled numeric fields reasonably well. If none of the 12 features are present, it falls back to all available numeric fields.

**Step 2: Embed with MiniLM-L6-v2.** `SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')` encodes every alert text to a 384-dimensional float32 vector. Our training set has 2,125,158 rows; embedding in batches of 64 on the RTX 2070 SUPER takes about 20 minutes.

**Step 3: L2-normalize.** Each embedding vector is normalized to unit length. This converts inner-product similarity to cosine similarity: `dot(a, b) = cos(angle(a, b))` when `|a| = |b| = 1`. Cosine similarity is the right measure here -- we care about the direction of the embedding (the "shape" of the alert) not its magnitude.

**Step 4: FAISS IndexFlatIP.** The normalized vectors are stored in a flat inner-product index. "Flat" means brute-force exact search (no approximation). At 2.1M vectors × 384 dimensions × 4 bytes = ~3.1 GB in memory. For a POC this is fine; a production system would use IVF or HNSW indexing for sub-linear query time.

**Step 5: Save both artifacts.** The FAISS index (`models/faiss_index.bin`) and the original training DataFrame (`models/training_df.parquet`) are saved together. At query time the index returns integer row indices; the DataFrame is used to look up the verified Label for each index to include in the prompt.

**What the index does NOT contain:** The index holds training data only (days 1-4). Day-5 (Friday) attack types -- DDoS, PortScan, Bot -- are not in the index. When an alert from those attack types reaches Stage 2, the retrieved neighbors are mostly BENIGN alerts (the closest things in embedding space), which biases the historical context in the prompt. This is the same distribution-shift problem seen in Stage 1 metrics, now visible in the RAG layer.

---

## How RAG helps Stage 2

Without RAG, the LLM prompt would contain only feature values and SHAP scores. A language model has no inherent knowledge of what a CICIDS2017 `Flow Duration` of 1,823,441 microseconds means in a SOC context, or whether a `Fwd Packet Length Max` of 1,460 bytes is suspicious.

RAG grounds the decision in historical precedent: "5 of the 5 most similar past alerts were labeled DoS Hulk by analysts." Even if the model cannot interpret the raw numbers, it can reason: "the feature profile matches known DoS Hulk patterns; confidence is high."

RAG also handles concept drift naturally. If the training set is periodically updated with new analyst-verified dispositions and re-indexed, the retrieved context reflects recent patterns without retraining the LLM or the ML classifier.

The limit of RAG in this system is index coverage. Queries for attack types absent from the training set return poor neighbors (benign alerts with some numeric similarity), and the historical context is misleading rather than helpful. That is the case for Friday DDoS and PortScan in this POC.

---

## Adversarial Validation

Source: `src/llm/adversarial.py`.

After Stage 2 produces a verdict, a second LLM call challenges it. This is inspired by Cloudflare's Project Glasswing multi-agent harness: a second independent agent with a different directive arguing against the first agent's conclusion.

### What the adversarial agent gets

The adversarial prompt contains:
- The Stage 2 verdict and confidence score
- The Stage 2 rationale (full text)
- The alert summary (same text used for RAG embedding)
- The SHAP top-5 summary

The adversarial agent does not get the full sanitized feature block or the historical RAG context. It gets the alert as a short text summary (the same 12-feature `alert_to_text()` output used for RAG embedding, sanitized before passing). Its job is to attack the argument Stage 2 already made, not to re-derive a verdict from scratch.

### What it is instructed to do

The system prompt is explicitly adversarial:

> "You are a skeptical SOC analyst tasked with challenging security alert verdicts. Your goal is to find weaknesses in the initial verdict and argue the opposing case. Be rigorous: identify the weakest evidence, propose alternative explanations, and provide a counter-verdict."

The model is asked to identify the weakest piece of evidence in the Stage 2 rationale and propose an alternative explanation. It returns:

```json
{
  "counter_verdict": "false_positive",
  "confidence": 0.72,
  "counter_rationale": "The high flow duration could indicate a long-lived idle connection rather than an attack. Similar DoS Hulk signatures can arise from CDN keep-alive traffic.",
  "weakest_evidence": "Similarity to DoS Hulk neighbors is based on flow duration alone; packet counts are within normal range for bulk transfers."
}
```

### Why this matters

A single LLM call can be confidently wrong. The model can latch onto one strong signal (high SHAP feature, matching historical label) and ignore contradictory evidence. A second call with a directive to find flaws produces a genuine counter-argument. If the counter-argument is persuasive, it should lower confidence in the original verdict.

The adversarial agent does not share state with the Stage 2 agent. Each call is independent. The disagreement between them is a signal, not a bug -- it means the alert is genuinely ambiguous.

---

## Final Verdict: Reconciliation Logic

Source: `src/llm/adversarial.py: reconcile()`.

`reconcile()` takes the Stage 2 verdict and the adversarial verdict and applies a deterministic rule set:

### Case 1: Adversarial call failed

If the adversarial API call timed out, returned a parse error, or was unavailable, Stage 2 is accepted unchanged:

```
final_verdict = stage2.verdict
confidence = stage2.confidence
note = "Adversarial agent unavailable; Stage 2 verdict accepted."
```

No auto-close happens on a missing adversarial check -- the system only downgrades, never upgrades, on failure.

### Case 2: Both agents agree

```
final_verdict = stage2.verdict
confidence = (stage2.confidence + adversarial.confidence) / 2
note = "Stage 2 and adversarial agent agree."
```

Agreement increases confidence (both independent callers reached the same conclusion). The averaged confidence is lower than Stage 2 alone if the adversarial agent agreed with lower confidence, higher if it agreed with higher confidence.

### Case 3: Disagreement, Stage 2 confidence high (> 0.80)

```
final_verdict = stage2.verdict
confidence = stage2.confidence
note = "Stage 2 confidence exceeds 0.80 threshold; Stage 2 wins."
```

If Stage 2 expressed very high confidence, a counter-argument from the adversarial agent is treated as noise. The threshold of 0.80 is configurable in `config.yaml` (`adversarial.confidence_threshold_high`).

### Case 4: Disagreement, Stage 2 confidence low (<= 0.80)

```
final_verdict = needs_review
confidence = min(stage2.confidence, adversarial.confidence)
note = "Stage 2 confidence below threshold; downgraded to needs_review."
```

This is the most important rule. When two independent LLM calls disagree and neither is highly confident, the system refuses to make a call. The alert is routed to a human analyst. This is intentionally conservative: the cost of a missed true positive is higher than the cost of an analyst reviewing one more alert.

### Verdict flow summary

```
Stage 1 uncertain band alert
        |
        v
[RAG: retrieve 5 similar historical alerts]
        |
        v
[Stage 2 LLM: feature values + SHAP + RAG context -> verdict + confidence + rationale]
        |
        v
[Adversarial LLM: Stage 2 verdict + rationale -> counter_verdict + counter_rationale]
        |
        v
[reconcile()]
    agree?          -> final = agreed verdict, avg confidence
    disagree, high confidence -> final = Stage 2 verdict
    disagree, low confidence  -> final = needs_review
    adversarial failed        -> final = Stage 2 verdict
        |
        v
DispositionRecord: final_verdict, final_confidence, reconciliation_note,
                   stage2_rationale, adversarial_rationale, recommended_actions
```

---

## Evaluation: cross-checking final verdicts against ground truth

When the input dataset contains a `Label` column (as `fixture_10k.csv` does), the pipeline automatically cross-checks every final verdict against the ground truth and computes evaluation metrics.

### How the labels flow through

`orchestrator.py` reads the `Label` column and encodes it to binary: `BENIGN=0`, any attack class`=1`. Each `DispositionRecord` carries a `true_label` field (0 or 1). The results parquet written by `run_pipeline.py` contains both `true_label` and `final_verdict` columns side by side, so per-alert correctness can be inspected directly.

### How `_compute_metrics()` cross-checks

`run_pipeline.py: _compute_metrics()` maps each final verdict to a predicted label:

```python
y_pred = 1 if r.final_verdict == "true_positive" else 0
```

This covers all three bands together:
- `auto_fp` → `final_verdict="false_positive"` → y_pred=0
- `auto_tp` → `final_verdict="true_positive"` → y_pred=1
- `uncertain` → `final_verdict=stage2 verdict` → y_pred=1 if "true_positive", 0 otherwise

From `y_true` (ground truth) and `y_pred` (pipeline decision), it computes precision, recall, F1, confusion matrix, and the PR-AUC from the raw ML score. All values land in `metrics/evaluation_<timestamp>.json`.

### The `needs_review` nuance

`needs_review` verdicts are mapped to `y_pred=0` (predicted negative). This is intentionally conservative: the system refused to commit, so it does not claim TP. The consequence is that any true attack that ends up `needs_review` -- because Stage 2 and the adversarial agent disagreed at low confidence, or because the Anthropic client was unavailable -- counts as a false negative in recall.

This means:
- **Recall** will look worse than the ML model's raw ranking quality (PR-AUC), especially if Stage 2 frequently returns `needs_review`.
- **PR-AUC** (computed from `ml_score`, not `final_verdict`) is the more honest measure of discriminative power.
- The `needs_review` rate in the metrics tells you how often the two-agent system deferred rather than deciding.

In practice, a high `needs_review` rate on the uncertain band is expected on out-of-distribution attack types (like Friday DDoS/PortScan on a Mon-Thu-trained model) because both agents will hedge when the alert doesn't match anything in the RAG index.

### What the dashboard shows

The dashboard reads the results parquet and has both `true_label` and `final_verdict` per alert. You can filter to misclassifications (e.g. `true_label=1` but `final_verdict="false_positive"`) to review specific errors. The metrics page displays the confusion matrix and PR curve from the evaluation JSON.

---

## What the system does NOT do

- **It does not learn.** Each pipeline run is stateless. The LLM does not update based on analyst feedback. The RAG index does not grow automatically. Both require manual retraining or re-indexing to incorporate new data.
- **It does not query live threat feeds.** The only external knowledge is the RAG index (historical analyst dispositions). No VirusTotal, no MITRE ATT&CK lookups, no IP reputation checks in this POC.
- **It does not guarantee correctness.** The conformal prediction guarantee applies to Stage 1 auto-FP/auto-TP bands only. Stage 2 verdicts have no statistical coverage guarantee -- the model can be wrong, which is why the adversarial check and the `needs_review` fallback exist.
- **It is not fast.** Each uncertain alert requires two API calls and two RAG queries. At ~10-20 seconds per alert pair, processing 1,000 uncertain alerts takes 3-6 hours. Parallelism (asyncio or thread pool) is the obvious next step for production use.
