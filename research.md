# AI-Driven False Positive Reduction for Enterprise SOC Alert Triage: A Build-Ready Research Synthesis (May 2026)

**TL;DR**
- Build a two-stage hybrid: a calibrated gradient-boosted classifier (LightGBM/XGBoost) auto-closes high-confidence false positives, a sentence-transformer + RAG layer over an open security-tuned LLM (Cisco's Foundation-Sec-8B-Instruct) adjudicates the ambiguous 10–20%, and conformal prediction defines the "send-to-analyst" band. Published evidence (CSIRO L2DHF, June 2025; Simbian benchmark, June 2025) supports this design, but **no current published study claims simultaneous >90% FP reduction AND >95% TP recall on production enterprise alerts** — targeting that is realistic only on benchmark datasets, not on raw enterprise SIEM feeds.
- The dataset bottleneck dominates: every public dataset (CICIDS2017/2018, UNSW-NB15, ToN-IoT, BETH) is a network-IDS or honeypot dataset, not a multi-source enterprise alert dataset. Compose a unified corpus by mapping all sources to OCSF v1.3.0+, then synthesize minority "hard-positive" samples with CTGAN (preferred over SMOTE for security tabular data, per XIDINTFL-VAE 2024 results: precision 99.67%, F1 94.74%).
- Highest practical risk is not model accuracy but adversarial evasion (Hoang et al. 2024: FGSM attacks dropped ML-IDS accuracy from 99.7% to 1.14% on CSE-CIC-IDS2018) and analyst-feedback poisoning. Bound false-negative risk with mandatory shadow-mode operation, conformal "uncertain" routing, and a tripwire that re-escalates any auto-closed alert sharing IOCs with a later-confirmed incident in a 7-day window.

---

## 1. Problem Definition and Scope

**The quantitative baseline (2024–2026).** Industry data converges on these enterprise figures:

- **Alert volume**: The Crogl/Ponemon 2026 *State of SecOps* report (n=649 practitioners) finds the average enterprise SOC receives **4,330 alerts/day**, of which **37% are investigated** and 63% never get worked. Vectra AI's 2026 figure is **2,992/day with 63% unaddressed**; the AI SOC Market Landscape 2025 puts >20,000-employee enterprises above **3,000/day**.
- **False-positive rate**: The Microsoft/Omdia 2026 *State of the SOC* report found **46% of all alerts prove to be false positives**. The 2025 SANS Detection & Response Survey reports **73% of organizations name false positives their #1 detection challenge** (up sharply from the prior year). Underdefense's 2025 playbook cites peaks of **174 alerts/analyst/day, only 22% genuinely investigable**, with FPs consuming **52% of analyst time**. CyberDefenders (2025) reports enterprise FP rates frequently exceeding 50% and reaching 80%. Expel's benchmark: "world-class" SOCs maintain FP rates <10%.
- **Mean time to detect/respond**: The **IBM 2025 *Cost of a Data Breach Report*** (conducted by Ponemon Institute, sponsored by IBM, based on 600 organizations, breaches March 2024–February 2025; IBM newsroom, July 30, 2025) states "organizations using AI and automation extensively throughout their security operations saved an average $1.9 million in breach costs and reduced the breach lifecycle by an average of 80 days." Prophet Security recommends MTTR targets of critical 1h, high 2h, medium 4h, low 8h.
- **Analyst workforce impact**: The **ISC2 2024 Cybersecurity Workforce Study** (n=15,852, released Sept 11, 2024) reported "the global cybersecurity workforce gap reached a new high with an estimated **4.8 million professionals needed** to effectively secure organizations, a 19% year-on-year increase." (The 2025 ISC2 study, n=16,029, deliberately moved away from a single gap headcount in favor of a skills-shortage framing.) The **2022 Tines "Voice of the SOC Analyst" report** (n=468 US security analysts, March 2022) found **71% of analysts "burned out to some extent"**; the 2023 Tines follow-up reported a lower 63%. Ponemon's SOC effectiveness research finds 60% of SOC staff have considered leaving due to stress.
- **Economic baseline**: Per Ponemon research cited in *The Hacker News* (Sept 2025), the average enterprise SOC costs **$5.3M/year, up 20% YoY**, with **75% of SIEM TCO going to maintenance not licensing**, and **78% of users saying SIEMs take significant effort to configure**.
- **AI satisfaction caveat**: The SANS 2024 SOC Survey rated AI/ML tooling the **lowest satisfaction among 47 SOC technologies**, and only **18% of organizations report fully deployed AI with measured effectiveness**. Implementation is harder than vendor marketing suggests.

**Baseline you should measure against.** For a POC on benchmarked data, baseline against the **untuned upstream detection rule's confusion matrix** on the same alert corpus. For an enterprise simulation, use these targets: pre-AI FP rate 46% (Microsoft/Omdia 2026), target post-AI FP rate ≤10% on auto-closed cases, post-AI true-positive recall ≥98% measured on a held-out, time-forward slice.

**Per-source FP root causes (structural reasons):**

| Source | Typical FP driver | Structural reason |
|---|---|---|
| SIEM correlation rules | Generic vendor rules + lack of business context | Rules ship calibrated for a hypothetical environment; per-deployment baselining rarely gets done. |
| IDS/IPS (Snort/Suricata) | Signature breadth, encrypted-traffic blindness | Signatures match byte patterns, not intent; vuln scanners and admin scripts trigger the same bytes. |
| EDR | Behavioral heuristics on developer/admin workstations | PowerShell, code-signing, and IT automation legitimately exhibit "living-off-the-land" patterns. |
| Email gateway / phishing | URL/attachment heuristics + employee-reported phish | Marketing newsletters, shortened URLs, legitimate password-reset emails all match heuristic features. |
| DLP | Regex/classifier-based content matching | Source code, encoded blobs, and base64 attachments match exfil patterns; lacks intent context. |
| Cloud workload (CSPM/CNAPP) | Misconfiguration detection at policy-graph scale | One misconfigured IAM role generates N alerts per dependent resource; lacks dedup graph. |
| Identity/UEBA | Statistical baseline drift | New hires, travel, BYOD, and quarterly batch jobs all look like "unusual" behavior. |

---

## 2. Datasets for Training and Evaluation

### 2.1 Publicly available labeled datasets

| Dataset | Size | Label distribution | Features | Key limitations |
|---|---|---|---|---|
| **CICIDS2017** | 2,830,743 flows over 5 days | 71.32% benign; Heartbleed = 11 samples (0.0004%); SQL Injection = 21 (0.0007%) | 78 numeric flow features + label | Imbalance ratio >2×10⁵:1 on rare classes; no payload; documented labeling errors (Engelen et al., "Troubleshooting an Intrusion Detection Dataset"). |
| **CSE-CIC-IDS2018** | ~16M flows, 10 days | Heavily imbalanced; infiltration detection rate 0% in some published benchmarks despite 99% overall accuracy | Same 80-column CICFlowMeter schema | Same TCP-appendix and labeling issues as CICIDS2017. |
| **UNSW-NB15** (Moustafa & Slay, 2015) | 2,540,044 vectors, ~100 GB pcap | 9 attack categories + benign; ~13% attack | 49 features incl. transaction/flow stats | Synthetic background traffic; dated; overuse → overfit risk. |
| **ToN-IoT** (UNSW Canberra) | Training 22.3M records / test 461K, 43 features | Multi-class heterogeneous: telemetry, Win7/10 OS, Ubuntu 14/18, TLS, network | Multi-modal | IoT-focused; not enterprise EDR/SIEM-realistic. |
| **BETH** (Highnam et al., ICML UDL 2021) | 8,004,918 events across 23 honeypots | Per host: benign + ≤1 attack; dual labels (`sus`, `evil`) | Kernel process calls + network flows | Honeypot, not enterprise; limited attack diversity. |
| **AIT Log Data Set V1.1** | Multi-host Suricata + system logs | Multi-stage attack with ground-truth | Mixed structured logs | Used by Univ. Oslo / NDRE 2025 LLM-triage study; small. |
| **CIC IoT-DIAD 2024** | Recent IoT dataset | Multi-class | Flow + behavioral | New (2024) — limited prior-art comparisons. |

**There is no enterprise multi-source labeled alert dataset in the public domain.** All of the above are network-flow or host-process datasets. The closest to "alerts with disposition labels" is the synthetic AIT V1.1 + Suricata pipeline. This is the single largest gap in the field and the reason most strong vendor numbers come from internal datasets.

### 2.2 Synthetic generation: CTGAN-family beats SMOTE for security data

- **CTGAN** (Xu et al., NeurIPS 2019; the **SDV `ctgan` library**) uses mode-specific normalization and a conditional generator. The 2024 **XIDINTFL-VAE** study (Springer *Journal of Supercomputing*) directly compared SMOTE, Borderline-SMOTE, ADASYN, and a class-wise focal-loss VAE on NSL-KDD and CSE-CIC-IDS2018. The VAE/CTGAN-family approach achieved **precision 99.67%, F1 94.74%, recall 89.41%** — outperforming SMOTE variants specifically on reducing false positives.
- **SMOTE/ADASYN/Borderline-SMOTE** still beat doing nothing but interpolate in feature space, producing unrealistic "in-between" samples for security data where categorical fields (protocol, signature ID) and skewed distributions don't interpolate meaningfully.
- **Cost-sensitive learning** (XGBoost `scale_pos_weight`, focal loss with α/γ tuning) is the **most cost-effective single intervention**. The 2024 weighted-XGBoost study on IoTID20 achieved 99.32% accuracy/precision/recall/F1 using `scale_pos_weight` alone.
- **Recommended stack for security alerts**: (a) cost-sensitive boosting as baseline; (b) CTGAN to synthesize *additional* minority-class samples that preserve correlations; (c) Borderline-SMOTE as a third tier for the hardest decision boundary. **Apply sampling only to training folds — pre-split oversampling inflates test metrics catastrophically** (Kabane 2024 credit-card-fraud study).

### 2.3 Compositing into a unified corpus via OCSF

**OCSF (Open Cybersecurity Schema Framework)** joined the Linux Foundation on **November 19, 2024** (Linux Foundation Member Summit, Napa, CA), with over 900 contributors and 200 participating organizations; founding sponsors are AWS, Cisco, IBM, Splunk, and Broadcom (the schema derives from Symantec's prior work). The current schema at that announcement was **OCSF 1.3.0** (released August 2024). AWS Security Lake natively converts to OCSF Parquet on S3.

Mapping: CICIDS/UNSW/ToN-IoT flows → OCSF `network_activity` event class; BETH process events → `process_activity`; synthetic alerts → `detection_finding` (Category 2). Common schema: `activity_id`, `severity_id`, `disposition_id`, `src_endpoint`, `dst_endpoint`, `actor.user`, `device`, `metadata`.

The composited training corpus should be partitioned by **time slice**, not random split — alert distributions are non-stationary.

---

## 3. Feature Engineering

### Tier 1 — Alert-intrinsic
Raw OCSF fields: `severity_id`, `category_uid`, `class_uid`, src/dst IP and port, protocol, signature/rule_uid, confidence score, byte counts, flag patterns. On UNSW-NB15, SHAP analyses (Frontiers *Computer Science*, 2025) consistently rank **`sttl` (source-to-destination TTL), `ct_dst_sport_ltm`, `ct_srv_dst`, `ct_srv_src`, `smean`** as top predictors. On CICIDS2017, flow duration, packet-length statistics, and inter-arrival-time features dominate (Talukder et al., 2024).

### Tier 2 — Contextual enrichment
- Asset criticality score (CMDB lookup → 1–5)
- Vulnerability state of target at alert time (Tenable/Qualys join)
- User role/privilege (Entra ID / Okta group membership)
- Threat-intel match (src IP/domain on MISP, AbuseIPDB, Mandiant feeds — boolean + reputation)
- Geo/ASN anomaly (is this src/dst country/ASN pair novel for this user?)
- Time-of-day, day-of-week, holiday calendar
- Historical alert frequency for this src↔dst pair (rolling 7/30/90 day)
- Recent disposition rate for this rule (if the rule has been 99% FP for 30 days, that's a feature)

### Tier 3 — Temporal/sequential
- Alert burst rate (alerts/min from same src in last 5/15/60 min)
- Time since last similar alert (same rule_uid + same src)
- Sliding-window co-occurrence
- **Kill-chain stage signals** — map each alert's rule to MITRE ATT&CK technique and track tactics present in a sliding window per asset. The CORTEX paper (arXiv:2510.00311, October 2025) explicitly uses MITRE technique presence as a feature.

### Tier 4 — Graph-based
Alert correlation graph: nodes = entities (IP, user, host, process hash); edges = alerts touching both. Compute degree, betweenness, PageRank, connected-component size, and GraphSAGE/GAT inductive embeddings.

Key references: **"A Graph-Based Approach to Alert Contextualisation in Security Operations Centres"** (arXiv:2509.12923, September 2025) uses Graph Matching Networks to correlate incoming alert groups with historical incidents. Foundational: **Huang et al., "Graph neural networks and cross-protocol analysis for detecting malicious IP addresses"** (Complex & Intelligent Systems, 2022, McAfee authors). The 2025 SIGCOMM paper reports inference speedup proportional to node count but does not solve over-smoothing — keep GNN depth ≤3 layers.

### Tier 5 — NLP/text
Embed alert description, rule name, and log message excerpt with `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~50ms CPU) as default; upgrade to `BAAI/bge-large-en-v1.5` (1024-dim) for higher quality. For security-domain text, Cisco **Foundation-Sec-8B** hidden-state embeddings outperform Llama-3.1-8B by **>10% on internal classification benchmarks** (Cisco Foundation AI technical report, arXiv:2504.21039, April 2025).

### Predictive power ranking (consolidated from 2024–2025 SHAP studies)
1. **Recent disposition rate for the same rule** (Tier 2)
2. **Asset criticality × vulnerability state** interaction (Tier 2)
3. **Threat-intel boolean + reputation** (Tier 2)
4. **`sttl` and packet-statistical features** (Tier 1, for network-IDS subset)
5. **Alert burst rate** (Tier 3)
6. **Embedding similarity to historical TPs** (Tier 5 via Tier 4 retrieval)

---

## 4. Model Architecture and Selection

### 4.1 Classical ML — the workhorse layer

**XGBoost and LightGBM remain the strongest single models for tabular security data.** The foundational benchmark on this question is **Shwartz-Ziv & Armon (Intel), "Tabular Data: Deep Learning is Not All You Need" (arXiv:2106.03253, 2021)**, which evaluated XGBoost against NODE, DNF-Net, TabNet, and 1D-CNN across 11 tabular datasets and concluded XGBoost outperformed all DL models for both classification and regression, with the verbatim finding that "tuning hyperparameters does not make DL models outperform the ML models." The 2024 follow-up benchmark (arXiv:2408.14817) extended this to 111 datasets and confirmed the result.

Recommended hyperparameter ranges for high-imbalance binary FP vs TP:
- **XGBoost**: `max_depth=6–10`, `learning_rate=0.03–0.1`, `n_estimators=500–2000` with early stopping on PR-AUC, `scale_pos_weight=(n_FP/n_TP)`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5–10`, `tree_method='hist'`
- **LightGBM**: same ranges + `is_unbalance=True` or `class_weight='balanced'`, `num_leaves=31–127`
- Use **focal loss** (`gamma=1–2, alpha=0.25`) when imbalance exceeds 100:1 — the XIDINTFL-VAE study showed class-wise focal loss decisively outperforms plain weighting at 200:1+ ratios.

**CatBoost** has a slight edge on categorical-heavy alert data (rule names, vendor codes) thanks to ordered target encoding; the Frontiers 2025 SHAP study reported **XGBoost and CatBoost tied at 87% accuracy with FPR 0.07** on UNSW-NB15.

### 4.2 Deep Learning

- **Feed-forward MLPs**: rarely beat GBM.
- **1D-CNN + XGBoost hybrid**: The MDPI 2025 *Explainable Hybrid CNN–XGBoost* paper on CIC IoT-DIAD 2024 uses a 1D-CNN to learn a 128-dim embedding fed to XGBoost — sound when raw byte/flow sequences exist.
- **LSTM/GRU on alert sequences**: useful for kill-chain progression; the CNN-LSTM hybrid on UNSW-NB15 reached near-100% multi-class F1 with SMOTE preprocessing.
- **TabNet / FT-Transformer / SAINT**: The 2025 KBS paper on TabNet + Google Vizier tuning reports binary classification of **99.9% / 99.92% / 99.16% on NSL-KDD / CICIoT2023 / RT_IoT2022**. FT-Transformer wins specifically when feature interactions are dense; the 2025 OS-fingerprinting paper (arXiv:2502.09084) reports FT-T beating state-of-the-art by ~12% F1.

**When DL beats GBM**: when (a) you have >1M rows AND (b) features have strong interactions AND (c) you can afford a GPU. For a single-tenant POC, conditions (a) and (c) typically aren't met.

### 4.3 Graph Neural Networks

GCN / GAT / GraphSAGE on alert correlation graphs improves recall for **multi-stage attacks** where individual alerts are low-severity. Use GraphSAGE for inductive learning (new entities appear constantly in SOCs). Keep depth ≤3 layers to avoid over-smoothing (Nature Scientific Reports, February 2025). The Graph Matching Network approach in arXiv:2509.12923 (September 2025) is the most directly applicable recent method. **For a POC, GNNs are optional — defer to v2.**

### 4.4 LLM-based approaches

**Critical finding from University of Oslo + Norwegian Defence Research Establishment (2025) on the AIT Log Data Set V1.1**: When GPT-5-mini, Claude 3 Haiku, Qwen3:30B, and Gemma 3:27B were given only an alert description + brief log summary, **all four achieved 0% detection of true-positive malicious cases**; Gemma 3:27B labeled everything benign regardless of content. When the same models were wrapped in a constrained agent workflow (planner issues predefined SQL queries against Suricata logs, summarizer consolidates evidence, adjudicator issues verdict), accuracy rose to **~93% average across the four models, with GPT-5-mini correctly identifying all malicious cases across 100 runs**. **Lesson: zero-shot prompting on raw alerts does not work; structured agent workflows with tool use do.**

**Simbian AI SOC LLM benchmark (June 2025)**: 100 full-kill-chain scenarios; top frontier models (Anthropic, OpenAI, Google, DeepSeek as of May 2025) complete **61–67% of investigation tasks**; Simbian's tuned agent at "extra effort" reached 72%; human analysts powered by AI scored **73–85%**.

**RAG over historical dispositions** is well-supported by the analogous medical-triage **MECR-RAG study (JMIR Medical Informatics 2026)**, which retrieved from a 3,000-case database and achieved **QWK 0.902 / accuracy 80.2% vs. baseline LLM 0.801 / 54.2%**. The security analog: embed each historical alert + final disposition with a sentence transformer, store in pgvector or FAISS, retrieve top-k by cosine similarity, and let the LLM reason from precedent.

**Fine-tuning small open models**: Use Cisco's **Foundation-Sec-8B** (Llama-3.1-8B base, continued pretraining on 4 TiB of security corpus distilled to 0.6% via an F1=0.92 relevancy classifier) or its **Foundation-Sec-8B-Instruct** and **Foundation-Sec-8B-Reasoning** variants. Cisco reports **>10% improvement over Llama-3.1-8B** on internal classification tasks. Splunk replaced Llama-3.1-70B with Foundation-Sec-8B-1.1-Instruct for incident summarization, citing better latency and consistency.

**LLM-as-judge**: Use the LLM only on the conformal-uncertain band (see §5). Claude 3.5/4 and GPT-4o/5-mini are appropriate adjudicators when self-hosted models are not viable.

### 4.5 Hybrid/ensemble — the recommended architecture

**Two-stage cascade (recommended):**
1. **Stage 1 — LightGBM/XGBoost binary classifier** scores every alert 0–1 for P(TP).
2. **Conformal calibration layer** splits alerts into three bands: P(TP) < 0.05 → auto-FP; P(TP) > 0.85 → auto-TP escalate; else → Stage 2.
3. **Stage 2 — Foundation-Sec-8B or Claude/GPT** receives the alert + top-5 most similar historical alerts (RAG) + Tier-2 enrichment, produces verdict + natural-language explanation.
4. **Stage 3 (low-confidence)** — Human analyst sees only Stage-2 "uncertain" outputs.

**Caveat**: I could not find any 2024–2026 peer-reviewed paper or vendor disclosure that benchmarks this specific XGBoost→LLM cascade with measured precision/recall. The closest published evidence is:
- **L2DHF — Jalalvand, Baruwal Chhetri, Nepal, Paris (CSIRO Data61), "Adaptive alert prioritisation in security operations centres via learning to defer with human feedback," arXiv:2506.18462, June 23, 2025**. Uses Deep RL from Human Feedback over a base classifier. Headline results (verbatim): "it achieves 13–16% higher AP accuracy for critical alerts on UNSW-NB15 and 60–67% on CICIDS2017. It also reduces misprioritisations, for example, by 98% for high-category alerts on CICIDS2017. Moreover, L2DHF decreases deferrals, for example, by 37% on UNSW-NB15, directly reducing analyst workload." (This learns to defer to humans, not to LLM.)
- **CORTEX — "Collaborative LLM Agents for High-Stakes Alert Triage," arXiv:2510.00311, October 2025**. Multi-agent LLM (not classifier+LLM cascade). Reports **+0.15 F1 improvement over single-LLM baselines at 5.44× higher latency**.
- **Google SecOps Triage and Investigation Agent** (Public Preview 2025) — Gemini-based, no numeric FP/recall published.
- Vendor marketing claims (Radiant ~90% FP reduction, Dropzone 75–95% MTTC reduction, Simbian 92% auto-resolution) are not independently benchmarked.

**Expected metrics for the POC two-stage cascade** (extrapolating from L2DHF and Simbian numbers):
- Stage-1-only: PR-AUC 0.85–0.92 on CICIDS2017; precision 0.90 / recall 0.85 on FP class at threshold 0.5.
- Stage 1 + 2: expected **alert volume reduction 70–85%** with **TP recall 95–98%** on benchmark data.
- **The >90% FP reduction AND >95% TP recall goal is achievable on benchmark datasets but is not yet demonstrated on production enterprise multi-source alerts in any published study.**

---

## 5. Explainability and Analyst Trust

### 5.1 SHAP / LIME
- Use `shap.TreeExplainer(model)` on the LightGBM/XGBoost — exact and ~1ms/alert for shallow trees. Output: per-alert top-5 contributing features with sign and magnitude.
- The Frontiers 2025 study (UNSW-NB15) and MDPI 2025 CIC IoT-DIAD study both demonstrate SHAP force plots driving analyst trust — surface them in the analyst UI.
- LIME is slower and stochastic; use it only as a development-time sanity check.

### 5.2 Attention-based for transformers
TabNet exposes per-decision-step feature masks natively; FT-Transformer's `[CLS]` token attention weights identify which features the model focused on. The 2025 KBS TabNet+Vizier paper uses SHAP on TabNet outputs as the gold standard for tabular DL explainability.

### 5.3 LLM-generated explanations — prompt template

```
SYSTEM: You are a Tier-1 SOC analyst assistant. Given a security alert and 
similar historical alerts with their final dispositions, output a JSON:
{
  "verdict": "true_positive" | "false_positive" | "needs_review",
  "confidence": 0.0-1.0,
  "rationale": "<2-4 sentences referencing specific alert fields>",
  "supporting_history": [<list of historical alert IDs>],
  "recommended_actions": ["<concrete next step>", ...]
}

USER:
Alert (OCSF JSON): {alert}
Enrichment: asset_criticality={crit}, vuln_state={vuln}, user_role={role},
            ti_match={ti}, geo_anomaly={geo}
Similar historical alerts (top-5 by embedding cosine):
  1. ID={id1}, disposition={d1}, summary={s1}
  ...
ML model score: P(TP)={score}, top SHAP features: {shap_top5}

Reason step by step, then output the JSON.
```

### 5.4 Confidence calibration
- **Platt scaling** for binary boosted trees: `sklearn.calibration.CalibratedClassifierCV(method='sigmoid', cv=5)`.
- **Isotonic regression** when you have >10K calibration samples: `method='isotonic'`.
- **Conformal prediction** for distribution-free guarantees: use `mapie` (`mapie.classification.MapieClassifier`) or `crepes`. Set α=0.05 → predictions come with a 95% coverage guarantee. The 2026 ScienceDirect paper on conformal anomaly detection in industrial CPS shows this materially reduces false-alarm rates. The 2025 arXiv **C-PP-COAD paper (arXiv:2505.01783)** addresses the calibration-data scarcity that plagues SOCs with online FDR control — directly applicable.

### 5.5 Analyst feedback loop
**L2DHF (CSIRO Data61, arXiv:2506.18462, June 2025)** is the strongest published reference. Its concrete numbers — **13–16% higher accuracy on UNSW-NB15, 60–67% on CICIDS2017, 98% reduction in high-category misprioritization, 37% reduction in deferrals** — set realistic expectations for what an analyst-feedback loop can deliver on benchmark data. Implementation pattern: maintain an `analyst_corrections` table; every N corrections (or every 7 days), retrain a "feedback head" with the original features + analyst label, weighted higher than original training data. Run shadow predictions for 24–48h before promoting to production scoring. **Cap per-analyst influence** so a single mislabeling cannot poison the model (see §8.4).

---

## 6. System Architecture for POC

### 6.1 Reference architecture (Claude Code-buildable)

```
┌─────────────────────────────────────────────────────────────────┐
│ Replay Layer: CSV/JSON file → Kafka topic / Python iterator     │
│  (CICIDS2017 + UNSW-NB15 + synthetic alerts in OCSF format)     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ Ingestion & Normalization                                       │
│  • parse to OCSF v1.3.0+ event class                            │
│  • Pydantic v2 validation                                       │
│  • store raw in DuckDB / Postgres                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ Feature Extraction (Tiers 1-5)                                  │
│  • alert-intrinsic via pandas                                   │
│  • enrichment via stub services (CMDB, TI, Tenable mocks)       │
│  • temporal via rolling DuckDB windows                          │
│  • graph features via networkx (POC) / DGL or PyG (scale)       │
│  • text embeddings via sentence-transformers MiniLM-L6-v2       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│ Stage 1 — LightGBM/XGBoost classifier + Platt/conformal         │
│  • output: P(TP), SHAP top-5, conformal prediction set          │
└──────────────────────────┬──────────────────────────────────────┘
              ┌────────────┼────────────┐
              │            │            │
        P(TP)<0.05    0.05–0.85    P(TP)>0.85
              │            │            │
        AUTO-CLOSE   ┌─────▼─────┐   ESCALATE-TP
        (log + LLM   │ Stage 2:  │
         spot-check) │ RAG+LLM   │
                     └─────┬─────┘
                           │
              ┌────────────┼────────────┐
           Confident-FP  Uncertain   Confident-TP
              │            │            │
         AUTO-CLOSE  HUMAN-REVIEW    ESCALATE
                           │
              ┌────────────▼────────────┐
              │ Analyst UI (Streamlit)  │
              │ • alert + SHAP + LLM    │
              │   rationale + similar   │
              │ • feedback capture      │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │ Feedback DB + nightly   │
              │ retrain (L2DHF-style)   │
              └─────────────────────────┘
```

### 6.2 Tech stack (concrete versions)

```python
# Python 3.11
pandas==2.2.*           # core dataframe
numpy==1.26.*
scikit-learn==1.5.*     # CalibratedClassifierCV, metrics
lightgbm==4.5.*         # primary classifier
xgboost==2.1.*          # alternative / ensemble member
catboost==1.2.*         # categorical-heavy features
shap==0.46.*            # TreeExplainer
mapie==0.9.*            # conformal prediction
imbalanced-learn==0.12.*# SMOTE/ADASYN/Borderline-SMOTE
ctgan==0.10.*           # SDV CTGAN for minority synthesis
transformers==4.45.*    # HF transformers
sentence-transformers==3.2.*
torch==2.4.*            # PyTorch (CUDA 12.1)
peft==0.13.*            # LoRA for Foundation-Sec-8B
bitsandbytes==0.44.*    # 4-bit quantization
langchain==0.3.*        # RAG orchestration
langchain-aws==0.2.*    # Bedrock integration
faiss-cpu==1.8.*        # vector store (or pgvector)
networkx==3.3.*         # graph features (POC)
torch-geometric==2.6.*  # PyG for GNN v2
duckdb==1.1.*           # local OLAP for windowed features
pydantic==2.9.*         # OCSF validation
streamlit==1.39.*       # analyst UI
mlflow==2.17.*          # experiment tracking
boto3==1.35.*           # SageMaker / Bedrock / Security Lake
adversarial-robustness-toolbox==1.18.*  # FGSM/PGD testing
river==0.21.*           # ADWIN drift detector
```

### 6.3 Infrastructure

**AWS path (recommended for enterprise parity)**:
- **SageMaker Studio** notebook on `ml.g5.2xlarge` (1× A10G 24GB) for development.
- **SageMaker Training Jobs** for LightGBM/XGBoost on CPU `ml.m5.4xlarge` (~$0.77/hr).
- **SageMaker Training** on `ml.g5.12xlarge` (4× A10G) for Foundation-Sec-8B QLoRA fine-tune (~$5.67/hr; full run on 100K examples ~4–6h).
- **Amazon Bedrock** for Claude inference at the LLM layer; alternatively SageMaker JumpStart endpoint for self-hosted Foundation-Sec-8B.
- **Amazon Security Lake** (OCSF Parquet on S3) as the persistent data store.
- **DynamoDB** for the analyst-feedback table.
- **OpenSearch Serverless** with k-NN for the RAG vector store.

**GCP equivalent**: Vertex AI Workbench + Vertex AI Training (NVIDIA L4 or A100) + Vertex AI Vector Search + BigQuery for OCSF storage.

**Colab path (POC only)**: Colab Pro+ A100 40GB is sufficient for the full pipeline including Foundation-Sec-8B QLoRA fine-tune at 4-bit precision. Estimated training time: 6–10h on 100K labeled alerts.

**GPU requirement summary**:
- Tier 1 (GBM training): CPU only.
- Tier 4 GNN training: 1× T4 or A10G sufficient.
- LLM full fine-tune of 8B: 2× A100 80GB or 1× H100.
- LLM QLoRA 4-bit: 1× A10G/L4 24GB sufficient.
- LLM inference (Foundation-Sec-8B Q4): 1× T4 16GB sufficient at batch 1, ~25 tok/sec.

---

## 7. Evaluation Framework

### 7.1 Primary ML metrics
- **Precision, recall, F1** at the operating threshold — report on both classes (FP-class and TP-class), not just macro.
- **Precision-Recall AUC** is the primary metric under extreme imbalance (**not ROC-AUC**, which is misleadingly high).
- **False Positive Rate, False Negative Rate** at the chosen threshold.
- **Brier score** and **Expected Calibration Error (ECE)** with 10 bins for calibration quality.

### 7.2 Operational metrics
- **Alert volume reduction**: (auto-closed / total alerts ingested) × 100. Compare to the modern-triage benchmark cited in *The Hacker News* (Sept 2025) of "**61% alert reduction with 1.36% false negatives**".
- **Analyst time saved**: (auto-closed × mean human triage time). Use **7 min/alert** as the median (Dropzone AI / ECS case study figure).
- **Median confidence score for correct vs. incorrect predictions** — should be well-separated.
- **Median time-to-decision** end-to-end (target <500ms Stage-1, <10s Stage-2).
- **Coverage** (fraction of alerts in the confident-FP or confident-TP bands, not the uncertain band).

### 7.3 Evaluation protocol
1. **Stratified k-fold (k=5)** for hyperparameter tuning **only** — never use for the final number.
2. **Temporal hold-out**: train on days 1–4 of CICIDS2017, test on day 5 (or week 1–3 vs. week 4 in any composited corpus). This is the only valid generalization estimate for non-stationary data.
3. **Deployment simulation**: replay the held-out set in time order, simulate the analyst-feedback loop, measure drift in calibration.
4. **Adversarial stress test**: regenerate the test set with FGSM/PGD perturbations on the feature vector (see §8.2).

### 7.4 Statistical significance
- **McNemar's test** for paired classifier comparisons on the same test set.
- **Bootstrap 95% CIs** (n=1000 resamples) on all reported metrics.
- **DeLong's test** for paired AUC comparisons.
- Significance threshold α=0.01 (not 0.05) given the multiple-comparison risk across many model variants.

---

## 8. Risks, Failure Modes, and Adversarial Considerations

### 8.1 False negatives (auto-suppressing a true positive)
**The worst failure mode in this system.** Bounding strategies:
1. **Conformal guarantee**: use `mapie` with α=0.01 → the auto-FP band has ≤1% empirical false-negative rate by construction (under the exchangeability assumption).
2. **Tripwire**: maintain a 7-day rolling window of every auto-closed alert. When any alert in the SOC is later confirmed TP, retroactively check all auto-closed alerts sharing IOCs (IP, hash, user, asset) and re-open them. This is non-negotiable.
3. **Shadow-mode launch**: run the model in "advisory only" mode for 4–6 weeks. Auto-closures are logged but not acted on. Compare against human analyst dispositions before activating.
4. **Severity gate**: never auto-close `severity_id >= 4` (high/critical) regardless of P(TP). Force to Stage 2 minimum.
5. **Rule-blacklist**: keep an opt-out list of detection rules (e.g., specific MITRE techniques) that bypass auto-closure entirely.

### 8.2 Adversarial evasion
**Concrete published threat**: The Hoang et al. 2024 APELID study on CSE-CIC-IDS2018 reports **FGSM attacks dropping IDS accuracy from 99.7% to 1.14%**, with PGD/CW/DeepFool comparably devastating. Defenses with documented effect:
- **Adversarial training**: include FGSM-perturbed samples in training set (APELID+ recovers most of the lost accuracy).
- **Feature squeezing**: round/discretize float features at inference.
- **Continuous retraining**: the 2024 arXiv attack-tree paper (arXiv:2306.05494) shows dynamic retraining alone reduces attack effectiveness even without adversarial training.
- **Defense-in-depth**: do not let the ML model be the only filter. Keep deterministic signatures for known-bad indicators in parallel.
- **Problem-space vs. feature-space gap**: an external attacker rarely has direct feature-vector access — they must manipulate raw packets/behavior, which constrains the perturbation space.

### 8.3 Concept drift
- **Detection**: ADWIN (in `river`), KS test on feature distributions per week, Isolation Forest on a sliding window per Mink et al. (MDPI *Future Internet*, 2025) validated on CICIDS2017/2018 and UNSW-NB15.
- **Strategy**: (a) continuous monitoring of PR-AUC and ECE on the most recent N=1000 labeled alerts; (b) trigger incremental retraining at >5% degradation; (c) full retraining quarterly; (d) explicit model-card versioning with rollback.
- **Drift-invariant features**: the 2024 arXiv malware concept-drift paper (arXiv:2407.13918) shows GNN-based drift-invariant features outperform simple retraining — relevant for v2.

### 8.4 Data poisoning via analyst feedback
- **Per-analyst influence cap**: weight each analyst's labels by historical agreement with senior analysts on a held-out review sample. Limit a single analyst's contribution to ≤10% of any retraining batch.
- **Label sanity check**: require quorum of 2 analysts for any disposition that overrides the model with confidence >0.95.
- **Influence functions** (Koh & Liang, 2017, foundational) to identify which historical training points most affect a given prediction — use for audit.
- **Anomalous-feedback detection**: an analyst whose corrections suddenly cluster in feature space (e.g., all flipping high-severity IDS alerts to FP) triggers review.
- **Provenance**: log analyst ID, timestamp, and rationale for every label correction; require it before the label enters the retraining pool.

---

## Recommendations (decision-ready, staged)

**Week 1–2 — Foundation**
1. Pull CICIDS2017 + UNSW-NB15 + BETH from canonical sources (CIC, UNSW Canberra, Kaggle).
2. Build the OCSF v1.3.0+ mapping layer (Pydantic models).
3. Stand up DuckDB with the composited corpus, time-partitioned.
4. Baseline LightGBM with `is_unbalance=True`, report PR-AUC on temporal hold-out — this is the number to beat.

**Week 3–4 — Stage 1 productionization**
5. Add Tier-2 enrichment features (mock CMDB, TI, vuln) — measure SHAP-attributed lift.
6. Add CTGAN minority synthesis for the hardest class (e.g., Heartbleed-analog).
7. Calibrate with `mapie` conformal at α=0.05; measure coverage.
8. Decide threshold bands on the dev set, **not** test set.

**Week 5–6 — Stage 2 LLM layer**
9. Stand up FAISS or pgvector with sentence-transformer embeddings of historical dispositions.
10. Implement the RAG+LLM adjudication prompt with structured JSON output.
11. Run Foundation-Sec-8B-Instruct locally (or Bedrock Claude) for the ambiguous band.
12. Measure end-to-end F1 / PR-AUC / FP-reduction / TP-recall vs. Stage-1-only.

**Week 7–8 — Trust and feedback**
13. Build the Streamlit analyst UI surfacing SHAP + LLM rationale + retrieved precedents.
14. Implement the feedback DB and weekly retraining job (L2DHF-style if time permits).
15. Adversarial stress test with FGSM perturbations using `adversarial-robustness-toolbox`.
16. Run the deployment simulation; produce the operational metrics table.

**Trigger thresholds for changing the recommendation:**
- If Stage 1 PR-AUC < 0.80 on temporal hold-out → re-examine features and class definitions before adding LLM complexity.
- If LLM Stage 2 adds < 5 percentage points of F1 on the uncertain band → drop it; ship Stage-1-only and use conformal "uncertain" as the direct analyst-route.
- If FGSM stress test drops Stage 1 below 50% accuracy → mandate adversarial training before any production launch.
- If analyst-feedback retraining ever degrades held-out PR-AUC → freeze feedback retraining and audit per §8.4.

---

## Caveats

1. **No public dataset is enterprise-realistic.** Every benchmark number in this report is from network-IDS or honeypot data. The first time you run the system on real multi-source SIEM data, expect 10–20 points of performance loss vs. published numbers.
2. **The >90% FP reduction + >95% TP recall goal is not demonstrated in any peer-reviewed publication on enterprise multi-source alerts as of May 2026.** It is achievable on CICIDS2017 (where Heartbleed has 11 samples and overfitting trivially gets you there) but extrapolation is unsafe.
3. **SANS 2024 rated AI/ML the lowest-satisfaction technology in SOCs.** The dominant failure mode is not model accuracy — it is integration friction, analyst trust loss after one bad auto-closure, and concept drift. Budget at least as much time for the feedback loop and tripwire system as for the model itself.
4. **Vendor numbers are not benchmark numbers.** Radiant's "90% FP reduction," Dropzone's "75–95% MTTC reduction," and Simbian's "92% auto-resolution" are marketing claims without published methodology. Treat them as upper-bound aspiration, not validated baseline.
5. **Adversarial robustness is presently weak.** Hoang et al.'s 2024 result (white-box attacks dropping accuracy by 98 percentage points) means ML alert filtering must be defense-in-depth with deterministic signatures, not a replacement, if your threat model includes sophisticated adversaries.
6. **Source-version specificity matters.** The widely circulated "71% SOC burnout" stat is from Tines' 2022 (not 2025) report; the "4.8M cybersecurity workforce gap" is from the ISC2 2024 study; the 2025 ISC2 study explicitly moved away from a single headcount figure. Workforce numbers cited by vendors often lag the underlying primary source by 1–3 years.
7. **The published XGBoost-vs-DL conclusion** ("XGBoost outperformed all DL models … tuning hyperparameters does not make DL models outperform the ML models") originates in Shwartz-Ziv & Armon's 2021 paper (arXiv:2106.03253), confirmed by the 2024 111-dataset benchmark (arXiv:2408.14817). This is now well-established and unlikely to flip for SOC-style tabular data unless raw multimodal inputs (packets, scripts, network sequences) are present in the feature pipeline.