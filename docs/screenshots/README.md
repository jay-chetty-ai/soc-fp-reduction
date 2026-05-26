# Dashboard Screenshots

Screenshots of the Streamlit analyst dashboard running against real pipeline results.
Filenames will be updated once captures are provided.

---

## 01 -- Alert List Overview

**File:** `01_alert_list_overview.png`

Main alert list with the full 10K fixture results loaded, band filter set to "All".
Shows the three-band routing distribution (auto_fp / uncertain / auto_tp), final verdict
counts, volume reduction percentage, and per-alert confidence scores at a glance.
Demonstrates the scale at which the pipeline operates without analyst intervention.

---

## 02 -- Metrics Summary

**File:** `02_metrics_summary.png`

Metrics page showing the confusion matrix, PR-AUC curve, and band distribution pie chart.
Quantitative proof of system performance. The PR-AUC curve is computed from the raw
LightGBM score (not the final verdict) and is the primary accuracy metric; the confusion
matrix reflects final pipeline decisions including Stage 2 verdicts.

---

## 03a -- Uncertain Alert: Top Half

**File:** `03a_uncertain_alert_top.png`

Detail view of an uncertain-band alert where the adversarial agent disagreed with Stage 2
but Stage 2 won on confidence. Top half shows the alert feature values (redacted per S6
field allowlist), the SHAP top-5 feature importance bars, and the Stage 2 verdict with
confidence score and full natural language rationale.

---

## 03b -- Uncertain Alert: Bottom Half

**File:** `03b_uncertain_alert_bottom.png`

Continuation of the same uncertain alert detail view. Shows the adversarial agent's
counter-argument and the weakest evidence it identified, the RAG similar historical alerts
(5 nearest neighbors with labels and cosine similarity scores), and the reconciliation note
explaining which agent's verdict was accepted and why.

---

## 04 -- Auto-FP Alert Detail

**File:** `04_auto_fp_alert_detail.png`

Detail view of an auto-FP alert (Stage 1 band, no Stage 2 call made). Shows the SHAP
top-5 explanation for the auto-close decision and the false_positive verdict with no LLM
fields populated. Contrasts with screenshots 03a/03b to show that Stage 2 only fires on
the uncertain band -- auto-FP and auto-TP alerts are decided by the ML model alone.

---

## 05 -- Band Filter: Uncertain Only

**File:** `05_band_filter_uncertain.png`

Alert list with the band filter set to "uncertain". Shows only the alerts that passed
through Stage 2 LLM adjudication, with their final verdicts, confidence scores, and
reconciliation outcomes visible in the list. Demonstrates the filtering UI and shows
what proportion of alerts required LLM reasoning.

---

## 06 -- Analyst Feedback Capture

**File:** `06_feedback_capture.png`

Analyst feedback section on an alert detail view. Shows the override verdict dropdown,
rationale text box, and submit button. The feedback workflow allows an analyst to correct
a pipeline verdict; overrides are written to the audit log with the analyst's rationale
and a timestamp. Role-based access control limits this action to the analyst role.

---

## How to update this file

When screenshots are provided, verify each filename matches the entry above and update
any filenames that differ. Reference the three key screenshots (01, 02, 03a) in the
main README.md dashboard section so they appear in the project overview.
