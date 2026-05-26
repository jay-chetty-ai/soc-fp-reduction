"""Streamlit analyst dashboard for SOC false positive triage results.

Roles:
  analyst -- can view all data and submit feedback overrides
  viewer  -- read-only access; feedback panel hidden

Run:
    streamlit run src/ui/dashboard.py
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")  # non-interactive backend for server rendering

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme CSS
# ---------------------------------------------------------------------------

_DARK_CSS = """
<style>
[data-testid="stAppViewContainer"] { background-color: #1e1e2e !important; }
[data-testid="stHeader"] { background-color: #1e1e2e !important; }
[data-testid="stSidebar"] > div:first-child { background-color: #181825 !important; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label { color: #cdd6f4 !important; }
.stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown li, p, label, h1, h2, h3 { color: #cdd6f4 !important; }
[data-testid="stMetricValue"] { color: #89b4fa !important; }
[data-testid="stMetricLabel"] { color: #a6adc8 !important; }
.stDataFrame { background-color: #313244 !important; }
</style>
"""

_LIGHT_CSS = """
<style>
[data-testid="stAppViewContainer"] { background-color: #ffffff !important; }
[data-testid="stHeader"] { background-color: #ffffff !important; }
[data-testid="stSidebar"] > div:first-child { background-color: #f5f5f5 !important; }
</style>
"""

# Band colour mapping for display
_BAND_COLOUR = {
    "auto_fp": "🟢",
    "uncertain": "🟡",
    "auto_tp": "🔴",
}

_VERDICT_COLOUR = {
    "false_positive": "🟢",
    "true_positive": "🔴",
    "needs_review": "🟡",
}


# ---------------------------------------------------------------------------
# Pure utility functions (importable and testable without Streamlit runtime)
# ---------------------------------------------------------------------------

def load_config(config_path: Path = Path("config.yaml")) -> dict:
    """Load and return the parsed config.yaml.

    Args:
        config_path: Path to the config file.

    Returns:
        Parsed configuration dict.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def list_available_runs(results_dir: Path, metrics_dir: Path) -> list[dict]:
    """Return metadata for all available pipeline runs, newest first.

    Each entry has:
        label         -- human-readable string for the sidebar selector
        parquet_path  -- Path to the results parquet
        metrics_path  -- Path to the matching metrics JSON, or None

    Args:
        results_dir: Directory containing evaluation_*.parquet files.
        metrics_dir: Directory containing evaluation_*.json files.

    Returns:
        List of run dicts ordered newest-first, empty if no runs found.
    """
    import pyarrow.parquet as pq

    parquet_files = sorted(results_dir.glob("evaluation_*.parquet"), reverse=True)
    runs = []
    for p in parquet_files:
        stem = p.stem  # evaluation_YYYYMMDD_HHMMSS
        ts_str = stem.replace("evaluation_", "")
        try:
            dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            date_label = dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            date_label = ts_str
        try:
            n_rows = pq.read_metadata(p).num_rows
        except Exception:
            n_rows = 0
        metrics_path = metrics_dir / f"{stem}.json"
        runs.append({
            "label": f"{n_rows:,} alerts — {date_label}",
            "parquet_path": p,
            "metrics_path": metrics_path if metrics_path.exists() else None,
        })
    return runs


def load_results(path: Path) -> pd.DataFrame | None:
    """Load results from a specific parquet file.

    Args:
        path: Path to an evaluation_*.parquet file.

    Returns:
        DataFrame of DispositionRecord rows, or None on failure.
    """
    try:
        df = pd.read_parquet(path)
        logger.info("Loaded %d results from %s.", len(df), path)
        return df
    except Exception as exc:
        logger.error("Failed to load results from %s: %s", path, exc)
        return None


def load_metrics(path: Path) -> dict | None:
    """Load metrics from a specific JSON file.

    Args:
        path: Path to an evaluation_*.json file.

    Returns:
        Metrics dict, or None on failure.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        logger.info("Loaded metrics from %s.", path)
        return data
    except Exception as exc:
        logger.error("Failed to load metrics from %s: %s", path, exc)
        return None


def filter_by_band(df: pd.DataFrame, band: str | None) -> pd.DataFrame:
    """Return rows matching the given band, or all rows if band is None.

    Args:
        df: Results DataFrame.
        band: Band value to filter on ("auto_fp", "uncertain", "auto_tp"), or None for all.

    Returns:
        Filtered DataFrame.
    """
    if band is None:
        return df
    return df[df["band"] == band].reset_index(drop=True)


def get_user_role(config: dict, username: str) -> str:
    """Resolve a username to its configured role.

    Falls back to "viewer" (least privilege) for unknown users.

    Args:
        config: Parsed config.yaml dict.
        username: Authenticated username from session state.

    Returns:
        Role string: "analyst" or "viewer".
    """
    users = config.get("auth", {}).get("credentials", {}).get("usernames", {})
    user_data = users.get(username, {})
    return user_data.get("role", "viewer")


def write_feedback(record: dict, feedback_path: Path, audit_logger: Any = None) -> None:
    """Append an analyst feedback record to the feedback JSONL file.

    Args:
        record: Dict with keys: timestamp, alert_id, analyst_id, original_verdict,
            override_verdict, rationale.
        feedback_path: Path to the feedback.jsonl file (created if absent).
        audit_logger: Optional AuditLogger instance; if provided, logs the feedback
            to the audit chain.
    """
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feedback_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info(
        "Feedback written for alert %s by %s.",
        record.get("alert_id"),
        record.get("analyst_id"),
    )

    if audit_logger is not None:
        try:
            from src.utils.audit import FeedbackEntry
            entry = FeedbackEntry(
                timestamp=record["timestamp"],
                alert_id=record["alert_id"],
                analyst_id=record["analyst_id"],
                override_verdict=record["override_verdict"],
                original_verdict=record["original_verdict"],
                rationale=record["rationale"],
                previous_entry_hash="",  # AuditLogger fills this in
            )
            audit_logger.log_feedback(entry)
        except Exception as exc:
            logger.warning("Failed to write feedback to audit log: %s", exc)


def make_shap_chart(shap_top5: list[dict], dark_mode: bool = False) -> plt.Figure:
    """Build a horizontal bar chart of SHAP feature contributions.

    Positive values push toward true_positive; negative toward false_positive.

    Args:
        shap_top5: List of dicts with keys feature, shap_value, feature_value.
        dark_mode: If True, render with dark background.

    Returns:
        Matplotlib Figure.
    """
    bg_colour = "#1e1e2e" if dark_mode else "#ffffff"
    text_colour = "#cdd6f4" if dark_mode else "#1e1e2e"
    grid_colour = "#313244" if dark_mode else "#e5e7eb"

    if not shap_top5:
        fig, ax = plt.subplots(figsize=(7, 2))
        fig.patch.set_facecolor(bg_colour)
        ax.set_facecolor(bg_colour)
        ax.text(0.5, 0.5, "No SHAP data available", ha="center", va="center",
                color=text_colour, transform=ax.transAxes)
        ax.axis("off")
        return fig

    features = [d["feature"] for d in shap_top5]
    values = [d["shap_value"] for d in shap_top5]
    feat_vals = [d.get("feature_value", 0.0) for d in shap_top5]
    colours = ["#ef4444" if v > 0 else "#3b82f6" for v in values]

    fig, ax = plt.subplots(figsize=(7, max(2, len(features) * 0.55 + 0.5)))
    fig.patch.set_facecolor(bg_colour)
    ax.set_facecolor(bg_colour)

    bars = ax.barh(features, values, color=colours, height=0.6)
    ax.axvline(x=0, color=text_colour, linewidth=0.8, alpha=0.6)

    # Annotate bars with feature values
    for bar, fv in zip(bars, feat_vals):
        width = bar.get_width()
        x_pos = width + 0.002 if width >= 0 else width - 0.002
        ha = "left" if width >= 0 else "right"
        ax.text(
            x_pos, bar.get_y() + bar.get_height() / 2,
            f"  val={fv:.2g}",
            va="center", ha=ha, fontsize=7, color=text_colour, alpha=0.7,
        )

    ax.set_xlabel("SHAP value  (red = toward TP,  blue = toward FP)", color=text_colour, fontsize=8)
    ax.tick_params(colors=text_colour, labelsize=8)
    ax.xaxis.label.set_color(text_colour)
    for spine in ax.spines.values():
        spine.set_edgecolor(grid_colour)
    ax.grid(axis="x", color=grid_colour, linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    return fig


def make_band_distribution_chart(metrics: dict, dark_mode: bool = False) -> plt.Figure:
    """Pie chart of alert band distribution.

    Args:
        metrics: Metrics dict from _compute_metrics().
        dark_mode: If True, render with dark background.

    Returns:
        Matplotlib Figure.
    """
    bg_colour = "#1e1e2e" if dark_mode else "#ffffff"
    text_colour = "#cdd6f4" if dark_mode else "#1e1e2e"

    band_counts = metrics.get("band_counts", {})
    labels = list(band_counts.keys())
    sizes = list(band_counts.values())
    colours = {"auto_fp": "#22c55e", "uncertain": "#f59e0b", "auto_tp": "#ef4444"}
    pie_colours = [colours.get(lbl, "#6b7280") for lbl in labels]

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(bg_colour)
    ax.set_facecolor(bg_colour)

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=pie_colours,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"color": text_colour, "fontsize": 9},
    )
    for at in autotexts:
        at.set_color(bg_colour)
        at.set_fontweight("bold")

    ax.set_title("Band Distribution", color=text_colour, fontsize=10)
    plt.tight_layout()
    return fig


def make_pr_curve_chart(metrics: dict, dark_mode: bool = False) -> plt.Figure | None:
    """Precision-Recall curve chart.

    Args:
        metrics: Metrics dict; must contain pr_curve key.
        dark_mode: If True, render with dark background.

    Returns:
        Matplotlib Figure, or None if PR curve data is absent.
    """
    if "pr_curve" not in metrics:
        return None

    bg_colour = "#1e1e2e" if dark_mode else "#ffffff"
    text_colour = "#cdd6f4" if dark_mode else "#1e1e2e"
    grid_colour = "#313244" if dark_mode else "#e5e7eb"
    line_colour = "#89b4fa" if dark_mode else "#2563eb"

    precision = metrics["pr_curve"]["precision"]
    recall = metrics["pr_curve"]["recall"]
    pr_auc = metrics.get("pr_auc", 0.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(bg_colour)
    ax.set_facecolor(bg_colour)

    ax.plot(recall, precision, color=line_colour, linewidth=2,
            label=f"PR-AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall", color=text_colour, fontsize=9)
    ax.set_ylabel("Precision", color=text_colour, fontsize=9)
    ax.set_title("Precision-Recall Curve", color=text_colour, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors=text_colour, labelsize=8)
    ax.legend(fontsize=8, labelcolor=text_colour, facecolor=bg_colour,
              edgecolor=grid_colour)
    ax.grid(color=grid_colour, linewidth=0.5, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(grid_colour)

    plt.tight_layout()
    return fig


def make_confusion_matrix_chart(metrics: dict, dark_mode: bool = False) -> plt.Figure | None:
    """Confusion matrix heatmap.

    Args:
        metrics: Metrics dict; must contain confusion_matrix key.
        dark_mode: If True, render with dark background.

    Returns:
        Matplotlib Figure, or None if confusion matrix data is absent.
    """
    if "confusion_matrix" not in metrics:
        return None

    import seaborn as sns

    bg_colour = "#1e1e2e" if dark_mode else "#ffffff"
    text_colour = "#cdd6f4" if dark_mode else "#1e1e2e"
    cmap = "Blues" if not dark_mode else "crest"

    cm = np.array(metrics["confusion_matrix"])
    labels = ["Benign (0)", "Attack (1)"]

    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.patch.set_facecolor(bg_colour)
    ax.set_facecolor(bg_colour)

    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=labels, yticklabels=labels,
        ax=ax, cbar=False,
        annot_kws={"size": 11, "color": text_colour},
    )
    ax.set_xlabel("Predicted", color=text_colour, fontsize=9)
    ax.set_ylabel("Actual", color=text_colour, fontsize=9)
    ax.set_title("Confusion Matrix", color=text_colour, fontsize=10)
    ax.tick_params(colors=text_colour, labelsize=8)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit page renderers
# ---------------------------------------------------------------------------

def _render_no_data_warning(data_type: str) -> None:
    """Show a guidance message when pipeline output is absent."""
    import streamlit as st
    st.warning(
        f"No {data_type} found. Run the pipeline first:\n\n"
        "```bash\n"
        "python scripts/train_stage1.py --skip-tuning\n"
        "python scripts/build_rag_index.py --sample-size 50000\n"
        "python scripts/run_pipeline.py --input data/fixtures/fixture_10k.csv\n"
        "```",
        icon="⚠️",
    )


def _render_alert_detail(row: pd.Series, username: str, role: str,
                          config: dict, dark_mode: bool) -> None:
    """Render the detail panel for a single selected alert.

    Args:
        row: A single row from the results DataFrame.
        username: Authenticated username.
        role: User role ("analyst" or "viewer").
        config: Parsed config dict.
        dark_mode: Current theme mode.
    """
    import streamlit as st

    alert_id = row["alert_id"]
    band = row["band"]
    verdict = row["final_verdict"]

    st.subheader(f"Alert {alert_id}")
    band_icon = _BAND_COLOUR.get(band, "⚪")
    verdict_icon = _VERDICT_COLOUR.get(verdict, "⚪")

    col1, col2, col3 = st.columns(3)
    col1.metric("Band", f"{band_icon} {band}")
    col2.metric("ML Score", f"{row['ml_score']:.3f}")
    col3.metric(
        "Final Verdict",
        f"{verdict_icon} {verdict}",
        delta=f"conf={row['final_confidence']:.2f}" if pd.notna(row.get("final_confidence")) else None,
    )

    # SHAP explanation
    shap_data = row.get("shap_top5", [])
    if isinstance(shap_data, str):
        try:
            shap_data = json.loads(shap_data)
        except (json.JSONDecodeError, ValueError):
            shap_data = []
    elif hasattr(shap_data, "tolist"):
        # pyarrow reads list columns back as numpy object arrays
        shap_data = shap_data.tolist()

    st.markdown("**SHAP Feature Contributions**")
    if shap_data:
        fig = make_shap_chart(shap_data, dark_mode=dark_mode)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.caption("SHAP data not available for this alert (auto-routed band).")

    # Stage 2 LLM rationale
    if pd.notna(row.get("stage2_rationale")):
        st.markdown("**Stage 2 LLM Rationale**")
        st.info(row["stage2_rationale"])

        if pd.notna(row.get("adversarial_rationale")):
            st.markdown("**Adversarial Counter-Argument**")
            st.warning(row["adversarial_rationale"])

        if pd.notna(row.get("reconciliation_note")) and row["reconciliation_note"]:
            st.caption(f"Reconciliation: {row['reconciliation_note']}")

        actions = row.get("recommended_actions", [])
        if isinstance(actions, str):
            try:
                actions = json.loads(actions)
            except (json.JSONDecodeError, ValueError):
                actions = []
        elif hasattr(actions, "tolist"):
            actions = actions.tolist()
        if actions:
            st.markdown("**Recommended Actions**")
            for action in actions:
                st.markdown(f"- {action}")

    # Similar historical alerts
    similar = row.get("similar_alerts", [])
    if isinstance(similar, str):
        try:
            similar = json.loads(similar)
        except (json.JSONDecodeError, ValueError):
            similar = []
    elif hasattr(similar, "tolist"):
        similar = similar.tolist()

    if similar:
        st.markdown("**Similar Historical Alerts (RAG)**")
        sim_rows = [
            {
                "ID": s["alert_id"],
                "Label": s["label"],
                "Cosine Similarity": f"{s['similarity']:.3f}",
            }
            for s in similar
        ]
        st.dataframe(pd.DataFrame(sim_rows), use_container_width=True, hide_index=True)

    # Feedback capture (analyst role only)
    if role == "analyst":
        st.markdown("---")
        st.markdown("**Submit Analyst Feedback**")
        with st.form(key=f"feedback_{alert_id}"):
            override = st.selectbox(
                "Override verdict",
                options=["true_positive", "false_positive", "needs_review"],
                index=["true_positive", "false_positive", "needs_review"].index(
                    verdict if verdict in ("true_positive", "false_positive", "needs_review")
                    else "needs_review"
                ),
            )
            rationale = st.text_area("Rationale (required)", height=80)
            submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            if not rationale.strip():
                st.error("Rationale cannot be empty.")
            else:
                feedback_path = Path(config.get("dashboard", {}).get(
                    "feedback_path", "data/processed/feedback.jsonl"
                ))
                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "alert_id": alert_id,
                    "analyst_id": username,
                    "original_verdict": verdict,
                    "override_verdict": override,
                    "rationale": rationale.strip(),
                }
                write_feedback(record, feedback_path)
                st.success(f"Feedback recorded: {verdict} -> {override}")
    else:
        st.caption("Feedback submission requires analyst role.")


def render_alert_list_page(results_df: pd.DataFrame, config: dict,
                            username: str, role: str, dark_mode: bool) -> None:
    """Render the alert list and detail panel.

    Args:
        results_df: Full results DataFrame.
        config: Parsed config dict.
        username: Authenticated username.
        role: User role.
        dark_mode: Current theme mode.
    """
    import streamlit as st

    st.header("Alert Triage Results")

    # Band filter
    col_filter, col_stats = st.columns([2, 3])
    with col_filter:
        band_options = ["All"] + sorted(results_df["band"].unique().tolist())
        selected_band_label = st.selectbox("Filter by band", band_options)
        selected_band = None if selected_band_label == "All" else selected_band_label

    filtered_df = filter_by_band(results_df, selected_band)

    with col_stats:
        total = len(filtered_df)
        verdicts = filtered_df["final_verdict"].value_counts().to_dict()
        st.metric("Showing", f"{total} alerts")
        st.caption(
            "  ".join(
                f"{_VERDICT_COLOUR.get(k, '⚪')} {k}: {v}"
                for k, v in sorted(verdicts.items())
            )
        )

    # Display table with band/verdict icons
    display_df = filtered_df[
        ["alert_id", "band", "ml_score", "final_verdict", "final_confidence",
         "stage2_verdict", "true_label"]
    ].copy()
    display_df["band"] = display_df["band"].map(
        lambda b: f"{_BAND_COLOUR.get(b, '')} {b}"
    )
    display_df["final_verdict"] = display_df["final_verdict"].map(
        lambda v: f"{_VERDICT_COLOUR.get(v, '')} {v}"
    )
    display_df = display_df.rename(columns={
        "alert_id": "Alert ID",
        "band": "Band",
        "ml_score": "ML Score",
        "final_verdict": "Verdict",
        "final_confidence": "Confidence",
        "stage2_verdict": "Stage 2",
        "true_label": "True Label",
    })

    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        key="alert_table",
    )

    # Detail panel for the selected row
    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        selected_idx = selected_rows[0]
        selected_row = filtered_df.iloc[selected_idx]
        st.markdown("---")
        _render_alert_detail(selected_row, username, role, config, dark_mode)


def render_metrics_page(metrics: dict, dark_mode: bool) -> None:
    """Render the metrics and evaluation dashboard page.

    Args:
        metrics: Metrics dict from _compute_metrics() / load_latest_metrics().
        dark_mode: Current theme mode.
    """
    import streamlit as st

    st.header("Pipeline Evaluation Metrics")

    ts = metrics.get("run_timestamp", "unknown")
    st.caption(f"Run: {ts}  |  LLM: {'enabled' if metrics.get('llm_enabled') else 'disabled'}")

    # Summary metrics row
    cols = st.columns(4)
    cols[0].metric("Total Alerts", f"{metrics.get('total_alerts', 0):,}")
    cols[1].metric("Volume Reduction", f"{100 * metrics.get('volume_reduction', 0):.1f}%",
                   delta="target ≥ 70%")
    cols[2].metric("Time Saved", f"{metrics.get('analyst_time_saved_hours', 0):.1f} h",
                   delta="at 7 min/alert")
    cols[3].metric(
        "PR-AUC",
        f"{metrics['pr_auc']:.3f}" if "pr_auc" in metrics else "N/A",
        delta="target ≥ 0.85" if "pr_auc" in metrics else None,
    )

    # Second row: precision/recall/F1
    if "precision" in metrics:
        cols2 = st.columns(3)
        cols2[0].metric("Precision", f"{metrics['precision']:.3f}")
        cols2[1].metric("Recall", f"{metrics['recall']:.3f}", delta="target ≥ 0.95")
        cols2[2].metric("F1", f"{metrics['f1']:.3f}")

    st.markdown("---")

    # Band distribution + PR curve side by side
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Band Distribution**")
        fig_band = make_band_distribution_chart(metrics, dark_mode=dark_mode)
        st.pyplot(fig_band, use_container_width=True)
        plt.close(fig_band)

        # Numeric breakdown table
        band_counts = metrics.get("band_counts", {})
        band_pct = metrics.get("band_pct", {})
        band_rows = [
            {"Band": f"{_BAND_COLOUR.get(k, '')} {k}", "Count": v,
             "Pct": f"{band_pct.get(k, 0):.1f}%"}
            for k, v in band_counts.items()
        ]
        if band_rows:
            st.dataframe(pd.DataFrame(band_rows), use_container_width=True, hide_index=True)

    with chart_col2:
        pr_fig = make_pr_curve_chart(metrics, dark_mode=dark_mode)
        if pr_fig is not None:
            st.markdown("**Precision-Recall Curve**")
            st.pyplot(pr_fig, use_container_width=True)
            plt.close(pr_fig)
        else:
            st.info("PR curve available after running with labeled data (CICIDS2017 CSVs).")

    # Confusion matrix
    cm_fig = make_confusion_matrix_chart(metrics, dark_mode=dark_mode)
    if cm_fig is not None:
        st.markdown("---")
        cm_col, _ = st.columns([1, 1])
        with cm_col:
            st.markdown("**Confusion Matrix**")
            st.pyplot(cm_fig, use_container_width=True)
            plt.close(cm_fig)

    # Verdict breakdown table
    st.markdown("---")
    st.markdown("**Verdict Breakdown**")
    verdict_counts = metrics.get("verdict_counts", {})
    total = metrics.get("total_alerts", 1)
    verdict_rows = [
        {
            "Verdict": f"{_VERDICT_COLOUR.get(k, '')} {k}",
            "Count": v,
            "Pct": f"{100 * v / total:.1f}%",
        }
        for k, v in sorted(verdict_counts.items())
    ]
    if verdict_rows:
        st.dataframe(pd.DataFrame(verdict_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the Streamlit dashboard."""
    import streamlit as st
    import streamlit_authenticator as stauth

    st.set_page_config(
        page_title="SOC Triage Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    config_path = Path("config.yaml")
    if not config_path.exists():
        st.error("config.yaml not found. Run from the project root directory.")
        return
    config = load_config(config_path)

    # ------------------------------------------------------------------
    # Authentication (S8)
    # ------------------------------------------------------------------
    auth_cfg = config.get("auth", {})
    credentials = auth_cfg.get("credentials", {"usernames": {}})
    cookie_cfg = auth_cfg.get("cookie", {})

    authenticator = stauth.Authenticate(
        credentials=credentials,
        cookie_name=cookie_cfg.get("name", "soc_dashboard_auth"),
        cookie_key=cookie_cfg.get("key", "change-me"),
        cookie_expiry_days=float(cookie_cfg.get("expiry_days", 1)),
        auto_hash=False,  # passwords already bcrypt-hashed in config.yaml
    )

    authenticator.login(location="main")

    auth_status = st.session_state.get("authentication_status")
    username = st.session_state.get("username", "")
    name = st.session_state.get("name", "")

    if auth_status is False:
        st.error("Incorrect username or password.")
        return

    if auth_status is None:
        st.info("Please log in to access the dashboard.")
        return

    # ------------------------------------------------------------------
    # Authenticated -- render sidebar and main content
    # ------------------------------------------------------------------
    role = get_user_role(config, username)

    with st.sidebar:
        st.markdown(f"**{name}** `({role})`")
        authenticator.logout(location="sidebar")
        st.markdown("---")

        page = st.radio("Navigation", ["Alert List", "Metrics"])
        st.markdown("---")

        dark_mode = st.toggle("Dark mode", value=False, key="dark_mode_toggle")

        st.markdown("---")

        # Run selector: lists all available evaluation runs newest-first
        results_dir = Path(config.get("dashboard", {}).get("results_dir", "results"))
        metrics_dir = Path(config.get("dashboard", {}).get("metrics_dir", "metrics"))

        runs = list_available_runs(results_dir, metrics_dir)
        if runs:
            st.markdown("**Pipeline run**")
            run_labels = [r["label"] for r in runs]
            selected_idx = st.selectbox(
                "Select run",
                options=range(len(run_labels)),
                format_func=lambda i: run_labels[i],
                label_visibility="collapsed",
            )
            selected_run = runs[selected_idx]
        else:
            selected_run = None

    # Apply CSS theme
    st.markdown(_DARK_CSS if dark_mode else _LIGHT_CSS, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Page routing
    # ------------------------------------------------------------------
    if selected_run is None:
        _render_no_data_warning("pipeline results")
        return

    if page == "Alert List":
        results_df = load_results(selected_run["parquet_path"])
        if results_df is None:
            _render_no_data_warning("pipeline results")
        else:
            render_alert_list_page(results_df, config, username, role, dark_mode)

    elif page == "Metrics":
        if selected_run["metrics_path"] is None:
            _render_no_data_warning("metrics")
        else:
            metrics = load_metrics(selected_run["metrics_path"])
            if metrics is None:
                _render_no_data_warning("metrics")
            else:
                render_metrics_page(metrics, dark_mode)


if __name__ == "__main__":
    main()
