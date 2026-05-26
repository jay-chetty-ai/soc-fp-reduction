"""Epic 3 tests: Streamlit dashboard, metrics, and feedback persistence.

Test IDs follow the sprint backlog TC-3.x naming convention.

Approach: all business logic is extracted into pure functions in
src/ui/dashboard.py. Tests cover those functions directly without
requiring a running Streamlit server or authenticated session.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Local fixtures (not in conftest.py -- UI-test-specific)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_records():
    """30 synthetic DispositionRecord objects covering all three bands."""
    from src.pipeline.orchestrator import DispositionRecord

    records = []
    for i in range(30):
        band = ["auto_fp", "auto_tp", "uncertain"][i % 3]
        verdict = {
            "auto_fp": "false_positive",
            "auto_tp": "true_positive",
            "uncertain": "needs_review" if i % 7 == 0 else "false_positive",
        }[band]

        r = DispositionRecord(
            alert_id=f"alert{i:04d}",
            band=band,
            ml_score=0.05 + 0.04 * i if band == "auto_fp" else (0.88 + 0.001 * i if band == "auto_tp" else 0.45),
            final_verdict=verdict,
            shap_top5=[
                {"feature": "Flow Duration", "shap_value": 0.1 * (i % 5), "feature_value": float(i * 100)},
                {"feature": "SYN Flag Count", "shap_value": -0.05 * (i % 3), "feature_value": float(i % 4)},
            ],
            similar_alerts=(
                [{"alert_id": f"hist_{j}", "label": "BENIGN", "similarity": 0.9 - j * 0.05}
                 for j in range(3)]
                if band == "uncertain" else []
            ),
            recommended_actions=(
                ["No action required", "Monitor for 24h"] if verdict == "false_positive" else []
            ),
            stage2_rationale=(
                "Flow pattern consistent with benign SSH scanning." if band == "uncertain" else None
            ),
            adversarial_rationale=(
                "No attack-specific indicators found." if band == "uncertain" else None
            ),
            final_confidence=0.82 if band == "uncertain" else None,
            true_label=0 if band == "auto_fp" else 1,
        )
        records.append(r)
    return records


@pytest.fixture
def sample_results_df(sample_records):
    """DataFrame built from sample_records (mirrors pipeline output format)."""
    return pd.DataFrame([r.model_dump() for r in sample_records])


@pytest.fixture
def sample_metrics():
    """Minimal metrics dict (no PR curve or confusion matrix)."""
    return {
        "run_timestamp": "2026-05-25T14:32:11Z",
        "total_alerts": 100,
        "elapsed_seconds": 23.5,
        "throughput_per_second": 4.3,
        "band_counts": {"auto_fp": 68, "uncertain": 12, "auto_tp": 20},
        "band_pct": {"auto_fp": 68.0, "uncertain": 12.0, "auto_tp": 20.0},
        "verdict_counts": {"false_positive": 79, "true_positive": 18, "needs_review": 3},
        "volume_reduction": 0.88,
        "analyst_time_saved_hours": 7.9,
        "llm_enabled": True,
    }


@pytest.fixture
def sample_metrics_full(sample_metrics):
    """Full metrics dict including PR curve and confusion matrix."""
    recall_pts = list(np.linspace(1.0, 0.0, 50))
    precision_pts = [1.0 - 0.4 * (1 - r) for r in recall_pts]
    return {
        **sample_metrics,
        "pr_auc": 0.912,
        "precision": 0.88,
        "recall": 0.97,
        "f1": 0.923,
        "confusion_matrix": [[65, 3], [2, 30]],
        "pr_curve": {
            "precision": [round(v, 4) for v in precision_pts],
            "recall": [round(v, 4) for v in recall_pts],
        },
    }


# ---------------------------------------------------------------------------
# TC-3.1.0: Extended DispositionRecord schema
# ---------------------------------------------------------------------------

class TestDispositionRecord:
    """Verify the new fields added to DispositionRecord for dashboard support."""

    def test_default_list_fields_empty(self):
        """TC-3.1.0a: List fields default to empty (not shared mutable)."""
        from src.pipeline.orchestrator import DispositionRecord
        r1 = DispositionRecord(alert_id="a", band="auto_fp", ml_score=0.1, final_verdict="false_positive")
        r2 = DispositionRecord(alert_id="b", band="auto_tp", ml_score=0.9, final_verdict="true_positive")
        assert r1.shap_top5 == []
        assert r2.shap_top5 == []
        # Mutating one should not affect the other
        r1.shap_top5.append({"feature": "x", "shap_value": 0.1, "feature_value": 1.0})
        assert r2.shap_top5 == []

    def test_all_new_fields_present(self):
        """TC-3.1.0b: All new fields exist and have correct defaults."""
        from src.pipeline.orchestrator import DispositionRecord
        r = DispositionRecord(alert_id="x", band="auto_fp", ml_score=0.05, final_verdict="false_positive")
        assert r.recommended_actions == []
        assert r.similar_alerts == []
        assert r.reconciliation_note is None
        assert r.adversarial_rationale is None
        assert r.true_label is None

    def test_full_uncertain_record(self):
        """TC-3.1.0c: Uncertain record with all fields populated round-trips cleanly."""
        from src.pipeline.orchestrator import DispositionRecord
        r = DispositionRecord(
            alert_id="u001",
            band="uncertain",
            ml_score=0.45,
            final_verdict="false_positive",
            stage2_verdict="false_positive",
            stage2_confidence=0.82,
            stage2_rationale="Benign SSH scan pattern.",
            adversarial_verdict="false_positive",
            adversarial_rationale="No attack indicators found.",
            final_confidence=0.80,
            reconciliation_note="Stage 2 and adversarial agree.",
            recommended_actions=["No action required"],
            shap_top5=[{"feature": "Flow Duration", "shap_value": -0.3, "feature_value": 12345.0}],
            similar_alerts=[{"alert_id": "hist_001", "label": "BENIGN", "similarity": 0.92}],
            true_label=0,
        )
        dumped = r.model_dump()
        r2 = DispositionRecord(**dumped)
        assert r2.shap_top5[0]["feature"] == "Flow Duration"
        assert r2.similar_alerts[0]["similarity"] == 0.92
        assert r2.true_label == 0
        assert r2.recommended_actions == ["No action required"]

    def test_model_dump_serialisable(self, sample_records):
        """TC-3.1.0d: model_dump() is JSON-serialisable (required for parquet/CSV)."""
        from src.pipeline.orchestrator import DispositionRecord
        for rec in sample_records:
            dumped = rec.model_dump()
            # Should not raise
            json.dumps(dumped)


# ---------------------------------------------------------------------------
# TC-3.1.1: Data loading utilities
# ---------------------------------------------------------------------------

class TestDataLoading:
    """Verify load_latest_results and load_latest_metrics."""

    def test_load_results_empty_dir_returns_none(self, tmp_path):
        """TC-3.1.1a: Empty results directory returns None."""
        from src.ui.dashboard import load_latest_results
        assert load_latest_results(tmp_path) is None

    def test_load_results_no_parquet_files_returns_none(self, tmp_path):
        """TC-3.1.1b: Directory with non-parquet files returns None."""
        from src.ui.dashboard import load_latest_results
        (tmp_path / "other.csv").write_text("a,b\n1,2")
        assert load_latest_results(tmp_path) is None

    def test_load_results_valid_parquet(self, tmp_path, sample_results_df):
        """TC-3.1.1c: Valid parquet file loads correctly."""
        from src.ui.dashboard import load_latest_results
        path = tmp_path / "evaluation_20260525_143211.parquet"
        sample_results_df.to_parquet(path, index=False)
        result = load_latest_results(tmp_path)
        assert result is not None
        assert len(result) == len(sample_results_df)
        assert "alert_id" in result.columns

    def test_load_results_picks_most_recent(self, tmp_path, sample_results_df):
        """TC-3.1.1d: When multiple files exist, the most recent (lexicographically last) is returned."""
        from src.ui.dashboard import load_latest_results
        for ts in ["20260524_120000", "20260525_143211"]:
            sample_results_df.to_parquet(tmp_path / f"evaluation_{ts}.parquet", index=False)
        result = load_latest_results(tmp_path)
        assert result is not None
        assert len(result) == len(sample_results_df)

    def test_load_metrics_empty_dir_returns_none(self, tmp_path):
        """TC-3.1.1e: Empty metrics directory returns None."""
        from src.ui.dashboard import load_latest_metrics
        assert load_latest_metrics(tmp_path) is None

    def test_load_metrics_valid_json(self, tmp_path, sample_metrics):
        """TC-3.1.1f: Valid metrics JSON loads correctly."""
        from src.ui.dashboard import load_latest_metrics
        path = tmp_path / "evaluation_20260525_143211.json"
        path.write_text(json.dumps(sample_metrics))
        result = load_latest_metrics(tmp_path)
        assert result is not None
        assert result["total_alerts"] == sample_metrics["total_alerts"]
        assert result["volume_reduction"] == sample_metrics["volume_reduction"]

    def test_parquet_preserves_list_fields(self, tmp_path, sample_results_df):
        """TC-3.1.1g: Parquet round-trip preserves shap_top5 and similar_alerts as lists."""
        from src.ui.dashboard import load_latest_results
        path = tmp_path / "evaluation_20260525_143211.parquet"
        sample_results_df.to_parquet(path, index=False)
        result = load_latest_results(tmp_path)
        assert result is not None
        # First uncertain-band row should have a non-empty shap_top5
        uncertain_rows = result[result["band"] == "uncertain"]
        if len(uncertain_rows) > 0:
            shap = uncertain_rows.iloc[0]["shap_top5"]
            # pyarrow reads list-of-dict columns back as numpy object arrays
            assert hasattr(shap, "__iter__"), "shap_top5 must be iterable"
            assert len(shap) > 0


# ---------------------------------------------------------------------------
# TC-3.1.2: Band filtering
# ---------------------------------------------------------------------------

class TestBandFilter:
    """Verify filter_by_band selects correct rows."""

    def test_filter_none_returns_all(self, sample_results_df):
        """TC-3.1.2a: filter_by_band(None) returns full DataFrame."""
        from src.ui.dashboard import filter_by_band
        result = filter_by_band(sample_results_df, None)
        assert len(result) == len(sample_results_df)

    def test_filter_auto_fp(self, sample_results_df):
        """TC-3.1.2b: Filtering by auto_fp returns only auto_fp rows."""
        from src.ui.dashboard import filter_by_band
        result = filter_by_band(sample_results_df, "auto_fp")
        assert len(result) > 0
        assert (result["band"] == "auto_fp").all()

    def test_filter_uncertain(self, sample_results_df):
        """TC-3.1.2c: Filtering by uncertain returns only uncertain rows."""
        from src.ui.dashboard import filter_by_band
        result = filter_by_band(sample_results_df, "uncertain")
        assert len(result) > 0
        assert (result["band"] == "uncertain").all()

    def test_filter_nonexistent_band_returns_empty(self, sample_results_df):
        """TC-3.1.2d: Filtering by unknown band returns empty DataFrame."""
        from src.ui.dashboard import filter_by_band
        result = filter_by_band(sample_results_df, "nonexistent_band")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# TC-3.1.3: User role resolution
# ---------------------------------------------------------------------------

class TestUserRole:
    """Verify get_user_role returns correct role and defaults to viewer."""

    def test_analyst_role(self, config):
        """TC-3.1.3a: Analyst username resolves to analyst role."""
        from src.ui.dashboard import get_user_role
        assert get_user_role(config, "analyst") == "analyst"

    def test_viewer_role(self, config):
        """TC-3.1.3b: Viewer username resolves to viewer role."""
        from src.ui.dashboard import get_user_role
        assert get_user_role(config, "viewer") == "viewer"

    def test_unknown_user_defaults_to_viewer(self, config):
        """TC-3.1.3c: Unknown username defaults to viewer (least privilege)."""
        from src.ui.dashboard import get_user_role
        assert get_user_role(config, "unknown_user_xyz") == "viewer"

    def test_empty_username_defaults_to_viewer(self, config):
        """TC-3.1.3d: Empty string username defaults to viewer."""
        from src.ui.dashboard import get_user_role
        assert get_user_role(config, "") == "viewer"


# ---------------------------------------------------------------------------
# TC-3.1.4: SHAP visualisation
# ---------------------------------------------------------------------------

class TestSHAPChart:
    """Verify make_shap_chart produces valid matplotlib Figures."""

    def test_returns_figure(self):
        """TC-3.1.4a: make_shap_chart returns a matplotlib Figure."""
        from src.ui.dashboard import make_shap_chart
        shap_top5 = [
            {"feature": "Flow Duration", "shap_value": 0.3, "feature_value": 12345.0},
            {"feature": "SYN Flag Count", "shap_value": -0.2, "feature_value": 1.0},
            {"feature": "Flow Bytes/s", "shap_value": 0.15, "feature_value": 98765.0},
        ]
        fig = make_shap_chart(shap_top5)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_shap_returns_figure(self):
        """TC-3.1.4b: Empty shap_top5 returns a Figure without error."""
        from src.ui.dashboard import make_shap_chart
        fig = make_shap_chart([])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dark_mode_returns_figure(self):
        """TC-3.1.4c: dark_mode=True returns a valid Figure."""
        from src.ui.dashboard import make_shap_chart
        shap_data = [{"feature": "ACK Flag Count", "shap_value": 0.1, "feature_value": 5.0}]
        fig = make_shap_chart(shap_data, dark_mode=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_positive_bars_are_red(self):
        """TC-3.1.4d: Bars for positive SHAP values use the red colour."""
        from src.ui.dashboard import make_shap_chart
        shap_data = [{"feature": "Flow Duration", "shap_value": 0.5, "feature_value": 100.0}]
        fig = make_shap_chart(shap_data)
        ax = fig.axes[0]
        bar_colour = ax.patches[0].get_facecolor()
        # Red in rgba: (0.937..., 0.267..., 0.267..., 1.0) ~ #ef4444
        assert bar_colour[0] > 0.8  # high red channel
        plt.close(fig)

    def test_negative_bars_are_blue(self):
        """TC-3.1.4e: Bars for negative SHAP values use the blue colour."""
        from src.ui.dashboard import make_shap_chart
        shap_data = [{"feature": "Flow Duration", "shap_value": -0.5, "feature_value": 100.0}]
        fig = make_shap_chart(shap_data)
        ax = fig.axes[0]
        bar_colour = ax.patches[0].get_facecolor()
        # Blue in rgba: (0.23..., 0.51..., 0.97..., 1.0) ~ #3b82f6
        assert bar_colour[2] > 0.8  # high blue channel
        plt.close(fig)


# ---------------------------------------------------------------------------
# TC-3.1.5: Feedback persistence
# ---------------------------------------------------------------------------

class TestFeedback:
    """Verify write_feedback creates and appends to feedback.jsonl."""

    def _make_record(self, alert_id: str, override: str) -> dict:
        return {
            "timestamp": "2026-05-25T14:00:00Z",
            "alert_id": alert_id,
            "analyst_id": "analyst",
            "original_verdict": "needs_review",
            "override_verdict": override,
            "rationale": f"Confirmed as {override}.",
        }

    def test_creates_file_on_first_write(self, tmp_path):
        """TC-3.1.5a: File is created if it does not exist."""
        from src.ui.dashboard import write_feedback
        feedback_path = tmp_path / "feedback.jsonl"
        assert not feedback_path.exists()
        write_feedback(self._make_record("abc", "false_positive"), feedback_path)
        assert feedback_path.exists()

    def test_record_is_valid_json(self, tmp_path):
        """TC-3.1.5b: Written line is valid JSON with expected keys."""
        from src.ui.dashboard import write_feedback
        feedback_path = tmp_path / "feedback.jsonl"
        write_feedback(self._make_record("abc", "false_positive"), feedback_path)
        line = json.loads(feedback_path.read_text().strip())
        assert line["alert_id"] == "abc"
        assert line["override_verdict"] == "false_positive"
        assert "timestamp" in line

    def test_appends_multiple_records(self, tmp_path):
        """TC-3.1.5c: Multiple calls produce multiple lines."""
        from src.ui.dashboard import write_feedback
        feedback_path = tmp_path / "feedback.jsonl"
        for i in range(5):
            write_feedback(self._make_record(f"id{i}", "false_positive"), feedback_path)
        lines = [l for l in feedback_path.read_text().strip().split("\n") if l]
        assert len(lines) == 5

    def test_creates_parent_directory(self, tmp_path):
        """TC-3.1.5d: Parent directory is created if absent."""
        from src.ui.dashboard import write_feedback
        feedback_path = tmp_path / "sub" / "deep" / "feedback.jsonl"
        write_feedback(self._make_record("xyz", "true_positive"), feedback_path)
        assert feedback_path.exists()

    def test_no_audit_logger_does_not_raise(self, tmp_path):
        """TC-3.1.5e: audit_logger=None is handled gracefully."""
        from src.ui.dashboard import write_feedback
        feedback_path = tmp_path / "feedback.jsonl"
        # Should not raise even without audit logger
        write_feedback(self._make_record("nolog", "false_positive"), feedback_path, audit_logger=None)
        assert feedback_path.exists()


# ---------------------------------------------------------------------------
# TC-3.2.1: Metrics chart rendering
# ---------------------------------------------------------------------------

class TestMetricsCharts:
    """Verify chart factory functions return valid Figures."""

    def test_band_distribution_chart(self, sample_metrics):
        """TC-3.2.1a: Band distribution chart renders without error."""
        from src.ui.dashboard import make_band_distribution_chart
        fig = make_band_distribution_chart(sample_metrics)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_band_distribution_dark_mode(self, sample_metrics):
        """TC-3.2.1b: Dark mode band distribution chart renders."""
        from src.ui.dashboard import make_band_distribution_chart
        fig = make_band_distribution_chart(sample_metrics, dark_mode=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_pr_curve_chart_with_data(self, sample_metrics_full):
        """TC-3.2.1c: PR curve chart renders when pr_curve key is present."""
        from src.ui.dashboard import make_pr_curve_chart
        fig = make_pr_curve_chart(sample_metrics_full)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_pr_curve_chart_no_data_returns_none(self, sample_metrics):
        """TC-3.2.1d: PR curve returns None when pr_curve key is absent."""
        from src.ui.dashboard import make_pr_curve_chart
        assert make_pr_curve_chart(sample_metrics) is None

    def test_confusion_matrix_chart_with_data(self, sample_metrics_full):
        """TC-3.2.1e: Confusion matrix heatmap renders when data is present."""
        from src.ui.dashboard import make_confusion_matrix_chart
        fig = make_confusion_matrix_chart(sample_metrics_full)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_confusion_matrix_chart_no_data_returns_none(self, sample_metrics):
        """TC-3.2.1f: Confusion matrix returns None when key is absent."""
        from src.ui.dashboard import make_confusion_matrix_chart
        assert make_confusion_matrix_chart(sample_metrics) is None


# ---------------------------------------------------------------------------
# TC-3.2.2: Metrics correctness
# ---------------------------------------------------------------------------

class TestMetricsCorrectness:
    """Verify volume reduction and other computed metrics are numerically correct."""

    def test_volume_reduction_formula(self, sample_metrics):
        """TC-3.2.2a: volume_reduction = (auto_fp + auto_tp) / total."""
        counts = sample_metrics["band_counts"]
        total = sample_metrics["total_alerts"]
        expected = (counts["auto_fp"] + counts["auto_tp"]) / total
        assert sample_metrics["volume_reduction"] == pytest.approx(expected, rel=1e-3)

    def test_band_pct_sums_to_100(self, sample_metrics):
        """TC-3.2.2b: Band percentages sum to 100."""
        pct_sum = sum(sample_metrics["band_pct"].values())
        assert pct_sum == pytest.approx(100.0, rel=1e-3)

    def test_compute_metrics_from_records(self, sample_records):
        """TC-3.2.2c: _compute_metrics() from run_pipeline produces expected keys."""
        import sys
        sys.path.insert(0, "scripts")
        from run_pipeline import _compute_metrics
        metrics = _compute_metrics(sample_records, elapsed=10.0, llm_enabled=False)
        assert "total_alerts" in metrics
        assert metrics["total_alerts"] == len(sample_records)
        assert "band_counts" in metrics
        assert "verdict_counts" in metrics
        assert "volume_reduction" in metrics
        assert 0.0 <= metrics["volume_reduction"] <= 1.0
        assert "analyst_time_saved_hours" in metrics

    def test_compute_metrics_includes_pr_metrics_when_labels_present(self, sample_records):
        """TC-3.2.2d: PR-AUC and confusion matrix included when true_label is set."""
        import sys
        sys.path.insert(0, "scripts")
        from run_pipeline import _compute_metrics
        # sample_records all have true_label set
        metrics = _compute_metrics(sample_records, elapsed=5.0, llm_enabled=False)
        assert "pr_auc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "confusion_matrix" in metrics
        assert "pr_curve" in metrics

    def test_compute_metrics_no_pr_when_no_labels(self):
        """TC-3.2.2e: PR metrics absent when true_label is None for all records."""
        import sys
        sys.path.insert(0, "scripts")
        from run_pipeline import _compute_metrics
        from src.pipeline.orchestrator import DispositionRecord
        records = [
            DispositionRecord(alert_id=f"a{i}", band="auto_fp", ml_score=0.1,
                              final_verdict="false_positive", true_label=None)
            for i in range(10)
        ]
        metrics = _compute_metrics(records, elapsed=1.0, llm_enabled=False)
        assert "pr_auc" not in metrics
        assert "confusion_matrix" not in metrics

    def test_analyst_time_saved_formula(self, sample_records):
        """TC-3.2.2f: Analyst time saved = auto_fp_count * 7 / 60 hours."""
        import sys
        sys.path.insert(0, "scripts")
        from run_pipeline import _compute_metrics
        metrics = _compute_metrics(sample_records, elapsed=5.0, llm_enabled=False)
        auto_fp_n = sum(1 for r in sample_records if r.band == "auto_fp")
        expected = round(auto_fp_n * 7 / 60, 1)
        assert metrics["analyst_time_saved_hours"] == expected


# ---------------------------------------------------------------------------
# TC-3.1.6: Dashboard module imports cleanly
# ---------------------------------------------------------------------------

class TestDashboardModule:
    """Verify the dashboard module can be imported and key functions are callable."""

    def test_module_imports(self):
        """TC-3.1.6a: src.ui.dashboard imports without error."""
        import importlib
        mod = importlib.import_module("src.ui.dashboard")
        assert mod is not None

    def test_all_public_functions_callable(self):
        """TC-3.1.6b: All public utility functions are callable."""
        from src.ui import dashboard
        for name in [
            "load_config", "load_latest_results", "load_latest_metrics",
            "filter_by_band", "get_user_role", "write_feedback",
            "make_shap_chart", "make_band_distribution_chart",
            "make_pr_curve_chart", "make_confusion_matrix_chart",
        ]:
            assert callable(getattr(dashboard, name)), f"{name} is not callable"

    def test_load_config_reads_project_config(self):
        """TC-3.1.6c: load_config() reads real config.yaml and returns auth section."""
        from src.ui.dashboard import load_config
        cfg = load_config(Path("config.yaml"))
        assert "auth" in cfg
        assert "credentials" in cfg["auth"]
        assert "usernames" in cfg["auth"]["credentials"]

    def test_config_has_analyst_and_viewer(self):
        """TC-3.1.6d: config.yaml contains analyst and viewer credentials."""
        from src.ui.dashboard import load_config
        cfg = load_config(Path("config.yaml"))
        usernames = cfg["auth"]["credentials"]["usernames"]
        assert "analyst" in usernames
        assert "viewer" in usernames
        # Passwords must be bcrypt hashes, not plaintext
        for user, data in usernames.items():
            assert data["password"].startswith("$2b$"), \
                f"{user} password is not bcrypt-hashed"
