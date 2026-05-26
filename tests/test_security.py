"""Security control tests for S2 (secrets), S3 (audit), and S4 (model integrity).

Additional S1/S5/S6/S7 tests are added in Story 2.3.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from unittest.mock import patch

import lightgbm as lgb
import pytest

from src.models.integrity import ModelIntegrityError, save_hash, verify_hash
from src.utils.audit import AuditLogger, AuditEntry
from src.utils.secrets import RedactionFilter, load_api_key, redact_secrets


# =============================================================================
# S3: Audit Hash Chain
# =============================================================================

_GENESIS_HASH = hashlib.sha256(b"GENESIS").hexdigest()


class TestAuditHashChain:

    def test_tc_s_15_first_entry_uses_genesis_hash(self, tmp_audit_path):
        """TC-S.15: First audit entry references the GENESIS hash."""
        audit = AuditLogger(tmp_audit_path)
        audit.log_decision(
            alert_id="alert_001",
            stage="stage1",
            verdict="auto_fp",
            model_version="1.0",
            band="auto_fp",
        )
        with open(tmp_audit_path) as f:
            entry = json.loads(f.readline())
        assert entry["previous_entry_hash"] == _GENESIS_HASH

    def test_tc_s_16_second_entry_references_first(self, tmp_audit_path):
        """TC-S.16: Second entry's previous_hash equals SHA-256 of first entry line."""
        audit = AuditLogger(tmp_audit_path)
        audit.log_decision(
            alert_id="alert_001", stage="stage1", verdict="auto_fp",
            model_version="1.0",
        )
        audit.log_decision(
            alert_id="alert_002", stage="stage1", verdict="auto_tp",
            model_version="1.0",
        )
        with open(tmp_audit_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2
        expected = hashlib.sha256(lines[0].encode()).hexdigest()
        second = json.loads(lines[1])
        assert second["previous_entry_hash"] == expected

    def test_tc_s_17_chain_validates_on_five_entries(self, tmp_audit_path):
        """TC-S.17: Hash chain validates correctly for 5 sequential entries."""
        audit = AuditLogger(tmp_audit_path)
        for i in range(5):
            audit.log_decision(
                alert_id=f"alert_{i:03d}",
                stage="stage1",
                verdict="auto_fp",
                model_version="1.0",
            )
        assert audit.validate_chain() is True

    def test_tc_s_18_tampered_entry_breaks_chain(self, tmp_audit_path):
        """TC-S.18: Manually modified entry causes validate_chain to raise."""
        audit = AuditLogger(tmp_audit_path)
        for i in range(3):
            audit.log_decision(
                alert_id=f"alert_{i:03d}",
                stage="stage1",
                verdict="auto_fp",
                model_version="1.0",
            )
        with open(tmp_audit_path) as f:
            lines = f.readlines()
        # Tamper: change verdict in entry 2 (index 1)
        tampered = json.loads(lines[1])
        tampered["verdict"] = "auto_tp"
        lines[1] = json.dumps(tampered) + "\n"
        with open(tmp_audit_path, "w") as f:
            f.writelines(lines)

        fresh_audit = AuditLogger.__new__(AuditLogger)
        fresh_audit._path = tmp_audit_path
        fresh_audit._last_hash = ""
        with pytest.raises(ValueError, match="break"):
            fresh_audit.validate_chain()

    def test_log_feedback_extends_chain(self, tmp_audit_path):
        """log_feedback entries participate in the hash chain."""
        audit = AuditLogger(tmp_audit_path)
        audit.log_decision(
            alert_id="alert_001", stage="stage1", verdict="auto_fp",
            model_version="1.0",
        )
        audit.log_feedback(
            alert_id="alert_001",
            analyst_id="analyst_01",
            override_verdict="true_positive",
            original_verdict="auto_fp",
            rationale="Analyst confirmed attack.",
        )
        assert audit.validate_chain() is True

    def test_empty_log_validates(self, tmp_audit_path):
        """validate_chain returns True on an empty log file."""
        audit = AuditLogger(tmp_audit_path)
        assert audit.validate_chain() is True


# =============================================================================
# S4: Model Artifact Integrity
# =============================================================================

class TestModelIntegrity:

    def test_tc_s_19_hash_saved_at_save_time(self, mock_lgb_model, tmp_model_path):
        """TC-S.19: checksums.json is created with a valid SHA-256 after save_model."""
        from src.models.classifier import save_model
        save_model(mock_lgb_model, tmp_model_path)
        cs_path = tmp_model_path.parent / "checksums.json"
        assert cs_path.exists()
        checksums = json.loads(cs_path.read_text())
        assert tmp_model_path.name in checksums
        digest = checksums[tmp_model_path.name]
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_tc_s_20_correct_model_loads_without_error(self, mock_lgb_model, tmp_model_path):
        """TC-S.20: Unmodified model loads without raising ModelIntegrityError."""
        from src.models.classifier import load_model, save_model
        save_model(mock_lgb_model, tmp_model_path)
        loaded = load_model(tmp_model_path)
        assert isinstance(loaded, lgb.Booster)

    def test_tc_s_21_tampered_model_raises_integrity_error(self, mock_lgb_model, tmp_model_path):
        """TC-S.21: Appending a byte to the model file raises ModelIntegrityError."""
        from src.models.classifier import load_model, save_model
        save_model(mock_lgb_model, tmp_model_path)
        with open(tmp_model_path, "ab") as f:
            f.write(b"\xff")
        with pytest.raises(ModelIntegrityError) as exc_info:
            load_model(tmp_model_path)
        assert exc_info.value.expected != exc_info.value.actual

    def test_load_raises_when_no_checksums_file(self, mock_lgb_model, tmp_model_path):
        """load_model raises FileNotFoundError when checksums.json is absent."""
        from src.models.classifier import load_model, save_model
        import pickle
        # Write pickle without saving hash
        with open(tmp_model_path, "wb") as f:
            pickle.dump(mock_lgb_model, f)
        with pytest.raises(FileNotFoundError):
            load_model(tmp_model_path)


# =============================================================================
# S2: Secrets Management
# =============================================================================

class TestSecretsManagement:

    def test_tc_s_22_api_key_redacted_in_logs(self):
        """TC-S.22: RedactionFilter replaces sk-ant-... with [REDACTED] in log records."""
        key = "sk-ant-api03-test-XXXX1234"
        msg = f"Connecting with key {key} to endpoint"
        result = redact_secrets(msg)
        assert "[REDACTED]" in result
        assert "sk-ant-" not in result

    def test_tc_s_22b_redaction_filter_on_log_record(self):
        """TC-S.22 (variant): RedactionFilter applied to a logging record."""
        key = "sk-ant-api03-test-YYYY5678"
        logger = logging.getLogger("test_redaction")
        handler = logging.handlers.MemoryHandler(capacity=100)
        flt = RedactionFilter()
        handler.addFilter(flt)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0,
            msg=f"key={key}", args=(), exc_info=None,
        )
        flt.filter(record)
        assert "[REDACTED]" in record.msg
        assert "sk-ant-" not in record.msg

    def test_tc_s_23_load_api_key_fails_on_missing(self):
        """TC-S.23: load_api_key raises when ANTHROPIC_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            if "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]
            # Temporarily unset the key
            env_copy = os.environ.copy()
            env_copy.pop("ANTHROPIC_API_KEY", None)
            with patch.dict(os.environ, env_copy, clear=True):
                with pytest.raises((ValueError, KeyError)):
                    load_api_key()

    def test_tc_s_24_load_api_key_fails_on_malformed(self):
        """TC-S.24: load_api_key raises on a key that lacks the sk-ant- prefix."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "invalid-key-format"}):
            with pytest.raises(ValueError, match="sk-ant-"):
                load_api_key()

    def test_redact_preserves_normal_text(self):
        """Normal messages without API keys pass through redact_secrets unchanged."""
        msg = "Processing alert_001 at 2026-05-25T12:00:00Z"
        assert redact_secrets(msg) == msg

    def test_redact_multiple_keys(self):
        """redact_secrets handles multiple key patterns in one string."""
        msg = "key1=sk-ant-abc123 key2=sk-ant-def456"
        result = redact_secrets(msg)
        assert "sk-ant-abc123" not in result
        assert "sk-ant-def456" not in result
        assert result.count("[REDACTED]") == 2


# Need logging.handlers for MemoryHandler
import logging.handlers
