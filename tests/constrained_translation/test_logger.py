"""Tests for constrained_translation.logger — Task 10 (JSONLLogger).

Strict TDD: all tests are written *before* the implementation.
Run them first to confirm RED, then implement logger.py to go GREEN.

Invariants exercised
────────────────────
1. Constructor creates the log file (even if empty).
2. Each log() call appends exactly one valid JSON object per line.
3. Timestamps are ISO-8601 UTC strings.
4. Required fields (ts, event, item_id) are always present.
5. Multiple events produce multiple lines (one per event).
6. All documented event names are accepted without error.
7. Extra keyword fields are preserved in the JSON line.
8. Unicode is serialized without ASCII escaping (ensure_ascii=False).
9. Each event is flushed immediately so crash-safe.
10. Attempting to open an unwritable path raises an error loudly.
"""
from __future__ import annotations

import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ── Import under test ───────────────────────────────────────────────────────
# These will fail (ImportError) until logger.py is created.
from constrained_translation.logger import JSONLLogger


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _read_lines(path: Path) -> list[dict]:
    """Return parsed JSON objects for every non-empty line in *path*."""
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            lines.append(json.loads(raw))
    return lines


# ════════════════════════════════════════════════════════════════════════════
# Test suite
# ════════════════════════════════════════════════════════════════════════════


class TestLoggerCreatesFile:
    """test_logger_creates_file — constructor creates the log file."""

    def test_logger_creates_file(self, tmp_path: Path):
        """Instantiating JSONLLogger must create the file immediately."""
        log_path = tmp_path / "run.jsonl"
        assert not log_path.exists(), "Pre-condition: file must not exist yet"

        _ = JSONLLogger(log_path)

        assert log_path.exists(), "JSONLLogger must create the log file on construction"

    def test_logger_creates_parent_path(self, tmp_path: Path):
        """Constructor accepts a Path object as well as a str."""
        log_path = str(tmp_path / "events.jsonl")
        logger = JSONLLogger(log_path)
        assert Path(log_path).exists()


class TestLoggerWritesValidJsonPerLine:
    """test_logger_writes_valid_json_per_line — each log() writes one JSON object."""

    def test_logger_writes_valid_json_per_line(self, tmp_path: Path):
        """log() must append exactly one valid JSON line per call."""
        log_path = tmp_path / "run.jsonl"
        logger = JSONLLogger(log_path)

        logger.log("coverage_failure", item_id="GEN_1_1")

        lines = _read_lines(log_path)
        assert len(lines) == 1
        assert isinstance(lines[0], dict)

    def test_each_line_is_complete_json_object(self, tmp_path: Path):
        """Every line in the file must be a self-contained JSON object."""
        log_path = tmp_path / "run.jsonl"
        logger = JSONLLogger(log_path)

        logger.log("hard_failure", item_id="GEN_1_2")
        logger.log("coverage_retry", item_id="GEN_1_3")

        raw = log_path.read_text(encoding="utf-8").splitlines()
        raw = [l for l in raw if l.strip()]
        assert len(raw) == 2
        for line in raw:
            obj = json.loads(line)  # must not raise
            assert isinstance(obj, dict)


class TestLoggerTsIsIso8601Utc:
    """test_logger_ts_is_iso8601_utc — timestamp must be ISO-8601 UTC."""

    def test_logger_ts_is_iso8601_utc(self, tmp_path: Path):
        """'ts' field must be parseable as an ISO-8601 UTC datetime."""
        log_path = tmp_path / "run.jsonl"
        logger = JSONLLogger(log_path)

        logger.log("token_audit", item_id="GEN_1_1")

        lines = _read_lines(log_path)
        ts_str = lines[0]["ts"]

        # Must parse without error.
        dt = datetime.fromisoformat(ts_str)

        # Must be UTC (tzinfo == UTC or offset == 0).
        assert dt.tzinfo is not None, "'ts' must include timezone info"
        utc_offset = dt.utcoffset()
        from datetime import timedelta
        assert utc_offset == timedelta(0), f"Expected UTC offset 0, got {utc_offset}"

    def test_ts_is_recent(self, tmp_path: Path):
        """Timestamp must be close to the current UTC time (within 5 seconds)."""
        log_path = tmp_path / "run.jsonl"
        logger = JSONLLogger(log_path)
        before = datetime.now(timezone.utc)

        logger.log("provenance", item_id="GEN_1_1")

        after = datetime.now(timezone.utc)
        lines = _read_lines(log_path)
        ts = datetime.fromisoformat(lines[0]["ts"])

        assert before <= ts <= after, (
            f"Timestamp {ts} is not between {before} and {after}"
        )


class TestLoggerAllRequiredFieldsPresent:
    """test_logger_all_required_fields_present — required fields always appear."""

    def test_logger_all_required_fields_present(self, tmp_path: Path):
        """Every JSON line must contain 'ts', 'event', and 'item_id'."""
        log_path = tmp_path / "run.jsonl"
        logger = JSONLLogger(log_path)

        logger.log("coverage_failure", item_id="GEN_1_1")

        lines = _read_lines(log_path)
        obj = lines[0]
        assert "ts" in obj, "Required field 'ts' missing"
        assert "event" in obj, "Required field 'event' missing"
        assert "item_id" in obj, "Required field 'item_id' missing"

    def test_event_field_matches_argument(self, tmp_path: Path):
        """The 'event' field must equal the event string passed to log()."""
        log_path = tmp_path / "run.jsonl"
        logger = JSONLLogger(log_path)

        logger.log("batch_summary", item_id="batch_0")

        lines = _read_lines(log_path)
        assert lines[0]["event"] == "batch_summary"

    def test_item_id_field_matches_argument(self, tmp_path: Path):
        """The 'item_id' field must equal the value passed as item_id kwarg."""
        log_path = tmp_path / "run.jsonl"
        logger = JSONLLogger(log_path)

        logger.log("hard_failure", item_id="REV_22_21")

        lines = _read_lines(log_path)
        assert lines[0]["item_id"] == "REV_22_21"

    def test_extra_kwargs_are_preserved(self, tmp_path: Path):
        """Keyword arguments beyond item_id are written into the JSON object."""
        log_path = tmp_path / "run.jsonl"
        logger = JSONLLogger(log_path)

        logger.log(
            "coverage_retry",
            item_id="GEN_1_5",
            attempt=2,
            uncovered_count=3,
        )

        lines = _read_lines(log_path)
        obj = lines[0]
        assert obj["attempt"] == 2
        assert obj["uncovered_count"] == 3


class TestLoggerMultipleEventsMultipleLines:
    """test_logger_multiple_events_multiple_lines — N calls → N lines."""

    def test_logger_multiple_events_multiple_lines(self, tmp_path: Path):
        """Each log() call appends exactly one line; N calls → N lines."""
        log_path = tmp_path / "run.jsonl"
        logger = JSONLLogger(log_path)

        events = [
            ("coverage_failure", "GEN_1_1"),
            ("coverage_retry",   "GEN_1_1"),
            ("hard_failure",     "GEN_1_1"),
            ("token_audit",      "GEN_1_2"),
            ("provenance",       "GEN_1_2"),
            ("batch_summary",    "batch_0"),
        ]
        for event, item_id in events:
            logger.log(event, item_id=item_id)

        lines = _read_lines(log_path)
        assert len(lines) == len(events), (
            f"Expected {len(events)} lines, got {len(lines)}"
        )

    def test_lines_preserve_event_order(self, tmp_path: Path):
        """Lines must appear in the order log() was called."""
        log_path = tmp_path / "run.jsonl"
        logger = JSONLLogger(log_path)

        ordered = ["coverage_failure", "coverage_retry", "hard_failure"]
        for ev in ordered:
            logger.log(ev, item_id="GEN_1_1")

        lines = _read_lines(log_path)
        for i, ev in enumerate(ordered):
            assert lines[i]["event"] == ev, (
                f"Line {i}: expected event={ev!r}, got {lines[i]['event']!r}"
            )


class TestLoggerAllDocumentedEventNames:
    """All event names from the task spec are accepted without error."""

    @pytest.mark.parametrize("event", [
        "coverage_failure",
        "coverage_retry",
        "hard_failure",
        "token_audit",
        "provenance",
        "batch_summary",
    ])
    def test_documented_event_accepted(self, tmp_path: Path, event: str):
        """log() must not raise for any documented event name."""
        log_path = tmp_path / f"{event}.jsonl"
        logger = JSONLLogger(log_path)

        logger.log(event, item_id="GEN_1_1")  # must not raise

        lines = _read_lines(log_path)
        assert lines[0]["event"] == event


class TestLoggerUnicodeNoAsciiEscape:
    """Unicode characters must be written literally, not \\uXXXX-escaped."""

    def test_unicode_not_ascii_escaped(self, tmp_path: Path):
        """Non-ASCII characters must appear as UTF-8 bytes, not \\uXXXX."""
        log_path = tmp_path / "run.jsonl"
        logger = JSONLLogger(log_path)

        unicode_text = "Au commencement Dieu créa — 神 начало"
        logger.log("provenance", item_id="GEN_1_1", translation=unicode_text)

        raw_bytes = log_path.read_bytes()
        raw_str = raw_bytes.decode("utf-8")

        # The file should contain the actual Unicode characters, not escapes.
        assert "créa" in raw_str, "French accented chars should not be ASCII-escaped"
        assert "神" in raw_str, "CJK character should not be ASCII-escaped"
        assert "начало" in raw_str, "Cyrillic should not be ASCII-escaped"
        assert "\\u00e9" not in raw_str.lower(), "Should not escape é as \\u00e9"


class TestLoggerFlushOnWrite:
    """Each event must be flushed immediately (crash-safe)."""

    def test_flush_after_each_write(self, tmp_path: Path):
        """Data must be visible on disk without closing the logger."""
        log_path = tmp_path / "run.jsonl"
        logger = JSONLLogger(log_path)

        logger.log("coverage_failure", item_id="GEN_1_1")

        # Read the file while the logger is still open (not closed/deleted).
        # If flush() is called, this will succeed.
        lines = _read_lines(log_path)
        assert len(lines) == 1, (
            "Event must be flushed to disk without closing the logger"
        )

    def test_second_logger_sees_first_logger_data(self, tmp_path: Path):
        """A second logger opened on same path appends (does not truncate)."""
        log_path = tmp_path / "run.jsonl"

        logger1 = JSONLLogger(log_path)
        logger1.log("coverage_failure", item_id="GEN_1_1")
        del logger1  # close first logger

        logger2 = JSONLLogger(log_path)
        logger2.log("hard_failure", item_id="GEN_1_2")

        lines = _read_lines(log_path)
        assert len(lines) == 2, "Second logger must append, not truncate"
        assert lines[0]["event"] == "coverage_failure"
        assert lines[1]["event"] == "hard_failure"


class TestLoggerUnwritablePath:
    """Opening an unwritable path must raise loudly."""

    def test_unwritable_path_raises_on_construction(self, tmp_path: Path):
        """JSONLLogger must raise an exception for a path that cannot be written."""
        # Create a read-only directory.
        ro_dir = tmp_path / "readonly"
        ro_dir.mkdir()
        ro_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)  # r-x --- ---

        log_path = ro_dir / "run.jsonl"

        try:
            with pytest.raises((OSError, PermissionError, IOError)):
                JSONLLogger(log_path)
        finally:
            # Restore permissions so tmp_path cleanup works.
            ro_dir.chmod(stat.S_IRWXU)
