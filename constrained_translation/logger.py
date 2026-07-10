"""constrained_translation.logger — Durable JSONL event logger (Task 10).

Overview
--------
JSONLLogger appends one UTF-8 JSON object per line to a log file.  Every
event is flushed immediately after writing so that crashes do not lose
records.

Design goals
────────────
* **Narrow API** — a single ``log(event, **fields)`` method, suitable for
  direct use by BatchRunner.
* **Crash-safe** — ``flush()`` is called after every write; the OS buffer
  is also sync'd via ``os.fsync`` on platforms that support it.
* **Unicode-clean** — ``ensure_ascii=False`` so accented characters, CJK,
  Cyrillic, etc. appear as real UTF-8 bytes rather than ``\\uXXXX`` escapes.
* **Append-only** — opening the logger a second time on the same file
  appends rather than truncating, preserving previous runs.
* **Loud on failure** — if the path cannot be opened for writing the
  constructor raises immediately (no silent swallowing).

Supported event names (non-exhaustive)
───────────────────────────────────────
``coverage_failure``, ``coverage_retry``, ``hard_failure``,
``token_audit``, ``provenance``, ``batch_summary``

Required fields present in every line
──────────────────────────────────────
* ``ts``      — ISO-8601 UTC timestamp (e.g. ``"2025-01-01T12:00:00+00:00"``)
* ``event``   — the event name string
* ``item_id`` — caller-supplied identifier (verse ID, batch ID, etc.)

Any additional keyword arguments passed to ``log()`` are merged into the
JSON object.

Usage
-----
.. code-block:: python

    from constrained_translation.logger import JSONLLogger

    logger = JSONLLogger("/tmp/run.jsonl")
    logger.log("coverage_failure", item_id="GEN_1_1", uncovered_count=2)
    logger.log("batch_summary",    item_id="batch_0", total=100, failed=3)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union


class JSONLLogger:
    """Append-only, flush-on-write JSONL event logger.

    Parameters
    ----------
    log_path:
        Destination file path (``str`` or :class:`~pathlib.Path`).
        The file is created (or opened for appending) immediately.  Parent
        directories must already exist.  An :exc:`OSError` /
        :exc:`PermissionError` is raised if the path is not writable.
    """

    def __init__(self, log_path: Union[str, Path]) -> None:
        self._path = Path(log_path)
        # Open in append mode; 'a' creates the file if it does not exist.
        # buffering=1 enables line-buffering, though we also call flush() explicitly.
        # errors='strict' ensures encoding problems surface immediately.
        self._fh = open(  # noqa: WPS515 (open without context manager is intentional)
            self._path,
            mode="a",
            encoding="utf-8",
            errors="strict",
            buffering=1,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, event: str, **fields: Any) -> None:
        """Append one JSON event line and flush to disk immediately.

        Parameters
        ----------
        event:
            Event name (e.g. ``"coverage_failure"``).
        **fields:
            Arbitrary keyword arguments merged into the JSON object.
            ``item_id`` is a required convention (callers must supply it).

        Notes
        -----
        Field order in the serialised object is:
        ``ts`` → ``event`` → ``item_id`` (if present) → remaining fields
        (insertion order, Python 3.7+).
        """
        record: dict[str, Any] = {
            "ts": _utc_now_iso(),
            "event": event,
        }
        # Place item_id right after event if provided, then the rest.
        if "item_id" in fields:
            record["item_id"] = fields.pop("item_id")
        record.update(fields)

        line = json.dumps(record, ensure_ascii=False)
        self._fh.write(line + "\n")
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())
        except OSError:
            # fsync is best-effort; some virtual/pipe fds don't support it.
            pass

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "JSONLLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with timezone offset."""
    return datetime.now(timezone.utc).isoformat()
