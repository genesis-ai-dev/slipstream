"""constrained_translation.fake_backend — FakeBackend for offline tests.

Satisfies BackendProtocol with no external dependencies, no GPU, and no
model downloads (invariant I6).

Design:
- Constructor accepts ``responses: dict[str, str]`` mapping the first 60
  characters of a prompt to a canned response text.
- ``tokenize()`` splits on whitespace and maps each surface token to a
  stable integer via a deterministic hash (built-in dict, populated on
  demand).
- ``decode_token()`` reverses that map.
- ``generate()`` looks up the canned response by prefix, falls back to a
  deterministic default derived from the prompt, then returns a
  GenerationResult whose ``token_ids`` are derived from the response text.
"""

from __future__ import annotations

import time
from typing import Optional

from constrained_translation.protocol import GenerationResult


class FakeBackend:
    """Deterministic in-memory backend for unit and integration tests.

    Parameters
    ----------
    responses:
        Mapping of prompt-prefix (first 60 chars) → canned response text.
        If a prompt's first 60 characters match a key, that response is used.
        Otherwise a deterministic default is returned.
    """

    _DEFAULT_RESPONSE = "fake translation output"

    def __init__(self, responses: Optional[dict[str, str]] = None) -> None:
        self._responses: dict[str, str] = responses or {}
        # Bidirectional vocabulary: surface_token → int, int → surface_token
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_assign_id(self, token: str) -> int:
        """Return a stable integer ID for *token*, assigning one if needed.

        IDs are assigned via Python's built-in ``hash()`` seeded by a
        counter to guarantee no collisions in the (small) fake vocabulary.
        We use a simple incrementing counter for determinism across runs
        within a single process; ties are broken by the counter so that
        two different tokens never share an ID.
        """
        if token in self._token_to_id:
            return self._token_to_id[token]

        # Assign the next sequential ID (0-indexed, grows monotonically)
        new_id = len(self._token_to_id)
        self._token_to_id[token] = new_id
        self._id_to_token[new_id] = token
        return new_id

    def _lookup_response(self, prompt: str) -> str:
        """Return the canned response for *prompt* or the default."""
        prefix = prompt[:60]
        if prefix in self._responses:
            return self._responses[prefix]
        # Deterministic default: always the same fixed string so that
        # repeated calls with the same prompt return the same result.
        return self._DEFAULT_RESPONSE

    # ------------------------------------------------------------------
    # BackendProtocol interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Always returns True — the fake backend is always ready."""
        return True

    def tokenize(self, text: str) -> list[int]:
        """Split *text* on whitespace and map each token to a stable int ID.

        Empty string returns an empty list.
        """
        if not text:
            return []
        tokens = text.split()
        return [self._get_or_assign_id(tok) for tok in tokens]

    def decode_token(self, token_id: int) -> str:
        """Return the surface string for *token_id*.

        Raises ValueError (not bare KeyError) if *token_id* has never been
        assigned by this instance (i.e. tokenize() was never called for that
        token).  The error message includes the offending token ID.
        """
        try:
            return self._id_to_token[token_id]
        except KeyError:
            raise ValueError(
                f"FakeBackend.decode_token: unknown token_id={token_id!r} "
                f"(has never been assigned by this FakeBackend instance)"
            ) from None

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Return the full surface string for a sequence of *token_ids*.

        For the FakeBackend's whitespace-split tokeniser, the correct
        full-sequence detokenisation is simply joining the individual
        decoded tokens with a space — this mirrors what a whitespace
        tokeniser would do and avoids the BPE partial-word problem at the
        FakeBackend level.

        An empty sequence returns an empty string.

        Raises ValueError if any token_id has never been assigned.
        """
        if not token_ids:
            return ""
        return " ".join(self.decode_token(tid) for tid in token_ids)

    def generate(
        self,
        prompt: str,
        grammar: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Return a deterministic GenerationResult for *prompt*.

        - Looks up the canned response by the first 60 chars of *prompt*.
        - Derives ``token_ids`` by tokenizing the response text.
        - ``prompt_tokens`` = ``len(tokenize(prompt))`` (whitespace-split).
        - ``output_tokens`` = ``len(token_ids)``.
        - ``generation_ms`` is a small fixed value (no real latency).
        """
        t_start = time.monotonic()

        response_text = self._lookup_response(prompt)

        # Ensure the response tokens are in the vocabulary
        token_ids = self.tokenize(response_text)

        # Prompt token count (whitespace approximation)
        prompt_token_ids = self.tokenize(prompt)
        prompt_tokens = len(prompt_token_ids)

        generation_ms = (time.monotonic() - t_start) * 1000.0

        return GenerationResult(
            text=response_text,
            token_ids=token_ids,
            prompt_tokens=prompt_tokens,
            output_tokens=len(token_ids),
            generation_ms=generation_ms,
        )
