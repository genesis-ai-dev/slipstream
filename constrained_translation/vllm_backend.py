"""constrained_translation.vllm_backend — VLLMBackend for vLLM 0.19+.

Uses the vLLM 0.19 structured-output HTTP API:

  POST {base_url}/v1/completions
    {model, prompt, max_tokens, temperature,
     structured_outputs: {grammar: <str>}, return_token_ids: true}

  POST {base_url}/tokenize
    {model, prompt: <text>, add_special_tokens: false, return_token_strs: false}
    → {tokens: [int, ...]}

  POST {base_url}/detokenize
    {model, tokens: [token_id]}
    → {prompt: <str>}

  GET {base_url}/health
    → 200 when ready

Design decisions
----------------
- Uses ``requests`` directly; no OpenAI SDK dependency.
- Explicit connect/read timeout; no silent retry.
- Base URL trailing slash is stripped at construction.
- ``generate()`` always sends a non-empty grammar — no unconstrained fallback.
- Missing ``token_ids`` in the response is a hard ``BackendError``; provenance
  audit cannot be guaranteed without them (invariant I2).
- All HTTP-level and connection-level errors are surfaced as ``BackendError``,
  never as bare ``requests.exceptions.*`` exceptions.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple, Union

import requests

from constrained_translation.protocol import GenerationResult


# ---------------------------------------------------------------------------
# BackendError
# ---------------------------------------------------------------------------

class BackendError(Exception):
    """Raised by VLLMBackend for any HTTP or protocol error.

    Parameters
    ----------
    operation:
        Name of the backend operation that failed:
        ``"generate"``, ``"tokenize"``, ``"decode_token"``, or
        ``"is_available"``.
    status:
        HTTP status code, or ``None`` for connection-level errors (timeout,
        refused, etc.).
    body:
        Raw response body text or error message string.
    """

    def __init__(self, operation: str, status: Optional[int], body: str) -> None:
        self.operation = operation
        self.status = status
        self.body = body
        super().__init__(
            f"BackendError(operation={operation!r}, status={status!r}, body={body!r})"
        )


# ---------------------------------------------------------------------------
# VLLMBackend
# ---------------------------------------------------------------------------

class VLLMBackend:
    """HTTP backend for a locally-served vLLM 0.19 instance.

    Parameters
    ----------
    base_url:
        Base URL of the vLLM server, e.g. ``http://localhost:8000``.
        Trailing slashes are stripped automatically.
    model:
        Model name as registered in vLLM, e.g.
        ``"Qwen/Qwen2.5-7B-Instruct"``.
    timeout:
        ``(connect_timeout, read_timeout)`` in seconds forwarded to every
        ``requests`` call.  Defaults to ``(10, 120)``.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: Tuple[float, float] = (10, 120),
    ) -> None:
        self.base_url: str = base_url.rstrip("/")
        self.model: str = model
        self.timeout: Tuple[float, float] = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict, operation: str) -> dict:
        """POST *payload* to *path* and return the parsed JSON body.

        Raises
        ------
        BackendError
            On any HTTP error, connection error, timeout, or JSON parse
            failure.
        """
        url = f"{self.base_url}{path}"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.Timeout as exc:
            raise BackendError(operation, None, str(exc)) from exc
        except requests.ConnectionError as exc:
            raise BackendError(operation, None, str(exc)) from exc
        except requests.RequestException as exc:
            raise BackendError(operation, None, str(exc)) from exc

        if resp.status_code != 200:
            raise BackendError(operation, resp.status_code, resp.text)

        try:
            return resp.json()
        except (ValueError, Exception) as exc:
            raise BackendError(operation, resp.status_code, resp.text) from exc

    def _get(self, path: str, operation: str) -> requests.Response:
        """GET *path* and return the response.

        Returns the raw ``requests.Response`` so callers can inspect the
        status code.  Connection errors bubble up as ``BackendError``.
        """
        url = f"{self.base_url}{path}"
        try:
            return requests.get(url, timeout=self.timeout)
        except requests.Timeout as exc:
            raise BackendError(operation, None, str(exc)) from exc
        except requests.ConnectionError as exc:
            raise BackendError(operation, None, str(exc)) from exc
        except requests.RequestException as exc:
            raise BackendError(operation, None, str(exc)) from exc

    # ------------------------------------------------------------------
    # BackendProtocol interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        grammar: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Run constrained generation via the vLLM 0.19 completions API.

        The grammar is sent via ``structured_outputs.grammar`` and
        ``return_token_ids=true`` is always set so that the post-decode
        token-ID audit can be performed (invariant I2).

        Parameters
        ----------
        prompt:
            The assembled few-shot prompt string.
        grammar:
            An XGrammar/BNF grammar string.  Must be non-empty — no
            unconstrained generation path exists in this backend.
        max_tokens:
            Maximum number of tokens to generate.
        temperature:
            Sampling temperature (0.0 = greedy).

        Returns
        -------
        GenerationResult
            Parsed result including text, token_ids, usage, and latency.

        Raises
        ------
        BackendError
            On HTTP error, malformed response, or missing token_ids.
        """
        t_start = time.monotonic()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "structured_outputs": {"grammar": grammar},
            "return_token_ids": True,
        }

        data = self._post("/v1/completions", payload, "generate")

        # Parse choices
        choices = data.get("choices")
        if not choices:
            raise BackendError(
                "generate",
                None,
                f"Response has no choices: {data!r}",
            )

        choice = choices[0]
        text = choice.get("text", "")

        # token_ids are mandatory — provenance audit cannot be guaranteed otherwise
        token_ids = choice.get("token_ids")
        if token_ids is None:
            raise BackendError(
                "generate",
                None,
                "Response is missing token_ids; provenance audit cannot be guaranteed",
            )

        # Parse usage
        usage = data.get("usage", {})
        prompt_tokens: int = usage.get("prompt_tokens", 0)
        output_tokens: int = usage.get("completion_tokens", 0)

        generation_ms = (time.monotonic() - t_start) * 1000.0

        return GenerationResult(
            text=text,
            token_ids=list(token_ids),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            generation_ms=generation_ms,
        )

    def tokenize(self, text: str) -> list[int]:
        """Tokenize *text* using the vLLM /tokenize endpoint.

        Parameters
        ----------
        text:
            Raw text to tokenise.

        Returns
        -------
        list[int]
            List of integer token IDs.

        Raises
        ------
        BackendError
            On HTTP error, malformed response, or missing ``tokens`` key.
        """
        payload = {
            "model": self.model,
            "prompt": text,
            "add_special_tokens": False,
            "return_token_strs": False,
        }

        data = self._post("/tokenize", payload, "tokenize")

        if "tokens" not in data:
            raise BackendError(
                "tokenize",
                None,
                f"Response missing 'tokens' key: {data!r}",
            )

        return list(data["tokens"])

    def decode_token(self, token_id: int) -> str:
        """Decode a single *token_id* to its surface string.

        Parameters
        ----------
        token_id:
            Integer token ID.

        Returns
        -------
        str
            Decoded surface string for this token.

        Raises
        ------
        BackendError
            On HTTP error, malformed response, or missing ``prompt`` key.
        """
        payload = {
            "model": self.model,
            "tokens": [token_id],
        }

        data = self._post("/detokenize", payload, "decode_token")

        if "prompt" not in data:
            raise BackendError(
                "decode_token",
                None,
                f"Response missing 'prompt' key: {data!r}",
            )

        return data["prompt"]

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Decode a full sequence of *token_ids* to a surface string.

        Unlike ``decode_token`` (single-ID), this method sends the **entire**
        token list to the vLLM ``/detokenize`` endpoint in one request.  This
        is the correct approach for BPE tokenisers (e.g. Qwen3.5) and
        byte-level tokenisers where decoding tokens individually produces
        garbled output (e.g. U+FFFD byte fragments for multi-byte Unicode
        characters).

        Parameters
        ----------
        token_ids:
            List of integer token IDs to detokenise.

        Returns
        -------
        str
            The full detokenised string (as returned by the tokeniser's
            decode method).  Empty list returns empty string without any
            HTTP request.

        Raises
        ------
        BackendError
            On HTTP error, malformed response, or missing ``prompt`` key.
        """
        if not token_ids:
            return ""

        payload = {
            "model": self.model,
            "tokens": token_ids,
        }

        data = self._post("/detokenize", payload, "decode_tokens")

        if "prompt" not in data:
            raise BackendError(
                "decode_tokens",
                None,
                f"Response missing 'prompt' key: {data!r}",
            )

        return data["prompt"]

    def is_available(self) -> bool:
        """Return True if the vLLM server is healthy and ready.

        Performs a GET /health request.  Returns False on any error
        (HTTP error, connection error, timeout) rather than raising.
        """
        try:
            resp = self._get("/health", "is_available")
        except BackendError:
            return False

        return resp.status_code == 200
