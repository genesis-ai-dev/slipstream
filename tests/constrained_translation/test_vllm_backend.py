"""Tests for Task 14: VLLMBackend — mocked HTTP tests.

Strict TDD: these tests are written BEFORE the implementation.
Run `pytest tests/constrained_translation/test_vllm_backend.py -x -v` to
observe RED, then implement constrained_translation/vllm_backend.py.

Coverage:
  §1  BackendError — carries operation / status / body
  §2  VLLMBackend construction — base URL normalisation, model storage
  §3  generate() — exact payload inspection, success path, token_ids required
  §4  generate() — HTTP error handling (4xx, 5xx)
  §5  generate() — malformed / truncated responses
  §6  generate() — timeout propagation
  §7  generate() — grammar required (never called without grammar)
  §8  tokenize() — exact payload, success, HTTP errors, malformed
  §9  decode_token() — exact payload, success, HTTP errors, malformed
  §10 is_available() — health GET, up/down, HTTP error, connection error
  §11 Unicode round-trip
  §12 BackendProtocol conformance
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests


# ---------------------------------------------------------------------------
# Module-level autouse fixture — clear decode_token cache before every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_decode_token_cache():
    """Flush the class-level decode_token cache before and after every test.

    This prevents cross-test cache pollution without altering any test logic.
    The cache is a private implementation detail; tests should observe only the
    externally-visible behaviours (HTTP call counts, return values, errors).
    """
    from constrained_translation.vllm_backend import VLLMBackend
    VLLMBackend._decode_token_cache.clear()
    yield
    VLLMBackend._decode_token_cache.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status_code: int, json_body: object) -> MagicMock:
    """Create a mock requests.Response with .status_code and .json()."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    resp.text = json.dumps(json_body)
    return resp


def _make_backend(base_url="http://localhost:8000", model="test-model"):
    from constrained_translation.vllm_backend import VLLMBackend
    return VLLMBackend(base_url=base_url, model=model)


# ---------------------------------------------------------------------------
# §1  BackendError
# ---------------------------------------------------------------------------

class TestBackendError:
    def test_importable(self):
        from constrained_translation.vllm_backend import BackendError
        assert BackendError is not None

    def test_is_exception(self):
        from constrained_translation.vllm_backend import BackendError
        err = BackendError("generate", 500, "internal error")
        assert isinstance(err, Exception)

    def test_carries_operation(self):
        from constrained_translation.vllm_backend import BackendError
        err = BackendError("tokenize", 400, "bad request")
        assert err.operation == "tokenize"

    def test_carries_status(self):
        from constrained_translation.vllm_backend import BackendError
        err = BackendError("generate", 503, "unavailable")
        assert err.status == 503

    def test_carries_body(self):
        from constrained_translation.vllm_backend import BackendError
        err = BackendError("decode_token", 422, "unprocessable")
        assert err.body == "unprocessable"

    def test_str_contains_operation(self):
        from constrained_translation.vllm_backend import BackendError
        err = BackendError("generate", 500, "oops")
        assert "generate" in str(err)

    def test_status_none_allowed(self):
        """Connection-level errors have status=None."""
        from constrained_translation.vllm_backend import BackendError
        err = BackendError("tokenize", None, "connection refused")
        assert err.status is None

    def test_subclass_of_exception(self):
        from constrained_translation.vllm_backend import BackendError
        assert issubclass(BackendError, Exception)


# ---------------------------------------------------------------------------
# §2  VLLMBackend construction
# ---------------------------------------------------------------------------

class TestVLLMBackendConstruction:
    def test_importable(self):
        from constrained_translation.vllm_backend import VLLMBackend
        assert VLLMBackend is not None

    def test_trailing_slash_stripped(self):
        from constrained_translation.vllm_backend import VLLMBackend
        b = VLLMBackend(base_url="http://localhost:8000/", model="m")
        assert not b.base_url.endswith("/")

    def test_no_trailing_slash_preserved(self):
        from constrained_translation.vllm_backend import VLLMBackend
        b = VLLMBackend(base_url="http://localhost:8000", model="m")
        assert b.base_url == "http://localhost:8000"

    def test_model_stored(self):
        from constrained_translation.vllm_backend import VLLMBackend
        b = VLLMBackend(base_url="http://localhost:8000", model="Qwen/Qwen2.5-7B")
        assert b.model == "Qwen/Qwen2.5-7B"

    def test_default_timeouts_present(self):
        """VLLMBackend must store explicit connect + read timeouts."""
        from constrained_translation.vllm_backend import VLLMBackend
        b = VLLMBackend(base_url="http://localhost:8000", model="m")
        assert hasattr(b, "timeout")

    def test_custom_timeout_accepted(self):
        from constrained_translation.vllm_backend import VLLMBackend
        b = VLLMBackend(base_url="http://localhost:8000", model="m", timeout=(5, 120))
        assert b.timeout == (5, 120)


# ---------------------------------------------------------------------------
# §3  generate() — success path + exact payload inspection
# ---------------------------------------------------------------------------

class TestGenerate:
    """VLLMBackend.generate() must POST the exact vLLM 0.19 payload."""

    _SUCCESS = {
        "choices": [
            {
                "text": "Au commencement",
                "token_ids": [101, 202, 303],
            }
        ],
        "usage": {
            "prompt_tokens": 42,
            "completion_tokens": 3,
        },
    }

    def test_returns_generation_result(self):
        from constrained_translation.protocol import GenerationResult
        b = _make_backend()
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp):
            result = b.generate(
                prompt="In the beginning",
                grammar="root ::= \"Au commencement\"",
                max_tokens=32,
            )
        assert isinstance(result, GenerationResult)

    def test_posts_to_completions_endpoint(self):
        b = _make_backend()
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp) as mock_post:
            b.generate(
                prompt="In the beginning",
                grammar="root ::= \"Au\"",
                max_tokens=10,
            )
        url = mock_post.call_args[0][0]
        assert url.endswith("/v1/completions")

    def test_payload_contains_model(self):
        b = _make_backend(model="my-model")
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp) as mock_post:
            b.generate(prompt="hello", grammar="root ::= \"hi\"", max_tokens=5)
        body = mock_post.call_args[1]["json"]
        assert body["model"] == "my-model"

    def test_payload_contains_prompt(self):
        b = _make_backend()
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp) as mock_post:
            b.generate(prompt="my prompt text", grammar="root ::= \"x\"", max_tokens=5)
        body = mock_post.call_args[1]["json"]
        assert body["prompt"] == "my prompt text"

    def test_payload_contains_max_tokens(self):
        b = _make_backend()
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp) as mock_post:
            b.generate(prompt="p", grammar="root ::= \"x\"", max_tokens=128)
        body = mock_post.call_args[1]["json"]
        assert body["max_tokens"] == 128

    def test_payload_contains_temperature(self):
        b = _make_backend()
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp) as mock_post:
            b.generate(prompt="p", grammar="root ::= \"x\"", max_tokens=5, temperature=0.7)
        body = mock_post.call_args[1]["json"]
        assert body["temperature"] == pytest.approx(0.7)

    def test_payload_structured_outputs_grammar(self):
        """structured_outputs.grammar must carry the grammar string."""
        b = _make_backend()
        resp = _make_response(200, self._SUCCESS)
        grammar_str = 'root ::= "hello"'
        with patch("requests.post", return_value=resp) as mock_post:
            b.generate(prompt="p", grammar=grammar_str, max_tokens=5)
        body = mock_post.call_args[1]["json"]
        assert body["structured_outputs"]["grammar"] == grammar_str

    def test_payload_return_token_ids_true(self):
        """return_token_ids must be True for provenance audit."""
        b = _make_backend()
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp) as mock_post:
            b.generate(prompt="p", grammar="root ::= \"x\"", max_tokens=5)
        body = mock_post.call_args[1]["json"]
        assert body["return_token_ids"] is True

    def test_result_text(self):
        b = _make_backend()
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp):
            result = b.generate(prompt="p", grammar="root ::= \"Au\"", max_tokens=10)
        assert result.text == "Au commencement"

    def test_result_token_ids(self):
        b = _make_backend()
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp):
            result = b.generate(prompt="p", grammar="root ::= \"Au\"", max_tokens=10)
        assert result.token_ids == [101, 202, 303]

    def test_result_prompt_tokens_from_usage(self):
        b = _make_backend()
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp):
            result = b.generate(prompt="p", grammar="root ::= \"Au\"", max_tokens=10)
        assert result.prompt_tokens == 42

    def test_result_output_tokens_from_usage(self):
        b = _make_backend()
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp):
            result = b.generate(prompt="p", grammar="root ::= \"Au\"", max_tokens=10)
        assert result.output_tokens == 3

    def test_result_generation_ms_positive(self):
        b = _make_backend()
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp):
            result = b.generate(prompt="p", grammar="root ::= \"Au\"", max_tokens=10)
        assert result.generation_ms >= 0.0

    def test_timeout_passed_to_requests(self):
        """Explicit timeout must be forwarded to requests.post."""
        from constrained_translation.vllm_backend import VLLMBackend
        b = VLLMBackend(base_url="http://localhost:8000", model="m", timeout=(3, 60))
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp) as mock_post:
            b.generate(prompt="p", grammar="root ::= \"x\"", max_tokens=5)
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["timeout"] == (3, 60)


# ---------------------------------------------------------------------------
# §4  generate() — HTTP error handling
# ---------------------------------------------------------------------------

class TestGenerateHTTPErrors:
    def test_400_raises_backend_error(self):
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        resp = _make_response(400, {"error": "bad request"})
        with patch("requests.post", return_value=resp):
            with pytest.raises(BackendError) as exc_info:
                b.generate(prompt="p", grammar="root ::= \"x\"", max_tokens=5)
        assert exc_info.value.status == 400

    def test_500_raises_backend_error(self):
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        resp = _make_response(500, {"error": "internal server error"})
        with patch("requests.post", return_value=resp):
            with pytest.raises(BackendError) as exc_info:
                b.generate(prompt="p", grammar="root ::= \"x\"", max_tokens=5)
        assert exc_info.value.status == 500

    def test_422_raises_backend_error(self):
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        resp = _make_response(422, {"detail": "unprocessable"})
        with patch("requests.post", return_value=resp):
            with pytest.raises(BackendError) as exc_info:
                b.generate(prompt="p", grammar="root ::= \"x\"", max_tokens=5)
        assert exc_info.value.operation == "generate"

    def test_backend_error_not_plain_exception(self):
        """BackendError must be raised, not a bare Exception."""
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        resp = _make_response(503, {"error": "service unavailable"})
        with patch("requests.post", return_value=resp):
            with pytest.raises(BackendError):
                b.generate(prompt="p", grammar="root ::= \"x\"", max_tokens=5)


# ---------------------------------------------------------------------------
# §5  generate() — malformed / truncated responses
# ---------------------------------------------------------------------------

class TestGenerateMalformed:
    def test_missing_choices_raises_backend_error(self):
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        resp = _make_response(200, {"usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        with patch("requests.post", return_value=resp):
            with pytest.raises(BackendError) as exc_info:
                b.generate(prompt="p", grammar="root ::= \"x\"", max_tokens=5)
        assert exc_info.value.operation == "generate"

    def test_empty_choices_raises_backend_error(self):
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        resp = _make_response(200, {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        with patch("requests.post", return_value=resp):
            with pytest.raises(BackendError):
                b.generate(prompt="p", grammar="root ::= \"x\"", max_tokens=5)

    def test_missing_token_ids_raises_backend_error(self):
        """Missing token_ids is a BackendError — provenance audit cannot be guaranteed."""
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        resp = _make_response(200, {
            "choices": [{"text": "hello"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })
        with patch("requests.post", return_value=resp):
            with pytest.raises(BackendError) as exc_info:
                b.generate(prompt="p", grammar="root ::= \"hello\"", max_tokens=5)
        assert "token_ids" in str(exc_info.value).lower() or exc_info.value.operation == "generate"

    def test_null_token_ids_raises_backend_error(self):
        """token_ids=null is treated the same as missing — BackendError."""
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        resp = _make_response(200, {
            "choices": [{"text": "hello", "token_ids": None}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })
        with patch("requests.post", return_value=resp):
            with pytest.raises(BackendError):
                b.generate(prompt="p", grammar="root ::= \"hello\"", max_tokens=5)

    def test_non_json_response_raises_backend_error(self):
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("not JSON")
        resp.text = "not json at all"
        with patch("requests.post", return_value=resp):
            with pytest.raises(BackendError):
                b.generate(prompt="p", grammar="root ::= \"x\"", max_tokens=5)


# ---------------------------------------------------------------------------
# §6  generate() — timeout propagation
# ---------------------------------------------------------------------------

class TestGenerateTimeout:
    def test_requests_timeout_raises_backend_error(self):
        """requests.Timeout must be caught and re-raised as BackendError."""
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        with patch("requests.post", side_effect=requests.Timeout("timed out")):
            with pytest.raises(BackendError) as exc_info:
                b.generate(prompt="p", grammar="root ::= \"x\"", max_tokens=5)
        assert exc_info.value.status is None

    def test_requests_connection_error_raises_backend_error(self):
        """requests.ConnectionError must be caught and re-raised as BackendError."""
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        with patch("requests.post", side_effect=requests.ConnectionError("refused")):
            with pytest.raises(BackendError) as exc_info:
                b.generate(prompt="p", grammar="root ::= \"x\"", max_tokens=5)
        assert exc_info.value.status is None


# ---------------------------------------------------------------------------
# §7  generate() — grammar required
# ---------------------------------------------------------------------------

class TestGenerateGrammarRequired:
    def test_grammar_sent_in_every_call(self):
        """Every generate() call must include a non-empty grammar in the payload."""
        b = _make_backend()
        grammar_str = 'root ::= "translation"'
        resp = _make_response(200, {
            "choices": [{"text": "translation", "token_ids": [5]}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        })
        with patch("requests.post", return_value=resp) as mock_post:
            b.generate(prompt="p", grammar=grammar_str, max_tokens=5)
        body = mock_post.call_args[1]["json"]
        # Grammar must be non-empty in structured_outputs
        so = body.get("structured_outputs", {})
        assert so.get("grammar") == grammar_str
        assert so.get("grammar") != ""


# ---------------------------------------------------------------------------
# §8  tokenize()
# ---------------------------------------------------------------------------

class TestTokenize:
    _SUCCESS = {"tokens": [10, 20, 30]}

    def test_returns_list_of_ints(self):
        b = _make_backend()
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp):
            ids = b.tokenize("hello world foo")
        assert ids == [10, 20, 30]

    def test_posts_to_tokenize_endpoint(self):
        b = _make_backend()
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp) as mock_post:
            b.tokenize("hello")
        url = mock_post.call_args[0][0]
        assert url.endswith("/tokenize")

    def test_payload_contains_model(self):
        b = _make_backend(model="qwen-model")
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp) as mock_post:
            b.tokenize("hello")
        body = mock_post.call_args[1]["json"]
        assert body["model"] == "qwen-model"

    def test_payload_contains_prompt(self):
        b = _make_backend()
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp) as mock_post:
            b.tokenize("my text here")
        body = mock_post.call_args[1]["json"]
        assert body["prompt"] == "my text here"

    def test_payload_add_special_tokens_false(self):
        b = _make_backend()
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp) as mock_post:
            b.tokenize("some text")
        body = mock_post.call_args[1]["json"]
        assert body["add_special_tokens"] is False

    def test_payload_return_token_strs_false(self):
        b = _make_backend()
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp) as mock_post:
            b.tokenize("some text")
        body = mock_post.call_args[1]["json"]
        assert body["return_token_strs"] is False

    def test_http_error_raises_backend_error(self):
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        resp = _make_response(500, {"error": "fail"})
        with patch("requests.post", return_value=resp):
            with pytest.raises(BackendError) as exc_info:
                b.tokenize("hello")
        assert exc_info.value.operation == "tokenize"

    def test_missing_tokens_key_raises_backend_error(self):
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        resp = _make_response(200, {"ids": [1, 2]})  # wrong key
        with patch("requests.post", return_value=resp):
            with pytest.raises(BackendError):
                b.tokenize("hello")

    def test_connection_error_raises_backend_error(self):
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        with patch("requests.post", side_effect=requests.ConnectionError("fail")):
            with pytest.raises(BackendError):
                b.tokenize("hello")

    def test_timeout_raises_backend_error(self):
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        with patch("requests.post", side_effect=requests.Timeout("timed out")):
            with pytest.raises(BackendError):
                b.tokenize("hello")

    def test_timeout_forwarded(self):
        """Explicit timeout must be forwarded to requests.post for tokenize."""
        from constrained_translation.vllm_backend import VLLMBackend
        b = VLLMBackend(base_url="http://localhost:8000", model="m", timeout=(3, 30))
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp) as mock_post:
            b.tokenize("test")
        assert mock_post.call_args[1]["timeout"] == (3, 30)


# ---------------------------------------------------------------------------
# §9  decode_token()
# ---------------------------------------------------------------------------

class TestDecodeToken:
    _SUCCESS = {"prompt": "hello"}

    def test_returns_string(self):
        b = _make_backend()
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp):
            result = b.decode_token(42)
        assert result == "hello"

    def test_posts_to_detokenize_endpoint(self):
        b = _make_backend()
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp) as mock_post:
            b.decode_token(42)
        url = mock_post.call_args[0][0]
        assert url.endswith("/detokenize")

    def test_payload_contains_model(self):
        b = _make_backend(model="test-m")
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp) as mock_post:
            b.decode_token(7)
        body = mock_post.call_args[1]["json"]
        assert body["model"] == "test-m"

    def test_payload_contains_token_id_in_tokens_list(self):
        b = _make_backend()
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp) as mock_post:
            b.decode_token(99)
        body = mock_post.call_args[1]["json"]
        assert body["tokens"] == [99]

    def test_http_error_raises_backend_error(self):
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        resp = _make_response(404, {"error": "not found"})
        with patch("requests.post", return_value=resp):
            with pytest.raises(BackendError) as exc_info:
                b.decode_token(1)
        assert exc_info.value.operation == "decode_token"

    def test_missing_prompt_key_raises_backend_error(self):
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        resp = _make_response(200, {"text": "hello"})  # wrong key
        with patch("requests.post", return_value=resp):
            with pytest.raises(BackendError):
                b.decode_token(1)

    def test_connection_error_raises_backend_error(self):
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        with patch("requests.post", side_effect=requests.ConnectionError("fail")):
            with pytest.raises(BackendError):
                b.decode_token(1)

    def test_timeout_raises_backend_error(self):
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        with patch("requests.post", side_effect=requests.Timeout("timed out")):
            with pytest.raises(BackendError):
                b.decode_token(1)

    def test_timeout_forwarded(self):
        from constrained_translation.vllm_backend import VLLMBackend
        b = VLLMBackend(base_url="http://localhost:8000", model="m", timeout=(2, 10))
        resp = _make_response(200, self._SUCCESS)
        with patch("requests.post", return_value=resp) as mock_post:
            b.decode_token(5)
        assert mock_post.call_args[1]["timeout"] == (2, 10)


# ---------------------------------------------------------------------------
# §10  is_available()
# ---------------------------------------------------------------------------

class TestIsAvailable:
    def test_health_ok_returns_true(self):
        b = _make_backend()
        resp = _make_response(200, {"status": "ok"})
        with patch("requests.get", return_value=resp):
            assert b.is_available() is True

    def test_gets_health_endpoint(self):
        b = _make_backend()
        resp = _make_response(200, {"status": "ok"})
        with patch("requests.get", return_value=resp) as mock_get:
            b.is_available()
        url = mock_get.call_args[0][0]
        assert url.endswith("/health")

    def test_503_returns_false(self):
        b = _make_backend()
        resp = _make_response(503, {"status": "loading"})
        with patch("requests.get", return_value=resp):
            assert b.is_available() is False

    def test_500_returns_false(self):
        b = _make_backend()
        resp = _make_response(500, {"error": "internal"})
        with patch("requests.get", return_value=resp):
            assert b.is_available() is False

    def test_connection_error_returns_false(self):
        b = _make_backend()
        with patch("requests.get", side_effect=requests.ConnectionError("refused")):
            assert b.is_available() is False

    def test_timeout_returns_false(self):
        b = _make_backend()
        with patch("requests.get", side_effect=requests.Timeout("timed out")):
            assert b.is_available() is False

    def test_timeout_forwarded_to_get(self):
        from constrained_translation.vllm_backend import VLLMBackend
        b = VLLMBackend(base_url="http://localhost:8000", model="m", timeout=(5, 10))
        resp = _make_response(200, {"status": "ok"})
        with patch("requests.get", return_value=resp) as mock_get:
            b.is_available()
        assert mock_get.call_args[1]["timeout"] == (5, 10)


# ---------------------------------------------------------------------------
# §11  Unicode round-trip
# ---------------------------------------------------------------------------

class TestUnicode:
    def test_generate_unicode_prompt(self):
        """Unicode prompt must survive serialisation and response parsing."""
        from constrained_translation.protocol import GenerationResult
        b = _make_backend()
        resp = _make_response(200, {
            "choices": [{"text": "Dieu créa les cieux", "token_ids": [1, 2, 3, 4]}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4},
        })
        with patch("requests.post", return_value=resp):
            result = b.generate(
                prompt="Au commencement Dieu créa les cieux et la terre",
                grammar='root ::= "Dieu créa les cieux"',
                max_tokens=20,
            )
        assert result.text == "Dieu créa les cieux"
        assert result.token_ids == [1, 2, 3, 4]

    def test_tokenize_unicode_text(self):
        b = _make_backend()
        resp = _make_response(200, {"tokens": [100, 200, 300, 400]})
        with patch("requests.post", return_value=resp) as mock_post:
            ids = b.tokenize("Dieu créa les cieux")
        assert ids == [100, 200, 300, 400]
        body = mock_post.call_args[1]["json"]
        assert body["prompt"] == "Dieu créa les cieux"

    def test_decode_token_unicode_result(self):
        b = _make_backend()
        resp = _make_response(200, {"prompt": "créa"})
        with patch("requests.post", return_value=resp):
            result = b.decode_token(888)
        assert result == "créa"


# ---------------------------------------------------------------------------
# §12  BackendProtocol conformance
# ---------------------------------------------------------------------------

class TestProtocolConformance:
    def test_vllm_backend_satisfies_protocol(self):
        from constrained_translation.protocol import BackendProtocol
        from constrained_translation.vllm_backend import VLLMBackend
        b = VLLMBackend(base_url="http://localhost:8000", model="m")
        assert isinstance(b, BackendProtocol)

    def test_vllm_backend_has_generate(self):
        from constrained_translation.vllm_backend import VLLMBackend
        b = VLLMBackend(base_url="http://localhost:8000", model="m")
        assert callable(b.generate)

    def test_vllm_backend_has_tokenize(self):
        from constrained_translation.vllm_backend import VLLMBackend
        b = VLLMBackend(base_url="http://localhost:8000", model="m")
        assert callable(b.tokenize)

    def test_vllm_backend_has_decode_token(self):
        from constrained_translation.vllm_backend import VLLMBackend
        b = VLLMBackend(base_url="http://localhost:8000", model="m")
        assert callable(b.decode_token)

    def test_vllm_backend_has_is_available(self):
        from constrained_translation.vllm_backend import VLLMBackend
        b = VLLMBackend(base_url="http://localhost:8000", model="m")
        assert callable(b.is_available)


# ---------------------------------------------------------------------------
# §13  decode_token() — process-local LRU/FIFO cache
# ---------------------------------------------------------------------------

class TestDecodeTokenCache:
    """Process-local bounded cache for VLLMBackend.decode_token().

    Key: (normalized base_url, model, token_id).
    Max entries: 8192.  Successes only.  Failures must NOT be cached.
    decode_tokens() (full-sequence) must remain uncached/per-call.
    """

    @staticmethod
    def _clear_cache():
        """Clear the class-level decode_token cache between tests."""
        from constrained_translation.vllm_backend import VLLMBackend
        VLLMBackend._decode_token_cache.clear()

    # setup_method/teardown_method are intentionally omitted here;
    # the module-level _clear_decode_token_cache autouse fixture handles cleanup.

    def test_cache_exists_on_class(self):
        """VLLMBackend must expose a class-level _decode_token_cache dict-like object."""
        from constrained_translation.vllm_backend import VLLMBackend
        assert hasattr(VLLMBackend, "_decode_token_cache")
        cache = VLLMBackend._decode_token_cache
        # Must support len() and clear()
        assert hasattr(cache, "__len__")
        assert hasattr(cache, "clear")

    def test_same_instance_hit_avoids_second_post(self):
        """Second call for same token_id on same instance must NOT make an HTTP request."""
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        resp = _make_response(200, {"prompt": "hello"})
        with patch("requests.post", return_value=resp) as mock_post:
            r1 = b.decode_token(42)
            r2 = b.decode_token(42)
        # HTTP POST called exactly once
        assert mock_post.call_count == 1
        assert r1 == "hello"
        assert r2 == "hello"

    def test_same_instance_different_tokens_both_posted(self):
        """Different token_ids must each make their own HTTP request."""
        b = _make_backend()
        with patch("requests.post", side_effect=[
            _make_response(200, {"prompt": "hello"}),
            _make_response(200, {"prompt": "world"}),
        ]) as mock_post:
            r1 = b.decode_token(1)
            r2 = b.decode_token(2)
        assert mock_post.call_count == 2
        assert r1 == "hello"
        assert r2 == "world"

    def test_cross_instance_same_endpoint_model_hit(self):
        """A second instance with the same base_url+model must hit the cache."""
        resp = _make_response(200, {"prompt": "cached_result"})
        with patch("requests.post", return_value=resp) as mock_post:
            b1 = _make_backend(base_url="http://localhost:8000", model="m")
            b2 = _make_backend(base_url="http://localhost:8000", model="m")
            r1 = b1.decode_token(99)
            r2 = b2.decode_token(99)
        assert mock_post.call_count == 1
        assert r1 == "cached_result"
        assert r2 == "cached_result"

    def test_cross_instance_different_endpoint_separate_cache(self):
        """Different base_urls must NOT share cached entries."""
        with patch("requests.post", side_effect=[
            _make_response(200, {"prompt": "from_a"}),
            _make_response(200, {"prompt": "from_b"}),
        ]) as mock_post:
            b1 = _make_backend(base_url="http://server-a:8000", model="m")
            b2 = _make_backend(base_url="http://server-b:8000", model="m")
            r1 = b1.decode_token(5)
            r2 = b2.decode_token(5)
        assert mock_post.call_count == 2
        assert r1 == "from_a"
        assert r2 == "from_b"

    def test_cross_instance_different_model_separate_cache(self):
        """Different model names on the same base_url must NOT share cached entries."""
        with patch("requests.post", side_effect=[
            _make_response(200, {"prompt": "from_model_a"}),
            _make_response(200, {"prompt": "from_model_b"}),
        ]) as mock_post:
            b1 = _make_backend(base_url="http://localhost:8000", model="model-A")
            b2 = _make_backend(base_url="http://localhost:8000", model="model-B")
            r1 = b1.decode_token(7)
            r2 = b2.decode_token(7)
        assert mock_post.call_count == 2
        assert r1 == "from_model_a"
        assert r2 == "from_model_b"

    def test_failure_not_cached(self):
        """A failed decode_token (BackendError) must NOT be stored in the cache.

        A subsequent call for the same token_id must retry the HTTP request.
        """
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        with patch("requests.post", side_effect=[
            _make_response(500, {"error": "server error"}),   # first call — fails
            _make_response(200, {"prompt": "recovered"}),     # second call — succeeds
        ]) as mock_post:
            with pytest.raises(BackendError):
                b.decode_token(10)
            result = b.decode_token(10)
        assert mock_post.call_count == 2
        assert result == "recovered"

    def test_failure_http_not_cached_on_error_response(self):
        """HTTP 4xx errors must not be cached; retry must hit network."""
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        with patch("requests.post", side_effect=[
            _make_response(404, {"error": "not found"}),
            _make_response(200, {"prompt": "ok"}),
        ]) as mock_post:
            with pytest.raises(BackendError):
                b.decode_token(20)
            result = b.decode_token(20)
        assert mock_post.call_count == 2
        assert result == "ok"

    def test_cache_size_bounded_at_8192(self):
        """Cache must not exceed 8192 entries (eviction must occur)."""
        from constrained_translation.vllm_backend import VLLMBackend
        b = _make_backend()

        def _resp_for(tid):
            return _make_response(200, {"prompt": f"tok_{tid}"})

        # Fill 8193 distinct token IDs
        with patch("requests.post", side_effect=[_resp_for(i) for i in range(8193)]):
            for i in range(8193):
                b.decode_token(i)

        assert len(VLLMBackend._decode_token_cache) <= 8192

    def test_decode_tokens_sequence_always_posts(self):
        """decode_tokens (full sequence) must NOT use the single-token cache."""
        b = _make_backend()
        resp = _make_response(200, {"prompt": "hello world"})
        with patch("requests.post", return_value=resp) as mock_post:
            r1 = b.decode_tokens([1, 2])
            r2 = b.decode_tokens([1, 2])
        # Both calls must go to the network — no caching for full sequences
        assert mock_post.call_count == 2
        assert r1 == "hello world"
        assert r2 == "hello world"

    def test_trailing_slash_normalisation_hits_cache(self):
        """base_url with trailing slash must normalise to same key as without."""
        resp = _make_response(200, {"prompt": "normalised"})
        with patch("requests.post", return_value=resp) as mock_post:
            from constrained_translation.vllm_backend import VLLMBackend
            b1 = VLLMBackend(base_url="http://localhost:8000/", model="m")
            b2 = VLLMBackend(base_url="http://localhost:8000", model="m")
            r1 = b1.decode_token(3)
            r2 = b2.decode_token(3)
        assert mock_post.call_count == 1
        assert r1 == "normalised"
        assert r2 == "normalised"

    def test_connection_error_not_cached(self):
        """A connection-level error must not be cached; retry must hit network."""
        from constrained_translation.vllm_backend import BackendError
        b = _make_backend()
        with patch("requests.post", side_effect=[
            requests.ConnectionError("refused"),
            _make_response(200, {"prompt": "retry_ok"}),
        ]) as mock_post:
            with pytest.raises(BackendError):
                b.decode_token(30)
            result = b.decode_token(30)
        assert mock_post.call_count == 2
        assert result == "retry_ok"
