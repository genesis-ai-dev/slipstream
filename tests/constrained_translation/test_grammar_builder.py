"""Tests for constrained_translation.grammar_builder — Task 6.

TDD: these tests are written BEFORE the implementation.

All tests are fully offline — they inspect the grammar string directly and do
not import xgrammar.  An optional compile test is added at the bottom,
skipped if xgrammar is not installed.
"""

from __future__ import annotations

import importlib
import re
import sys

import pytest


# ---------------------------------------------------------------------------
# Helper: import the module under test
# ---------------------------------------------------------------------------

def _import_builder():
    """Import grammar_builder, failing fast with ImportError if absent."""
    import constrained_translation.grammar_builder as m
    return m


# ---------------------------------------------------------------------------
# test_grammar_builder_empty_vocab_raises_grammar_build_error
# ---------------------------------------------------------------------------

def test_grammar_builder_empty_vocab_raises_grammar_build_error():
    """An empty attested_vocab must raise GrammarBuildError (invariant I7)."""
    m = _import_builder()
    with pytest.raises(m.GrammarBuildError) as exc_info:
        m.GrammarBuilder().build(frozenset(), item_id="GEN 1:1")
    # Exception must carry item_id
    assert exc_info.value.item_id == "GEN 1:1"


# ---------------------------------------------------------------------------
# test_grammar_build_error_carries_item_id
# ---------------------------------------------------------------------------

def test_grammar_build_error_carries_item_id():
    """GrammarBuildError.item_id and .reason are accessible as attributes."""
    m = _import_builder()
    try:
        m.GrammarBuilder().build(frozenset(), item_id="REV 22:21")
    except m.GrammarBuildError as exc:
        assert exc.item_id == "REV 22:21"
        assert isinstance(exc.reason, str)
        assert exc.reason  # non-empty
    else:
        pytest.fail("GrammarBuildError not raised for empty vocab")


# ---------------------------------------------------------------------------
# test_grammar_builder_output_is_string
# ---------------------------------------------------------------------------

def test_grammar_builder_output_is_string():
    """build() must return a str."""
    m = _import_builder()
    vocab = frozenset(["hello", "world"])
    result = m.GrammarBuilder().build(vocab, item_id="GEN 1:1")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# test_grammar_builder_attested_tokens_present_in_grammar
# ---------------------------------------------------------------------------

def test_grammar_builder_attested_tokens_present_in_grammar():
    """Every attested surface token must appear as a quoted literal in the grammar."""
    m = _import_builder()
    vocab = frozenset(["blessed", "God", "earth"])
    grammar = m.GrammarBuilder().build(vocab, item_id="GEN 1:1")

    for token in vocab:
        # Tokens appear inside double-quoted string literals in EBNF.
        # We check the raw token string is present somewhere in the grammar
        # (it may be escaped, so we also accept escaped variants).
        assert token in grammar or repr(token)[1:-1] in grammar, (
            f"Token {token!r} not found in grammar:\n{grammar}"
        )


# ---------------------------------------------------------------------------
# test_grammar_builder_unk_markers_absent_from_grammar
# ---------------------------------------------------------------------------

def test_grammar_builder_unk_markers_absent_from_grammar():
    """[UNK:…] must NEVER appear in the grammar string (invariant I3).

    [UNK:…] markers are inserted deterministically by UNKDetector after
    constrained generation; the model must never generate them.
    """
    m = _import_builder()
    vocab = frozenset(["hello", "world", "foo"])
    grammar = m.GrammarBuilder().build(vocab, item_id="GEN 1:1")
    assert "[UNK:" not in grammar, (
        "[UNK: marker found in grammar — model must never generate UNK markers"
    )
    assert "UNK" not in grammar, (
        "UNK token found in grammar — grammar must only license attested surface forms"
    )


# ---------------------------------------------------------------------------
# test_grammar_builder_special_chars_escaped
# ---------------------------------------------------------------------------

def test_grammar_builder_special_chars_escaped():
    """Special EBNF/regex characters inside token literals must be properly escaped.

    Tokens that contain double quotes, backslashes, or other EBNF meta-characters
    must appear safely inside double-quoted string literals.
    """
    m = _import_builder()
    # Tokens with characters that need escaping in EBNF string literals
    vocab = frozenset([
        'it\'s',       # apostrophe — fine as-is
        'said"he',     # embedded double-quote — must be escaped as \"
        'back\\slash', # backslash — must be escaped as \\
        'café',        # non-ASCII — must be preserved or unicode-escaped
        'end.',        # trailing period
    ])
    # Must not raise
    grammar = m.GrammarBuilder().build(vocab, item_id="SPEC 1:1")
    assert isinstance(grammar, str)
    # The grammar string must be syntactically sane: no unbalanced quotes
    # at the top level (rough check: count unescaped quotes).
    # We simply verify the function completes and returns a non-empty string.
    assert len(grammar) > 0


def test_grammar_builder_double_quote_in_token_is_escaped():
    """A token containing a double-quote must escape it so the grammar is valid EBNF."""
    m = _import_builder()
    vocab = frozenset(['say"hello'])
    grammar = m.GrammarBuilder().build(vocab, item_id="SPEC 1:2")
    # The escaped form \" must appear, not the raw unescaped double-quote inside a literal.
    # Valid EBNF would represent it as: "say\"hello" or use single quotes.
    assert '\\"' in grammar or "say\"hello" not in grammar.replace('\\"', ''), (
        "Double-quote in token was not escaped in grammar"
    )


def test_grammar_builder_backslash_in_token_is_escaped():
    """A token containing a backslash must be double-escaped in the grammar literal."""
    m = _import_builder()
    vocab = frozenset(["back\\slash"])
    grammar = m.GrammarBuilder().build(vocab, item_id="SPEC 1:3")
    # Should contain \\\\ (two backslashes) or the literal \\ at minimum
    assert "\\\\" in grammar, (
        f"Backslash was not properly escaped in grammar:\n{grammar}"
    )


# ---------------------------------------------------------------------------
# test_grammar_builder_no_raw_subword_pieces
# ---------------------------------------------------------------------------

def test_grammar_builder_no_raw_subword_pieces():
    """The grammar must contain only surface strings, never token IDs or subword markers.

    XGrammar grammars are built over surface strings, not over tokenizer
    vocabulary indices or subword piece markers like '▁' or '<0x41>' (invariant I2).
    """
    m = _import_builder()
    vocab = frozenset(["hello", "world", "the"])
    grammar = m.GrammarBuilder().build(vocab, item_id="GEN 1:1")
    # No raw integer token IDs should appear as standalone grammar terminals
    assert not re.search(r'"\d{3,}"', grammar), (
        "Raw token ID (3+ digit integer in quotes) found in grammar"
    )
    # No subword sentencepiece/BPE markers
    assert "▁" not in grammar, "SentencePiece prefix marker ▁ found in grammar"
    assert "<0x" not in grammar, "Hex subword piece marker found in grammar"


# ---------------------------------------------------------------------------
# test_grammar_builder_grammar_has_required_rules
# ---------------------------------------------------------------------------

def test_grammar_builder_grammar_has_required_rules():
    """The grammar string must define the key rules: translation, attested_token, WS."""
    m = _import_builder()
    vocab = frozenset(["In", "the", "beginning"])
    grammar = m.GrammarBuilder().build(vocab, item_id="GEN 1:1")
    # Root rule must reference translation (or be named root/translation)
    assert "translation" in grammar or "root" in grammar, (
        "Grammar missing a root/translation rule"
    )
    # whitespace or WS rule, or an inline space literal
    assert "WS" in grammar or '" "' in grammar or "' '" in grammar or "[ \\t\\n]" in grammar, (
        "Grammar missing whitespace handling"
    )


def test_grammar_builder_grammar_allows_newline():
    """The grammar must allow a sentence-final newline (plan §7.6)."""
    m = _import_builder()
    vocab = frozenset(["hello", "world"])
    grammar = m.GrammarBuilder().build(vocab, item_id="GEN 1:1")
    # \\n or \n should appear in the grammar to license a trailing newline
    assert "\\n" in grammar or r"\n" in grammar, (
        "Grammar does not allow a sentence-final newline"
    )


# ---------------------------------------------------------------------------
# test_grammar_builder_punctuation_licensed
# ---------------------------------------------------------------------------

def test_grammar_builder_punctuation_licensed():
    """The grammar must include the minimal explicit punctuation set from the plan."""
    m = _import_builder()
    vocab = frozenset(["hello"])
    grammar = m.GrammarBuilder().build(vocab, item_id="GEN 1:1")
    # Plan §7.6 lists: . , ! ? ; : ' "
    for ch in [".", ",", "!", "?"]:
        assert ch in grammar, f"Punctuation character {ch!r} not licensed in grammar"


# ---------------------------------------------------------------------------
# test_grammar_builder_single_token_vocab
# ---------------------------------------------------------------------------

def test_grammar_builder_single_token_vocab():
    """A vocab with a single token must not raise and must produce a valid grammar."""
    m = _import_builder()
    vocab = frozenset(["Amen"])
    grammar = m.GrammarBuilder().build(vocab, item_id="REV 22:20")
    assert isinstance(grammar, str)
    assert "Amen" in grammar


# ---------------------------------------------------------------------------
# test_grammar_builder_large_vocab
# ---------------------------------------------------------------------------

def test_grammar_builder_large_vocab():
    """A vocab with 500 tokens must complete quickly and produce a valid grammar."""
    import time
    m = _import_builder()
    # Generate 500 distinct ASCII tokens
    vocab = frozenset(f"word{i}" for i in range(500))
    t0 = time.monotonic()
    grammar = m.GrammarBuilder().build(vocab, item_id="PERF 1:1")
    elapsed = time.monotonic() - t0
    assert isinstance(grammar, str)
    assert elapsed < 2.0, f"GrammarBuilder too slow for 500 tokens: {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Optional: xgrammar compile test (skipped if xgrammar not installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    importlib.util.find_spec("xgrammar") is None,
    reason="xgrammar not installed; skipping grammar compilation test",
)
def test_grammar_compiles_with_xgrammar():
    """If xgrammar is installed, the produced grammar must compile without error."""
    import xgrammar  # noqa: F401 — guarded by skipif

    m = _import_builder()
    vocab = frozenset(["In", "the", "beginning", "God", "created", "heavens", "earth"])
    grammar_str = m.GrammarBuilder().build(vocab, item_id="GEN 1:1")

    # xgrammar.Grammar.from_ebnf_string(grammar_str) is the typical API call.
    # We try multiple plausible API entry points to be robust to version changes.
    compiled = False
    errors = []
    for attr in ["Grammar", "BNFGrammar", "GrammarCompiler"]:
        cls = getattr(xgrammar, attr, None)
        if cls is None:
            continue
        for method in ["from_ebnf_string", "from_ebnf", "from_string"]:
            fn = getattr(cls, method, None)
            if fn is None:
                continue
            try:
                fn(grammar_str)
                compiled = True
                break
            except Exception as e:
                errors.append(f"{attr}.{method}: {e}")
        if compiled:
            break

    assert compiled, (
        f"xgrammar failed to compile grammar.\nErrors:\n" + "\n".join(errors) +
        f"\n\nGrammar:\n{grammar_str}"
    )


# ---------------------------------------------------------------------------
# test_grammar_builder_rejects_control_characters — new tests (quality review)
# ---------------------------------------------------------------------------

def test_grammar_builder_rejects_nul_character():
    """A token containing NUL (U+0000, Cc) must raise GrammarBuildError."""
    m = _import_builder()
    vocab = frozenset(["hello\x00world"])
    with pytest.raises(m.GrammarBuildError) as exc_info:
        m.GrammarBuilder().build(vocab, item_id="CTRL 1:1")
    assert exc_info.value.item_id == "CTRL 1:1"
    assert "0000" in exc_info.value.reason or "Cc" in exc_info.value.reason


def test_grammar_builder_rejects_tab_character():
    """A token containing TAB (U+0009, Cc) must raise GrammarBuildError."""
    m = _import_builder()
    vocab = frozenset(["word\t"])
    with pytest.raises(m.GrammarBuildError) as exc_info:
        m.GrammarBuilder().build(vocab, item_id="CTRL 1:2")
    assert exc_info.value.item_id == "CTRL 1:2"


def test_grammar_builder_rejects_carriage_return():
    """A token containing CR (U+000D, Cc) must raise GrammarBuildError."""
    m = _import_builder()
    vocab = frozenset(["word\r"])
    with pytest.raises(m.GrammarBuildError) as exc_info:
        m.GrammarBuilder().build(vocab, item_id="CTRL 1:3")
    assert exc_info.value.item_id == "CTRL 1:3"


def test_grammar_builder_rejects_bidi_control():
    """A token containing a BiDi control character (U+200F, Cf) must raise GrammarBuildError."""
    m = _import_builder()
    # U+200F RIGHT-TO-LEFT MARK (category Cf)
    vocab = frozenset(["word\u200f"])
    with pytest.raises(m.GrammarBuildError) as exc_info:
        m.GrammarBuilder().build(vocab, item_id="CTRL 1:4")
    assert exc_info.value.item_id == "CTRL 1:4"
    # Must mention the character codepoint or Cf category
    assert "200F" in exc_info.value.reason.upper() or "Cf" in exc_info.value.reason


def test_grammar_builder_control_char_error_carries_item_id():
    """GrammarBuildError raised for a control character must carry item_id."""
    m = _import_builder()
    vocab = frozenset(["bad\x00token"])
    try:
        m.GrammarBuilder().build(vocab, item_id="MYITEM 42:7")
    except m.GrammarBuildError as exc:
        assert exc.item_id == "MYITEM 42:7"
        assert exc.reason
    else:
        pytest.fail("Expected GrammarBuildError for NUL character")


def test_grammar_builder_normal_unicode_not_rejected():
    """Normal Unicode letters/accents/CJK must NOT be rejected."""
    m = _import_builder()
    vocab = frozenset(["café", "über", "日本語", "العربية", "Привет"])
    grammar = m.GrammarBuilder().build(vocab, item_id="UNICODE 1:1")
    assert isinstance(grammar, str)
    assert len(grammar) > 0


# ---------------------------------------------------------------------------
# xgrammar compile test using vllm-env interpreter (non-skipped when available)
# ---------------------------------------------------------------------------

def test_grammar_compiles_with_vllm_env_xgrammar():
    """Compile the grammar using the vllm-env Python interpreter's xgrammar.

    Invokes /home/clear/vllm-env/bin/python with a tiny compile script.
    Skipped if that interpreter path does not exist.
    The standard offline test suite is kept independent from this check.
    """
    import subprocess
    import os

    vllm_python = "/home/clear/vllm-env/bin/python"
    if not os.path.exists(vllm_python):
        pytest.skip(f"{vllm_python} not found — skipping vllm-env xgrammar compile test")

    m = _import_builder()
    vocab = frozenset(["In", "the", "beginning", "God", "created", "heavens", "earth"])
    grammar_str = m.GrammarBuilder().build(vocab, item_id="GEN 1:1")

    # Escape the grammar string for safe inline embedding.
    grammar_repr = repr(grammar_str)

    script = f"""
import sys
try:
    import xgrammar
except ImportError:
    print("SKIP:xgrammar_not_importable")
    sys.exit(0)

grammar_str = {grammar_repr}

compiled = False
errors = []
for attr in ["Grammar", "BNFGrammar", "GrammarCompiler"]:
    cls = getattr(xgrammar, attr, None)
    if cls is None:
        continue
    for method in ["from_ebnf_string", "from_ebnf", "from_string"]:
        fn = getattr(cls, method, None)
        if fn is None:
            continue
        try:
            fn(grammar_str)
            compiled = True
            break
        except Exception as e:
            errors.append(f"{{attr}}.{{method}}: {{e}}")
    if compiled:
        break

if compiled:
    print("OK")
else:
    print("FAIL:" + "; ".join(errors))
    sys.exit(1)
"""

    result = subprocess.run(
        [vllm_python, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )

    stdout = result.stdout.strip()
    if stdout.startswith("SKIP:"):
        pytest.skip(f"xgrammar not importable in vllm-env: {stdout}")

    assert result.returncode == 0, (
        f"vllm-env xgrammar compile failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert stdout == "OK", (
        f"vllm-env xgrammar compile reported failure: {stdout}"
    )
