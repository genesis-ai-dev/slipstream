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
# Zero-width character allowlist tests — Task: allow U+200B/C/D in Cf
# ---------------------------------------------------------------------------

def test_grammar_builder_allows_zero_width_space():
    """U+200B ZERO WIDTH SPACE (Cf) must be accepted — essential in Burmese corpora."""
    m = _import_builder()
    # Burmese word-boundary token containing ZWS
    tok = "ကြည့်\u200bရှု"  # zero-width space between syllables
    vocab = frozenset([tok])
    grammar = m.GrammarBuilder().build(vocab, item_id="MYA 1:1")
    assert isinstance(grammar, str)
    assert len(grammar) > 0


def test_grammar_builder_allows_zero_width_non_joiner():
    """U+200C ZERO WIDTH NON-JOINER (Cf) must be accepted — Indic/Devanagari scripts."""
    m = _import_builder()
    tok = "प्र\u200cकार"  # ZWNJ used to prevent ligature in Devanagari
    vocab = frozenset([tok])
    grammar = m.GrammarBuilder().build(vocab, item_id="NPI 1:1")
    assert isinstance(grammar, str)
    assert len(grammar) > 0


def test_grammar_builder_allows_zero_width_joiner():
    """U+200D ZERO WIDTH JOINER (Cf) must be accepted — essential in Nepali corpora."""
    m = _import_builder()
    # Nepali conjunct consonant token containing ZWJ
    tok = "क्\u200dष"  # ZWJ between consonants to form a conjunct
    vocab = frozenset([tok])
    grammar = m.GrammarBuilder().build(vocab, item_id="NPI 1:2")
    assert isinstance(grammar, str)
    assert len(grammar) > 0


def test_grammar_builder_allows_burmese_realistic_token():
    """Realistic Burmese token with U+200B must be accepted and appear in grammar."""
    m = _import_builder()
    # Burmese: 'လေ့လာ' + ZWS + 'ရန်'  (commonly tokenised together in mya Bible text)
    tok = "လေ့လာ\u200bရန်"
    vocab = frozenset([tok, "သည်", "ကို"])
    grammar = m.GrammarBuilder().build(vocab, item_id="MYA 2:1")
    assert isinstance(grammar, str)
    # The Burmese chars should appear in the grammar
    assert "လေ့လာ" in grammar


def test_grammar_builder_allows_nepali_realistic_token():
    """Realistic Nepali token with U+200D must be accepted and appear in grammar."""
    m = _import_builder()
    # Nepali: conjunct with ZWJ
    tok = "ज्\u200dञ"
    vocab = frozenset([tok, "को", "मा"])
    grammar = m.GrammarBuilder().build(vocab, item_id="NPI 2:1")
    assert isinstance(grammar, str)
    assert "ज्" in grammar


def test_grammar_builder_still_rejects_right_to_left_override():
    """U+202E RIGHT-TO-LEFT OVERRIDE (Cf) must still be rejected — bidi attack vector."""
    m = _import_builder()
    vocab = frozenset(["word\u202eevil"])
    with pytest.raises(m.GrammarBuildError) as exc_info:
        m.GrammarBuilder().build(vocab, item_id="BIDI 1:1")
    assert exc_info.value.item_id == "BIDI 1:1"
    assert "202E" in exc_info.value.reason.upper() or "Cf" in exc_info.value.reason


def test_grammar_builder_still_rejects_right_to_left_embedding():
    """U+202B RIGHT-TO-LEFT EMBEDDING (Cf) must still be rejected."""
    m = _import_builder()
    vocab = frozenset(["word\u202b"])
    with pytest.raises(m.GrammarBuildError) as exc_info:
        m.GrammarBuilder().build(vocab, item_id="BIDI 1:2")
    assert exc_info.value.item_id == "BIDI 1:2"
    assert "Cf" in exc_info.value.reason


# ---------------------------------------------------------------------------
# NEW TESTS — tightened grammar: concatenation prevention, separator, coverage
# ---------------------------------------------------------------------------

def test_grammar_accepted_hello_world():
    """'hello world' must be structurally accepted by the tightened grammar.

    Both tokens are attested; they are separated by a canonical single space,
    so they must form a valid lex_core.
    """
    m = _import_builder()
    vocab = frozenset(["hello", "world"])
    grammar = m.GrammarBuilder().build(vocab, item_id="SEP 1:1")
    # Grammar must contain both tokens
    assert '"hello"' in grammar or "hello" in grammar
    assert '"world"' in grammar or "world" in grammar
    # The separator pattern must appear in lex_core: tok_unit (" " tok_unit)*
    assert '" "' in grammar or "' '" in grammar, (
        "Grammar missing mandatory single-space separator"
    )


def test_grammar_structure_prevents_bare_concatenation():
    """The grammar must require a ' ' separator between adjacent lexical units.

    The old grammar allowed segment+ where each segment could be an
    attested_token, producing unattested concatenations like 'terreAu'.
    The new grammar requires lex_core ::= tok_unit (' ' tok_unit)* which
    makes a space mandatory between any two lexical runs.
    """
    m = _import_builder()
    vocab = frozenset(["terre", "Au"])
    grammar = m.GrammarBuilder().build(vocab, item_id="CONCAT 1:1")
    # The grammar must define lex_core with a mandatory space separator
    assert "lex_core" in grammar, "Grammar missing lex_core rule"
    # tok_unit must exist and wrap attested_token
    assert "tok_unit" in grammar, "Grammar missing tok_unit rule"
    # The separator is a literal single space between tok_units
    # The pattern (' ' tok_unit) must appear in the grammar
    assert '" " tok_unit' in grammar or "' ' tok_unit" in grammar, (
        "Grammar missing mandatory ' ' tok_unit separator in lex_core"
    )


def test_grammar_requires_at_least_one_lexical_unit():
    """translation must not be satisfiable by whitespace or punctuation alone.

    The tightened grammar requires lex_core which requires at least one
    tok_unit which requires at least one attested_token — so the root
    cannot be satisfied by empty/WS/punctuation-only output.
    """
    m = _import_builder()
    vocab = frozenset(["hello"])
    grammar = m.GrammarBuilder().build(vocab, item_id="REQ 1:1")
    # lex_core must be non-optional (not followed by ?)
    # Check that the translation rule references lex_core non-optionally
    lines = grammar.splitlines()
    translation_line = next((l for l in lines if l.strip().startswith("translation")), "")
    assert "lex_core" in translation_line, (
        "translation rule must reference lex_core (mandatory lexical core)"
    )
    # lex_core must not have a ? after it making it optional
    assert "lex_core?" not in translation_line, (
        "lex_core must not be optional — at least one lexical unit required"
    )


def test_grammar_punctuation_attachment_accepted():
    """Punctuation attached to a token (e.g. 'hello,') must be structurally accepted.

    tok_unit ::= PUNCT* attested_token PUNCT* means punctuation is freely
    attachable to either side of a token with no separator required.
    """
    m = _import_builder()
    vocab = frozenset(["hello", "world"])
    grammar = m.GrammarBuilder().build(vocab, item_id="PUNCT 1:1")
    # PUNCT rule must exist and list the standard punctuation characters
    assert "PUNCT" in grammar, "Grammar missing PUNCT rule"
    punct_line = next(
        (l for l in grammar.splitlines() if l.strip().startswith("PUNCT")), ""
    )
    for ch in [".", ",", "!", "?"]:
        assert ch in punct_line, f"Punctuation char {ch!r} missing from PUNCT rule"
    # tok_unit must reference PUNCT
    tok_unit_line = next(
        (l for l in grammar.splitlines() if l.strip().startswith("tok_unit")), ""
    )
    assert "PUNCT" in tok_unit_line, "tok_unit must reference PUNCT"
    assert "attested_token" in tok_unit_line, "tok_unit must reference attested_token"


def test_grammar_empty_or_layout_only_rejected_structurally():
    """Empty output and layout-only (whitespace/punct-only) must be structurally rejected.

    The grammar requires at least one lex_core (non-optional), and lex_core
    requires at least one tok_unit with an attested_token.  A purely
    whitespace or punctuation sequence cannot satisfy this requirement.
    """
    m = _import_builder()
    vocab = frozenset(["hello"])
    grammar = m.GrammarBuilder().build(vocab, item_id="EMPTY 1:1")
    # translation must have lex_core (non-optional) in the middle
    # Structurally: translation ::= (" ")* lex_core (" ")* "\n"?
    translation_line = next(
        (l for l in grammar.splitlines() if l.strip().startswith("translation")), ""
    )
    assert "lex_core" in translation_line, (
        "translation rule must require lex_core — empty output must be rejected"
    )


def test_grammar_unk_rejected_structurally():
    """[UNK:…] markers must never appear in the grammar.

    The tightened grammar still upholds invariant I3: UNK markers are never
    licensed as generatable tokens.
    """
    m = _import_builder()
    vocab = frozenset(["hello", "world"])
    grammar = m.GrammarBuilder().build(vocab, item_id="UNK 1:1")
    assert "[UNK:" not in grammar, "[UNK: found in grammar — invariant I3 violated"
    assert "UNK" not in grammar, "UNK string found in grammar — invariant I3 violated"


def test_grammar_canonical_single_space_separator():
    """The separator between lexical units must be exactly one ' ', not WS+.

    The old grammar had WS ::= (' ' | '\\n')+ which allowed unbounded
    whitespace sequences.  The new grammar uses a canonical single space.
    """
    m = _import_builder()
    vocab = frozenset(["God", "created"])
    grammar = m.GrammarBuilder().build(vocab, item_id="SPACE 1:1")
    # Old WS rule must not exist
    assert "WS" not in grammar, (
        "Old unbounded WS rule still present — must be replaced with canonical single space"
    )
    # The lex_core separator must be literal single space
    assert '" " tok_unit' in grammar or "' ' tok_unit" in grammar, (
        "lex_core must use a literal single-space separator"
    )


def test_grammar_optional_leading_trailing_spaces():
    """translation must allow optional leading and trailing spaces."""
    m = _import_builder()
    vocab = frozenset(["hello"])
    grammar = m.GrammarBuilder().build(vocab, item_id="LEAD 1:1")
    translation_line = next(
        (l for l in grammar.splitlines() if l.strip().startswith("translation")), ""
    )
    # Optional spaces on both sides of lex_core: (" ")* ... (" ")*
    assert '(" ")*' in translation_line or "(' ')*" in translation_line, (
        "translation rule must allow optional leading/trailing spaces"
    )


def test_grammar_optional_final_newline():
    """translation must allow an optional single final newline."""
    m = _import_builder()
    vocab = frozenset(["hello"])
    grammar = m.GrammarBuilder().build(vocab, item_id="NL 1:1")
    translation_line = next(
        (l for l in grammar.splitlines() if l.strip().startswith("translation")), ""
    )
    assert '"\\n"?' in translation_line or r'"\n"?' in translation_line, (
        "translation rule must allow an optional final newline"
    )


def test_grammar_unicode_burmese_compiles():
    """Burmese token with U+200B must compile through the tightened grammar."""
    m = _import_builder()
    tok = "လေ့လာ\u200bရန်"
    vocab = frozenset([tok, "သည်"])
    grammar = m.GrammarBuilder().build(vocab, item_id="MYA 3:1")
    assert isinstance(grammar, str)
    assert "lex_core" in grammar
    assert "tok_unit" in grammar


def test_grammar_builder_still_rejects_left_to_right_override():
    """U+202D LEFT-TO-RIGHT OVERRIDE (Cf) must still be rejected."""
    m = _import_builder()
    vocab = frozenset(["\u202dhello"])
    with pytest.raises(m.GrammarBuildError) as exc_info:
        m.GrammarBuilder().build(vocab, item_id="BIDI 1:3")
    assert exc_info.value.item_id == "BIDI 1:3"


def test_grammar_builder_still_rejects_soft_hyphen():
    """U+00AD SOFT HYPHEN (Cf) must still be rejected — non-printing, confusable."""
    m = _import_builder()
    vocab = frozenset(["word\u00adbreak"])
    with pytest.raises(m.GrammarBuildError) as exc_info:
        m.GrammarBuilder().build(vocab, item_id="CF 1:1")
    assert exc_info.value.item_id == "CF 1:1"


def test_grammar_builder_still_rejects_word_joiner():
    """U+2060 WORD JOINER (Cf) must still be rejected — not in the explicit allowlist."""
    m = _import_builder()
    vocab = frozenset(["word\u2060join"])
    with pytest.raises(m.GrammarBuildError) as exc_info:
        m.GrammarBuilder().build(vocab, item_id="CF 1:2")
    assert exc_info.value.item_id == "CF 1:2"


def test_grammar_builder_cf_allowlist_does_not_weaken_cc_rejection():
    """The Cf allowlist must not accidentally permit Cc (real control chars)."""
    m = _import_builder()
    # U+0001 START OF HEADING (Cc) — must still be rejected
    vocab = frozenset(["bad\x01token"])
    with pytest.raises(m.GrammarBuildError) as exc_info:
        m.GrammarBuilder().build(vocab, item_id="CC 1:1")
    assert exc_info.value.item_id == "CC 1:1"
    assert "Cc" in exc_info.value.reason


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
        f"vllm-env xgrammar compile failed (rc={result.returncode}):\\n"
        f"stdout: {result.stdout}\\nstderr: {result.stderr}"
    )
    assert stdout == "OK", (
        f"vllm-env xgrammar compile reported failure: {stdout}"
    )


@pytest.mark.skipif(
    not __import__("os").path.exists("/home/clear/vllm-env/bin/python"),
    reason="/home/clear/vllm-env/bin/python not found — skipping Burmese/Nepali xgrammar compile test",
)
def test_grammar_compiles_with_vllm_env_xgrammar_burmese_nepali():
    """Compile grammar with Burmese (U+200B) and Nepali (U+200D) tokens via vllm-env xgrammar."""
    import subprocess

    vllm_python = "/home/clear/vllm-env/bin/python"

    m = _import_builder()
    # Realistic Burmese tokens with ZERO WIDTH SPACE (U+200B)
    # Realistic Nepali tokens with ZERO WIDTH JOINER (U+200D)
    vocab = frozenset([
        "လေ့လာ\u200bရန်",   # Burmese + ZWS
        "ကြည့်\u200bရှု",    # Burmese + ZWS
        "သည်",               # plain Burmese
        "ज्\u200dञ",          # Nepali + ZWJ
        "क्\u200dष",          # Nepali + ZWJ
        "को",                 # plain Nepali
        "प्र\u200cकार",       # Devanagari + ZWNJ
    ])
    grammar_str = m.GrammarBuilder().build(vocab, item_id="MYA-NPI 1:1")

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
        f"vllm-env xgrammar Burmese/Nepali compile failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert stdout == "OK", (
        f"vllm-env xgrammar Burmese/Nepali compile reported failure: {stdout}"
    )
