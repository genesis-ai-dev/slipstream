"""Tests for constrained_translation.prompt_builder — Task 7.

TDD: these tests are written BEFORE the implementation.

All tests are fully offline — no model, no GPU, no corpus download.

Key behaviours under test:
- Semantic examples appear before coverage examples in the output.
- De-duplication by verse_idx (semantic copy wins when both lists share an idx).
- Exact example source / target text appears verbatim in the prompt.
- UNK spans are substituted right-to-left by character offset in the SOURCE
  line only; target examples are never modified.
- Overlapping spans raise ValueError loudly rather than silently corrupting
  offsets.
- Invalid spans (start >= end, or out-of-range) raise ValueError.
- The translation cue `[Translation]:` ends the prompt (no trailing text).
- The module exposes a PROMPT_TEMPLATE constant.
- [UNK:…] markers in the prompt are not model output; the prompt itself makes
  this clear (system instruction line present).
- PromptBuilder.build() accepts a TranslationRequest and returns str.
"""

from __future__ import annotations

import pytest

from constrained_translation.protocol import (
    AlignedExample,
    TranslationRequest,
    UNKSpan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_example(
    verse_idx: int,
    source: str,
    target: str,
    method: str = "semantic",
    score: float = 1.0,
) -> AlignedExample:
    return AlignedExample(
        verse_idx=verse_idx,
        source=source,
        target=target,
        selection_score=score,
        selection_method=method,
        evidence_tier=1,
    )


def _make_request(
    source_text: str = "In the beginning God created the heavens",
    semantic: list[AlignedExample] | None = None,
    coverage: list[AlignedExample] | None = None,
    unk_spans: list[UNKSpan] | None = None,
    item_id: str = "GEN 1:1",
) -> TranslationRequest:
    if semantic is None:
        semantic = []
    if coverage is None:
        coverage = []
    if unk_spans is None:
        unk_spans = []
    return TranslationRequest(
        item_id=item_id,
        source_text=source_text,
        semantic_examples=tuple(semantic),
        coverage_examples=tuple(coverage),
        attested_vocab=frozenset(["In", "the", "beginning"]),
        unk_spans=tuple(unk_spans),
        grammar_str="root ::= \"dummy\"",
        prompt="",  # will be filled by PromptBuilder
    )


def _import_builder():
    import constrained_translation.prompt_builder as m
    return m


# ---------------------------------------------------------------------------
# test_module_has_prompt_template_constant
# ---------------------------------------------------------------------------

def test_module_has_prompt_template_constant():
    """PromptBuilder module must expose a PROMPT_TEMPLATE constant (str)."""
    m = _import_builder()
    assert hasattr(m, "PROMPT_TEMPLATE"), "Missing PROMPT_TEMPLATE constant"
    assert isinstance(m.PROMPT_TEMPLATE, str)
    assert len(m.PROMPT_TEMPLATE) > 0


# ---------------------------------------------------------------------------
# test_build_returns_str
# ---------------------------------------------------------------------------

def test_build_returns_str():
    """PromptBuilder.build() must return a str."""
    m = _import_builder()
    req = _make_request()
    result = m.PromptBuilder().build(req)
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# test_translation_cue_at_end
# ---------------------------------------------------------------------------

def test_translation_cue_at_end():
    """The prompt must end with the '[Translation]:' cue (possibly trailing whitespace)."""
    m = _import_builder()
    req = _make_request()
    result = m.PromptBuilder().build(req)
    assert result.rstrip().endswith("[Translation]:"), (
        f"Prompt does not end with '[Translation]:':\n{result!r}"
    )


# ---------------------------------------------------------------------------
# test_system_instruction_present
# ---------------------------------------------------------------------------

def test_system_instruction_present():
    """The prompt must contain the system instruction about vocabulary and UNK markers."""
    m = _import_builder()
    req = _make_request()
    result = m.PromptBuilder().build(req)
    # The plan template says:
    #   "Only use vocabulary attested in the examples."
    #   "Uncovered spans are already marked as [UNK:span]."
    # Both phrases (or similar) must appear.
    assert "vocabulary" in result.lower() or "attested" in result.lower(), (
        "Prompt missing vocabulary instruction"
    )
    assert "[UNK:" in result or "UNK:span" in result or "uncovered" in result.lower(), (
        "Prompt missing UNK marker instruction"
    )


# ---------------------------------------------------------------------------
# test_unk_markers_are_instructions_not_model_output
# ---------------------------------------------------------------------------

def test_unk_markers_are_instructions_not_model_output():
    """The prompt must state that [UNK:…] markers are input/instruction, not model output.

    This ensures the prompt is unambiguous about what the model is expected to
    produce.  The system instruction line must make clear that UNK markers are
    pre-inserted (deterministic), not tokens the model should generate.
    """
    m = _import_builder()
    req = _make_request(
        source_text="foo bar baz",
        unk_spans=[UNKSpan(surface="foo bar baz", start_char=0, end_char=11)],
    )
    result = m.PromptBuilder().build(req)
    # The system instruction line says "Uncovered spans are already marked as [UNK:span]"
    # OR an equivalent phrasing making clear markers are pre-inserted.
    lower = result.lower()
    assert "already" in lower or "pre-" in lower or "deterministic" in lower or (
        "marked" in lower and "unk" in lower
    ), (
        "Prompt does not clarify that [UNK:…] markers are pre-inserted instructions, "
        "not model output:\n" + result
    )


# ---------------------------------------------------------------------------
# test_semantic_examples_before_coverage_examples
# ---------------------------------------------------------------------------

def test_semantic_examples_before_coverage_examples():
    """Semantic examples must appear before coverage examples in the prompt."""
    m = _import_builder()
    sem = _make_example(1, "Semantic source", "Semantic target", method="semantic")
    cov = _make_example(2, "Coverage source", "Coverage target", method="coverage")
    req = _make_request(semantic=[sem], coverage=[cov])
    result = m.PromptBuilder().build(req)
    sem_pos = result.index("Semantic source")
    cov_pos = result.index("Coverage source")
    assert sem_pos < cov_pos, (
        "Semantic example does not appear before coverage example"
    )


# ---------------------------------------------------------------------------
# test_exact_example_source_and_target_in_prompt
# ---------------------------------------------------------------------------

def test_exact_example_source_and_target_in_prompt():
    """Exact example source and target text must appear verbatim in the prompt."""
    m = _import_builder()
    sem = _make_example(
        10,
        "Et Dieu créa les cieux",
        "And God created the heavens",
        method="semantic",
    )
    req = _make_request(semantic=[sem])
    result = m.PromptBuilder().build(req)
    assert "Et Dieu créa les cieux" in result, "Example source text missing from prompt"
    assert "And God created the heavens" in result, "Example target text missing from prompt"


# ---------------------------------------------------------------------------
# test_deduplication_semantic_wins
# ---------------------------------------------------------------------------

def test_deduplication_semantic_wins():
    """If the same verse_idx appears in both lists, only the semantic copy is shown."""
    m = _import_builder()
    sem = _make_example(5, "Semantic source", "Semantic target", method="semantic")
    cov = _make_example(5, "Coverage source", "Coverage target", method="coverage")
    req = _make_request(semantic=[sem], coverage=[cov])
    result = m.PromptBuilder().build(req)
    # Semantic copy present
    assert "Semantic source" in result
    assert "Semantic target" in result
    # Coverage duplicate absent
    assert "Coverage source" not in result
    assert "Coverage target" not in result


# ---------------------------------------------------------------------------
# test_deduplication_no_double_verse
# ---------------------------------------------------------------------------

def test_deduplication_no_double_verse():
    """Each verse_idx must appear at most once in the prompt."""
    m = _import_builder()
    # Three semantic examples, two with same verse_idx
    a = _make_example(1, "Source A", "Target A", method="semantic")
    b = _make_example(1, "Source B", "Target B", method="semantic")
    c = _make_example(2, "Source C", "Target C", method="semantic")
    req = _make_request(semantic=[a, b, c])
    result = m.PromptBuilder().build(req)
    # verse_idx 1 should appear exactly once
    count_a = result.count("Source A")
    count_b = result.count("Source B")
    assert count_a + count_b == 1, (
        "verse_idx 1 appears more than once in the prompt"
    )
    assert "Source C" in result


# ---------------------------------------------------------------------------
# test_unk_spans_inserted_into_source_only
# ---------------------------------------------------------------------------

def test_unk_spans_inserted_into_source_only():
    """UNK markers must be inserted in the query source line, NOT in example targets."""
    m = _import_builder()
    # Example target contains the word "selah" — it must NOT gain a [UNK:…] wrapper
    ex = _make_example(
        1,
        "Praise the Lord selah amen",
        "Praise the Lord selah amen",
        method="semantic",
    )
    # The query source contains "selah" which is an unknown span
    source_text = "Praise the Lord selah forever"
    selah_start = source_text.index("selah")
    selah_end = selah_start + len("selah")
    req = _make_request(
        source_text=source_text,
        semantic=[ex],
        unk_spans=[UNKSpan(surface="selah", start_char=selah_start, end_char=selah_end)],
    )
    result = m.PromptBuilder().build(req)
    # The NOW-translate source line must have [UNK:selah]
    assert "[UNK:selah]" in result, "UNK marker not inserted in query source line"
    # The example target must be untouched
    assert "Praise the Lord selah amen" in result, (
        "Example target was modified (UNK marker inserted into example)"
    )


# ---------------------------------------------------------------------------
# test_unk_insertion_right_to_left_preserves_offsets
# ---------------------------------------------------------------------------

def test_unk_insertion_right_to_left_preserves_offsets():
    """Multiple UNK spans must be inserted right-to-left so earlier offsets stay valid."""
    m = _import_builder()
    # Two uncovered spans: "foo" at 0-3 and "baz" at 8-11
    source_text = "foo bar baz"
    req = _make_request(
        source_text=source_text,
        unk_spans=[
            UNKSpan(surface="foo", start_char=0, end_char=3),
            UNKSpan(surface="baz", start_char=8, end_char=11),
        ],
    )
    result = m.PromptBuilder().build(req)
    assert "[UNK:foo]" in result, "First UNK span not inserted"
    assert "[UNK:baz]" in result, "Second UNK span not inserted"
    # "bar" is covered — must appear as-is
    assert "bar" in result, "Covered token 'bar' was removed"
    # The resulting source line in the "Now translate" section:
    assert "[UNK:foo] bar [UNK:baz]" in result, (
        "Source line after UNK insertion is incorrect"
    )


# ---------------------------------------------------------------------------
# test_unk_insertion_single_span
# ---------------------------------------------------------------------------

def test_unk_insertion_single_span():
    """A single UNK span is replaced correctly."""
    m = _import_builder()
    source_text = "God created selah the heavens"
    span_start = source_text.index("selah")
    span_end = span_start + len("selah")
    req = _make_request(
        source_text=source_text,
        unk_spans=[UNKSpan(surface="selah", start_char=span_start, end_char=span_end)],
    )
    result = m.PromptBuilder().build(req)
    assert "[UNK:selah]" in result
    assert "God created" in result
    assert "the heavens" in result


# ---------------------------------------------------------------------------
# test_overlapping_spans_raise_value_error
# ---------------------------------------------------------------------------

def test_overlapping_spans_raise_value_error():
    """Overlapping UNK spans must raise ValueError loudly (not corrupt offsets)."""
    m = _import_builder()
    source_text = "hello world foo"
    req = _make_request(
        source_text=source_text,
        unk_spans=[
            UNKSpan(surface="hello world", start_char=0, end_char=11),
            UNKSpan(surface="world foo", start_char=6, end_char=15),
        ],
    )
    with pytest.raises(ValueError, match="[Oo]verlap"):
        m.PromptBuilder().build(req)


# ---------------------------------------------------------------------------
# test_invalid_span_start_ge_end_raises
# ---------------------------------------------------------------------------

def test_invalid_span_start_ge_end_raises():
    """A span with start_char >= end_char must raise ValueError."""
    m = _import_builder()
    source_text = "hello world"
    req = _make_request(
        source_text=source_text,
        unk_spans=[UNKSpan(surface="hello", start_char=5, end_char=5)],
    )
    with pytest.raises(ValueError, match="[Ii]nvalid"):
        m.PromptBuilder().build(req)


# ---------------------------------------------------------------------------
# test_invalid_span_out_of_range_raises
# ---------------------------------------------------------------------------

def test_invalid_span_out_of_range_raises():
    """A span with end_char beyond len(source_text) must raise ValueError."""
    m = _import_builder()
    source_text = "hi"
    req = _make_request(
        source_text=source_text,
        unk_spans=[UNKSpan(surface="hi", start_char=0, end_char=99)],
    )
    with pytest.raises(ValueError, match="[Ii]nvalid|[Oo]ut.of.range|[Bb]ound"):
        m.PromptBuilder().build(req)


# ---------------------------------------------------------------------------
# test_no_unk_spans_source_unchanged
# ---------------------------------------------------------------------------

def test_no_unk_spans_source_unchanged():
    """With no unk_spans the query source line must appear verbatim."""
    m = _import_builder()
    source_text = "God saw the light and it was good"
    req = _make_request(source_text=source_text, unk_spans=[])
    result = m.PromptBuilder().build(req)
    assert source_text in result, "Source text not found verbatim when no UNK spans"


# ---------------------------------------------------------------------------
# test_prompt_contains_now_translate_section
# ---------------------------------------------------------------------------

def test_prompt_contains_now_translate_section():
    """The prompt must contain a 'Now translate' section with the query source."""
    m = _import_builder()
    req = _make_request(source_text="In the beginning God created")
    result = m.PromptBuilder().build(req)
    assert "Now translate" in result or "now translate" in result.lower(), (
        "Prompt missing 'Now translate' section"
    )
    assert "In the beginning God created" in result


# ---------------------------------------------------------------------------
# test_examples_section_header_present
# ---------------------------------------------------------------------------

def test_examples_section_header_present():
    """The prompt must contain an 'Examples:' section header (from PROMPT_TEMPLATE)."""
    m = _import_builder()
    req = _make_request(
        semantic=[_make_example(1, "Source", "Target")],
    )
    result = m.PromptBuilder().build(req)
    assert "Examples" in result, "Missing 'Examples' section header in prompt"


# ---------------------------------------------------------------------------
# test_no_arbitrary_target_vocab_beyond_examples
# ---------------------------------------------------------------------------

def test_no_arbitrary_target_vocab_beyond_examples():
    """PromptBuilder must not inject arbitrary target vocabulary beyond the examples.

    The only target text in the prompt is the verbatim example targets.  No
    word list, no attested_vocab dump, no extra tokens.
    """
    m = _import_builder()
    # Example target contains only "Alpha Beta"
    ex = _make_example(1, "source", "Alpha Beta", method="semantic")
    req = _make_request(
        source_text="source text",
        semantic=[ex],
        # attested_vocab has extra words that must NOT appear as standalone vocab
    )
    # Override attested_vocab to include an extra word not in examples
    req.attested_vocab = frozenset(["Alpha", "Beta", "INJECTED_EXTRA"])
    result = m.PromptBuilder().build(req)
    # INJECTED_EXTRA must NOT appear in the prompt
    assert "INJECTED_EXTRA" not in result, (
        "PromptBuilder injected attested_vocab words beyond example text"
    )


# ---------------------------------------------------------------------------
# test_multiple_semantic_and_coverage_examples_order
# ---------------------------------------------------------------------------

def test_multiple_semantic_and_coverage_examples_order():
    """With multiple examples of each type, all semantic come before all coverage."""
    m = _import_builder()
    sem1 = _make_example(1, "Sem1 source", "Sem1 target", method="semantic")
    sem2 = _make_example(2, "Sem2 source", "Sem2 target", method="semantic")
    cov1 = _make_example(3, "Cov1 source", "Cov1 target", method="coverage")
    cov2 = _make_example(4, "Cov2 source", "Cov2 target", method="coverage")
    req = _make_request(semantic=[sem1, sem2], coverage=[cov1, cov2])
    result = m.PromptBuilder().build(req)
    pos_sem1 = result.index("Sem1 source")
    pos_sem2 = result.index("Sem2 source")
    pos_cov1 = result.index("Cov1 source")
    pos_cov2 = result.index("Cov2 source")
    assert pos_sem1 < pos_cov1, "Sem1 must be before Cov1"
    assert pos_sem2 < pos_cov1, "Sem2 must be before Cov1"
    assert pos_cov1 < pos_cov2, "Cov1 must be before Cov2 (order preserved)"


# ---------------------------------------------------------------------------
# test_source_label_and_translation_label_in_examples
# ---------------------------------------------------------------------------

def test_source_label_and_translation_label_in_examples():
    """Each example must be prefixed with [Source]: and [Translation]: labels."""
    m = _import_builder()
    ex = _make_example(1, "Source sentence", "Target sentence", method="semantic")
    req = _make_request(semantic=[ex])
    result = m.PromptBuilder().build(req)
    assert "[Source]:" in result, "Missing [Source]: label in example"
    assert "[Translation]:" in result, "Missing [Translation]: label in example"


# ---------------------------------------------------------------------------
# test_empty_examples_still_produces_valid_prompt
# ---------------------------------------------------------------------------

def test_empty_examples_still_produces_valid_prompt():
    """With no examples the prompt must still be well-formed (header + cue)."""
    m = _import_builder()
    req = _make_request(source_text="hello world", semantic=[], coverage=[])
    result = m.PromptBuilder().build(req)
    assert isinstance(result, str)
    assert result.rstrip().endswith("[Translation]:")
    assert "hello world" in result


# ---------------------------------------------------------------------------
# test_deterministic_output
# ---------------------------------------------------------------------------

def test_deterministic_output():
    """build() must return the same string for the same request every time."""
    m = _import_builder()
    sem = _make_example(1, "Sem source", "Sem target", method="semantic")
    cov = _make_example(2, "Cov source", "Cov target", method="coverage")
    req = _make_request(
        source_text="hello world",
        semantic=[sem],
        coverage=[cov],
    )
    result_a = m.PromptBuilder().build(req)
    result_b = m.PromptBuilder().build(req)
    assert result_a == result_b, "PromptBuilder is not deterministic"
