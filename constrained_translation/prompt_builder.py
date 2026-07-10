"""constrained_translation.prompt_builder — Few-shot prompt assembly (Task 7).

This module assembles the final few-shot prompt for one translation unit.

### Design notes

**UNK markers are instructions / input placeholders — not licensed model output.**

``[UNK:surface]`` markers that appear in the *query* ``[Source]:`` line are
deterministic substitutions inserted by this module *before* the prompt is
handed to the model.  They are part of the instruction that tells the model
"this span has no attested translation — leave it as-is or transliterate it."
They are NOT tokens the model is licensed to generate freely.  The grammar
produced by ``GrammarBuilder`` never includes ``[UNK:…]`` as a generatable
alternative (invariant I3).

### Example ordering (deterministic)

1. Semantic examples (from ``TranslationRequest.semantic_examples``) — in the
   order they are stored (preserved from ExampleSelector).
2. Coverage examples (from ``TranslationRequest.coverage_examples``) — in the
   order they are stored.

De-duplication by ``verse_idx`` is applied across the combined list: if a
verse index appears in both lists the *semantic* copy is kept and the coverage
copy is silently dropped.  If a verse index appears more than once within a
single list, the first occurrence wins.

### UNK span insertion (right-to-left)

Spans are applied to the query source string in **descending** ``start_char``
order so that inserting a marker at a later position does not shift the
character offsets of earlier spans.

**Overlap / invalid span detection** is loud:

- A span with ``start_char >= end_char`` raises ``ValueError``.
- A span with ``end_char > len(source_text)`` or ``start_char < 0`` raises
  ``ValueError``.
- Two spans that overlap (i.e. the earlier span's ``end_char`` > the later
  span's ``start_char`` after sorting) raise ``ValueError``.

### What this module does NOT inject

- It does NOT dump ``attested_vocab`` into the prompt as a word list.
- It does NOT modify example target strings.
- It does NOT call any backend or model.
"""

from __future__ import annotations

from constrained_translation.protocol import TranslationRequest, UNKSpan


# ---------------------------------------------------------------------------
# Module-level constant (plan §7.7 requirement)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
You are a translator. Translate the source sentence into the target language.
Only use vocabulary attested in the examples. \
Uncovered spans are already marked as [UNK:span] — \
these are deterministic input placeholders inserted before model generation, \
not tokens you should generate yourself.

Examples:
{examples}
Now translate:
[Source]: {source_with_unk}
[Translation]:\
"""


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Assemble a deterministic few-shot prompt from a TranslationRequest.

    Usage::

        prompt = PromptBuilder().build(request)

    The returned string ends with ``[Translation]:`` so the model knows exactly
    where to begin its output.
    """

    def build(self, request: TranslationRequest) -> str:
        """Return the assembled few-shot prompt string.

        Args:
            request: A fully populated ``TranslationRequest``.  The ``prompt``
                field on the request is *not* read — this method produces the
                value that should be stored there.

        Returns:
            A prompt string ending with ``[Translation]:``.

        Raises:
            ValueError: If any ``UNKSpan`` in ``request.unk_spans`` is invalid
                (start >= end, out-of-range) or if two spans overlap.
        """
        # 1. Collect examples: semantic first, then coverage; deduplicate.
        examples_block = self._render_examples(request)

        # 2. Build the query source line with UNK markers inserted.
        source_with_unk = self._insert_unk_markers(
            request.source_text, request.unk_spans
        )

        # 3. Fill the template.
        return PROMPT_TEMPLATE.format(
            examples=examples_block,
            source_with_unk=source_with_unk,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _render_examples(request: TranslationRequest) -> str:
        """Return the 'Examples' block as a string, de-duplicated by verse_idx.

        Semantic examples are emitted first (in their original order), then
        coverage examples (in their original order).  If a verse_idx has
        already been emitted, subsequent occurrences are skipped.
        """
        seen: set[int] = set()
        lines: list[str] = []

        for example in (*request.semantic_examples, *request.coverage_examples):
            if example.verse_idx in seen:
                continue
            seen.add(example.verse_idx)
            lines.append(f"[Source]: {example.source}")
            lines.append(f"[Translation]: {example.target}")
            lines.append("")  # blank line between examples

        return "\n".join(lines)

    @staticmethod
    def _insert_unk_markers(source_text: str, unk_spans: tuple[UNKSpan, ...]) -> str:
        """Return *source_text* with each UNKSpan replaced by ``[UNK:surface]``.

        Substitutions are applied right-to-left (descending ``start_char``) so
        earlier character offsets remain valid when multiple spans are present.

        Raises:
            ValueError: For invalid or overlapping spans.
        """
        if not unk_spans:
            return source_text

        n = len(source_text)

        # Validate every span individually first.
        for span in unk_spans:
            if span.start_char >= span.end_char:
                raise ValueError(
                    f"Invalid UNKSpan: start_char ({span.start_char}) >= "
                    f"end_char ({span.end_char}) for surface={span.surface!r}"
                )
            if span.start_char < 0 or span.end_char > n:
                raise ValueError(
                    f"Invalid UNKSpan: out-of-range offsets "
                    f"[{span.start_char}, {span.end_char}) for source of length {n}; "
                    f"surface={span.surface!r}"
                )

        # Sort descending by start_char for right-to-left application.
        sorted_spans = sorted(unk_spans, key=lambda s: s.start_char, reverse=True)

        # Check for overlaps in the sorted list.
        # After sorting descending, overlap occurs when sorted_spans[i].start_char
        # < sorted_spans[i-1].end_char  (i.e. a later span starts before the
        # previous one ends).
        for i in range(1, len(sorted_spans)):
            prev = sorted_spans[i - 1]  # higher start_char
            curr = sorted_spans[i]      # lower start_char
            if curr.end_char > prev.start_char:
                raise ValueError(
                    f"Overlapping UNKSpans: "
                    f"[{curr.start_char}, {curr.end_char}) surface={curr.surface!r} "
                    f"overlaps with "
                    f"[{prev.start_char}, {prev.end_char}) surface={prev.surface!r}"
                )

        # Apply substitutions right-to-left.
        result = source_text
        for span in sorted_spans:
            marker = f"[UNK:{span.surface}]"
            result = result[: span.start_char] + marker + result[span.end_char :]

        return result
