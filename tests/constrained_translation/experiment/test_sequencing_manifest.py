"""tests/constrained_translation/experiment/test_sequencing_manifest.py

Strict TDD tests for the sequencing manifest builder.
Written BEFORE the implementation (RED phase).

Required behaviour (per docs/plans/2026-07-10-slipstream-sequencing.md §Task 1):
- Shared source indices across mya, npi, ckb, tpi.
- Four disjoint pools after normalized-source equivalence expansion:
    seed (default 10), acquisition (40), fixed eval (60), remaining (all other eligible).
- Structural marker filtering (<range>) consistent with existing manifest.py.
- Deterministic seed.
- Stable manifest digest.
- Source/target row provenance preserved.
- Reject missing or misaligned corpora.
- Fixed eval never enters seed, acquisition, or remaining pool.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from constrained_translation.experiment.sequencing_manifest import (
    SequencingManifest,
    SequencingPoolItem,
    build_sequencing_manifest,
    load_sequencing_manifest,
    manifest_digest,
    save_sequencing_manifest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LANGUAGES = ["mya", "npi", "ckb", "tpi"]


def _eng_line(i: int) -> str:
    """Generate a unique, long-enough English source line for row i."""
    phrases = [
        "In the beginning God created the heavens and the earth",
        "The earth was without form and void and darkness was upon the face",
        "God said let there be light and there was light in the world",
        "God saw that the light was good and he divided it from darkness",
        "God called the light Day and the darkness he called Night",
        "And there was evening and morning the first day of creation",
        "God made the firmament to divide the waters above from below",
        "God created great whales and every living creature that moves",
        "God blessed them saying be fruitful and multiply and fill the earth",
        "God saw every thing that he had made and it was very good indeed",
        "Thus the heavens and the earth were finished in all their host",
        "On the seventh day God rested from all his work that he had done",
        "God blessed the seventh day and sanctified it because he rested",
        "These are the generations of the heavens and of the earth created",
        "The Lord God formed man of the dust of the ground and breathed life",
        "The Lord God planted a garden eastward in Eden and there put man",
        "Out of the ground God made every tree that is pleasant to the sight",
        "The river went out of Eden to water the garden and from thence",
        "The Lord God took the man and put him in the garden to dress it",
        "Of every tree of the garden you may freely eat the fruit thereof",
        "But of the tree of knowledge of good and evil you shall not eat",
        "The Lord God said it is not good that man should be alone herein",
        "Out of the ground God formed every beast of the field and fowl",
        "Adam gave names to all cattle and to the fowl and every beast",
        "The Lord God caused a deep sleep to fall upon Adam while he slept",
        "God took one of Adam's ribs and closed up the flesh in its place",
        "Of the rib God made a woman and brought her unto the man Adam",
        "Adam said this is now bone of my bones and flesh of my flesh truly",
        "Therefore shall a man leave his father and mother and cleave to wife",
        "They were both naked the man and his wife and were not ashamed",
    ]
    base = phrases[i % len(phrases)]
    if i >= len(phrases):
        base = f"{base} verse {i}"
    return base


def _tgt_line(lang: str, i: int) -> str:
    """Generate a unique target line for language lang, row i."""
    return f"[{lang}] translation of verse {i} text here for completeness"


@pytest.fixture
def small_corpora(tmp_path) -> dict:
    """
    30-row, 4-language synthetic corpus with these deliberate anomalies:

    Row 5:  English source is "<range>"  → excluded entirely (eng marker)
    Row 20: English normalized source == normalized(row 2)
            ("god said let there be light" appears twice) → equivalence group {2, 20}
    Row 24: ckb target is "<range>"  → excluded because shared eligibility requires
            every language to pass filters
    Row 27: English source is "  <RANGE>  " (whitespace/case) → excluded

    Eligible rows (30 - 4 excluded): 26 rows.
    Equivalence groups: {2, 20} plus 24 singletons = 25 groups total.
    """
    # Build English corpus lines
    eng_lines = [_eng_line(i) for i in range(30)]

    # Anomaly: row 5 — eng marker
    eng_lines[5] = "<range>"
    # Anomaly: row 20 — normalized source same as row 2
    # Row 2 normalizes to: "god said let there be light and there was light in the world"
    # This sentence (with punctuation variations) normalizes identically.
    eng_lines[20] = "God said: let there be light! And there was light, in the world."
    # Anomaly: row 27 — whitespace-padded, uppercase marker
    eng_lines[27] = "  <RANGE>  "

    # Build per-language corpora
    tgt_lines: dict[str, list[str]] = {}
    for lang in LANGUAGES:
        lines = [_tgt_line(lang, i) for i in range(30)]
        # Anomaly: row 24 — ckb target is a corpus marker
        if lang == "ckb":
            lines[24] = "<range>"
        tgt_lines[lang] = lines

    # Write files
    vref_lines = [f"GEN {i + 1}:{i + 1}" for i in range(30)]
    vref_path = tmp_path / "vref.txt"
    vref_path.write_text("\n".join(vref_lines) + "\n", encoding="utf-8")

    eng_path = tmp_path / "eng.txt"
    eng_path.write_text("\n".join(eng_lines) + "\n", encoding="utf-8")

    lang_paths: dict[str, Path] = {}
    for lang in LANGUAGES:
        p = tmp_path / f"{lang}.txt"
        p.write_text("\n".join(tgt_lines[lang]) + "\n", encoding="utf-8")
        lang_paths[lang] = p

    return {
        "eng_path": eng_path,
        "lang_paths": lang_paths,
        "vref_path": vref_path,
        "eng_lines": eng_lines,
        "tgt_lines": tgt_lines,
        "vref_lines": vref_lines,
        "n_rows": 30,
    }


def _build(corpora: dict, tmp_path: Path, *, seed: int = 42,
           seed_size: int = 3, acq_size: int = 5, eval_size: int = 7) -> SequencingManifest:
    """Helper: build manifest from the small_corpora fixture."""
    return build_sequencing_manifest(
        eng_corpus_path=corpora["eng_path"],
        lang_corpus_paths={lang: corpora["lang_paths"][lang] for lang in LANGUAGES},
        vref_path=corpora["vref_path"],
        languages=LANGUAGES,
        seed_size=seed_size,
        acq_size=acq_size,
        eval_size=eval_size,
        random_seed=seed,
    )


# ---------------------------------------------------------------------------
# Pool item data-model tests
# ---------------------------------------------------------------------------

class TestSequencingPoolItem:
    def test_has_corpus_idx(self):
        item = SequencingPoolItem(
            corpus_idx=3,
            item_id="GEN 4:4",
            source_text="some source text here",
            target_texts={"mya": "mya text", "npi": "npi text"},
        )
        assert item.corpus_idx == 3

    def test_has_item_id(self):
        item = SequencingPoolItem(
            corpus_idx=3,
            item_id="GEN 4:4",
            source_text="some source text here",
            target_texts={"mya": "mya text"},
        )
        assert item.item_id == "GEN 4:4"

    def test_has_source_text(self):
        item = SequencingPoolItem(
            corpus_idx=3,
            item_id="GEN 4:4",
            source_text="some source text here",
            target_texts={},
        )
        assert item.source_text == "some source text here"

    def test_has_target_texts_dict(self):
        tts = {"mya": "mya text", "ckb": "ckb text"}
        item = SequencingPoolItem(
            corpus_idx=3,
            item_id="GEN 4:4",
            source_text="some source text",
            target_texts=tts,
        )
        assert item.target_texts == tts


# ---------------------------------------------------------------------------
# Build: structure and counts
# ---------------------------------------------------------------------------

class TestBuildSequencingManifestBasic:
    def test_returns_sequencing_manifest(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path)
        assert isinstance(m, SequencingManifest)

    def test_four_pools_present(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path)
        assert hasattr(m, "seed")
        assert hasattr(m, "acquisition")
        assert hasattr(m, "fixed_eval")
        assert hasattr(m, "remaining")

    def test_seed_at_least_seed_size(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path, seed_size=3, acq_size=5, eval_size=7)
        assert len(m.seed) >= 3

    def test_acquisition_at_least_acq_size(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path, seed_size=3, acq_size=5, eval_size=7)
        assert len(m.acquisition) >= 5

    def test_fixed_eval_at_least_eval_size(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path, seed_size=3, acq_size=5, eval_size=7)
        assert len(m.fixed_eval) >= 7

    def test_remaining_nonempty(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path, seed_size=3, acq_size=5, eval_size=7)
        assert len(m.remaining) > 0

    def test_pool_items_are_sequencing_pool_item(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path)
        for pool in (m.seed, m.acquisition, m.fixed_eval, m.remaining):
            for item in pool:
                assert isinstance(item, SequencingPoolItem)

    def test_manifest_carries_languages(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path)
        assert set(m.languages) == set(LANGUAGES)

    def test_manifest_carries_random_seed(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path, seed=99)
        assert m.random_seed == 99

    def test_manifest_carries_pool_size_params(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path, seed_size=3, acq_size=5, eval_size=7)
        assert m.seed_size == 3
        assert m.acq_size == 5
        assert m.eval_size == 7


# ---------------------------------------------------------------------------
# Disjoint pools: every eligible row in exactly one pool
# ---------------------------------------------------------------------------

class TestDisjointPools:
    def _all_indices(self, m: SequencingManifest) -> list[int]:
        indices = []
        for pool in (m.seed, m.acquisition, m.fixed_eval, m.remaining):
            for item in pool:
                indices.append(item.corpus_idx)
        return indices

    def test_pools_are_disjoint(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path)
        all_indices = self._all_indices(m)
        assert len(all_indices) == len(set(all_indices)), (
            "Duplicate corpus_idx found across pools — pools are not disjoint"
        )

    def test_all_eligible_rows_covered(self, small_corpora, tmp_path):
        """Every eligible row must appear in exactly one pool."""
        m = _build(small_corpora, tmp_path)
        all_indices = set(self._all_indices(m))

        # Manually compute expected eligible set (rows not excluded by any filter)
        # Excluded: row 5 (eng marker), row 20 (equiv to 2, but still eligible — just grouped),
        #           row 24 (ckb target marker), row 27 (eng marker with whitespace/caps).
        # Note row 20 is NOT excluded — it passes filters; it's just in the same group as row 2.
        excluded = {5, 24, 27}  # only truly filtered-out rows
        expected_eligible = set(range(30)) - excluded
        assert all_indices == expected_eligible, (
            f"Pool indices {all_indices} do not match expected eligible {expected_eligible}"
        )

    def test_no_excluded_rows_in_any_pool(self, small_corpora, tmp_path):
        """Rows filtered by corpus marker must never appear in any pool."""
        m = _build(small_corpora, tmp_path)
        excluded = {5, 24, 27}
        all_indices = set(self._all_indices(m))
        assert not (all_indices & excluded), (
            f"Excluded rows {excluded & all_indices} appeared in a pool"
        )


# ---------------------------------------------------------------------------
# Fixed-eval isolation
# ---------------------------------------------------------------------------

class TestFixedEvalIsolation:
    def test_fixed_eval_not_in_seed(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path)
        eval_idx = {item.corpus_idx for item in m.fixed_eval}
        seed_idx = {item.corpus_idx for item in m.seed}
        assert not (eval_idx & seed_idx), "Fixed eval rows leaked into seed"

    def test_fixed_eval_not_in_acquisition(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path)
        eval_idx = {item.corpus_idx for item in m.fixed_eval}
        acq_idx = {item.corpus_idx for item in m.acquisition}
        assert not (eval_idx & acq_idx), "Fixed eval rows leaked into acquisition"

    def test_fixed_eval_not_in_remaining(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path)
        eval_idx = {item.corpus_idx for item in m.fixed_eval}
        rem_idx = {item.corpus_idx for item in m.remaining}
        assert not (eval_idx & rem_idx), "Fixed eval rows leaked into remaining"


# ---------------------------------------------------------------------------
# Normalized-source equivalence groups
# ---------------------------------------------------------------------------

class TestEquivalenceGrouping:
    def test_equiv_group_rows_land_in_same_pool(self, small_corpora, tmp_path):
        """Rows 2 and 20 have the same normalized source — must be in the same pool."""
        m = _build(small_corpora, tmp_path, seed=42)
        # Find which pool contains row 2
        row2_pool = None
        row20_pool = None
        for pool_name, pool in [("seed", m.seed), ("acquisition", m.acquisition),
                                 ("fixed_eval", m.fixed_eval), ("remaining", m.remaining)]:
            for item in pool:
                if item.corpus_idx == 2:
                    row2_pool = pool_name
                if item.corpus_idx == 20:
                    row20_pool = pool_name
        assert row2_pool is not None, "Row 2 not found in any pool"
        assert row20_pool is not None, "Row 20 not found in any pool"
        assert row2_pool == row20_pool, (
            f"Rows 2 and 20 have identical normalized source but landed in "
            f"different pools ({row2_pool!r} vs {row20_pool!r})"
        )

    def test_equiv_group_rows_land_in_same_pool_different_seed(self, small_corpora, tmp_path):
        """Grouping holds regardless of random seed."""
        for seed in (0, 1, 7, 13, 99):
            m = _build(small_corpora, tmp_path, seed=seed)
            row2_pool = None
            row20_pool = None
            for pool_name, pool in [("seed", m.seed), ("acquisition", m.acquisition),
                                     ("fixed_eval", m.fixed_eval), ("remaining", m.remaining)]:
                for item in pool:
                    if item.corpus_idx == 2:
                        row2_pool = pool_name
                    if item.corpus_idx == 20:
                        row20_pool = pool_name
            assert row2_pool == row20_pool, (
                f"Seed={seed}: rows 2 and 20 in different pools "
                f"({row2_pool!r} vs {row20_pool!r})"
            )


# ---------------------------------------------------------------------------
# Structural marker filtering
# ---------------------------------------------------------------------------

class TestMarkerFiltering:
    def test_eng_range_marker_excluded(self, small_corpora, tmp_path):
        """Row 5 (eng '<range>') must not appear in any pool."""
        m = _build(small_corpora, tmp_path)
        all_indices = set()
        for pool in (m.seed, m.acquisition, m.fixed_eval, m.remaining):
            all_indices.update(item.corpus_idx for item in pool)
        assert 5 not in all_indices, "Row 5 (eng '<range>') appeared in a pool"

    def test_eng_range_marker_uppercase_whitespace_excluded(self, small_corpora, tmp_path):
        """Row 27 (eng '  <RANGE>  ') must not appear in any pool."""
        m = _build(small_corpora, tmp_path)
        all_indices = set()
        for pool in (m.seed, m.acquisition, m.fixed_eval, m.remaining):
            all_indices.update(item.corpus_idx for item in pool)
        assert 27 not in all_indices, "Row 27 (eng '<RANGE>' with whitespace) appeared in a pool"

    def test_target_range_marker_excluded_from_shared(self, small_corpora, tmp_path):
        """Row 24 (ckb target '<range>') must not appear in any pool
        because shared eligibility requires ALL languages to pass filters."""
        m = _build(small_corpora, tmp_path)
        all_indices = set()
        for pool in (m.seed, m.acquisition, m.fixed_eval, m.remaining):
            all_indices.update(item.corpus_idx for item in pool)
        assert 24 not in all_indices, (
            "Row 24 (ckb target '<range>') appeared in a pool — "
            "cross-language filtering is broken"
        )

    def test_no_pool_item_has_marker_source(self, small_corpora, tmp_path):
        """No pool item's source_text should be a corpus marker."""
        m = _build(small_corpora, tmp_path)
        for pool in (m.seed, m.acquisition, m.fixed_eval, m.remaining):
            for item in pool:
                assert item.source_text.strip().lower() != "<range>", (
                    f"Pool item {item.item_id} has '<range>' source_text"
                )

    def test_no_pool_item_has_marker_target(self, small_corpora, tmp_path):
        """No pool item's target_texts should contain a corpus marker."""
        m = _build(small_corpora, tmp_path)
        for pool in (m.seed, m.acquisition, m.fixed_eval, m.remaining):
            for item in pool:
                for lang, tgt in item.target_texts.items():
                    assert tgt.strip().lower() != "<range>", (
                        f"Pool item {item.item_id} lang={lang} has '<range>' target_text"
                    )


# ---------------------------------------------------------------------------
# Shared indices across all languages
# ---------------------------------------------------------------------------

class TestSharedIndices:
    def test_every_pool_item_has_all_language_targets(self, small_corpora, tmp_path):
        """Each SequencingPoolItem must have target_texts for all languages."""
        m = _build(small_corpora, tmp_path)
        for pool in (m.seed, m.acquisition, m.fixed_eval, m.remaining):
            for item in pool:
                for lang in LANGUAGES:
                    assert lang in item.target_texts, (
                        f"Item {item.item_id} (idx={item.corpus_idx}) "
                        f"missing target for lang={lang!r}"
                    )

    def test_same_corpus_idx_used_for_all_langs(self, small_corpora, tmp_path):
        """The corpus index must correspond to the same verse across languages."""
        m = _build(small_corpora, tmp_path)
        eng_lines = small_corpora["eng_lines"]
        tgt_lines = small_corpora["tgt_lines"]
        for pool in (m.seed, m.acquisition, m.fixed_eval, m.remaining):
            for item in pool:
                idx = item.corpus_idx
                assert item.source_text == eng_lines[idx], (
                    f"source_text mismatch at corpus_idx={idx}"
                )
                for lang in LANGUAGES:
                    assert item.target_texts[lang] == tgt_lines[lang][idx], (
                        f"target_text mismatch for lang={lang} at corpus_idx={idx}"
                    )


# ---------------------------------------------------------------------------
# Row provenance
# ---------------------------------------------------------------------------

class TestRowProvenance:
    def test_source_text_matches_corpus(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path)
        eng_lines = small_corpora["eng_lines"]
        for pool in (m.seed, m.acquisition, m.fixed_eval, m.remaining):
            for item in pool:
                assert item.source_text == eng_lines[item.corpus_idx]

    def test_target_text_matches_corpus(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path)
        tgt_lines = small_corpora["tgt_lines"]
        for pool in (m.seed, m.acquisition, m.fixed_eval, m.remaining):
            for item in pool:
                for lang in LANGUAGES:
                    assert item.target_texts[lang] == tgt_lines[lang][item.corpus_idx]

    def test_item_id_from_vref(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path)
        vrefs = small_corpora["vref_lines"]
        for pool in (m.seed, m.acquisition, m.fixed_eval, m.remaining):
            for item in pool:
                assert item.item_id == vrefs[item.corpus_idx], (
                    f"item_id={item.item_id!r} does not match vref {vrefs[item.corpus_idx]!r} "
                    f"at corpus_idx={item.corpus_idx}"
                )


# ---------------------------------------------------------------------------
# Deterministic seed
# ---------------------------------------------------------------------------

class TestDeterministicSeed:
    def _pool_indices(self, m: SequencingManifest) -> dict[str, list[int]]:
        return {
            "seed": sorted(item.corpus_idx for item in m.seed),
            "acquisition": sorted(item.corpus_idx for item in m.acquisition),
            "fixed_eval": sorted(item.corpus_idx for item in m.fixed_eval),
            "remaining": sorted(item.corpus_idx for item in m.remaining),
        }

    def test_same_seed_same_pools(self, small_corpora, tmp_path):
        m1 = _build(small_corpora, tmp_path, seed=42)
        m2 = _build(small_corpora, tmp_path, seed=42)
        assert self._pool_indices(m1) == self._pool_indices(m2)

    def test_different_seeds_differ(self, small_corpora, tmp_path):
        m1 = _build(small_corpora, tmp_path, seed=42)
        m2 = _build(small_corpora, tmp_path, seed=7)
        # Very unlikely to be identical; with 26 eligible rows and 4 pools this should differ
        assert self._pool_indices(m1) != self._pool_indices(m2), (
            "Different seeds produced identical pool assignments"
        )


# ---------------------------------------------------------------------------
# Stable manifest digest
# ---------------------------------------------------------------------------

class TestManifestDigest:
    def test_digest_present_and_nonempty(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path)
        d = manifest_digest(m)
        assert isinstance(d, str)
        assert len(d) > 0

    def test_digest_stable_same_inputs(self, small_corpora, tmp_path):
        m1 = _build(small_corpora, tmp_path, seed=42)
        m2 = _build(small_corpora, tmp_path, seed=42)
        assert manifest_digest(m1) == manifest_digest(m2)

    def test_digest_changes_with_different_seed(self, small_corpora, tmp_path):
        m1 = _build(small_corpora, tmp_path, seed=42)
        m2 = _build(small_corpora, tmp_path, seed=7)
        assert manifest_digest(m1) != manifest_digest(m2)

    def test_manifest_carries_digest(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path)
        # The manifest object itself should expose a digest attribute
        assert hasattr(m, "digest")
        assert isinstance(m.digest, str)
        assert len(m.digest) > 0

    def test_stored_digest_matches_computed(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path)
        assert m.digest == manifest_digest(m)


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_creates_file(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path)
        out = tmp_path / "seq_manifest.json"
        save_sequencing_manifest(m, out)
        assert out.exists()

    def test_load_roundtrip_pool_sizes(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path)
        out = tmp_path / "seq_manifest.json"
        save_sequencing_manifest(m, out)
        loaded = load_sequencing_manifest(out)
        assert len(loaded.seed) == len(m.seed)
        assert len(loaded.acquisition) == len(m.acquisition)
        assert len(loaded.fixed_eval) == len(m.fixed_eval)
        assert len(loaded.remaining) == len(m.remaining)

    def test_load_roundtrip_indices(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path)
        out = tmp_path / "seq_manifest.json"
        save_sequencing_manifest(m, out)
        loaded = load_sequencing_manifest(out)
        for pool_name in ("seed", "acquisition", "fixed_eval", "remaining"):
            orig_idx = sorted(i.corpus_idx for i in getattr(m, pool_name))
            load_idx = sorted(i.corpus_idx for i in getattr(loaded, pool_name))
            assert orig_idx == load_idx, f"Pool {pool_name} indices mismatch after roundtrip"

    def test_load_roundtrip_digest(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path)
        out = tmp_path / "seq_manifest.json"
        save_sequencing_manifest(m, out)
        loaded = load_sequencing_manifest(out)
        assert loaded.digest == m.digest

    def test_load_roundtrip_target_texts(self, small_corpora, tmp_path):
        m = _build(small_corpora, tmp_path)
        out = tmp_path / "seq_manifest.json"
        save_sequencing_manifest(m, out)
        loaded = load_sequencing_manifest(out)
        for pool_name in ("seed", "acquisition", "fixed_eval", "remaining"):
            orig_items = {i.corpus_idx: i for i in getattr(m, pool_name)}
            for item in getattr(loaded, pool_name):
                assert item.target_texts == orig_items[item.corpus_idx].target_texts


# ---------------------------------------------------------------------------
# Rejection of bad inputs
# ---------------------------------------------------------------------------

class TestRejection:
    def test_missing_eng_corpus_raises(self, small_corpora, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            build_sequencing_manifest(
                eng_corpus_path=tmp_path / "nonexistent_eng.txt",
                lang_corpus_paths={lang: small_corpora["lang_paths"][lang] for lang in LANGUAGES},
                vref_path=small_corpora["vref_path"],
                languages=LANGUAGES,
                seed_size=3,
                acq_size=5,
                eval_size=7,
                random_seed=42,
            )

    def test_missing_lang_corpus_raises(self, small_corpora, tmp_path):
        bad_paths = dict(small_corpora["lang_paths"])
        bad_paths["ckb"] = tmp_path / "nonexistent_ckb.txt"
        with pytest.raises((FileNotFoundError, OSError)):
            build_sequencing_manifest(
                eng_corpus_path=small_corpora["eng_path"],
                lang_corpus_paths=bad_paths,
                vref_path=small_corpora["vref_path"],
                languages=LANGUAGES,
                seed_size=3,
                acq_size=5,
                eval_size=7,
                random_seed=42,
            )

    def test_misaligned_corpus_raises(self, small_corpora, tmp_path):
        """A language corpus with fewer rows than eng must raise ValueError."""
        short_ckb = tmp_path / "short_ckb.txt"
        short_ckb.write_text("only one line\n", encoding="utf-8")
        bad_paths = {lang: small_corpora["lang_paths"][lang] for lang in LANGUAGES}
        bad_paths["ckb"] = short_ckb
        with pytest.raises(ValueError, match="misalign|length|rows|align"):
            build_sequencing_manifest(
                eng_corpus_path=small_corpora["eng_path"],
                lang_corpus_paths=bad_paths,
                vref_path=small_corpora["vref_path"],
                languages=LANGUAGES,
                seed_size=3,
                acq_size=5,
                eval_size=7,
                random_seed=42,
            )

    def test_insufficient_eligible_rows_raises(self, small_corpora, tmp_path):
        """Requesting more rows than are eligible must raise ValueError."""
        with pytest.raises(ValueError, match="eligible|insufficient|enough"):
            build_sequencing_manifest(
                eng_corpus_path=small_corpora["eng_path"],
                lang_corpus_paths={lang: small_corpora["lang_paths"][lang] for lang in LANGUAGES},
                vref_path=small_corpora["vref_path"],
                languages=LANGUAGES,
                seed_size=10,
                acq_size=20,
                eval_size=50,  # 80 total > 26 eligible → should raise
                random_seed=42,
            )
