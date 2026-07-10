"""tests/constrained_translation/test_experiment_manifest.py

Focused offline TDD tests for the experiment manifest builder.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from constrained_translation.experiment.manifest import (
    ManifestItem,
    build_manifest,
    load_manifest,
    _passes_filters,
    _digit_fraction,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_corpus(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_vrefs(path: Path, n: int) -> list[str]:
    vrefs = [f"GEN {i+1}:{i+1}" for i in range(n)]
    path.write_text("\n".join(vrefs) + "\n", encoding="utf-8")
    return vrefs


@pytest.fixture
def small_corpus(tmp_path):
    """20-line aligned corpus with a few empty/short/anomalous lines."""
    eng = [
        "In the beginning God created the heavens and the earth",   # 0 — good
        "The earth was without form and void darkness was upon",     # 1 — good
        "God said let there be light and there was light",           # 2 — good
        "God saw that the light was good and divided the light",     # 3 — good
        "God called the light Day and the darkness Night",           # 4 — good
        "And there was evening and there was morning the first day", # 5 — good
        "God made the firmament above the waters",                   # 6 — good
        "God created great whales and every living creature",        # 7 — good
        "God blessed them saying be fruitful and multiply",          # 8 — good
        "God saw every thing that he had made and it was very good", # 9 — good
        "Thus the heavens and the earth were finished",              # 10 — good
        "And on the seventh day God ended his work",                 # 11 — good
        "And God blessed the seventh day and sanctified it",         # 12 — good
        "",                                                          # 13 — empty eng
        "Short",                                                     # 14 — too short tgt
        "The waters brought forth abundantly after their kind",      # 15 — good
        "1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2",            # 16 — digit heavy
        "And God said let the earth bring forth grass herb seed",    # 17 — good
        "And it was so and it was good and evening morning",         # 18 — good
        "The fruit of the tree yielding seed after his kind",        # 19 — good
    ]
    tgt = [
        "Au commencement Dieu créa les cieux et la terre",
        "La terre était sans forme et vide les ténèbres",
        "Dieu dit que la lumière soit et la lumière fut",
        "Dieu vit que la lumière était bonne et la sépara",
        "Dieu appela la lumière Jour et les ténèbres Nuit",
        "Ce fut le soir et le matin du premier jour",
        "Dieu fit le firmament au-dessus des eaux",
        "Dieu créa les grands poissons et tout être vivant",
        "Dieu les bénit soyez féconds et multipliez-vous",
        "Dieu vit tout ce qu'il avait fait et c'était très bon",
        "Ainsi les cieux et la terre furent achevés",
        "Le septième jour Dieu acheva son œuvre",
        "Dieu bénit le septième jour et le sanctifia",
        "Non vide",                                                  # 13 — eng empty, tgt not
        "xyz",                                                       # 14 — too short
        "Les eaux produisirent en abondance des êtres vivants",
        "Un deux trois quatre cinq six sept huit neuf dix",
        "Dieu dit que la terre produise de la verdure",
        "Ce fut le soir et le matin du deuxième jour",
        "Les fruits des arbres portant leur semence",
    ]
    vref_path = tmp_path / "vref.txt"
    vrefs = _write_vrefs(vref_path, len(eng))

    eng_path = tmp_path / "eng.txt"
    tgt_path = tmp_path / "tgt.txt"
    _write_corpus(eng_path, eng)
    _write_corpus(tgt_path, tgt)

    return {
        "eng_path": eng_path,
        "tgt_path": tgt_path,
        "vref_path": vref_path,
        "eng": eng,
        "tgt": tgt,
        "vrefs": vrefs,
        "n_rows": len(eng),
    }


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------

class TestPassesFilters:
    def test_good_pair(self):
        ok, reason = _passes_filters("Hello world today", "Bonjour monde")
        assert ok is True
        assert reason == ""

    def test_empty_source(self):
        ok, reason = _passes_filters("", "Some target text here")
        assert not ok
        assert reason == "empty_source"

    def test_empty_target(self):
        ok, reason = _passes_filters("Some source text here", "")
        assert not ok
        assert reason == "empty_target"

    def test_whitespace_only_source(self):
        ok, reason = _passes_filters("   ", "Some target text here")
        assert not ok
        assert reason == "empty_source"

    def test_too_short_source(self):
        ok, reason = _passes_filters("Hi", "A valid and normal target sentence")
        assert not ok
        assert reason == "too_short"

    def test_too_short_target(self):
        ok, reason = _passes_filters("A valid and normal source sentence", "xy")
        assert not ok
        assert reason == "too_short"

    def test_len_ratio_extreme(self):
        src = "a" * 100
        tgt = "b" * 5000  # ratio = 50 > _MAX_LEN_RATIO=20
        ok, reason = _passes_filters(src, tgt)
        assert not ok
        assert reason == "len_ratio"

    def test_digit_heavy(self):
        src = "1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6"
        tgt = "Normal target text here"
        ok, reason = _passes_filters(src, tgt)
        assert not ok
        assert reason == "digit_heavy"


class TestDigitFraction:
    def test_empty(self):
        assert _digit_fraction("") == 0.0

    def test_all_digits(self):
        assert _digit_fraction("12345") == 1.0

    def test_no_digits(self):
        assert _digit_fraction("abcdef") == 0.0

    def test_half_digits(self):
        assert abs(_digit_fraction("12ab") - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# build_manifest tests
# ---------------------------------------------------------------------------

class TestBuildManifest:
    def test_basic(self, small_corpus, tmp_path):
        out = tmp_path / "manifest.jsonl"
        stats = tmp_path / "stats.json"
        items = build_manifest(
            eng_corpus_path=small_corpus["eng_path"],
            tgt_corpus_path=small_corpus["tgt_path"],
            vref_path=small_corpus["vref_path"],
            lang="tst",
            n=5,
            seed=42,
            output_path=out,
            stats_path=stats,
        )
        assert len(items) == 5
        assert out.exists()
        assert stats.exists()

    def test_deterministic_same_seed(self, small_corpus, tmp_path):
        """Same seed → same items both times."""
        out1 = tmp_path / "m1.jsonl"
        out2 = tmp_path / "m2.jsonl"
        items1 = build_manifest(
            eng_corpus_path=small_corpus["eng_path"],
            tgt_corpus_path=small_corpus["tgt_path"],
            vref_path=small_corpus["vref_path"],
            lang="tst", n=5, seed=42, output_path=out1,
        )
        items2 = build_manifest(
            eng_corpus_path=small_corpus["eng_path"],
            tgt_corpus_path=small_corpus["tgt_path"],
            vref_path=small_corpus["vref_path"],
            lang="tst", n=5, seed=42, output_path=out2,
        )
        assert [i.corpus_idx for i in items1] == [i.corpus_idx for i in items2]

    def test_different_seeds_differ(self, small_corpus, tmp_path):
        """Different seeds → different samples (with high probability)."""
        out1 = tmp_path / "m1.jsonl"
        out2 = tmp_path / "m2.jsonl"
        items1 = build_manifest(
            eng_corpus_path=small_corpus["eng_path"],
            tgt_corpus_path=small_corpus["tgt_path"],
            vref_path=small_corpus["vref_path"],
            lang="tst", n=5, seed=42, output_path=out1,
        )
        items2 = build_manifest(
            eng_corpus_path=small_corpus["eng_path"],
            tgt_corpus_path=small_corpus["tgt_path"],
            vref_path=small_corpus["vref_path"],
            lang="tst", n=5, seed=99, output_path=out2,
        )
        # With 14 eligible rows choosing 5, chance same == C(14,5) arrangements
        # Very unlikely to be identical.
        indices1 = [i.corpus_idx for i in items1]
        indices2 = [i.corpus_idx for i in items2]
        assert indices1 != indices2, "Different seeds should produce different samples"

    def test_exclude_idx_set(self, small_corpus, tmp_path):
        """exclude_idx == corpus_idx (held-out row identity)."""
        out = tmp_path / "manifest.jsonl"
        items = build_manifest(
            eng_corpus_path=small_corpus["eng_path"],
            tgt_corpus_path=small_corpus["tgt_path"],
            vref_path=small_corpus["vref_path"],
            lang="tst", n=5, seed=42, output_path=out,
        )
        for item in items:
            assert item.exclude_idx == item.corpus_idx

    def test_no_empty_rows(self, small_corpus, tmp_path):
        """Sampled items have non-empty source and target text."""
        out = tmp_path / "manifest.jsonl"
        items = build_manifest(
            eng_corpus_path=small_corpus["eng_path"],
            tgt_corpus_path=small_corpus["tgt_path"],
            vref_path=small_corpus["vref_path"],
            lang="tst", n=5, seed=42, output_path=out,
        )
        for item in items:
            assert item.source_text.strip()
            assert item.target_text.strip()

    def test_item_id_from_vref(self, small_corpus, tmp_path):
        """item_id comes from the vref file."""
        out = tmp_path / "manifest.jsonl"
        items = build_manifest(
            eng_corpus_path=small_corpus["eng_path"],
            tgt_corpus_path=small_corpus["tgt_path"],
            vref_path=small_corpus["vref_path"],
            lang="tst", n=5, seed=42, output_path=out,
        )
        vrefs = small_corpus["vrefs"]
        for item in items:
            assert item.item_id == vrefs[item.corpus_idx]

    def test_filter_stats_written(self, small_corpus, tmp_path):
        """Filter stats JSON contains expected keys."""
        out = tmp_path / "manifest.jsonl"
        stats_path = tmp_path / "stats.json"
        build_manifest(
            eng_corpus_path=small_corpus["eng_path"],
            tgt_corpus_path=small_corpus["tgt_path"],
            vref_path=small_corpus["vref_path"],
            lang="tst", n=5, seed=42, output_path=out, stats_path=stats_path,
        )
        stats = json.loads(stats_path.read_text())
        for key in ("total_corpus_rows", "empty_source", "eligible", "sampled"):
            assert key in stats
        assert stats["sampled"] == 5
        assert stats["empty_source"] >= 1   # row 13 has empty eng

    def test_n_exceeds_eligible_raises(self, small_corpus, tmp_path):
        """Requesting n > eligible rows raises ValueError."""
        out = tmp_path / "manifest.jsonl"
        with pytest.raises(ValueError, match="eligible"):
            build_manifest(
                eng_corpus_path=small_corpus["eng_path"],
                tgt_corpus_path=small_corpus["tgt_path"],
                vref_path=small_corpus["vref_path"],
                lang="tst", n=100, seed=42, output_path=out,
            )

    def test_cross_language_same_indices(self, small_corpus, tmp_path):
        """With shared exclude_corpus_indices, two languages sample same indices."""
        # Build eligible intersection manually (all same corpus here)
        from constrained_translation.experiment.manifest import (
            _load_corpus, _passes_filters,
        )
        eng_lines = _load_corpus(small_corpus["eng_path"])
        tgt_lines = _load_corpus(small_corpus["tgt_path"])
        shared = [
            i for i in range(len(eng_lines))
            if _passes_filters(eng_lines[i], tgt_lines[i])[0]
        ]

        out1 = tmp_path / "m1.jsonl"
        out2 = tmp_path / "m2.jsonl"
        items1 = build_manifest(
            eng_corpus_path=small_corpus["eng_path"],
            tgt_corpus_path=small_corpus["tgt_path"],
            vref_path=small_corpus["vref_path"],
            lang="tst1", n=5, seed=42, output_path=out1,
            exclude_corpus_indices=shared,
        )
        items2 = build_manifest(
            eng_corpus_path=small_corpus["eng_path"],
            tgt_corpus_path=small_corpus["tgt_path"],
            vref_path=small_corpus["vref_path"],
            lang="tst2", n=5, seed=42, output_path=out2,
            exclude_corpus_indices=shared,
        )
        assert [i.corpus_idx for i in items1] == [i.corpus_idx for i in items2]


class TestLoadManifest:
    def test_roundtrip(self, small_corpus, tmp_path):
        out = tmp_path / "manifest.jsonl"
        items = build_manifest(
            eng_corpus_path=small_corpus["eng_path"],
            tgt_corpus_path=small_corpus["tgt_path"],
            vref_path=small_corpus["vref_path"],
            lang="tst", n=5, seed=42, output_path=out,
        )
        loaded = load_manifest(out)
        assert len(loaded) == len(items)
        for orig, loaded_item in zip(items, loaded):
            assert orig.item_id == loaded_item.item_id
            assert orig.corpus_idx == loaded_item.corpus_idx
            assert orig.exclude_idx == loaded_item.exclude_idx
