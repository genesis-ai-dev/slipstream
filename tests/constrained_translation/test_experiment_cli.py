"""tests/constrained_translation/test_experiment_cli.py

Focused offline TDD tests for the experiment CLI subcommands.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from constrained_translation.experiment.cli import build_parser, main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_corpus_dir(tmp_path):
    """Write minimal eng + two language corpora (12 lines each)."""
    eng = [
        "In the beginning God created the heavens and the earth",
        "The earth was without form and void darkness was deep",
        "God said let there be light and there was light",
        "God saw that the light was good and divided it",
        "God called the light Day and the darkness Night",
        "And there was evening and there was morning the first day",
        "God made the firmament above the waters",
        "God created great whales and every living creature",
        "God blessed them saying be fruitful and multiply",
        "God saw every thing that he had made and it was very good",
        "Thus the heavens and the earth were finished",
        "And on the seventh day God ended his work",
    ]
    mya = [
        "ဘုရားသည်ကောင်းကင်နှင့်မြေကြီးကိုဖန်ဆင်းတော်မူ၏",
        "မြေကြီးသည်အသွင်မရှိ",
        "ဘုရားသည်အလင်းရောင်ပေါ်စေဟုအမိန့်ချ",
        "ဘုရားသည်အလင်းကောင်းသည်ဟုမြင်",
        "ဘုရားသည်အလင်းကိုနေ့ဟုခေါ်",
        "ညဦးနှင့်နံနက်ပထမနေ့ဖြစ်လေ",
        "ဘုရားသည်မိုးကောင်းကင်ပြုလုပ်",
        "ဘုရားသည်ငါးကြီးများဖန်ဆင်း",
        "ဘုရားသည်ကောင်းကြီးပေးတော်မူ",
        "ဘုရားသည်ပြုလုပ်ခဲ့သမျှကောင်းသည်",
        "ကောင်းကင်နှင့်မြေပြီးဆုံး",
        "သတ္တမနေ့ဘုရားအလုပ်ချုပ်",
    ]
    npi = [
        "आदिमा परमेश्वरले आकाश र पृथ्वी सृष्टि गर्नुभयो",
        "पृथ्वी बेढंग र सुनसान थियो",
        "परमेश्वरले भन्नुभयो उज्यालो होस्",
        "परमेश्वरले उज्यालो राम्रो छ भनी देख्नुभयो",
        "परमेश्वरले उज्यालोलाई दिन भन्नुभयो",
        "साँझ र बिहान भएर पहिलो दिन भयो",
        "परमेश्वरले आकाशको गुम्बज बनाउनुभयो",
        "परमेश्वरले ठूला ठूला माछाहरू सृष्टि गर्नुभयो",
        "परमेश्वरले तिनीहरूलाई आशिष्ले भन्नुभयो",
        "परमेश्वरले गर्नुभएको सबै राम्रो थियो",
        "यसरी आकाश र पृथ्वी सम्पन्न भयो",
        "सातौँ दिन परमेश्वरले आफ्नो काम सकाउनुभयो",
    ]

    corpus_dir = tmp_path / "Corpus"
    corpus_dir.mkdir()

    (corpus_dir / "eng-engULB.txt").write_text("\n".join(eng) + "\n", encoding="utf-8")
    (corpus_dir / "mya-mya.txt").write_text("\n".join(mya) + "\n", encoding="utf-8")
    (corpus_dir / "npi-npiulb.txt").write_text("\n".join(npi) + "\n", encoding="utf-8")

    vref_dir = tmp_path / "benchmarks" / "data"
    vref_dir.mkdir(parents=True)
    vrefs = [f"GEN 1:{i+1}" for i in range(12)]
    (vref_dir / "vref.txt").write_text("\n".join(vrefs) + "\n", encoding="utf-8")

    return {
        "corpus_dir": str(corpus_dir),
        "vref_path": str(vref_dir / "vref.txt"),
        "output_dir": str(tmp_path / "exp_out"),
        "tmp_path": tmp_path,
    }


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_parser_returns_parser(self):
        p = build_parser()
        assert p is not None

    def test_subcommands_present(self):
        p = build_parser()
        # Parsing help for subcommands should not raise
        import argparse
        # If we try to parse 'prepare --help' it raises SystemExit(0)
        with pytest.raises(SystemExit) as exc_info:
            p.parse_args(["prepare", "--help"])
        assert exc_info.value.code == 0

    def test_run_help(self):
        p = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            p.parse_args(["run", "--help"])
        assert exc_info.value.code == 0

    def test_analyze_help(self):
        p = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            p.parse_args(["analyze", "--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# prepare subcommand tests
# ---------------------------------------------------------------------------

class TestCLIPrepare:
    def test_prepare_creates_manifests(self, tiny_corpus_dir):
        ret = main([
            "prepare",
            "--corpus-dir", tiny_corpus_dir["corpus_dir"],
            "--vref",        tiny_corpus_dir["vref_path"],
            "--output-dir",  tiny_corpus_dir["output_dir"],
            "--languages", "mya", "npi",
            "--n", "3",
            "--seed", "42",
        ])
        assert ret == 0
        out = Path(tiny_corpus_dir["output_dir"])
        assert (out / "mya" / "mya_manifest.jsonl").exists()
        assert (out / "npi" / "npi_manifest.jsonl").exists()

    def test_prepare_manifests_have_same_indices(self, tiny_corpus_dir):
        """Manifests for different languages share the same corpus indices."""
        main([
            "prepare",
            "--corpus-dir", tiny_corpus_dir["corpus_dir"],
            "--vref",        tiny_corpus_dir["vref_path"],
            "--output-dir",  tiny_corpus_dir["output_dir"],
            "--languages", "mya", "npi",
            "--n", "3",
            "--seed", "42",
        ])
        out = Path(tiny_corpus_dir["output_dir"])
        mya_items = [json.loads(l) for l in (out / "mya" / "mya_manifest.jsonl").read_text().splitlines() if l.strip()]
        npi_items = [json.loads(l) for l in (out / "npi" / "npi_manifest.jsonl").read_text().splitlines() if l.strip()]
        mya_idx = [i["corpus_idx"] for i in mya_items]
        npi_idx = [i["corpus_idx"] for i in npi_items]
        assert mya_idx == npi_idx, "Cross-language manifests should have same corpus indices"

    def test_prepare_filter_stats_written(self, tiny_corpus_dir):
        main([
            "prepare",
            "--corpus-dir", tiny_corpus_dir["corpus_dir"],
            "--vref",        tiny_corpus_dir["vref_path"],
            "--output-dir",  tiny_corpus_dir["output_dir"],
            "--languages", "mya",
            "--n", "3",
            "--seed", "42",
        ])
        stats = json.loads(
            (Path(tiny_corpus_dir["output_dir"]) / "mya" / "mya_filter_stats.json").read_text()
        )
        assert stats["sampled"] == 3

    def test_prepare_missing_corpus_returns_nonzero(self, tiny_corpus_dir):
        ret = main([
            "prepare",
            "--corpus-dir", "/nonexistent/path",
            "--vref",        tiny_corpus_dir["vref_path"],
            "--output-dir",  tiny_corpus_dir["output_dir"],
            "--languages", "mya",
            "--n", "3",
            "--seed", "42",
        ])
        assert ret != 0


# ---------------------------------------------------------------------------
# run subcommand tests
# ---------------------------------------------------------------------------

class TestCLIRun:
    def _prepare(self, tiny_corpus_dir, n=3, langs=("mya", "npi")):
        main([
            "prepare",
            "--corpus-dir", tiny_corpus_dir["corpus_dir"],
            "--vref",        tiny_corpus_dir["vref_path"],
            "--output-dir",  tiny_corpus_dir["output_dir"],
            "--languages", *langs,
            "--n", str(n),
            "--seed", "42",
        ])

    def test_run_creates_results(self, tiny_corpus_dir):
        self._prepare(tiny_corpus_dir)
        ret = main([
            "run",
            "--corpus-dir", tiny_corpus_dir["corpus_dir"],
            "--output-dir",  tiny_corpus_dir["output_dir"],
            "--languages", "mya", "npi",
            "--seed", "42",
        ])
        assert ret == 0
        out = Path(tiny_corpus_dir["output_dir"])
        assert (out / "mya" / "semantic_only" / "results.jsonl").exists()
        assert (out / "mya" / "semantic_coverage" / "results.jsonl").exists()
        assert (out / "npi" / "semantic_only" / "results.jsonl").exists()

    def test_run_without_prepare_fails(self, tiny_corpus_dir):
        ret = main([
            "run",
            "--corpus-dir", tiny_corpus_dir["corpus_dir"],
            "--output-dir",  tiny_corpus_dir["output_dir"],
            "--languages", "mya",
            "--seed", "42",
        ])
        assert ret != 0

    def test_run_resumable(self, tiny_corpus_dir):
        """Running twice should not error; second run skips completed."""
        self._prepare(tiny_corpus_dir)
        main([
            "run",
            "--corpus-dir", tiny_corpus_dir["corpus_dir"],
            "--output-dir",  tiny_corpus_dir["output_dir"],
            "--languages", "mya",
            "--seed", "42",
        ])
        ret = main([
            "run",
            "--corpus-dir", tiny_corpus_dir["corpus_dir"],
            "--output-dir",  tiny_corpus_dir["output_dir"],
            "--languages", "mya",
            "--seed", "42",
        ])
        assert ret == 0


# ---------------------------------------------------------------------------
# analyze subcommand tests
# ---------------------------------------------------------------------------

class TestCLIAnalyze:
    def _prepare_and_run(self, tiny_corpus_dir, langs=("mya", "npi")):
        main([
            "prepare",
            "--corpus-dir", tiny_corpus_dir["corpus_dir"],
            "--vref",        tiny_corpus_dir["vref_path"],
            "--output-dir",  tiny_corpus_dir["output_dir"],
            "--languages", *langs,
            "--n", "3",
            "--seed", "42",
        ])
        main([
            "run",
            "--corpus-dir", tiny_corpus_dir["corpus_dir"],
            "--output-dir",  tiny_corpus_dir["output_dir"],
            "--languages", *langs,
            "--seed", "42",
        ])

    def test_analyze_creates_json(self, tiny_corpus_dir, capsys):
        self._prepare_and_run(tiny_corpus_dir)
        ret = main([
            "analyze",
            "--output-dir",  tiny_corpus_dir["output_dir"],
            "--languages", "mya", "npi",
            "--seed", "42",
            "--n-boot", "100",
            "--n-sign-flip", "100",
        ])
        assert ret == 0
        out = Path(tiny_corpus_dir["output_dir"])
        assert (out / "analysis" / "macro_analysis.json").exists()
        assert (out / "analysis" / "mya_analysis.json").exists()
        assert (out / "analysis" / "npi_analysis.json").exists()

    def test_analyze_creates_markdown(self, tiny_corpus_dir):
        self._prepare_and_run(tiny_corpus_dir)
        main([
            "analyze",
            "--output-dir",  tiny_corpus_dir["output_dir"],
            "--languages", "mya", "npi",
            "--seed", "42",
            "--n-boot", "100",
            "--n-sign-flip", "100",
        ])
        md_path = Path(tiny_corpus_dir["output_dir"]) / "analysis" / "summary.md"
        assert md_path.exists()
        md = md_path.read_text()
        assert "mya" in md
        assert "npi" in md

    def test_analyze_stdout_json(self, tiny_corpus_dir, capsys):
        self._prepare_and_run(tiny_corpus_dir)
        main([
            "analyze",
            "--output-dir",  tiny_corpus_dir["output_dir"],
            "--languages", "mya", "npi",
            "--seed", "42",
            "--n-boot", "100",
            "--n-sign-flip", "100",
        ])
        captured = capsys.readouterr()
        summary = json.loads(captured.out)
        assert "macro_sem_only_chrf" in summary
        assert "per_lang" in summary
        assert "mya" in summary["per_lang"]


# ---------------------------------------------------------------------------
# End-to-end: prepare → run → analyze
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_pipeline(self, tiny_corpus_dir, capsys):
        """Full offline pipeline with FakeBackend for both languages."""
        out_dir = tiny_corpus_dir["output_dir"]
        corpus_dir = tiny_corpus_dir["corpus_dir"]
        vref = tiny_corpus_dir["vref_path"]

        assert main(["prepare", "--corpus-dir", corpus_dir, "--vref", vref,
                     "--output-dir", out_dir, "--languages", "mya", "npi",
                     "--n", "3", "--seed", "7"]) == 0

        assert main(["run", "--corpus-dir", corpus_dir,
                     "--output-dir", out_dir,
                     "--languages", "mya", "npi", "--seed", "7"]) == 0

        assert main(["analyze", "--output-dir", out_dir,
                     "--languages", "mya", "npi", "--seed", "7",
                     "--n-boot", "50", "--n-sign-flip", "50"]) == 0

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["macro_mean_diff"] is not None
        assert "pooled_diff_ci" in result
        assert len(result["pooled_diff_ci"]) == 2
