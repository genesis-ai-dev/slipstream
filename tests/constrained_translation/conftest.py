"""Shared fixtures for constrained_translation tests.

A tiny 12-verse aligned corpus lives in tmp files.
No corpus download; all tests run fully offline.
"""

import pytest

# ---------------------------------------------------------------------------
# Tiny aligned corpus — 12 English/French verse pairs
# ---------------------------------------------------------------------------

_SOURCE = [
    "In the beginning God created the heavens and the earth",
    "The earth was without form and void",
    "And God said let there be light",
    "And God saw that the light was good",
    "God called the light Day and the darkness Night",
    "And God created great whales in the sea",
    "And God blessed them saying be fruitful and multiply",
    "And God made the firmament above the waters",
    "God saw every thing that he had made and it was very good",
    "Thus the heavens and the earth were finished",
    "And on the seventh day God ended his work",
    "And God blessed the seventh day and sanctified it",
]

_TARGET = [
    "Au commencement Dieu créa les cieux et la terre",
    "La terre était sans forme et vide",
    "Et Dieu dit que la lumière soit",
    "Et Dieu vit que la lumière était bonne",
    "Dieu appela la lumière Jour et les ténèbres Nuit",
    "Et Dieu créa les grands poissons dans la mer",
    "Et Dieu les bénit disant soyez féconds et multipliez",
    "Et Dieu fit le firmament au-dessus des eaux",
    "Dieu vit tout ce qu'il avait fait et c'était très bon",
    "Ainsi les cieux et la terre furent achevés",
    "Et le septième jour Dieu acheva son œuvre",
    "Et Dieu bénit le septième jour et le sanctifia",
]


@pytest.fixture
def tiny_corpus_files(tmp_path):
    """Write the tiny aligned corpus to temp files.

    Returns (source_path: str, target_path: str).
    Indices are 0-based:  verse 0 = "In the beginning…"
    """
    src = tmp_path / "source.txt"
    tgt = tmp_path / "target.txt"
    src.write_text("\n".join(_SOURCE) + "\n", encoding="utf-8")
    tgt.write_text("\n".join(_TARGET) + "\n", encoding="utf-8")
    return str(src), str(tgt)
