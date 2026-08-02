"""Loading and chunking.

The chunker tests are the ones that matter most: a chunk that ends mid-argument
is how a citation-first system starts hallucinating, because the model receives
a conclusion whose premise was left in a neighbouring chunk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from philo.config import Settings
from philo.corpus.chunker import chunk_work, merge_small, pack_paragraphs, split_blocks, split_long
from philo.corpus.loader import infer_from_filename, load_file, load_library, parse_front_matter
from philo.util import char_weight, split_sentences


# --------------------------------------------------------------------------
# Front matter
# --------------------------------------------------------------------------


def test_front_matter_inline_list_and_aliases():
    meta, body = parse_front_matter(
        "---\n"
        "author: Zhuangzi\n"
        "title: Inner Chapters\n"
        "tags: [freedom, dreams, relativity]\n"
        "---\n"
        "The body.\n"
    )
    # `author`/`title` are accepted as aliases for philosopher/work.
    assert meta["philosopher"] == "Zhuangzi"
    assert meta["work"] == "Inner Chapters"
    assert meta["tags"] == ["freedom", "dreams", "relativity"]
    assert body.strip() == "The body."


def test_front_matter_block_list_and_quotes():
    meta, _ = parse_front_matter(
        "---\n"
        'philosopher: "Marcus Aurelius"\n'
        "work: Meditations\n"
        "tags:\n"
        "  - death\n"
        "  - duty\n"
        "---\n"
        "text\n"
    )
    assert meta["philosopher"] == "Marcus Aurelius"
    assert meta["tags"] == ["death", "duty"]


def test_missing_front_matter_falls_back_to_filename(tmp_path: Path):
    path = tmp_path / "Simone Weil - Gravity and Grace.txt"
    path.write_text("Attention is the rarest and purest form of generosity.", encoding="utf-8")
    loaded = load_file(path)
    assert loaded.work.philosopher == "Simone Weil"
    assert loaded.work.title == "Gravity and Grace"


def test_infer_from_filename_handles_em_dash():
    assert infer_from_filename(Path("Plato — Republic.md"))["work"] == "Republic"


def test_library_reports_errors_without_aborting(tmp_path: Path, library: Path):
    (library / "broken.md").write_text("---\nphilosopher: X\nwork: Y\n---\n\n   \n", encoding="utf-8")
    works, errors = load_library(library)
    assert len(works) == 2          # the two good files still loaded
    assert any("broken.md" in e for e in errors)


def test_duplicate_work_ids_are_disambiguated(tmp_path: Path):
    directory = tmp_path / "lib"
    directory.mkdir()
    fm = "---\nphilosopher: Plato\nwork: Apology\n---\n\nSome text long enough to keep.\n"
    (directory / "a.md").write_text(fm, encoding="utf-8")
    (directory / "b.md").write_text(fm, encoding="utf-8")
    works, errors = load_library(directory)
    ids = {w.work.id for w in works}
    # Two works must never share an id: citations would become ambiguous.
    assert len(ids) == 2
    assert any("duplicate" in e for e in errors)


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def test_sections_become_chunk_metadata(settings: Settings, library: Path):
    loaded = load_file(library / "epictetus-enchiridion.md")
    chunks = chunk_work(loaded.work, loaded.body, settings)
    sections = {c.section for c in chunks}
    assert sections == {"Chapter I", "Chapter V"}
    assert all(c.philosopher == "Epictetus" for c in chunks)
    assert all(c.work_title == "Enchiridion" for c in chunks)


def test_chunks_never_cross_a_section_boundary(settings: Settings, library: Path):
    loaded = load_file(library / "epictetus-enchiridion.md")
    for chunk in chunk_work(loaded.work, loaded.body, settings):
        # Text from Chapter V must never appear in a chunk labelled Chapter I.
        if chunk.section == "Chapter I":
            assert "Men are disturbed not by the things" not in chunk.text


def test_short_aphorisms_are_merged_not_left_alone(settings: Settings, library: Path):
    """One-line aphorisms retrieve well and explain nothing on their own."""
    loaded = load_file(library / "laozi-tao-te-ching.md")
    chunks = chunk_work(loaded.work, loaded.body, settings)
    by_section: dict[str, int] = {}
    for c in chunks:
        by_section[c.section] = by_section.get(c.section, 0) + 1
    # Chapter 22 is four one-line aphorisms; they must collapse into one chunk.
    assert by_section["Chapter 22"] == 1
    assert "The worn out becomes new" in next(c.text for c in chunks if c.section == "Chapter 22")


def test_merge_small_converges_on_long_aphorism_chains(settings: Settings):
    units = ["short." for _ in range(40)]
    merged = merge_small(list(units), settings)
    assert len(merged) < len(units)
    # Fixed-point iteration, not a single pass: nothing under the minimum
    # should survive when a neighbour had room for it.
    assert all(char_weight(u) >= settings.chunk_min_chars or len(merged) == 1 for u in merged)


def test_long_paragraph_splits_only_between_sentences(settings: Settings):
    sentences = [
        f"This is sentence number {i} and it carries a complete thought to its end."
        for i in range(60)
    ]
    paragraph = " ".join(sentences)
    pieces = split_long(paragraph, settings)

    assert len(pieces) > 1
    for piece in pieces:
        # Every piece must begin and end on a sentence boundary.
        assert piece.strip().endswith(".")
        for sentence in split_sentences(piece):
            assert sentence.strip() in paragraph


def test_no_chunk_exceeds_the_ceiling(settings: Settings, library: Path):
    for name in ("epictetus-enchiridion.md", "laozi-tao-te-ching.md"):
        loaded = load_file(library / name)
        for chunk in chunk_work(loaded.work, loaded.body, settings):
            assert char_weight(chunk.text) <= settings.chunk_max_chars


def test_overlap_does_not_push_a_piece_over_the_ceiling(settings: Settings):
    long_sentence = "A sentence of considerable length that keeps going and going " * 12
    paragraph = ". ".join([long_sentence.strip()] * 4) + "."
    for piece in split_long(paragraph, settings):
        assert char_weight(piece) <= settings.chunk_max_chars


def test_cjk_length_is_weighted_more_heavily():
    # 100 Chinese characters carry far more than 100 Latin ones; the chunker
    # would otherwise pack Chinese chunks with several times the content.
    assert char_weight("道" * 100) > char_weight("a" * 100)


def test_hard_wrapped_lines_are_unwrapped_into_paragraphs(settings: Settings):
    body = "## Book I\n\nThis line is wrapped\nacross three separate\nsource lines.\n"
    blocks = split_blocks(body)
    assert len(blocks) == 1
    assert blocks[0].paragraphs[0] == "This line is wrapped across three separate source lines."


def test_pack_paragraphs_prefers_target_size(settings: Settings):
    paragraphs = ["word " * 60 for _ in range(10)]  # ~300 chars each
    packed = pack_paragraphs(paragraphs, settings)
    assert packed
    assert all(char_weight(p) <= settings.chunk_max_chars for p in packed)
    assert max(char_weight(p) for p in packed) >= settings.chunk_min_chars


@pytest.mark.parametrize("text", ["", "   ", "\n\n\n"])
def test_empty_bodies_produce_no_chunks(settings: Settings, text: str):
    from philo.models import Work

    assert chunk_work(Work(id="x", philosopher="A", title="B"), text, settings) == []
