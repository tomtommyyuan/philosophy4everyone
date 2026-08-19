"""The Chronicle: the record, what rhymes with it, and the recap.

The interesting properties are durability (an interrupted append must not
cost you the book) and honesty (resurfacing must not claim a connection the
texts do not support).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from philo.chronicle import Chronicle, Entry


def passage(text: str, **kw) -> Entry:
    return Entry(kind="passage", text=text, philosopher="Epictetus",
                 work_title="Enchiridion", **kw)


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------


def test_entries_survive_a_round_trip(tmp_path: Path):
    book = Chronicle(tmp_path / "me.jsonl")
    book.add(passage("Men are disturbed not by things.", chunk_id="c1"))
    book.add(Entry(kind="decision", text="take the job?", note="two cities"))

    reloaded = Chronicle.load(tmp_path / "me.jsonl")
    assert len(reloaded) == 2
    assert reloaded.counts() == {"passage": 1, "decision": 1, "question": 0}
    assert reloaded.entries[0].chunk_id == "c1"


def test_a_half_written_line_does_not_cost_you_the_book(tmp_path: Path):
    """Exactly what a crashed append leaves behind."""
    path = tmp_path / "me.jsonl"
    book = Chronicle(path)
    book.add(passage("first"))
    book.add(passage("second"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"kind": "passage", "text": "trunca')

    reloaded = Chronicle.load(path)
    assert [e.text for e in reloaded] == ["first", "second"]


def test_adding_never_rewrites_what_is_already_there(tmp_path: Path):
    path = tmp_path / "me.jsonl"
    book = Chronicle(path)
    book.add(passage("first"))
    before = path.read_text("utf-8")
    book.add(passage("second"))
    assert path.read_text("utf-8").startswith(before)


def test_deleting_is_the_one_operation_that_rewrites(tmp_path: Path):
    path = tmp_path / "me.jsonl"
    book = Chronicle(path)
    keep = book.add(passage("keep me"))
    drop = book.add(passage("drop me"))

    assert book.remove(drop.id) is True
    assert book.remove("nothing-with-this-id") is False
    assert [e.id for e in Chronicle.load(path)] == [keep.id]


def test_ids_are_stable_and_distinct(tmp_path: Path):
    a = Entry(kind="passage", text="same", created="2026-01-01T00:00:00+00:00")
    b = Entry(kind="passage", text="same", created="2026-01-01T00:00:00+00:00")
    c = Entry(kind="passage", text="other", created="2026-01-01T00:00:00+00:00")
    assert a.id == b.id
    assert a.id != c.id


def test_since_and_before_split_the_record_by_day(tmp_path: Path):
    book = Chronicle(tmp_path / "me.jsonl")
    book.add(passage("old", created="2026-01-01T09:00:00+00:00"))
    book.add(passage("new", created="2026-03-05T09:00:00+00:00"))
    assert [e.text for e in book.since("2026-02-01")] == ["new"]
    assert [e.text for e in book.before("2026-02-01")] == ["old"]


def test_a_missing_file_is_an_empty_book_not_an_error(tmp_path: Path):
    book = Chronicle.load(tmp_path / "never-written.jsonl")
    assert len(book) == 0
    assert book.counts()["passage"] == 0


def test_the_chronicle_lives_next_to_the_profiles(settings):
    assert settings.chronicle_dir.name == "chronicle"
    assert settings.chronicle_dir.parent == settings.profiles_dir.parent
