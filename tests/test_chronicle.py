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


# --------------------------------------------------------------------------
# Resurfacing
# --------------------------------------------------------------------------

from philo.chronicle import rhymes                      # noqa: E402
from philo.chronicle.resurface import Echo, _age_days   # noqa: E402
from philo.corpus.ingest import ingest                  # noqa: E402
from philo.providers.mock import MockProvider           # noqa: E402
from philo.store.vector_store import VectorStore        # noqa: E402

OLD = "2025-11-02T09:00:00+00:00"
TODAY = "2026-02-02"


@pytest.fixture
def indexed(settings, provider: MockProvider):
    ingest(settings, provider)
    return VectorStore(settings.index_dir).load()


def test_a_saved_passage_is_matched_by_its_vector_already_in_the_index(indexed, provider):
    """No second embedding call, and none stored: the index has it."""
    chunk = next(c for c in indexed.chunks if "disturbed" in c.text)
    book = Chronicle(Path("/dev/null"))
    book.entries = [Entry(kind="passage", text=chunk.text, chunk_id=chunk.id, created=OLD)]

    query = "why am I disturbed by things that happen?"
    echoes = rhymes(book, query, store=indexed, query_vec=provider.embed_query(query),
                    today=TODAY, vector_floor=0.05)
    assert len(echoes) == 1
    assert echoes[0].how == "vector"


def test_entries_with_no_chunk_fall_back_to_words(indexed, provider):
    book = Chronicle(Path("/dev/null"))
    book.entries = [Entry(kind="decision", text="should I take the job?",
                          note="the job in another city", created=OLD)]
    echoes = rhymes(book, "should I take the job or stay?", store=indexed,
                    query_vec=provider.embed_query("x"), today=TODAY)
    assert [e.how for e in echoes] == ["words"]


def test_one_shared_word_is_a_coincidence_not_an_echo(indexed, provider):
    book = Chronicle(Path("/dev/null"))
    book.entries = [Entry(kind="question", text="what is water?", created=OLD)]
    assert rhymes(book, "what is virtue, courage, water", store=indexed,
                  query_vec=provider.embed_query("x"), today=TODAY) == []


def test_vector_evidence_is_ranked_ahead_of_word_evidence(indexed, provider):
    """The two measures are not comparable, so they are not interleaved."""
    chunk = next(c for c in indexed.chunks if "disturbed" in c.text)
    book = Chronicle(Path("/dev/null"))
    book.entries = [
        Entry(kind="question", text="disturbed opinions things happen terrible", created=OLD),
        Entry(kind="passage", text=chunk.text, chunk_id=chunk.id, created=OLD),
    ]
    query = "disturbed opinions things happen terrible"
    echoes = rhymes(book, query, store=indexed, query_vec=provider.embed_query(query),
                    today=TODAY, vector_floor=0.05, limit=2)
    assert [e.how for e in echoes] == ["vector", "words"]


def test_something_saved_today_is_the_session_not_an_echo(indexed, provider):
    chunk = indexed.chunks[0]
    book = Chronicle(Path("/dev/null"))
    book.entries = [Entry(kind="passage", text=chunk.text, chunk_id=chunk.id,
                          created=f"{TODAY}T08:00:00+00:00")]
    assert rhymes(book, chunk.text, store=indexed, query_vec=provider.embed_query(chunk.text),
                  today=TODAY, vector_floor=0.0) == []


def test_resurfacing_works_with_no_index_at_all():
    book = Chronicle(Path("/dev/null"))
    book.entries = [Entry(kind="decision", text="move city or stay", created=OLD)]
    echoes = rhymes(book, "should I move city", today=TODAY)
    assert [e.how for e in echoes] == ["words"]


@pytest.mark.parametrize(
    "days,expected",
    [(3, "3 days ago"), (21, "3 weeks ago"), (95, "3 months ago"), (400, "a year ago")],
)
def test_age_is_stated_vaguely_because_the_exact_date_is_in_the_entry(days, expected):
    assert Echo(entry=Entry(), score=1.0, how="vector", age_days=days).when == expected


def test_an_unparseable_timestamp_is_treated_as_old_not_suppressed():
    assert _age_days("not a date", TODAY) > 1000


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------

from philo.chronicle.journal import log_decision, weekly_recap   # noqa: E402
from philo.generation.answerer import Engine                     # noqa: E402


# Offline embeddings are lexical, so a decision only retrieves anything if it
# shares vocabulary with the fixture texts — hence "control" and "disturbed".
DECISION = "someone disturbed me at work; is my reaction in my control?"


@pytest.fixture
def engine(settings, provider: MockProvider) -> Engine:
    ingest(settings, provider)
    return Engine(settings, provider)


def test_a_decision_is_recorded_with_what_the_texts_said(engine, tmp_path: Path):
    book = Chronicle(tmp_path / "me.jsonl")
    result = log_decision(engine, book, DECISION)

    assert result.grounded
    assert result.sources
    stored = Chronicle.load(tmp_path / "me.jsonl")
    assert len(stored) == 1
    assert stored.entries[0].kind == "decision"
    assert stored.entries[0].response
    assert stored.entries[0].citations[0]["work_title"]


def test_the_entry_is_kept_even_when_nothing_could_be_quoted(engine, tmp_path: Path, monkeypatch):
    """The record is the reader's; an empty retrieval is not a reason to lose it."""
    monkeypatch.setattr(engine.settings, "min_score", 0.99)
    book = Chronicle(tmp_path / "me.jsonl")
    result = log_decision(engine, book, "should I move to another city?")

    assert not result.grounded
    assert not result.sources
    assert len(Chronicle.load(tmp_path / "me.jsonl")) == 1


def test_no_model_call_is_made_without_sources(engine, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(engine.settings, "min_score", 0.99)
    monkeypatch.setattr(
        engine.provider, "chat",
        lambda *a, **kw: pytest.fail("the model must not be asked with no sources"),
    )
    log_decision(engine, Chronicle(tmp_path / "me.jsonl"), "anything at all")


def test_the_decision_prompt_refuses_to_give_a_verdict():
    from philo.generation.prompts import DECISION_SYSTEM

    assert "Do not tell them what to do" in DECISION_SYSTEM
    assert "fortune cookie" in DECISION_SYSTEM
    assert "The decision is the reader's" in DECISION_SYSTEM


def test_save_can_be_switched_off(engine, tmp_path: Path):
    book = Chronicle(tmp_path / "me.jsonl")
    log_decision(engine, book, DECISION, save=False)
    assert not (tmp_path / "me.jsonl").exists()


# --------------------------------------------------------------------------
# The weekly recap
# --------------------------------------------------------------------------


def test_an_empty_week_is_not_a_model_call(engine, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        engine.provider, "chat",
        lambda *a, **kw: pytest.fail("nothing happened this week; nothing to synthesise"),
    )
    recap = weekly_recap(engine, Chronicle(tmp_path / "me.jsonl"), today="2026-02-02")
    assert recap.empty
    assert not recap.grounded


def test_the_recap_quotes_the_readers_own_saves_first(engine, tmp_path: Path):
    """A recap that quotes back something they chose beats one that introduces
    a passage they have never seen."""
    chunk = next(c for c in engine.store.chunks if "disturbed" in c.text)
    book = Chronicle(tmp_path / "me.jsonl")
    book.add(Entry(kind="passage", text=chunk.text, chunk_id=chunk.id,
                   philosopher=chunk.philosopher, work_title=chunk.work_title,
                   created="2026-01-30T09:00:00+00:00"))
    book.add(Entry(kind="decision", text="should I answer the message tonight?",
                   created="2026-01-31T09:00:00+00:00"))

    recap = weekly_recap(engine, book, today="2026-02-02")
    assert recap.grounded
    assert recap.sources[0].chunk.id == chunk.id
    assert [s.marker for s in recap.sources] == list(range(1, len(recap.sources) + 1))
    assert len(recap.entries) == 2


def test_the_recap_only_resurfaces_things_from_before_the_week(engine, tmp_path: Path):
    """Something inside the span is already in the recap; calling it a
    resurfacing would be theatre."""
    chunk = next(c for c in engine.store.chunks if "disturbed" in c.text)
    book = Chronicle(tmp_path / "me.jsonl")
    book.add(Entry(kind="passage", text=chunk.text, chunk_id=chunk.id,
                   created="2026-01-31T09:00:00+00:00"))

    recap = weekly_recap(engine, book, today="2026-02-02")
    assert recap.echoes == []


def test_entries_outside_the_window_are_not_in_the_week(engine, tmp_path: Path):
    book = Chronicle(tmp_path / "me.jsonl")
    book.add(Entry(kind="question", text="old business", created="2025-06-01T09:00:00+00:00"))
    book.add(Entry(kind="question", text="this week", created="2026-02-01T09:00:00+00:00"))

    recap = weekly_recap(engine, book, today="2026-02-02")
    assert [e.text for e in recap.entries] == ["this week"]


def test_the_recap_prompt_forbids_a_manufactured_pattern():
    from philo.generation.prompts import RECAP_SYSTEM

    assert "no connecting thread, say that" in RECAP_SYSTEM
    assert "a false pattern is worse than" in RECAP_SYSTEM
    assert "no streak talk" in RECAP_SYSTEM
