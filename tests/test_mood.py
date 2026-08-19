"""The check-in.

The claim this feature makes is about *retrieval*, not about prose: that
naming a feeling in modern words fails against a library of Victorian
translations, and that widening the query with the words those translations
actually use fixes it. So that is what the tests are about.
"""

from __future__ import annotations

import collections
import pathlib

import pytest

from philo.config import Settings
from philo.corpus.ingest import ingest
from philo.generation.answerer import Engine
from philo.generation.mood import MOODS, read_mood, resolve, retrieval_query
from philo.providers.mock import MockProvider
from philo.util import tokenize


# The offline embedder matches on shared words, and the two test fixtures are
# a short Enchiridion and a shorter Tao Te Ching. A mood's period terms alone
# do not overlap them, so the grounded tests supply a reason written in the
# fixtures' own vocabulary. The real library is exercised separately, below.
REASON = "someone disturbed me and I lost my opinion of what is in my control"


@pytest.fixture
def engine(settings: Settings, provider: MockProvider) -> Engine:
    ingest(settings, provider)
    return Engine(settings, provider)


# --------------------------------------------------------------------------
# The vocabulary bridge
# --------------------------------------------------------------------------


def test_a_mood_is_findable_by_key_or_by_either_label():
    assert resolve("angry").key == "angry"
    assert resolve("Worried").key == "worried"
    assert resolve("生气").key == "angry"
    assert resolve("  ASHAMED ").key == "ashamed"
    assert resolve("elated") is None
    assert resolve("") is None


def test_the_query_carries_both_the_readers_words_and_the_texts():
    """The terms make a passage findable; the reason decides which one."""
    query = retrieval_query(resolve("angry"), "my colleague took credit for my work")
    assert "wrath" in query
    assert "colleague took credit" in query


def test_a_mood_with_no_reason_still_has_something_to_retrieve_on():
    query = retrieval_query(resolve("lonely"), "   ")
    assert "solitude" in query
    assert query.strip().endswith("retirement")


def test_every_mood_widens_to_terms_it_does_not_already_contain():
    for mood in MOODS:
        terms = {t for term in mood.terms for t in tokenize(term)}
        assert terms, mood.key
        assert mood.key not in terms, f"{mood.key} lists itself; that widens nothing"


def test_the_period_terms_are_actually_in_the_real_library():
    """Guards the whole premise: terms nobody wrote widen the query to nothing.

    Runs against `library/` when it is present — the test fixtures are two
    short works and cannot speak to this.
    """
    library = pathlib.Path(__file__).resolve().parent.parent / "library"
    if not library.is_dir():
        pytest.skip("the real library is not present")
    words = collections.Counter(
        tokenize("\n".join(p.read_text(errors="ignore") for p in library.glob("*.md")))
    )
    for mood in MOODS:
        found = sum(words[t] for term in mood.terms for t in tokenize(term))
        assert found >= 25, f"{mood.key}: its period terms appear only {found} times"


def test_worried_is_the_case_that_justifies_the_bridge():
    """"Worried" does not occur in this library at all; "fear" occurs constantly."""
    library = pathlib.Path(__file__).resolve().parent.parent / "library"
    if not library.is_dir():
        pytest.skip("the real library is not present")
    words = collections.Counter(
        tokenize("\n".join(p.read_text(errors="ignore") for p in library.glob("*.md")))
    )
    assert words["worried"] == 0
    assert words["fear"] > 100


# --------------------------------------------------------------------------
# Reading a mood
# --------------------------------------------------------------------------


def test_a_check_in_is_grounded_and_cites_real_passages(engine: Engine):
    reading = read_mood(engine, "worried", reason=REASON)
    assert reading.grounded
    assert reading.sources
    assert reading.feeling
    assert reading.schools
    assert not reading.invented_markers


def test_the_reason_is_optional(engine: Engine):
    """A tap on a card with nothing typed must still produce a reading."""
    reading = read_mood(engine, "restless")
    assert reading.reason == ""
    assert reading.mood == "restless"
    assert reading.feeling


def test_an_unknown_mood_is_rejected_with_the_list(engine: Engine):
    with pytest.raises(ValueError) as excinfo:
        read_mood(engine, "hangry")
    assert "angry" in str(excinfo.value)


def test_nothing_retrievable_means_no_model_call(engine: Engine, monkeypatch):
    monkeypatch.setattr(engine.settings, "min_score", 0.99)
    monkeypatch.setattr(
        engine.provider, "chat",
        lambda *a, **kw: pytest.fail("a check-in with no sources must not reach the model"),
    )
    reading = read_mood(engine, "sad")
    assert not reading.grounded
    assert "library" in reading.feeling


def test_the_check_in_serialises(engine: Engine):
    import json

    reading = read_mood(engine, "angry", reason=REASON)
    payload = json.loads(json.dumps(reading.to_dict(), ensure_ascii=False))
    assert payload["mood"] == "angry"
    assert payload["sources"]


def test_the_prompt_refuses_to_be_therapy():
    from philo.generation.prompts import MOOD_SYSTEM

    assert "Not therapy" in MOOD_SYSTEM
    assert "do not congratulate them for" in MOOD_SYSTEM
    assert "do not reconcile them" in MOOD_SYSTEM
