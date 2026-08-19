"""The Council: seating, independent positions, and the objection.

The thing worth testing here is not that N model calls happen. It is that the
room is chosen by the evidence, that each tradition is genuinely confined to
its own texts, and that the objection is itself sourced rather than invented.
"""

from __future__ import annotations

import pytest

from philo.generation.council import MIN_SEATS, Seat, seat_council
from philo.models import Chunk, ScoredChunk
from philo.retrieval.retriever import RetrievalResult


def hit(tradition: str, philosopher: str, score: float, work: str = "") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=f"{philosopher}-{score}",
            work_id=work or f"{philosopher}-w",
            text="…",
            philosopher=philosopher,
            work_title=work or "A Work",
            tradition=tradition,
        ),
        score=score,
    )


def survey(*hits: ScoredChunk) -> RetrievalResult:
    return RetrievalResult(query="q", hits=list(hits), n_candidates=len(hits))


# --------------------------------------------------------------------------
# Seating
# --------------------------------------------------------------------------


def test_seats_go_to_the_traditions_the_library_can_actually_support():
    seats = seat_council(
        survey(
            hit("Stoicism", "Epictetus", 0.90),
            hit("Stoicism", "Marcus Aurelius", 0.80),
            hit("Daoism", "Laozi", 0.60),
            hit("Empiricism", "David Hume", 0.20),
        ),
        size=2,
    )
    assert [s.tradition for s in seats] == ["Stoicism", "Daoism"]


def test_one_decisive_passage_still_earns_a_seat():
    """Scoring on the mean of the top two must not halve a single strong hit."""
    seats = seat_council(
        survey(
            hit("Analytic", "Bertrand Russell", 0.95),
            hit("Daoism", "Laozi", 0.70),
            hit("Daoism", "Laozi", 0.70, work="other"),
        ),
        size=2,
    )
    assert seats[0].tradition == "Analytic"
    assert seats[0].score == pytest.approx(0.95)


def test_consistent_coverage_outranks_one_lucky_passage():
    seats = seat_council(
        survey(
            hit("Stoicism", "Epictetus", 0.80),
            hit("Stoicism", "Marcus Aurelius", 0.78),
            hit("Daoism", "Laozi", 0.82),
            hit("Daoism", "Laozi", 0.30, work="other"),
        ),
        size=1,
    )
    assert seats[0].tradition == "Stoicism"


def test_a_work_with_no_tradition_cannot_hold_a_seat():
    """There would be nothing to confine its position's retrieval to."""
    seats = seat_council(
        survey(
            hit("", "Anonymous", 0.99),
            hit("Stoicism", "Epictetus", 0.40),
        ),
        size=3,
    )
    assert [s.tradition for s in seats] == ["Stoicism"]


def test_a_seat_names_the_philosophers_behind_it():
    seats = seat_council(
        survey(
            hit("Stoicism", "Epictetus", 0.90),
            hit("Stoicism", "Marcus Aurelius", 0.85),
            hit("Stoicism", "Epictetus", 0.70, work="other"),
        ),
        size=1,
    )
    assert seats[0].philosophers == ["Epictetus", "Marcus Aurelius"]
    assert "Epictetus" in seats[0].label
    assert seats[0].n_passages == 3


def test_seating_is_capped_even_when_asked_for_more():
    seats = seat_council(
        survey(*[hit(f"T{i}", f"P{i}", 0.9 - i / 100) for i in range(9)]),
        size=99,
    )
    assert len(seats) == 4


def test_an_empty_survey_seats_nobody():
    assert seat_council(survey(), size=3) == []
    assert MIN_SEATS == 2


def test_a_tradition_below_the_floor_is_not_seated():
    """Seated-but-unable-to-speak is worse than not seated at all.

    The survey keeps every hit as long as *something* cleared the floor, so a
    tradition can appear in it on passages that are all below it.
    """
    strong = hit("Stoicism", "Epictetus", 0.90)
    strong.dense = 0.40
    weak = hit("Daoism", "Laozi", 0.85)
    weak.dense = 0.02

    seats = seat_council(survey(strong, weak), size=3, min_dense=0.12)
    assert [s.tradition for s in seats] == ["Stoicism"]


# --------------------------------------------------------------------------
# Holding the council
# --------------------------------------------------------------------------


from philo.config import Settings                     # noqa: E402
from philo.corpus.ingest import ingest                # noqa: E402
from philo.generation.answerer import Engine          # noqa: E402
from philo.generation.council import hold_council     # noqa: E402
from philo.providers.mock import MockProvider         # noqa: E402


# The offline provider embeds by hashed word overlap, so a question only
# reaches a tradition whose fixture text shares vocabulary with it. This one
# straddles both: "control" is Epictetus, "excellence" is Laozi.
COUNCIL_Q = "what is excellence and what is in our control?"


@pytest.fixture
def engine(settings: Settings, provider: MockProvider) -> Engine:
    ingest(settings, provider)
    return Engine(settings, provider)


def test_every_position_is_confined_to_its_own_tradition(engine: Engine):
    council = hold_council(engine, COUNCIL_Q, seats=2, k=3)
    assert council.held
    for position in council.spoken:
        traditions = {s.chunk.tradition for s in position.answer.sources}
        assert traditions == {position.seat.tradition}


def test_positions_are_independent_answers_not_one_synthesis(engine: Engine):
    council = hold_council(engine, COUNCIL_Q, seats=2, k=3)
    assert len(council.spoken) >= MIN_SEATS
    assert len({p.seat.tradition for p in council.spoken}) == len(council.spoken)
    for position in council.spoken:
        assert position.answer.mode == "council"
        assert position.answer.sources


def test_markers_are_local_to_each_position(engine: Engine):
    """Each position numbers its own sources from 1."""
    council = hold_council(engine, COUNCIL_Q, seats=2, k=3)
    for position in council.spoken:
        assert [s.marker for s in position.answer.sources] == list(
            range(1, len(position.answer.sources) + 1)
        )


def test_the_question_is_embedded_once_not_once_per_seat(engine: Engine):
    """N seats differ by filter, not by query — N embeddings would be waste."""
    calls = {"n": 0}
    original = engine.provider.embed_query

    def counted(text: str):
        calls["n"] += 1
        return original(text)

    engine.provider.embed_query = counted  # type: ignore[method-assign]
    council = hold_council(engine, COUNCIL_Q, seats=2, k=3)
    assert council.held
    assert calls["n"] == 1


def test_a_library_of_one_tradition_declines_to_stage_a_debate(engine: Engine, monkeypatch):
    monkeypatch.setattr(
        "philo.generation.council.seat_council",
        lambda result, **kw: [Seat(tradition="Stoicism", score=0.9, n_passages=3)],
    )
    council = hold_council(engine, COUNCIL_Q, seats=3, k=3)
    assert not council.held
    assert len(council.spoken) == 1


def test_one_failing_seat_does_not_lose_the_others(engine: Engine):
    from philo.providers.base import ProviderError

    original = engine.provider.chat
    state = {"first": True}

    def flaky(messages, **kwargs):
        if state["first"]:
            state["first"] = False
            raise ProviderError("rate limited")
        return original(messages, **kwargs)

    engine.provider.chat = flaky  # type: ignore[method-assign]
    council = hold_council(engine, COUNCIL_Q, seats=2, k=3)
    assert any(p.error for p in council.positions)
    assert any(p.spoke for p in council.positions)


def test_council_serialises_to_json(engine: Engine):
    import json

    council = hold_council(engine, COUNCIL_Q, seats=2, k=3)
    payload = json.loads(json.dumps(council.to_dict(), ensure_ascii=False))
    assert payload["held"] is True
    assert len(payload["positions"]) >= MIN_SEATS
    assert payload["positions"][0]["answer"]["sources"]
