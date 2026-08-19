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
