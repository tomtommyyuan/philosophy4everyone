"""The Council — several traditions answering the same question, then arguing.

`philo ask` returns one voice. That voice is grounded and cited, but it is
still a single synthesis, and synthesis is where a plurality of traditions
quietly becomes a consensus that none of them actually holds. Averaging
Epictetus and Nietzsche does not produce a third philosopher; it produces
mush with citations attached.

The council answers the same question **several times over, independently**,
each time filtered to one tradition and sourced only from that tradition's
own texts. Then it does the thing a chatbot never does: it turns the
strongest position around and looks for the sharpest objection the *other*
traditions can document. Disagreement is the content, not a failure to
converge.

Two design decisions are load-bearing:

*Seats are earned, not assigned.* There is no hardcoded list of traditions.
One broad retrieval decides which traditions actually have relevant passages
for this question, and only those sit. A question about induction seats Hume
and Russell; a question about grief does not.

*Markers are local to each position.* Every position re-retrieves under its
own filter and numbers its own sources `[1]`, `[2]`, … A global numbering
would be tidier on screen and would also let a marker in one tradition's
prose point at another tradition's text — which is exactly the kind of
quiet mis-citation the rest of this codebase exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import ScoredChunk
from ..retrieval.retriever import RetrievalResult

# A council of one is a monologue. Below this we decline rather than dress a
# single tradition up as a debate.
MIN_SEATS = 2
DEFAULT_SEATS = 3
MAX_SEATS = 4

# How wide the seating survey looks before deciding who is in the room. Larger
# than a normal answer's k because the point is coverage, not a shortlist.
SURVEY_K = 48


@dataclass
class Seat:
    """One tradition, and the evidence that earned it a place."""

    tradition: str
    philosophers: list[str] = field(default_factory=list)
    score: float = 0.0
    n_passages: int = 0

    @property
    def label(self) -> str:
        who = ", ".join(self.philosophers[:3])
        return f"{self.tradition} ({who})" if who else self.tradition


def seat_council(
    result: RetrievalResult,
    *,
    size: int = DEFAULT_SEATS,
) -> list[Seat]:
    """Decide which traditions are in the room, from one survey retrieval.

    Pure: takes a retrieval result and returns seats, so seating is testable
    without a provider and the caller controls how the survey was run.

    Works whose front matter carries no `tradition:` cannot hold a seat —
    there would be nothing to filter their position's retrieval by, and a
    seat that silently retrieves from the whole library is not a tradition
    speaking, it is the general answer wearing a label.
    """
    grouped: dict[str, list[ScoredChunk]] = {}
    for hit in result.hits:
        tradition = (hit.chunk.tradition or "").strip()
        if not tradition:
            continue
        grouped.setdefault(tradition, []).append(hit)

    seats: list[Seat] = []
    for tradition, hits in grouped.items():
        ranked = sorted(hits, key=lambda h: h.score, reverse=True)
        # Mean of the top two: one lucky passage should not outrank a
        # tradition with consistent coverage, but a tradition holding exactly
        # one decisive passage keeps its full score rather than being halved.
        top = ranked[:2]
        seats.append(
            Seat(
                tradition=tradition,
                philosophers=_unique([h.chunk.philosopher for h in ranked]),
                score=round(sum(h.score for h in top) / len(top), 4),
                n_passages=len(hits),
            )
        )

    seats.sort(key=lambda s: (-s.score, s.tradition))
    return seats[: max(1, min(size, MAX_SEATS))]


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out
