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

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..models import Answer, ScoredChunk
from ..providers.base import ProviderError
from ..retrieval.retriever import RetrievalResult
from ..store.vector_store import Filters
from ..util import detect_language
from .prompts import audit_markers, build_council_messages, split_two_layer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .answerer import Engine

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
    min_dense: float = 0.0,
) -> list[Seat]:
    """Decide which traditions are in the room, from one survey retrieval.

    Pure: takes a retrieval result and returns seats, so seating is testable
    without a provider and the caller controls how the survey was run.

    Works whose front matter carries no `tradition:` cannot hold a seat —
    there would be nothing to filter their position's retrieval by, and a
    seat that silently retrieves from the whole library is not a tradition
    speaking, it is the general answer wearing a label.

    `min_dense` should be the retriever's relevance floor. The survey keeps
    every hit as long as *some* passage cleared the floor, so without this a
    tradition can be seated on passages that are all below it and then find
    nothing to say when it retrieves under its own filter — seated, and
    unable to speak.
    """
    grouped: dict[str, list[ScoredChunk]] = {}
    for hit in result.hits:
        tradition = (hit.chunk.tradition or "").strip()
        if not tradition or hit.dense < min_dense:
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


# --------------------------------------------------------------------------
# Positions
# --------------------------------------------------------------------------


@dataclass
class Position:
    """One tradition's independent answer, sourced only from its own texts."""

    seat: Seat
    answer: Answer | None = None
    error: str = ""

    @property
    def spoke(self) -> bool:
        return self.answer is not None and bool(self.answer.plain.strip())

    def to_dict(self) -> dict:
        return {
            "tradition": self.seat.tradition,
            "philosophers": self.seat.philosophers,
            "seat_score": self.seat.score,
            "error": self.error,
            "answer": self.answer.to_dict() if self.answer else None,
        }


@dataclass
class Council:
    """The whole session: who sat, what each said, and what was raised against it."""

    question: str
    positions: list[Position] = field(default_factory=list)
    seats: list[Seat] = field(default_factory=list)
    lang: str = "en"
    survey_candidates: int = 0
    best_score: float = 0.0
    took_ms: int = 0
    model: str = ""
    provider: str = ""

    @property
    def held(self) -> bool:
        """A council of one is a monologue with extra steps."""
        return len([p for p in self.positions if p.spoke]) >= MIN_SEATS

    @property
    def spoken(self) -> list[Position]:
        return [p for p in self.positions if p.spoke]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "held": self.held,
            "lang": self.lang,
            "seats": [s.tradition for s in self.seats],
            "positions": [p.to_dict() for p in self.positions],
            "survey_candidates": self.survey_candidates,
            "best_score": round(self.best_score, 4),
            "took_ms": self.took_ms,
            "model": self.model,
            "provider": self.provider,
        }


def hold_council(
    engine: "Engine",
    question: str,
    *,
    seats: int = DEFAULT_SEATS,
    k: int = 4,
    lang: str = "",
    reader_note: str = "",
    chat_model: str = "",
    chat_provider: str = "",
) -> Council:
    """Run the survey, seat the traditions, and let each answer independently.

    Cost is the honest headline: this is one embedding call and N chat
    completions, where N is the number of seats. It is deliberately not
    streamed — three or four completions arriving interleaved is noise, and
    the value of the feature is in reading them side by side anyway.
    """
    from .answerer import AskOptions

    started = time.perf_counter()
    lang = lang or detect_language(question)

    # Resolve the backend before spending anything on retrieval, for the same
    # reason `Engine.ask` does: a bad provider name should fail immediately
    # rather than after the embedding work.
    backend = engine.chat_backend(
        AskOptions(chat_model=chat_model, chat_provider=chat_provider)
    )

    # Embed once. Every seat asks the identical question; only the filter
    # differs, and a filter does not change the query vector.
    query_vec = engine.provider.embed_query(question)

    # Warm the lazily-built BM25 table before the fan-out, so N threads do not
    # each decide to build their own copy of it.
    engine.retriever.bm25

    survey = engine.retriever.search(
        question, k=SURVEY_K, diversify=False, query_vec=query_vec
    )
    council = Council(
        question=question,
        lang=lang,
        survey_candidates=survey.n_candidates,
        best_score=survey.best_score,
        provider=backend.name,
    )
    council.seats = seat_council(
        survey, size=seats, min_dense=engine.settings.min_score
    )
    if not council.seats:
        council.took_ms = int((time.perf_counter() - started) * 1000)
        return council

    def run(seat: Seat) -> Position:
        return _position(
            engine, backend, question, seat, query_vec,
            k=k, lang=lang, reader_note=reader_note, chat_model=chat_model,
        )

    with ThreadPoolExecutor(max_workers=len(council.seats)) as pool:
        council.positions = list(pool.map(run, council.seats))

    spoke = council.spoken
    council.model = spoke[0].answer.model if spoke and spoke[0].answer else ""
    council.took_ms = int((time.perf_counter() - started) * 1000)
    return council


def _position(
    engine: "Engine",
    backend,
    question: str,
    seat: Seat,
    query_vec,
    *,
    k: int,
    lang: str,
    reader_note: str,
    chat_model: str,
) -> Position:
    """One seat: retrieve inside the tradition, then answer from that alone."""
    result = engine.retriever.search(
        question,
        k=k,
        filters=Filters(tradition=seat.tradition),
        query_vec=query_vec,
    )
    if not result.hits:
        # Seating uses the same floor, so this should not happen — but a
        # silent empty position would read as "this tradition has nothing to
        # say", which is a claim about philosophy rather than about the index.
        return Position(seat=seat, error="no passage from this tradition cleared the relevance floor")

    messages = build_council_messages(
        question, result.hits, tradition=seat.tradition, lang=lang, reader_note=reader_note
    )
    try:
        completion = backend.chat(
            messages,
            temperature=engine.settings.temperature,
            max_tokens=min(engine.settings.max_tokens, 1100),
            task="council",
            model=chat_model,
        )
    except ProviderError as exc:
        # One seat failing must not lose the other three. The council is
        # still worth reading with a gap in it, and the gap is reported.
        return Position(seat=seat, error=str(exc))

    text, invented = audit_markers(completion.text, {h.marker for h in result.hits})
    plain, academic = split_two_layer(text)
    return Position(
        seat=seat,
        answer=Answer(
            question=question,
            plain=plain,
            academic=academic,
            sources=list(result.hits),
            raw=completion.text,
            grounded=True,
            mode="council",
            provider=completion.provider or backend.name,
            model=completion.model,
            latency_ms=completion.latency_ms,
            usage=completion.usage,
            invented_markers=invented,
            truncated=completion.truncated,
        ),
    )
