"""Bringing back what rhymes with what you are asking now.

"You saved this on grief three months ago" is the line that makes a record
feel like it remembers you. Getting there needs a similarity measure, and the
obvious implementation — embed every entry when it is written — costs an API
call per save and a second index to keep in step with the first.

It is unnecessary. A **saved passage is already in the index**: its vector is
sitting in the store under its chunk id, and the question being asked has
just been embedded for retrieval anyway. So the comparison is a dot product
between two vectors that both already exist. Zero extra calls.

Entries with no chunk — decisions, questions, passages whose work has since
left the library — fall back to word overlap, which also costs nothing.

The two measures are **not mixed into one ranking**. A cosine and a token
overlap are on incomparable scales, and blending them would produce an order
that means nothing (the same mistake the retriever's `minmax` exists to
avoid). Vector matches are stronger evidence, so they come first as a group;
word matches follow as a group; each is sorted within itself and each has its
own floor.

Recency is disqualifying, not helpful. Something saved an hour ago is not an
echo, it is the session you are in — so an entry has to be old enough to have
been forgotten before it can be brought back.

The vector threshold is best set by `floor_from()`, not by the constant
below. Cosine ranges are a property of the embedding model — the offline
mock's hashed vectors top out around 0.3 for near-verbatim text where a real
embedding model would say 0.8 — so any absolute number is right for one
provider and wrong for the next. What is portable is a *comparison*: a saved
passage comes back if this question would have retrieved it anyway. That
needs no tuning and means something the reader can be told in one sentence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

from ..util import dot, tokenize
from .store import Chronicle, Entry

# Fallback for callers with no retrieval to hand — see `floor_from()`, which
# is what the CLI and the web actually use.
VECTOR_FLOOR = 0.38

# Share of the current question's meaningful words that appear in the entry.
LEXICAL_FLOOR = 0.34
MIN_SHARED_WORDS = 2

# An entry younger than this is the conversation you are having, not an echo.
MIN_AGE_DAYS = 2


def floor_from(result, *, settings=None) -> float:
    """The bar a saved passage must clear: would this question have found it?

    The weakest passage that actually made it into the answer sets the bar.
    Anything scoring at least that much would have earned a place in the
    sources, so bringing it back is not a second, looser notion of relevance
    bolted on beside the retriever's — it is the retriever's own.
    """
    floor = getattr(settings, "min_score", 0.0) if settings is not None else 0.0
    hits = getattr(result, "hits", None) or []
    if hits:
        return max(floor, min(h.dense for h in hits))
    return max(floor, VECTOR_FLOOR) if floor else VECTOR_FLOOR


@dataclass
class Echo:
    """One old entry that rhymes with the current question."""

    entry: Entry
    score: float
    how: str            # "vector" | "words"
    age_days: int

    @property
    def when(self) -> str:
        """`three months ago` — vague on purpose; the exact date is in the entry."""
        days = self.age_days
        if days < 14:
            return f"{days} days ago"
        if days < 60:
            return f"{max(2, days // 7)} weeks ago"
        if days < 365:
            return f"{max(2, days // 30)} months ago"
        years = days // 365
        return "a year ago" if years == 1 else f"{years} years ago"

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "how": self.how,
            "age_days": self.age_days,
            "when": self.when,
            "entry": self.entry.to_dict(),
        }


def rhymes(
    book: Chronicle,
    query: str,
    *,
    store=None,
    query_vec: Sequence[float] | None = None,
    limit: int = 2,
    today: str = "",
    vector_floor: float = VECTOR_FLOOR,
    lexical_floor: float = LEXICAL_FLOOR,
    min_age_days: int = MIN_AGE_DAYS,
) -> list[Echo]:
    """Entries worth bringing back, strongest evidence first.

    `store` and `query_vec` are both optional: without them every entry is
    compared by words, which is worse but still free and still honest about
    which measure it used.
    """
    today = today or date.today().isoformat()
    rows = _by_chunk(store) if store is not None else {}
    terms = set(tokenize(query))

    vector_hits: list[Echo] = []
    word_hits: list[Echo] = []

    for entry in book:
        age = _age_days(entry.created, today)
        if age < min_age_days:
            continue

        row = rows.get(entry.chunk_id) if entry.chunk_id else None
        if row is not None and query_vec is not None:
            # Both vectors are L2-normalised in the store, so the dot product
            # is the cosine.
            score = float(dot(query_vec, store.vector_at(row)))
            if score >= vector_floor:
                vector_hits.append(Echo(entry=entry, score=score, how="vector", age_days=age))
            continue

        score, shared = _overlap(terms, entry)
        if shared >= MIN_SHARED_WORDS and score >= lexical_floor:
            word_hits.append(Echo(entry=entry, score=score, how="words", age_days=age))

    vector_hits.sort(key=lambda e: e.score, reverse=True)
    word_hits.sort(key=lambda e: e.score, reverse=True)
    return (vector_hits + word_hits)[:limit]


def _by_chunk(store) -> dict[str, int]:
    return {chunk.id: i for i, chunk in enumerate(store.chunks)}


def _overlap(terms: set[str], entry: Entry) -> tuple[float, int]:
    """What share of the question's words this entry already contains.

    Coverage of the *query*, not Jaccard: a long saved passage should not be
    penalised for being long when it contains everything being asked about.
    """
    if not terms:
        return 0.0, 0
    text = " ".join(filter(None, (entry.text, entry.note, entry.citation)))
    shared = terms & set(tokenize(text))
    return len(shared) / len(terms), len(shared)


def _age_days(created: str, today: str) -> int:
    try:
        then = datetime.fromisoformat(created).date()
        now = date.fromisoformat(today)
    except (TypeError, ValueError):
        return 10_000  # unparseable: treat as old rather than suppressing it
    return max(0, (now - then).days)
