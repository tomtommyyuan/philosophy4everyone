"""Core data types.

Everything that flows through the pipeline — ingestion, storage, retrieval,
generation — is one of these.  They are plain dataclasses with explicit
`to_dict`/`from_dict` so the on-disk index stays human-readable JSON.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable


# --------------------------------------------------------------------------
# Source material
# --------------------------------------------------------------------------


@dataclass
class Work:
    """A single source text: one philosopher, one book, one translation.

    The metadata here is what makes citation possible.  Every chunk carries a
    reference back to its Work, and every sentence the model is allowed to
    assert has to trace to one of these.
    """

    id: str
    philosopher: str
    title: str
    # Optional Chinese display names — the UI is bilingual.
    philosopher_zh: str = ""
    title_zh: str = ""
    translator: str = ""
    year: str = ""              # free-form: "c. 175 CE", "1785", "-350"
    tradition: str = ""         # Stoicism, Daoism, Existentialism, ...
    era: str = ""               # Ancient / Medieval / Modern / Contemporary
    language: str = "en"
    rights: str = "public-domain"
    source: str = ""            # e.g. "Project Gutenberg #2680"
    source_url: str = ""
    tags: list[str] = field(default_factory=list)
    path: str = ""              # file it was loaded from

    @property
    def label(self) -> str:
        """`Marcus Aurelius, Meditations` — used in citation lines."""
        return f"{self.philosopher}, {self.title}"

    @property
    def label_zh(self) -> str:
        p = self.philosopher_zh or self.philosopher
        t = self.title_zh or self.title
        return f"{p}《{t}》"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Work":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# --------------------------------------------------------------------------
# Chunks
# --------------------------------------------------------------------------


@dataclass
class Chunk:
    """One retrievable unit of text — ideally one complete argument.

    `section` and `chunk_index` exist so a citation can say *where* in the book
    a passage came from, not just which book.
    """

    id: str
    work_id: str
    text: str
    section: str = ""           # "Book II", "Chapter 3", "§17"
    chunk_index: int = 0
    # Denormalised from Work so retrieval results are self-describing and the
    # index stays greppable without a join.
    philosopher: str = ""
    philosopher_zh: str = ""
    work_title: str = ""
    work_title_zh: str = ""
    translator: str = ""
    tradition: str = ""
    era: str = ""
    rights: str = ""
    source: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def n_chars(self) -> int:
        return len(self.text)

    @property
    def n_words(self) -> int:
        return len(self.text.split())

    @property
    def locator(self) -> str:
        """`Meditations, Book II` — the human-facing 'where'."""
        return f"{self.work_title}, {self.section}" if self.section else self.work_title

    @property
    def citation(self) -> str:
        """Full citation string used in rendered source lists."""
        bits = [self.philosopher, self.work_title]
        if self.section:
            bits.append(self.section)
        line = " · ".join(b for b in bits if b)
        if self.translator:
            line += f" (tr. {self.translator})"
        return line

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @staticmethod
    def make_id(work_id: str, index: int, text: str) -> str:
        """Stable across re-ingests of unchanged text, so the index can be
        incrementally updated instead of always rebuilt from zero."""
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        return f"{work_id}#{index:04d}-{digest}"


@dataclass
class ScoredChunk:
    """A chunk plus why the retriever picked it."""

    chunk: Chunk
    score: float                # final blended score, 0..1
    dense: float = 0.0          # cosine similarity from embeddings
    lexical: float = 0.0        # BM25-ish keyword overlap
    marker: int = 0             # citation number [1], [2], ... assigned at render

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "score": round(self.score, 4),
            "dense": round(self.dense, 4),
            "lexical": round(self.lexical, 4),
            "chunk": self.chunk.to_dict(),
        }


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


@dataclass
class Answer:
    """The two-layer answer the whole project is organised around.

    `plain` is the everyday-language explanation; `academic` is the rigorous
    version with inline `[n]` citation markers.  `sources` are the chunks the
    model was actually shown, in marker order.
    """

    question: str
    plain: str = ""
    academic: str = ""
    sources: list[ScoredChunk] = field(default_factory=list)
    raw: str = ""               # unparsed model output, kept for debugging
    grounded: bool = True       # False when retrieval found nothing usable
    # "sources" = answered from retrieved text; "direct" = the model's own
    # recollection, no retrieval. Kept separate from `grounded` because a
    # sourced answer that found nothing and an intentionally unsourced one
    # need completely different presentation.
    mode: str = "sources"
    provider: str = ""
    model: str = ""
    latency_ms: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    # Citation markers the model produced that pointed at no real source.
    # Non-empty means the model tried to invent a citation and we caught it.
    invented_markers: set[int] = field(default_factory=set)
    truncated: bool = False

    @property
    def cited_markers(self) -> set[int]:
        """Which [n] markers the model actually used in its prose."""
        import re

        text = f"{self.plain}\n{self.academic}"
        return {int(m) for m in re.findall(r"\[(\d{1,2})\]", text)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "plain": self.plain,
            "academic": self.academic,
            "grounded": self.grounded,
            "mode": self.mode,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "usage": self.usage,
            "invented_markers": sorted(self.invented_markers),
            "sources": [s.to_dict() for s in self.sources],
        }


@dataclass
class DailyPiece:
    """One 'Daily Philosophy' card, personalised from a user profile."""

    date: str
    theme: str
    title: str = ""
    hook: str = ""              # the everyday-language opening
    quote: str = ""             # a real line from the corpus
    quote_marker: int = 0
    reflection: str = ""        # what it means / why it matters
    practice: str = ""          # something to actually try today
    sources: list[ScoredChunk] = field(default_factory=list)
    raw: str = ""
    provider: str = ""
    model: str = ""
    profile_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "theme": self.theme,
            "title": self.title,
            "hook": self.hook,
            "quote": self.quote,
            "reflection": self.reflection,
            "practice": self.practice,
            "profile": self.profile_name,
            "provider": self.provider,
            "model": self.model,
            "sources": [s.to_dict() for s in self.sources],
        }


# --------------------------------------------------------------------------
# Chat message helper
# --------------------------------------------------------------------------


Message = dict[str, str]


def system(content: str) -> Message:
    return {"role": "system", "content": content}


def user(content: str) -> Message:
    return {"role": "user", "content": content}


def assistant(content: str) -> Message:
    return {"role": "assistant", "content": content}


def join_messages(*groups: Iterable[Message]) -> list[Message]:
    out: list[Message] = []
    for g in groups:
        out.extend(g)
    return out
