"""Decisions and the weekly recap — the two things that read the record back.

Both share one shape, which is the shape everything grounded in this project
has: retrieve first, refuse if nothing was found, generate only from what was
retrieved, then audit the markers. The differences are what they retrieve
*on* and what they are forbidden to do.

A decision is forbidden to give a verdict. A recap is forbidden to invent a
pattern. Those are the two ways each feature fails into being worthless, and
both are stated in the prompt rather than hoped for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, timedelta

from ..generation.prompts import (
    audit_markers,
    build_decision_messages,
    build_recap_messages,
    parse_sections,
)
from ..models import ScoredChunk
from ..store.vector_store import Filters
from ..util import detect_language, truncate
from .resurface import Echo, floor_from, rhymes
from .store import Chronicle, Entry

RECAP_DAYS = 7

# How far back a resurfaced entry has to come from to be worth mentioning in
# a weekly recap: anything inside the week is already in the recap itself.
ECHO_MIN_AGE = RECAP_DAYS + 1


@dataclass
class DecisionResult:
    entry: Entry
    choice: str = ""
    tests: str = ""
    limits: str = ""
    sources: list[ScoredChunk] = field(default_factory=list)
    echoes: list[Echo] = field(default_factory=list)
    raw: str = ""
    grounded: bool = True
    provider: str = ""
    model: str = ""
    took_ms: int = 0
    invented_markers: set[int] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "id": self.entry.id,
            "situation": self.entry.text,
            "note": self.entry.note,
            "choice": self.choice,
            "tests": self.tests,
            "limits": self.limits,
            "grounded": self.grounded,
            "provider": self.provider,
            "model": self.model,
            "took_ms": self.took_ms,
            "invented_markers": sorted(self.invented_markers),
            "echoes": [e.to_dict() for e in self.echoes],
            "sources": [s.to_dict() for s in self.sources],
        }


@dataclass
class RecapResult:
    span: str
    week: str = ""
    thread: str = ""
    sit_with: str = ""
    entries: list[Entry] = field(default_factory=list)
    sources: list[ScoredChunk] = field(default_factory=list)
    echoes: list[Echo] = field(default_factory=list)
    raw: str = ""
    grounded: bool = True
    provider: str = ""
    model: str = ""
    took_ms: int = 0

    @property
    def empty(self) -> bool:
        return not self.entries

    def to_dict(self) -> dict:
        return {
            "span": self.span,
            "week": self.week,
            "thread": self.thread,
            "sit_with": self.sit_with,
            "grounded": self.grounded,
            "n_entries": len(self.entries),
            "counts": _counts(self.entries),
            "provider": self.provider,
            "model": self.model,
            "took_ms": self.took_ms,
            "echoes": [e.to_dict() for e in self.echoes],
            "entries": [e.to_dict() for e in self.entries],
            "sources": [s.to_dict() for s in self.sources],
        }


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------


def log_decision(
    engine,
    book: Chronicle,
    situation: str,
    *,
    note: str = "",
    lang: str = "",
    k: int = 6,
    reader_note: str = "",
    chat_model: str = "",
    chat_provider: str = "",
    save: bool = True,
    today: str = "",
) -> DecisionResult:
    """Put the texts to work on one real decision, and record it.

    The entry is written whether or not the model had anything to say. The
    record is the reader's; a retrieval that came up empty is not a reason to
    lose the fact that they faced this on this day.
    """
    from ..generation.answerer import AskOptions

    started = time.perf_counter()
    lang = lang or detect_language(situation)
    query = f"{situation}\n{note}".strip()

    backend = engine.chat_backend(
        AskOptions(chat_model=chat_model, chat_provider=chat_provider)
    )
    query_vec = engine.provider.embed_query(query)
    result = engine.retriever.search(query, k=k, filters=Filters(), query_vec=query_vec)
    echoes = rhymes(
        book, query, store=engine.store, query_vec=query_vec, today=today,
        vector_floor=floor_from(result, settings=engine.settings),
    )

    entry = Entry(kind="decision", text=situation.strip(), note=note.strip())

    if not result.hits:
        # No sources, no model call — the same rule as everywhere else.
        entry.response = ""
        if save:
            book.add(entry)
        return DecisionResult(
            entry=entry,
            choice=_nothing_found(lang),
            grounded=False,
            echoes=echoes,
            provider=backend.name,
            took_ms=int((time.perf_counter() - started) * 1000),
        )

    messages = build_decision_messages(
        situation,
        result.hits,
        note=note,
        lang=lang,
        reader_note=reader_note,
        echoes=[f"{e.when}: {e.entry.headline()}" for e in echoes],
    )
    completion = backend.chat(
        messages,
        temperature=engine.settings.temperature,
        max_tokens=min(engine.settings.max_tokens, 1200),
        task="decision",
        model=chat_model,
    )

    text, invented = audit_markers(completion.text, {h.marker for h in result.hits})
    sections = parse_sections(text)

    entry.response = text.strip()
    entry.citations = _citations(result.hits)
    entry.tags = sorted({h.chunk.tradition for h in result.hits if h.chunk.tradition})
    if save:
        book.add(entry)

    return DecisionResult(
        entry=entry,
        choice=sections.get("CHOICE", "").strip() or sections.get("_PREAMBLE", "").strip(),
        tests=sections.get("TESTS", "").strip(),
        limits=sections.get("LIMITS", "").strip(),
        sources=list(result.hits),
        echoes=echoes,
        raw=completion.text,
        grounded=True,
        provider=completion.provider or backend.name,
        model=completion.model,
        took_ms=int((time.perf_counter() - started) * 1000),
        invented_markers=invented,
    )


# --------------------------------------------------------------------------
# The weekly recap
# --------------------------------------------------------------------------


def weekly_recap(
    engine,
    book: Chronicle,
    *,
    days: int = RECAP_DAYS,
    lang: str = "",
    reader_note: str = "",
    chat_model: str = "",
    chat_provider: str = "",
    today: str = "",
    k: int = 6,
) -> RecapResult:
    """Stitch the week's entries into a short narrative.

    The sources are the passages the reader actually saved, topped up by a
    retrieval over everything they wrote. Their own saves come first: a recap
    that quotes back something they chose is worth more than one that
    introduces a passage they have never seen.
    """
    from ..generation.answerer import AskOptions

    started = time.perf_counter()
    today = today or date.today().isoformat()
    since = (date.fromisoformat(today) - timedelta(days=days - 1)).isoformat()
    span = f"{since} → {today}"

    entries = book.since(since)
    if not entries:
        return RecapResult(span=span, grounded=False,
                           took_ms=int((time.perf_counter() - started) * 1000))

    backend = engine.chat_backend(
        AskOptions(chat_model=chat_model, chat_provider=chat_provider)
    )
    lang = lang or detect_language(" ".join(e.text for e in entries))

    written = "\n".join(_line(e) for e in entries)
    query_vec = engine.provider.embed_query(truncate(written, 2000))

    saved = _saved_passages(engine, entries)
    retrieved = engine.retriever.search(
        truncate(written, 2000), k=k, filters=Filters(), query_vec=query_vec
    )
    hits = _merge(saved, retrieved.hits, limit=k)

    # Only from *before* this week: something inside the span is already in
    # the recap, and calling it a resurfacing would be theatre.
    older = Chronicle(book.path, book.before(since))
    echoes = rhymes(
        older, written, store=engine.store, query_vec=query_vec,
        today=today, min_age_days=ECHO_MIN_AGE, limit=1,
        vector_floor=floor_from(retrieved, settings=engine.settings),
    )

    if not hits:
        return RecapResult(
            span=span, entries=entries, echoes=echoes, grounded=False,
            week=_nothing_found(lang), provider=backend.name,
            took_ms=int((time.perf_counter() - started) * 1000),
        )

    messages = build_recap_messages(
        [_line(e) for e in entries],
        hits,
        span=span,
        lang=lang,
        reader_note=reader_note,
        echoes=[f"{e.when}: {e.entry.headline()}" for e in echoes],
    )
    completion = backend.chat(
        messages,
        temperature=min(0.8, engine.settings.temperature + 0.3),
        max_tokens=min(engine.settings.max_tokens, 1000),
        task="recap",
        model=chat_model,
    )

    text, _ = audit_markers(completion.text, {h.marker for h in hits})
    sections = parse_sections(text)

    return RecapResult(
        span=span,
        week=sections.get("WEEK", "").strip() or sections.get("_PREAMBLE", "").strip(),
        thread=sections.get("THREAD", "").strip(),
        sit_with=sections.get("SIT", "").strip(),
        entries=entries,
        sources=hits,
        echoes=echoes,
        raw=completion.text,
        grounded=True,
        provider=completion.provider or backend.name,
        model=completion.model,
        took_ms=int((time.perf_counter() - started) * 1000),
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _line(entry: Entry) -> str:
    """One entry as the model sees it — kind, day, and what was written."""
    label = {"passage": "saved", "decision": "decided", "question": "asked"}.get(
        entry.kind, entry.kind
    )
    body = truncate(entry.text.replace("\n", " "), 300)
    if entry.kind == "passage" and entry.citation:
        body = f"“{body}” — {entry.citation}"
    if entry.note:
        body += f"  (their note: {truncate(entry.note, 200)})"
    return f"[{entry.day}] {label}: {body}"


def _saved_passages(engine, entries) -> list[ScoredChunk]:
    """The reader's own saves, looked up in the index by chunk id."""
    rows = {chunk.id: i for i, chunk in enumerate(engine.store.chunks)}
    out: list[ScoredChunk] = []
    seen: set[str] = set()
    for entry in entries:
        row = rows.get(entry.chunk_id) if entry.chunk_id else None
        if row is None or entry.chunk_id in seen:
            continue
        seen.add(entry.chunk_id)
        out.append(ScoredChunk(chunk=engine.store.chunks[row], score=1.0))
    return out


def _merge(saved: list[ScoredChunk], retrieved: list[ScoredChunk], *, limit: int) -> list[ScoredChunk]:
    """Saved passages first, then retrieval, renumbered 1..N as one block."""
    from dataclasses import replace

    out: list[ScoredChunk] = []
    seen: set[str] = set()
    for hit in list(saved) + list(retrieved):
        if hit.chunk.id in seen:
            continue
        seen.add(hit.chunk.id)
        out.append(hit)
    return [replace(hit, marker=i) for i, hit in enumerate(out[:limit], 1)]


def _citations(hits) -> list[dict[str, str]]:
    return [
        {
            "chunk_id": h.chunk.id,
            "philosopher": h.chunk.philosopher,
            "work_title": h.chunk.work_title,
            "section": h.chunk.section,
        }
        for h in hits
    ]


def _counts(entries) -> dict[str, int]:
    out: dict[str, int] = {}
    for entry in entries:
        out[entry.kind] = out.get(entry.kind, 0) + 1
    return out


def _nothing_found(lang: str) -> str:
    if lang == "zh":
        return "当前library里没有能支撑这件事的原文，所以这里不作解读——与其套用一个不存在的框架，不如先把相关文本放进 library/。"
    return (
        "Nothing in this library bears on that closely enough to be worth quoting, "
        "so there is no reading here. The entry is still recorded. Adding the "
        "relevant texts to library/ and re-running `philo ingest` would change that."
    )
