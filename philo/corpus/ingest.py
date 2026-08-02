"""The ingestion pipeline: files → chunks → vectors → index.

Re-ingest is incremental.  Chunk ids embed a content hash, so a chunk whose
text has not changed keeps its existing vector and is never re-embedded —
which means editing one file in a large library costs one file's worth of API
calls, not the whole shelf.  `--rebuild` forces the full pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..config import Settings
from ..models import Chunk
from ..providers.base import Provider
from ..store.vector_store import VectorStore
from .chunker import chunk_work, stats_for
from .loader import LoadedWork, load_library

# stage, current, total, detail
EventCallback = Callable[[str, int, int, str], None]


@dataclass
class WorkReport:
    work_id: str
    philosopher: str
    title: str
    n_chunks: int
    n_chars: int
    rights: str
    reused: int = 0


@dataclass
class IngestReport:
    library: Path
    index: Path
    works: list[WorkReport] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    n_chunks: int = 0
    n_embedded: int = 0
    n_reused: int = 0
    avg_chunk_chars: int = 0
    min_chunk_chars: int = 0
    max_chunk_chars: int = 0
    embed_model: str = ""
    provider: str = ""
    dim: int = 0
    took_ms: int = 0

    @property
    def n_works(self) -> int:
        return len(self.works)

    @property
    def ok(self) -> bool:
        return self.n_chunks > 0


def ingest(
    settings: Settings,
    provider: Provider,
    *,
    rebuild: bool = False,
    strict: bool = False,
    library: Path | None = None,
    on_event: EventCallback | None = None,
) -> IngestReport:
    started = time.perf_counter()
    lib = Path(library or settings.library_dir)
    store = VectorStore(settings.index_dir)
    report = IngestReport(
        library=lib,
        index=settings.index_dir,
        embed_model=provider.embed_model,
        provider=provider.name,
    )

    def emit(stage: str, cur: int = 0, total: int = 0, detail: str = "") -> None:
        if on_event:
            on_event(stage, cur, total, detail)

    # -- 1. load -----------------------------------------------------------
    emit("load", 0, 0, str(lib))
    loaded, errors = load_library(lib, strict=strict)
    report.errors.extend(errors)
    if not loaded:
        report.took_ms = int((time.perf_counter() - started) * 1000)
        return report

    # -- 2. chunk ----------------------------------------------------------
    all_chunks: list[Chunk] = []
    for i, item in enumerate(loaded, 1):
        emit("chunk", i, len(loaded), item.work.label)
        chunks = chunk_work(item.work, item.body, settings)
        all_chunks.extend(chunks)
        stats = stats_for(chunks)
        report.works.append(
            WorkReport(
                work_id=item.work.id,
                philosopher=item.work.philosopher,
                title=item.work.title,
                n_chunks=stats.n_chunks,
                n_chars=stats.n_chars,
                rights=item.work.rights,
            )
        )

    if not all_chunks:
        report.errors.append("no chunks produced — are the source files empty?")
        report.took_ms = int((time.perf_counter() - started) * 1000)
        return report

    # -- 3. reuse ----------------------------------------------------------
    cached: dict[str, list[float]] = {}
    if not rebuild and store.exists():
        try:
            store.load(expect_model=provider.embed_model, expect_provider=provider.name)
            cached = {c.id: list(store.vector_at(i)) for i, c in enumerate(store.chunks)}
        except Exception as exc:
            # An unusable old index is not fatal here — we are about to
            # replace it. Note it and embed everything.
            report.errors.append(f"previous index not reusable ({exc}); embedding from scratch")
            cached = {}

    pending = [c for c in all_chunks if c.id not in cached]
    report.n_reused = len(all_chunks) - len(pending)
    report.n_embedded = len(pending)

    reused_by_work: dict[str, int] = {}
    for c in all_chunks:
        if c.id in cached:
            reused_by_work[c.work_id] = reused_by_work.get(c.work_id, 0) + 1
    for wr in report.works:
        wr.reused = reused_by_work.get(wr.work_id, 0)

    # -- 4. embed ----------------------------------------------------------
    if pending:
        emit("embed", 0, len(pending), provider.embed_model)
        vectors = provider.embed(
            [_embedding_text(c) for c in pending],
            on_progress=lambda done, total: emit("embed", done, total, provider.embed_model),
        )
        if len(vectors) != len(pending):
            raise RuntimeError(
                f"provider returned {len(vectors)} vectors for {len(pending)} chunks"
            )
        for chunk, vec in zip(pending, vectors):
            cached[chunk.id] = vec

    ordered = [cached[c.id] for c in all_chunks]

    # -- 5. store ----------------------------------------------------------
    emit("store", 0, len(all_chunks), str(settings.index_dir))
    store.replace_all(all_chunks, ordered)
    store.manifest.provider = provider.name
    store.manifest.embed_model = provider.embed_model
    store.manifest.n_works = len(report.works)
    store.manifest.built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    store.manifest.library = str(lib)
    store.manifest.chunk_settings = {
        "min": settings.chunk_min_chars,
        "target": settings.chunk_target_chars,
        "max": settings.chunk_max_chars,
    }
    store.manifest.works = [
        {
            "id": w.work.id,
            "philosopher": w.work.philosopher,
            "title": w.work.title,
            "translator": w.work.translator,
            "tradition": w.work.tradition,
            "rights": w.work.rights,
            "source": w.work.source,
        }
        for w in loaded
    ]
    store.save()

    overall = stats_for(all_chunks)
    report.n_chunks = overall.n_chunks
    report.avg_chunk_chars = overall.avg_chars
    report.min_chunk_chars = overall.min_chars
    report.max_chunk_chars = overall.max_chars
    report.dim = store.manifest.dim
    report.took_ms = int((time.perf_counter() - started) * 1000)
    emit("done", report.n_chunks, report.n_chunks, "")
    return report


def _embedding_text(chunk: Chunk) -> str:
    """What actually gets embedded.

    The citation header goes in alongside the text: it makes "what did
    Epictetus say about X" match passages by Epictetus, and it is how a
    query naming a work finds that work even when the prose never names it.
    """
    header = " · ".join(p for p in (chunk.philosopher, chunk.work_title, chunk.section) if p)
    tags = " ".join(chunk.tags)
    return f"{header}\n{tags}\n\n{chunk.text}" if header else chunk.text


def preview_chunks(
    settings: Settings, library: Path | None = None
) -> tuple[list[Chunk], list[str]]:
    """Chunk the whole library without embedding — `philo ingest --dry-run`.

    Deliberately chunks everything: the point of a dry run is to check the
    size distribution before spending money on embeddings, and statistics
    from a truncated sample would not answer that question.
    """
    lib = Path(library or settings.library_dir)
    loaded, errors = load_library(lib)
    chunks: list[Chunk] = []
    for item in loaded:
        chunks.extend(chunk_work(item.work, item.body, settings))
    return chunks, errors


__all__ = ["ingest", "IngestReport", "WorkReport", "preview_chunks", "LoadedWork"]
