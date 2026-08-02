"""A small, honest, local vector store.

No server, no Docker, no bespoke binary format you cannot inspect.  An index
is three files in a directory:

    .philo/index/
      manifest.json   what produced this index (provider, model, dimension)
      chunks.jsonl    one JSON object per chunk — greppable, diffable
      vectors.npy     float32 matrix, row i ↔ line i of chunks.jsonl

The manifest is not bookkeeping.  Vectors from different embedding models
occupy different spaces, and comparing them produces confident nonsense
rather than an error — so the store refuses to serve an index that was built
by a different model than the one now asking questions of it.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..models import Chunk, ScoredChunk
from ..util import HAS_NUMPY, cosine_batch, l2_normalize

if HAS_NUMPY:
    import numpy as np

SCHEMA_VERSION = 1


class IndexError_(RuntimeError):
    """Index is missing, corrupt, or built by an incompatible model."""

    def __init__(self, message: str, *, hint: str = ""):
        super().__init__(message)
        self.hint = hint


@dataclass
class Manifest:
    schema: int = SCHEMA_VERSION
    provider: str = ""
    embed_model: str = ""
    dim: int = 0
    n_chunks: int = 0
    n_works: int = 0
    built_at: str = ""
    library: str = ""
    chunk_settings: dict[str, int] = field(default_factory=dict)
    works: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "provider": self.provider,
            "embed_model": self.embed_model,
            "dim": self.dim,
            "n_chunks": self.n_chunks,
            "n_works": self.n_works,
            "built_at": self.built_at,
            "library": self.library,
            "chunk_settings": self.chunk_settings,
            "works": self.works,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Manifest":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Filters:
    """Metadata narrowing.  All conditions are ANDed; each is case-insensitive
    substring matching, so `--philosopher aurelius` finds Marcus Aurelius."""

    philosopher: str = ""
    work: str = ""
    tradition: str = ""
    era: str = ""
    tags: list[str] = field(default_factory=list)
    work_ids: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any([self.philosopher, self.work, self.tradition, self.era, self.tags, self.work_ids])

    def matches(self, c: Chunk) -> bool:
        if self.philosopher and not _contains(c.philosopher, self.philosopher) and not _contains(
            c.philosopher_zh, self.philosopher
        ):
            return False
        if self.work and not _contains(c.work_title, self.work) and not _contains(c.work_title_zh, self.work):
            return False
        if self.tradition and not _contains(c.tradition, self.tradition):
            return False
        if self.era and not _contains(c.era, self.era):
            return False
        if self.work_ids and c.work_id not in self.work_ids:
            return False
        if self.tags:
            have = {t.lower() for t in c.tags}
            if not any(t.lower() in have for t in self.tags):
                return False
        return True

    def describe(self) -> str:
        bits = []
        if self.philosopher:
            bits.append(f"philosopher~{self.philosopher}")
        if self.work:
            bits.append(f"work~{self.work}")
        if self.tradition:
            bits.append(f"tradition~{self.tradition}")
        if self.era:
            bits.append(f"era~{self.era}")
        if self.tags:
            bits.append("tags∈{" + ",".join(self.tags) + "}")
        return " · ".join(bits)


def _contains(haystack: str, needle: str) -> bool:
    return needle.strip().lower() in (haystack or "").lower()


class VectorStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.chunks: list[Chunk] = []
        self.vectors: Any = []          # np.ndarray when numpy is available
        self.manifest = Manifest()
        self._loaded = False

    # -- paths ------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    @property
    def chunks_path(self) -> Path:
        return self.path / "chunks.jsonl"

    @property
    def vectors_path(self) -> Path:
        return self.path / ("vectors.npy" if HAS_NUMPY else "vectors.json")

    def exists(self) -> bool:
        return self.manifest_path.is_file() and self.chunks_path.is_file()

    def __len__(self) -> int:
        return len(self.chunks)

    # -- io ---------------------------------------------------------------
    def load(self, *, expect_model: str = "", expect_provider: str = "") -> "VectorStore":
        if not self.exists():
            raise IndexError_(
                f"no index at {self.path}",
                hint="Build one with `philo ingest`.",
            )
        try:
            self.manifest = Manifest.from_dict(json.loads(self.manifest_path.read_text("utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            raise IndexError_(f"index manifest is unreadable: {exc}", hint="Rebuild with `philo ingest --rebuild`.") from exc

        if self.manifest.schema != SCHEMA_VERSION:
            raise IndexError_(
                f"index schema v{self.manifest.schema} was built by a different version of philo "
                f"(this build expects v{SCHEMA_VERSION})",
                hint="Rebuild with `philo ingest --rebuild`.",
            )

        # The check that prevents silently-wrong retrieval.
        if expect_model and self.manifest.embed_model and expect_model != self.manifest.embed_model:
            raise IndexError_(
                f"index was embedded with '{self.manifest.embed_model}' but the current "
                f"provider uses '{expect_model}' — the vectors are not comparable",
                hint="Rebuild with `philo ingest --rebuild`, or switch back to the original provider.",
            )
        if expect_provider and self.manifest.provider and expect_provider != self.manifest.provider:
            raise IndexError_(
                f"index was built with the '{self.manifest.provider}' provider but you are now "
                f"using '{expect_provider}'",
                hint="Rebuild with `philo ingest --rebuild`.",
            )

        self.chunks = []
        with self.chunks_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self.chunks.append(Chunk.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError) as exc:
                    raise IndexError_(
                        f"chunks.jsonl is corrupt at line {line_no}: {exc}",
                        hint="Rebuild with `philo ingest --rebuild`.",
                    ) from exc

        self.vectors = self._read_vectors()
        n_vecs = len(self.vectors)
        if n_vecs != len(self.chunks):
            raise IndexError_(
                f"index is inconsistent: {len(self.chunks)} chunks but {n_vecs} vectors",
                hint="Rebuild with `philo ingest --rebuild`.",
            )
        self._loaded = True
        return self

    def _read_vectors(self) -> Any:
        if HAS_NUMPY and self.vectors_path.suffix == ".npy" and self.vectors_path.is_file():
            return np.load(self.vectors_path)
        # Either numpy is absent, or the index was built without it.
        json_path = self.path / "vectors.json"
        if json_path.is_file():
            data = json.loads(json_path.read_text("utf-8"))
            return np.asarray(data, dtype="float32") if HAS_NUMPY else data
        npy_path = self.path / "vectors.npy"
        if npy_path.is_file() and not HAS_NUMPY:
            raise IndexError_(
                "this index stores vectors as .npy but numpy is not installed",
                hint="pip install numpy — or rebuild with `philo ingest --rebuild`.",
            )
        raise IndexError_("index has no vector file", hint="Rebuild with `philo ingest --rebuild`.")

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest.n_chunks = len(self.chunks)
        self.manifest.schema = SCHEMA_VERSION

        _atomic_write(
            self.chunks_path,
            "".join(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in self.chunks),
        )
        if HAS_NUMPY:
            self._atomic_npy(self.vectors_path, self.vectors)
            (self.path / "vectors.json").unlink(missing_ok=True)
        else:
            _atomic_write(
                self.path / "vectors.json",
                json.dumps([list(map(float, v)) for v in self.vectors]),
            )
        _atomic_write(
            self.manifest_path,
            json.dumps(self.manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        )

    @staticmethod
    def _atomic_npy(path: Path, matrix: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        os.close(fd)
        try:
            arr = matrix if isinstance(matrix, np.ndarray) else np.asarray(matrix, dtype="float32")
            if arr.size == 0:
                arr = arr.reshape(0, 0)
            # np.save appends .npy to a path that lacks it; write then move.
            with open(tmp, "wb") as fh:
                np.save(fh, arr.astype("float32", copy=False))
            os.replace(tmp, path)
        finally:
            Path(tmp).unlink(missing_ok=True)

    # -- mutation ---------------------------------------------------------
    def replace_all(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"chunk/vector count mismatch: {len(chunks)} vs {len(vectors)}")
        self.chunks = list(chunks)
        normalized = [l2_normalize(v) for v in vectors]
        self.vectors = np.asarray(normalized, dtype="float32") if HAS_NUMPY else normalized
        self.manifest.dim = len(normalized[0]) if normalized else 0
        self._loaded = True

    def clear(self) -> None:
        self.chunks = []
        self.vectors = np.zeros((0, 0), dtype="float32") if HAS_NUMPY else []
        self.manifest = Manifest()

    def destroy(self) -> None:
        for p in (self.manifest_path, self.chunks_path, self.path / "vectors.npy", self.path / "vectors.json"):
            p.unlink(missing_ok=True)
        self.clear()

    # -- query ------------------------------------------------------------
    def similarities(self, query_vec: Sequence[float]) -> list[float]:
        if not len(self.chunks):
            return []
        return cosine_batch(query_vec, self.vectors)

    def vector_at(self, i: int) -> Sequence[float]:
        return self.vectors[i]

    def candidates(self, filters: Filters | None) -> list[int]:
        """Row indices surviving metadata filters."""
        if filters is None or filters.is_empty():
            return list(range(len(self.chunks)))
        return [i for i, c in enumerate(self.chunks) if filters.matches(c)]

    def iter_chunks(self) -> Iterable[Chunk]:
        return iter(self.chunks)

    def find(self, chunk_id: str) -> Chunk | None:
        for c in self.chunks:
            if c.id == chunk_id:
                return c
        return None

    # -- introspection ----------------------------------------------------
    def works_summary(self) -> list[dict[str, Any]]:
        by_work: dict[str, dict[str, Any]] = {}
        for c in self.chunks:
            row = by_work.setdefault(
                c.work_id,
                {
                    "work_id": c.work_id,
                    "philosopher": c.philosopher,
                    "philosopher_zh": c.philosopher_zh,
                    "work_title": c.work_title,
                    "work_title_zh": c.work_title_zh,
                    "translator": c.translator,
                    "tradition": c.tradition,
                    "era": c.era,
                    "rights": c.rights,
                    "tags": list(c.tags),
                    "n_chunks": 0,
                    "n_chars": 0,
                    "sections": set(),
                },
            )
            row["n_chunks"] += 1
            row["n_chars"] += c.n_chars
            if c.section:
                row["sections"].add(c.section)
        out = []
        for row in by_work.values():
            row["n_sections"] = len(row.pop("sections"))
            out.append(row)
        out.sort(key=lambda r: (r["philosopher"], r["work_title"]))
        return out

    def philosophers(self) -> list[str]:
        return sorted({c.philosopher for c in self.chunks if c.philosopher})

    def tags(self) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for c in self.chunks:
            for t in c.tags:
                counts[t] = counts.get(t, 0) + 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def make_scored(chunk: Chunk, score: float, dense: float = 0.0, lexical: float = 0.0) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score, dense=dense, lexical=lexical)
