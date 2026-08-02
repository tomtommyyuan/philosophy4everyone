"""Retrieval: hybrid scoring, metadata filters, and diversity.

Three things beyond a plain nearest-neighbour lookup, each because pure
dense retrieval fails a specific way on philosophy:

*Lexical blending (BM25).*  Proper nouns and terms of art — `eudaimonia`,
`noumenon`, `無為` — are exactly where embeddings are weakest and exact
matching is strongest.  A question naming a concept should retrieve the
passage that names it.

*MMR diversity.*  The top-6 by cosine are frequently six near-identical
paragraphs from the same chapter.  That looks like six sources and is really
one, which is a citation illusion.  MMR trades a little relevance for genuine
coverage, and `max_per_work` stops any single book monopolising the answer.

*A floor.*  Cosine always returns *something*; a ranked list is not evidence
that the library contains an answer.  Below `min_score` we return nothing, so
the generator can say "not in the library" instead of quoting the least
irrelevant passage it could find.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from ..config import Settings
from ..models import Chunk, ScoredChunk
from ..providers.base import Provider
from ..store.vector_store import Filters, VectorStore
from ..util import minmax, tokenize


@dataclass
class RetrievalResult:
    query: str
    hits: list[ScoredChunk] = field(default_factory=list)
    n_candidates: int = 0
    filters: Filters | None = None
    took_ms: int = 0
    best_score: float = 0.0

    @property
    def grounded(self) -> bool:
        return bool(self.hits)

    def markers(self) -> dict[int, ScoredChunk]:
        return {h.marker: h for h in self.hits}


class BM25:
    """Compact BM25 over the chunk texts.

    Built lazily on first lexical query and cached for the life of the
    retriever — for a library of a few thousand chunks this costs
    milliseconds and removes the dense retriever's blind spot for rare terms.
    """

    __slots__ = ("df", "tf", "lengths", "avg_len", "n", "k1", "b")

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf: list[Counter[str]] = []
        self.df: Counter[str] = Counter()
        self.lengths: list[int] = []
        for text in corpus:
            toks = tokenize(text)
            counts = Counter(toks)
            self.tf.append(counts)
            self.lengths.append(len(toks))
            self.df.update(counts.keys())
        self.n = len(corpus)
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0

    def score(self, query_terms: Sequence[str], indices: Sequence[int]) -> list[float]:
        if not self.n or not query_terms:
            return [0.0] * len(indices)
        idf = {
            t: math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for t in set(query_terms)
            if self.df.get(t)
        }
        if not idf:
            return [0.0] * len(indices)

        out: list[float] = []
        for i in indices:
            counts = self.tf[i]
            length = self.lengths[i] or 1
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
            total = 0.0
            for term, weight in idf.items():
                f = counts.get(term, 0)
                if f:
                    total += weight * (f * (self.k1 + 1)) / (f + norm)
            out.append(total)
        return out


class Retriever:
    def __init__(self, store: VectorStore, provider: Provider, settings: Settings) -> None:
        self.store = store
        self.provider = provider
        self.settings = settings
        self._bm25: BM25 | None = None

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25([c.text for c in self.store.chunks])
        return self._bm25

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        filters: Filters | None = None,
        min_score: float | None = None,
        diversify: bool = True,
    ) -> RetrievalResult:
        import time

        started = time.perf_counter()
        k = k or self.settings.top_k
        floor = self.settings.min_score if min_score is None else min_score

        candidate_idx = self.store.candidates(filters)
        if not candidate_idx:
            return RetrievalResult(
                query=query, filters=filters, n_candidates=0,
                took_ms=int((time.perf_counter() - started) * 1000),
            )

        query_vec = self.provider.embed_query(query)
        dense_all = self.store.similarities(query_vec)
        dense = [dense_all[i] for i in candidate_idx]

        lexical_raw = self.bm25.score(tokenize(query), candidate_idx)

        # Normalise within the candidate pool: BM25 and cosine are on
        # incomparable scales, and cosine's usable range is corpus-dependent.
        dense_n = minmax(dense)
        lex_n = minmax(lexical_raw)
        alpha = self.settings.hybrid_alpha
        blended = [alpha * d + (1 - alpha) * l for d, l in zip(dense_n, lex_n)]

        # Rank on the blend, but keep raw cosine for the "is anything here at
        # all?" decision — a normalised score is relative to the pool and says
        # nothing about absolute relevance.
        order = sorted(range(len(candidate_idx)), key=lambda i: blended[i], reverse=True)
        pool_size = max(k * 4, self.settings.candidate_k)
        pool = order[:pool_size]

        best_dense = max(dense) if dense else 0.0
        if best_dense < floor:
            return RetrievalResult(
                query=query, filters=filters, n_candidates=len(candidate_idx),
                best_score=best_dense, took_ms=int((time.perf_counter() - started) * 1000),
            )

        chosen = (
            self._mmr(pool, candidate_idx, blended, k)
            if diversify
            else self._cap_per_work(pool, candidate_idx, k)
        )

        hits: list[ScoredChunk] = []
        for marker, local in enumerate(chosen, 1):
            row = candidate_idx[local]
            hits.append(
                ScoredChunk(
                    chunk=self.store.chunks[row],
                    score=round(blended[local], 4),
                    dense=round(dense[local], 4),
                    lexical=round(lexical_raw[local], 4),
                    marker=marker,
                )
            )

        return RetrievalResult(
            query=query,
            hits=hits,
            n_candidates=len(candidate_idx),
            filters=filters,
            best_score=best_dense,
            took_ms=int((time.perf_counter() - started) * 1000),
        )

    # ------------------------------------------------------------------
    def _mmr(
        self,
        pool: Sequence[int],
        candidate_idx: Sequence[int],
        relevance: Sequence[float],
        k: int,
    ) -> list[int]:
        """Maximal Marginal Relevance over the shortlist."""
        lam = self.settings.mmr_lambda
        selected: list[int] = []
        remaining = list(pool)
        per_work: Counter[str] = Counter()

        while remaining and len(selected) < k:
            best_local, best_value = None, -math.inf
            for local in remaining:
                chunk = self.store.chunks[candidate_idx[local]]
                if per_work[chunk.work_id] >= self.settings.max_per_work:
                    continue
                redundancy = 0.0
                for chosen in selected:
                    redundancy = max(
                        redundancy,
                        self._sim(candidate_idx[local], candidate_idx[chosen]),
                    )
                value = lam * relevance[local] - (1 - lam) * redundancy
                if value > best_value:
                    best_local, best_value = local, value

            if best_local is None:
                # Every remaining candidate is blocked by the per-work cap.
                # Better to return fewer, genuinely distinct sources than to
                # pad the answer with more of the same book.
                break
            selected.append(best_local)
            remaining.remove(best_local)
            per_work[self.store.chunks[candidate_idx[best_local]].work_id] += 1

        return selected

    def _cap_per_work(self, pool: Sequence[int], candidate_idx: Sequence[int], k: int) -> list[int]:
        selected: list[int] = []
        per_work: Counter[str] = Counter()
        for local in pool:
            chunk = self.store.chunks[candidate_idx[local]]
            if per_work[chunk.work_id] >= self.settings.max_per_work:
                continue
            selected.append(local)
            per_work[chunk.work_id] += 1
            if len(selected) >= k:
                break
        return selected

    def _sim(self, row_a: int, row_b: int) -> float:
        va, vb = self.store.vector_at(row_a), self.store.vector_at(row_b)
        # Stored vectors are L2-normalised, so the dot product is the cosine.
        return float(sum(x * y for x, y in zip(va, vb)))

    # ------------------------------------------------------------------
    def sample_by_tags(self, tags: Sequence[str], seed: int, limit: int = 40) -> list[Chunk]:
        """Deterministic sample of chunks matching any of `tags`.

        Used by the daily generator to pick a starting point that is stable
        for a given (date, profile) pair but varies across days.
        """
        wanted = {t.lower() for t in tags}
        pool = [
            c for c in self.store.chunks
            if wanted & {t.lower() for t in c.tags}
        ] or list(self.store.chunks)
        if not pool:
            return []
        step = max(1, len(pool) // max(1, limit))
        start = seed % len(pool)
        return [pool[(start + i * step) % len(pool)] for i in range(min(limit, len(pool)))]
