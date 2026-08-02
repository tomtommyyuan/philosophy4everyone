"""Storage and retrieval.

The load-bearing guarantees here: an index built by one embedding model is
never served to another, retrieval refuses to answer when nothing is relevant,
and no single work can monopolise the citation list.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from philo.config import Settings
from philo.corpus.ingest import ingest
from philo.providers.mock import MockProvider
from philo.retrieval.retriever import BM25, Retriever
from philo.store.vector_store import Filters, IndexError_, VectorStore


@pytest.fixture
def built(settings: Settings, provider: MockProvider):
    report = ingest(settings, provider)
    store = VectorStore(settings.index_dir).load()
    return settings, provider, store, report


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


def test_ingest_writes_a_readable_index(built):
    settings, provider, store, report = built
    assert report.n_chunks == len(store) > 0
    assert report.n_works == 2
    assert store.manifest.embed_model == provider.embed_model
    # chunks.jsonl is meant to stay greppable by hand.
    first = json.loads(store.chunks_path.read_text("utf-8").splitlines()[0])
    assert first["philosopher"]
    assert first["text"]


def test_index_refuses_a_different_embedding_model(built):
    settings, _, _, _ = built
    store = VectorStore(settings.index_dir)
    with pytest.raises(IndexError_) as excinfo:
        store.load(expect_model="text-embedding-3-large")
    assert "not comparable" in str(excinfo.value)
    assert excinfo.value.hint


def test_missing_index_gives_an_actionable_error(tmp_path: Path):
    with pytest.raises(IndexError_) as excinfo:
        VectorStore(tmp_path / "nope").load()
    assert "philo ingest" in excinfo.value.hint


def test_index_detects_chunk_vector_mismatch(built):
    settings, _, store, _ = built
    with store.chunks_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "x", "work_id": "w", "text": "orphan"}) + "\n")
    with pytest.raises(IndexError_) as excinfo:
        VectorStore(settings.index_dir).load()
    assert "inconsistent" in str(excinfo.value)


def test_reingest_reuses_unchanged_embeddings(settings: Settings, provider: MockProvider):
    ingest(settings, provider)
    second = ingest(settings, provider)
    assert second.n_embedded == 0
    assert second.n_reused == second.n_chunks


def test_rebuild_forces_reembedding(settings: Settings, provider: MockProvider):
    ingest(settings, provider)
    rebuilt = ingest(settings, provider, rebuild=True)
    assert rebuilt.n_reused == 0
    assert rebuilt.n_embedded == rebuilt.n_chunks


def test_editing_one_file_only_reembeds_that_file(settings: Settings, provider: MockProvider):
    ingest(settings, provider)
    path = settings.library_dir / "laozi-tao-te-ching.md"
    path.write_text(path.read_text("utf-8") + "\n\n## Chapter 78\n\nThere is nothing in the world more soft and weak than water.\n", encoding="utf-8")
    report = ingest(settings, provider)
    assert 0 < report.n_embedded < report.n_chunks


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------


def test_filters_match_case_insensitively_on_substrings(built):
    _, _, store, _ = built
    hits = [c for c in store.chunks if Filters(philosopher="epict").matches(c)]
    assert hits and all(c.philosopher == "Epictetus" for c in hits)
    assert not Filters(philosopher="laozi").matches(hits[0])
    assert Filters().is_empty()


def test_filters_match_chinese_names(built):
    _, _, store, _ = built
    hits = [c for c in store.chunks if Filters(philosopher="爱比克泰德").matches(c)]
    assert hits


def test_tag_filter(built):
    _, _, store, _ = built
    hits = [c for c in store.chunks if Filters(tags=["wu-wei"]).matches(c)]
    assert hits and all(c.philosopher == "Laozi" for c in hits)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


def test_retrieval_finds_the_relevant_passage(built):
    settings, provider, store, _ = built
    result = Retriever(store, provider, settings).search("what is in our control?")
    assert result.hits
    joined = " ".join(h.chunk.text for h in result.hits)
    assert "in our control" in joined
    # Markers are assigned 1..n in rank order and are what the model cites.
    assert [h.marker for h in result.hits] == list(range(1, len(result.hits) + 1))


def test_relevance_floor_returns_nothing_rather_than_the_least_bad_match(built):
    settings, provider, store, _ = built
    retriever = Retriever(store, provider, settings)
    result = retriever.search("quantum chromodynamics lattice gauge renormalisation", min_score=0.9)
    assert result.hits == []
    assert not result.grounded


def test_philosopher_filter_narrows_the_candidate_pool(built):
    settings, provider, store, _ = built
    retriever = Retriever(store, provider, settings)
    result = retriever.search("water", filters=Filters(philosopher="Laozi"))
    assert result.hits
    assert all(h.chunk.philosopher == "Laozi" for h in result.hits)
    assert result.n_candidates < len(store)


def test_no_single_work_can_monopolise_the_answer(built):
    settings, provider, store, _ = built
    settings.max_per_work = 1
    result = Retriever(store, provider, settings).search("control and desire", k=4)
    works = [h.chunk.work_id for h in result.hits]
    assert len(works) == len(set(works))


def test_bm25_ranks_exact_terms_of_art_highest(built):
    _, _, store, _ = built
    bm25 = BM25([c.text for c in store.chunks])
    indices = list(range(len(store.chunks)))
    scores = bm25.score(["socrates"], indices)
    best = max(range(len(scores)), key=lambda i: scores[i])
    assert "Socrates" in store.chunks[best].text


def test_bm25_of_unknown_term_scores_zero(built):
    _, _, store, _ = built
    bm25 = BM25([c.text for c in store.chunks])
    assert set(bm25.score(["zzzznotaword"], list(range(len(store.chunks))))) == {0.0}


def test_empty_filter_result_is_handled(built):
    settings, provider, store, _ = built
    result = Retriever(store, provider, settings).search(
        "anything", filters=Filters(philosopher="Nobody At All")
    )
    assert result.hits == []
    assert result.n_candidates == 0
