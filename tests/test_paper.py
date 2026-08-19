"""Reading a paper through one philosopher.

The risk this feature carries is not a bad summary — it is the blurring of
two different kinds of claim. "Hume held X" is a fact about the record and
must be cited. "Hume would say Y about this paper" is the model reasoning,
and must never wear a citation. Most of what follows is about that seam.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from philo.config import Settings
from philo.corpus.ingest import ingest
from philo.generation.answerer import Engine
from philo.generation.paper import (
    MIN_PAPER_CHARS,
    PaperError,
    condense,
    extract_pdf,
    extract_text,
    guess_title,
    read_paper,
)
from philo.providers.mock import MockProvider

# The offline provider echoes the paper's first three sentences as its
# "themes" output and then matches lexically, and its hashed embedding thins
# out badly over a long query (0.07 cosine at 600 characters against 0.37 at
# 40). So the abstract here is short and dense with the words the Enchiridion
# fixture uses. Translating a real paper into that vocabulary is exactly the
# job of the themes pass, and the one thing the mock cannot do.
PAPER = """\
Scaling Laws Do Not Predict Downstream Capability

Abstract. Practitioners are disturbed not by the loss curve but by the opinion
they hold about it. What is in our control is what we measured. The rest is
opinion.

1. Introduction. The dominant methodology treats the loss curve as a proxy for
capability. That inference has the structure of an induction from observed
regularity to unobserved cases, and it is rarely examined as such. Our results
across 41 checkpoints show benchmark accuracy is frequently flat or
non-monotonic in loss.

7. Conclusion. Capability claims should be grounded in direct measurement
rather than extrapolation. Practitioners are relying on a uniformity
assumption they have not defended.
"""


@pytest.fixture
def engine(settings: Settings, provider: MockProvider) -> Engine:
    ingest(settings, provider)
    return Engine(settings, provider)


# --------------------------------------------------------------------------
# Getting text out
# --------------------------------------------------------------------------


def test_a_long_paper_is_trimmed_from_the_middle_not_the_end():
    """The claims live in the abstract and the conclusion; the middle is method."""
    text = "ABSTRACT " * 200 + "METHOD " * 4000 + "CONCLUSION " * 200
    out = condense(text, budget=800)
    assert out.startswith("ABSTRACT")
    assert out.rstrip().endswith("CONCLUSION")
    assert "omitted" in out
    assert len(out) < len(text)


def test_a_short_paper_is_left_alone():
    assert condense("short enough already", budget=800) == "short enough already"


def test_extraction_repairs_what_pdfs_do_to_text():
    messy = "the argu-\nment is  that we\n\n\n\nknow"
    assert "argument" in condense(messy)
    assert " " not in condense(messy)


def test_the_title_is_the_first_line_that_is_not_furniture():
    assert guess_title("arXiv:2403.11111v2\n2024\nOn the Limits of Induction\n\nAbstract.") \
        == "On the Limits of Induction"


def test_no_plausible_title_is_better_than_a_wrong_one():
    assert guess_title("1\n2\n3\n") == ""


def test_a_paragraph_is_not_a_paper(engine: Engine):
    with pytest.raises(PaperError) as excinfo:
        read_paper(engine, "Too short to read as a paper.", "David Hume")
    assert str(MIN_PAPER_CHARS) in excinfo.value.hint


def test_a_philosopher_is_required(engine: Engine):
    with pytest.raises(PaperError):
        read_paper(engine, PAPER, "   ")


def test_a_missing_file_says_so(tmp_path: Path):
    with pytest.raises(PaperError):
        extract_text(tmp_path / "nope.pdf")


def test_an_unreadable_pdf_explains_itself():
    with pytest.raises(PaperError) as excinfo:
        extract_pdf(b"this is not a pdf at all")
    assert "Scanned PDFs" in excinfo.value.hint


def test_a_real_pdf_round_trips(tmp_path: Path):
    pypdf = pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    path = tmp_path / "blank.pdf"
    with path.open("wb") as fh:
        writer.write(fh)
    assert extract_text(path) == ""   # a blank page has no text layer, and says so


# --------------------------------------------------------------------------
# The reading
# --------------------------------------------------------------------------


def test_a_philosopher_in_the_library_is_read_from_their_own_texts(engine: Engine, monkeypatch):
    """The filter is what this feature owns; the floor belongs to the embedder.

    Whether a paper-length query clears the relevance floor is a property of
    the embedding model — the offline mock's hashed vectors thin out badly
    over long text — so the floor is dropped here and the assertion is about
    the thing that would be a real bug: sources leaking in from someone else.
    """
    monkeypatch.setattr(engine.settings, "min_score", 0.0)
    reading = read_paper(engine, PAPER, "Epictetus")
    assert reading.grounded
    assert reading.sources
    assert {s.chunk.philosopher for s in reading.sources} == {"Epictetus"}
    assert reading.title == "Scaling Laws Do Not Predict Downstream Capability"


def test_a_philosopher_not_in_the_library_is_still_answered_but_unsourced(engine: Engine):
    """Refusing "any philosopher" would be worse than answering it honestly."""
    reading = read_paper(engine, PAPER, "Karl Popper")
    assert reading.grounded is False
    assert reading.sources == []
    assert reading.invented_markers == set()


def test_an_unsourced_reading_may_not_carry_a_single_marker(engine: Engine, monkeypatch):
    """With no sources every [n] is invented, so every [n] must be stripped."""
    original = engine.provider.chat

    def with_citations(messages, **kwargs):
        if kwargs.get("task") == "paper":
            result = original(messages, **kwargs)
            result.text = "## WHERE THEY WOULD AGREE\nPopper wrote this [1] and also [7]."
            return result
        return original(messages, **kwargs)

    engine.provider.chat = with_citations  # type: ignore[method-assign]
    reading = read_paper(engine, PAPER, "Karl Popper")
    assert "[1]" not in reading.agree
    assert "[7]" not in reading.agree
    assert reading.invented_markers == {1, 7}


def test_the_library_is_searched_with_the_claims_not_the_raw_paper(engine: Engine):
    """The themes pass exists to be the retrieval query.

    Asserting the *contract* rather than the wording: a real model rewrites
    "benchmark" and "ablation" into the tradition's vocabulary, but the mock
    cannot, so checking for absent jargon would only be testing the mock.
    """
    seen = []
    original = engine.retriever.search

    def spy(query, **kwargs):
        seen.append(query)
        return original(query, **kwargs)

    engine.retriever.search = spy  # type: ignore[method-assign]
    reading = read_paper(engine, PAPER, "Epictetus")
    assert len(seen) == 1
    assert seen[0] == reading.claims
    assert seen[0] != PAPER


def test_the_reading_serialises(engine: Engine):
    import json

    payload = json.loads(json.dumps(read_paper(engine, PAPER, "Epictetus").to_dict()))
    assert payload["philosopher"] == "Epictetus"
    assert payload["n_chars"] == len(PAPER)


def test_the_prompt_keeps_the_record_and_the_inference_apart():
    from philo.generation.prompts import PAPER_SYSTEM, PAPER_UNSOURCED_NOTE

    assert "never read this paper" in PAPER_SYSTEM
    assert "not a séance" in PAPER_SYSTEM
    assert "do not impersonate" in PAPER_SYSTEM
    assert "Do not produce any `[n]` citation" in PAPER_UNSOURCED_NOTE


def test_the_themes_prompt_translates_rather_than_summarises():
    from philo.generation.prompts import THEMES_SYSTEM

    assert "translation task, not a summary task" in THEMES_SYSTEM
    assert "retrieves nothing" in THEMES_SYSTEM
