"""Read a modern paper through one philosopher.

Upload something written this decade, name a philosopher, and get their
reading of it: what they would recognise, what they would refuse, and the
question they would put to the authors.

**This is inference, and the whole design exists to keep that visible.** Kant
never read a paper on transformer interpretability. Anything of the form
"Kant would say…" is a conclusion drawn from what he wrote, not a thing he
wrote, and a system that blurs the two produces exactly the confident
fabrication the rest of this project is built against. So:

- the philosopher's *positions* are retrieved and cited like anything else —
  those are claims about the historical record and they carry `[n]`;
- the *application* to the paper is marked as the model's reasoning, and the
  prompt requires the two to stay visibly separate.

It runs as two completions, not one. The first reads the paper and says what
it claims — in the vocabulary the library would use, not the paper's own.
That output is worth showing on its own (it is the receipt that the paper was
actually read) and it doubles as the retrieval query: an ML abstract full of
"benchmark" and "ablation" retrieves nothing at all from a shelf that ends in
1929, while "what we can know from repeated observation" retrieves Hume.

A philosopher who is not in the library is still allowed. You asked for any
philosopher, and refusing the request would be worse than answering it
honestly — so that path runs unsourced, and says so in the same words the
`--no-sources` mode uses elsewhere.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from ..models import ScoredChunk
from ..store.vector_store import Filters
from ..util import char_weight, detect_language, truncate
from .prompts import (
    audit_markers,
    build_paper_messages,
    build_themes_messages,
    parse_sections,
)

# How much of the paper reaches the model. Enough for an abstract, an
# introduction and a conclusion; papers run far longer than any of this needs.
PAPER_BUDGET = 14000

# Below this a "paper" is a paragraph, and reading it as one is a category error.
MIN_PAPER_CHARS = 400

SUFFIXES = (".pdf", ".txt", ".md", ".markdown", ".rst", ".tex")


class PaperError(RuntimeError):
    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


@dataclass
class PaperReading:
    philosopher: str = ""
    title: str = ""
    claims: str = ""          # what the paper says, in the library's vocabulary
    agree: str = ""
    object: str = ""
    question: str = ""
    sources: list[ScoredChunk] = field(default_factory=list)
    grounded: bool = True     # False = the philosopher is not in this library
    n_chars: int = 0
    raw: str = ""
    lang: str = "en"
    provider: str = ""
    model: str = ""
    took_ms: int = 0
    invented_markers: set[int] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "philosopher": self.philosopher,
            "title": self.title,
            "claims": self.claims,
            "agree": self.agree,
            "object": self.object,
            "question": self.question,
            "grounded": self.grounded,
            "n_chars": self.n_chars,
            "lang": self.lang,
            "provider": self.provider,
            "model": self.model,
            "took_ms": self.took_ms,
            "invented_markers": sorted(self.invented_markers),
            "sources": [s.to_dict() for s in self.sources],
        }


# --------------------------------------------------------------------------
# Getting text out
# --------------------------------------------------------------------------


def extract_text(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise PaperError(f"no file at {path}")
    if path.suffix.lower() == ".pdf":
        return extract_pdf(path.read_bytes())
    try:
        return path.read_text("utf-8", errors="replace")
    except OSError as exc:
        raise PaperError(f"could not read {path}: {exc}") from exc


def extract_pdf(data: bytes) -> str:
    try:
        import pypdf
    except ImportError as exc:
        raise PaperError(
            "reading PDFs needs the `pypdf` package",
            hint="pip install pypdf  — or paste the text instead",
        ) from exc
    import io

    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pypdf raises a wide family of parse errors
        raise PaperError(
            f"this PDF could not be read: {exc}",
            hint="Scanned PDFs are images with no text layer. Paste the text instead.",
        ) from exc
    return "\n\n".join(p for p in pages if p.strip())


def condense(text: str, budget: int = PAPER_BUDGET) -> str:
    """Trim a long paper from the middle, not the end.

    A paper's claims live in its abstract and its conclusion; its middle is
    method and related work. Cutting only the tail throws away the half that
    says what the authors think they showed.
    """
    text = _tidy(text)
    if char_weight(text) <= budget:
        return text
    head = int(budget * 0.62)
    tail = budget - head
    return (
        text[:head].rstrip()
        + "\n\n[… middle of the paper omitted …]\n\n"
        + text[-tail:].lstrip()
    )


def _tidy(text: str) -> str:
    """PDF extraction leaves ragged line breaks and page furniture."""
    import re

    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)          # de-hyphenate across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def guess_title(text: str) -> str:
    """The first line that looks like a title rather than a header or a URL."""
    for line in _tidy(text).splitlines():
        line = line.strip()
        if len(line) < 12 or len(line) > 200:
            continue
        if line.lower().startswith(("arxiv:", "http", "doi:", "published", "preprint")):
            continue
        if sum(c.isdigit() for c in line) > len(line) / 3:
            continue
        return truncate(line, 140)
    return ""


# --------------------------------------------------------------------------
# The reading
# --------------------------------------------------------------------------


def read_paper(
    engine,
    text: str,
    philosopher: str,
    *,
    k: int = 6,
    lang: str = "",
    reader_note: str = "",
    chat_model: str = "",
    chat_provider: str = "",
) -> PaperReading:
    from .answerer import AskOptions

    started = time.perf_counter()
    body = condense(text)
    if char_weight(body) < MIN_PAPER_CHARS:
        raise PaperError(
            "there is not enough text here to read as a paper",
            hint=f"Give it at least {MIN_PAPER_CHARS} characters, or use `philo ask` instead.",
        )
    philosopher = (philosopher or "").strip()
    if not philosopher:
        raise PaperError("no philosopher named", hint="`philo sources` lists who is in the library.")

    lang = lang or detect_language(body[:2000])
    backend = engine.chat_backend(
        AskOptions(chat_model=chat_model, chat_provider=chat_provider)
    )

    # Pass one: what does this paper claim, in words a 19th-century shelf
    # would recognise? Shown to the reader, and used as the retrieval query.
    themes = backend.chat(
        build_themes_messages(body, philosopher=philosopher, lang=lang),
        temperature=engine.settings.temperature,
        max_tokens=600,
        task="themes",
        model=chat_model,
    )
    claims = parse_sections(themes.text).get("CLAIMS", "").strip() or themes.text.strip()

    # Pass two: the philosopher's own positions on those questions.
    known = {p.lower() for p in engine.store.philosophers()}
    in_library = any(philosopher.lower() in p or p in philosopher.lower() for p in known)
    hits: list[ScoredChunk] = []
    if in_library:
        result = engine.retriever.search(
            claims, k=k, filters=Filters(philosopher=philosopher)
        )
        hits = list(result.hits)

    messages = build_paper_messages(
        body,
        hits,
        philosopher=philosopher,
        claims=claims,
        lang=lang,
        reader_note=reader_note,
    )
    completion = backend.chat(
        messages,
        temperature=engine.settings.temperature,
        max_tokens=min(engine.settings.max_tokens, 1400),
        task="paper",
        model=chat_model,
    )

    text_out, invented = audit_markers(completion.text, {h.marker for h in hits})
    sections = parse_sections(text_out)

    return PaperReading(
        philosopher=philosopher,
        title=guess_title(text),
        claims=claims,
        agree=sections.get("AGREE", "").strip(),
        object=sections.get("OBJECT", "").strip(),
        question=sections.get("PUTQ", "").strip(),
        sources=hits,
        grounded=bool(hits),
        n_chars=len(text),
        raw=completion.text,
        lang=lang,
        provider=completion.provider or backend.name,
        model=completion.model,
        took_ms=int((time.perf_counter() - started) * 1000),
        invented_markers=invented,
    )
