"""Build the library from Project Gutenberg.

Why a fetcher instead of committed excerpts: this system's entire claim is
that its quotations are real.  Text written from memory — human or model —
cannot make that claim.  Everything in `library/` therefore comes from a
canonical public-domain edition, downloaded verbatim, with only structural
cleanup applied.

What "structural cleanup" means, precisely:

- Project Gutenberg's licence header and footer are removed.
- Editorial apparatus is removed: translators' introductions, tables of
  contents, appendices, glossaries, indexes.  These are the editor's words,
  not the philosopher's, and retrieving them would produce citations that
  attribute a Victorian scholar's opinion to Marcus Aurelius.
- Footnote reference markers of the form `[12]` are stripped, because they
  would be indistinguishable from this system's own citation markers once
  the passage reaches the model.
- Gutenberg's `_underscore italics_` are unwrapped.
- Book and chapter headings are converted to Markdown `##`/`###`, which is
  what lets a citation say "Meditations, Book IV" rather than just
  "Meditations".

Not one word of the philosophical text itself is altered.

Every `translator` field below was verified against the downloaded file
rather than assumed — PG #2680, for instance, is Meric Casaubon's 1634
translation, which the file states in its own notes.

This module ships inside the package, so an installed CLI can populate its
own library with `philo fetch` — a checkout is not required.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

USER_AGENT = "philosophy-for-everyone/0.1 (educational use)"
ROMAN = r"[IVXLCDM]+"


@dataclass
class Source:
    """One work, and how to carve the philosophy out of the file."""

    gid: int
    slug: str
    philosopher: str
    work: str
    translator: str
    year: str
    tradition: str
    era: str
    tags: list[str]
    philosopher_zh: str = ""
    work_zh: str = ""
    # (regex, which match) — text starts at the start of that match.
    start: tuple[str, int] | None = None
    # (regex, which match) — text ends at the start of that match.
    stop: tuple[str, int] | None = None
    # (regex, heading level) — standalone lines matching become `##`/`###`.
    headings: list[tuple[str, int]] = field(default_factory=list)
    # Lines matching any of these are deleted.
    drop_lines: list[str] = field(default_factory=list)
    # (from_regex, to_regex) — everything between is excised (editorial inserts).
    drop_ranges: list[tuple[str, str]] = field(default_factory=list)
    # Titles that run into the text, e.g. `OF PROVIDENCE.—Of all the things…`
    inline_titles: tuple[str, int] | None = None
    # Chapters delimited by blank-line blocks and numbered sequentially.
    block_chapters: tuple[str, int] | None = None
    # Join `CHAPTER I.` with the ALL-CAPS title line that follows it.
    join_title_lines: bool = False
    note: str = ""

    @property
    def url(self) -> str:
        return f"https://www.gutenberg.org/cache/epub/{self.gid}/pg{self.gid}.txt"


MEDITATIONS_BOOKS = (
    "FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH"
)

SOURCES: list[Source] = [
    Source(
        gid=2680,
        slug="marcus-aurelius-meditations",
        philosopher="Marcus Aurelius",
        philosopher_zh="马可·奥勒留",
        work="Meditations",
        work_zh="沉思录",
        translator="Meric Casaubon",
        year="c. 180 CE",
        tradition="Stoicism",
        era="Ancient",
        tags=["death", "virtue", "self-control", "acceptance", "duty", "impermanence", "anger", "time"],
        start=(r"^THE FIRST BOOK$", 0),
        stop=(r"^APPENDIX$", 0),
        headings=[(rf"^THE ({MEDITATIONS_BOOKS}) BOOK$", 2)],
    ),
    Source(
        gid=10661,
        slug="epictetus-discourses-enchiridion",
        philosopher="Epictetus",
        philosopher_zh="爱比克泰德",
        work="Discourses and the Encheiridion",
        work_zh="论说集与手册",
        translator="George Long",
        year="c. 108 CE",
        tradition="Stoicism",
        era="Ancient",
        tags=["freedom", "control", "desire", "fear", "adversity", "self-control", "judgement", "death"],
        start=(r"^A SELECTION FROM THE DISCOURSES OF EPICTETUS\.$", 1),
        headings=[
            (r"^A SELECTION FROM THE DISCOURSES OF EPICTETUS\.$", 2),
            (r"^THE ENCHEIRIDION, OR MANUAL\.$", 2),
            (rf"^{ROMAN}\.$", 3),
        ],
        # Discourse titles are ALL-CAPS runs terminated by `.—` before the text.
        inline_titles=(r"^([A-Z][A-Z0-9 ,;:'’\-\(\)]{8,}(?:\n[A-Z0-9 ,;:'’\-\(\)]{4,})*?)\.—", 3),
    ),
    Source(
        gid=1656,
        slug="plato-apology",
        philosopher="Plato",
        philosopher_zh="柏拉图",
        work="Apology",
        work_zh="申辩篇",
        translator="Benjamin Jowett",
        year="c. 399 BCE",
        tradition="Ancient Greek",
        era="Ancient",
        tags=["knowledge", "death", "justice", "examined-life", "courage", "wisdom", "ignorance"],
        start=(r"^APOLOGY$", 0),
        note="Socrates' defence at his trial, as written by Plato.",
    ),
    Source(
        gid=8438,
        slug="aristotle-nicomachean-ethics",
        philosopher="Aristotle",
        philosopher_zh="亚里士多德",
        work="Nicomachean Ethics",
        work_zh="尼各马可伦理学",
        translator="D. P. Chase",
        year="c. 340 BCE",
        tradition="Ancient Greek",
        era="Ancient",
        tags=["happiness", "virtue", "friendship", "habit", "moderation", "character", "the-good", "pleasure"],
        start=(r"^BOOK I$", 0),
        stop=(r"^NOTES$", 0),
        headings=[(rf"^BOOK {ROMAN}$", 2), (rf"^Chapter {ROMAN}\.$", 3)],
        # Chase's own appendices sit between books; they are his words, not Aristotle's.
        drop_ranges=[(r"^APPENDIX\.?$", rf"^BOOK {ROMAN}$")],
    ),
    Source(
        gid=5682,
        slug="kant-groundwork",
        philosopher="Immanuel Kant",
        philosopher_zh="康德",
        work="Fundamental Principles of the Metaphysic of Morals",
        work_zh="道德形而上学基础",
        translator="Thomas Kingsmill Abbott",
        year="1785",
        tradition="German Idealism",
        era="Modern",
        tags=["duty", "good-will", "categorical-imperative", "autonomy", "reason", "morality", "freedom"],
        start=(r"^PREFACE$", 0),
        headings=[(r"^PREFACE$", 2), (r"^(FIRST|SECOND|THIRD) SECTION\b.*$", 2)],
        drop_ranges=[(r"^[ \t]*1 FIRST SECTION\.", r"^[ \t]+FIRST SECTION$")],
        drop_lines=[r"^\s*1785\s*$"],
        join_title_lines=True,
    ),
    Source(
        gid=11224,
        slug="mill-utilitarianism",
        philosopher="John Stuart Mill",
        philosopher_zh="约翰·斯图尔特·密尔",
        work="Utilitarianism",
        work_zh="功利主义",
        translator="",
        year="1863",
        tradition="Utilitarianism",
        era="Modern",
        tags=["happiness", "pleasure", "morality", "justice", "consequences", "the-good", "duty"],
        start=(r"^CHAPTER I\.$", 0),
        headings=[(rf"^CHAPTER {ROMAN}\..*$", 2)],
        join_title_lines=True,
    ),
    Source(
        gid=34901,
        slug="mill-on-liberty",
        philosopher="John Stuart Mill",
        philosopher_zh="约翰·斯图尔特·密尔",
        work="On Liberty",
        work_zh="论自由",
        translator="",
        year="1859",
        tradition="Liberalism",
        era="Modern",
        tags=["freedom", "harm", "individuality", "speech", "authority", "society", "tyranny", "toleration"],
        start=(r"^CHAPTER I\.$", 1),
        headings=[(rf"^CHAPTER {ROMAN}\..*$", 2)],
        join_title_lines=True,
    ),
    Source(
        gid=59,
        slug="descartes-discourse-on-method",
        philosopher="René Descartes",
        philosopher_zh="笛卡尔",
        work="Discourse on the Method",
        work_zh="谈谈方法",
        translator="John Veitch",
        year="1637",
        tradition="Rationalism",
        era="Modern",
        tags=["doubt", "certainty", "reason", "mind", "existence", "method", "knowledge", "god"],
        start=(r"^PART I$", 0),
        headings=[(rf"^PART {ROMAN}$", 2)],
    ),
    Source(
        gid=9662,
        slug="hume-enquiry-human-understanding",
        philosopher="David Hume",
        philosopher_zh="大卫·休谟",
        work="An Enquiry Concerning Human Understanding",
        work_zh="人类理解研究",
        translator="",
        year="1748",
        tradition="Empiricism",
        era="Modern",
        tags=["causation", "induction", "experience", "belief", "scepticism", "knowledge", "miracles", "habit"],
        start=(r"^SECTION I\.$", 0),
        stop=(r"^INDEX$", 0),
        headings=[(rf"^SECTION {ROMAN}\..*$", 2)],
        join_title_lines=True,
    ),
    Source(
        gid=4363,
        slug="nietzsche-beyond-good-and-evil",
        philosopher="Friedrich Nietzsche",
        philosopher_zh="尼采",
        work="Beyond Good and Evil",
        work_zh="善恶的彼岸",
        translator="Helen Zimmern",
        year="1886",
        tradition="Existentialism",
        era="Modern",
        tags=["morality", "power", "truth", "values", "self-overcoming", "herd", "nobility", "suspicion"],
        start=(r"^CHAPTER I\. PREJUDICES OF PHILOSOPHERS$", 0),
        headings=[(rf"^CHAPTER {ROMAN}\. .+$", 2)],
    ),
    Source(
        gid=216,
        slug="laozi-tao-te-ching",
        philosopher="Laozi",
        philosopher_zh="老子",
        work="Tao Te Ching",
        work_zh="道德经",
        translator="James Legge",
        year="c. 400 BCE",
        tradition="Daoism",
        era="Ancient",
        tags=["wu-wei", "nature", "simplicity", "softness", "leadership", "desire", "the-way", "humility"],
        start=(r"^Ch\. 1\.", 0),
        block_chapters=("Chapter {}", 2),
        drop_lines=[r"^PART [I1V]+\.?$"],
    ),
    Source(
        gid=3330,
        slug="confucius-analects",
        philosopher="Confucius",
        philosopher_zh="孔子",
        work="The Analects",
        work_zh="论语",
        translator="James Legge",
        year="c. 400 BCE",
        tradition="Confucianism",
        era="Ancient",
        tags=["virtue", "learning", "ritual", "filial-piety", "government", "friendship", "character", "shame"],
        start=(r"^BOOK I\.\s+HSIO R\.", 0),
        headings=[(rf"^BOOK {ROMAN}\.\s+.+$", 2)],
        drop_lines=[r"^\s*\[?page \d+\]?\s*$"],
    ),
    Source(
        gid=5827,
        slug="russell-problems-of-philosophy",
        philosopher="Bertrand Russell",
        philosopher_zh="伯特兰·罗素",
        work="The Problems of Philosophy",
        work_zh="哲学问题",
        translator="",
        year="1912",
        tradition="Analytic",
        era="Contemporary",
        tags=["knowledge", "appearance", "reality", "induction", "truth", "a-priori", "philosophy", "doubt"],
        start=(rf"^CHAPTER I\. .+$", 1),
        stop=(r"^BIBLIOGRAPHICAL NOTE", 0),
        headings=[(rf"^CHAPTER {ROMAN}\. .+$", 2)],
    ),
    Source(
        gid=3420,
        slug="wollstonecraft-vindication-rights-of-woman",
        philosopher="Mary Wollstonecraft",
        philosopher_zh="玛丽·沃斯通克拉夫特",
        work="A Vindication of the Rights of Woman",
        work_zh="女权辩护",
        translator="",
        year="1792",
        tradition="Enlightenment Feminism",
        era="Modern",
        tags=["equality", "education", "reason", "virtue", "rights", "society", "freedom", "character"],
        start=(r"^INTRODUCTION\.$", 1),
        headings=[(r"^INTRODUCTION\.$", 2), (r"^CHAPTER \d+\.$", 2)],
    ),
]


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------


def fetch(source: Source, cache_dir: Path, *, force: bool = False) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"pg{source.gid}.txt"
    if cached.is_file() and not force:
        return cached.read_text(encoding="utf-8", errors="replace")

    request = urllib.request.Request(source.url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read().decode("utf-8", errors="replace")
    cached.write_text(raw, encoding="utf-8")
    return raw


# --------------------------------------------------------------------------
# Clean
# --------------------------------------------------------------------------

_PG_START = re.compile(r"\*\*\* ?START OF TH[EIS][^\n]*\*\*\*")
_PG_END = re.compile(r"\*\*\* ?END OF TH[EIS][^\n]*\*\*\*")
_FOOTNOTE_REF = re.compile(r"\[\d{1,4}\]")
_PAGE_MARK = re.compile(r"\[(?:Pg|page|p\.)\s*[\divxlc]+\]", re.IGNORECASE)
_ITALIC = re.compile(r"_([^_\n]{1,120}?)_")
_STRICT_ROMAN = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
_TITLE_LINE = re.compile(r"^(?:CHAPTER|SECTION|PART|BOOK|FIRST|SECOND|THIRD)\b.*$")

_SMALL_WORDS = {
    "of", "the", "and", "or", "in", "on", "to", "a", "an", "for", "is", "not",
    "from", "with", "by", "at", "as", "that", "which", "are", "be", "we",
    "our", "it", "his", "her", "their", "but", "into", "upon", "no", "nor",
}


_PG_LEGACY_END = re.compile(r"^End of (?:the )?Project Gutenberg", re.MULTILINE | re.IGNORECASE)


def strip_boilerplate(raw: str) -> str:
    start = _PG_START.search(raw)
    body = raw[start.end():] if start else raw
    end = _PG_END.search(body)
    if end:
        body = body[: end.start()]
    # Older texts also carry a plain-prose trailer *inside* the marked region.
    legacy = _PG_LEGACY_END.search(body)
    return body[: legacy.start()] if legacy else body


def _nth(pattern: str, text: str, index: int) -> int | None:
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    if not matches:
        return None
    return matches[min(index, len(matches) - 1)].start()


def _is_roman(token: str) -> bool:
    core = token.strip(".,;:")
    return bool(core) and bool(_STRICT_ROMAN.match(core))


def title_case(heading: str) -> str:
    """`SECOND BOOK` → `Second Book`.  Mixed-case headings are left alone."""
    heading = heading.strip().rstrip(".")
    if not heading.isupper():
        return heading
    words: list[str] = []
    for i, word in enumerate(heading.split()):
        # "Chapter II. Of the Free Spirit" — a small word still leads a
        # sub-title, so sentence position matters, not just index.
        starts_clause = i == 0 or words[-1].endswith((".", ":", "—"))
        if _is_roman(word):
            words.append(word)
        elif not starts_clause and word.strip(".,;:").lower() in _SMALL_WORDS:
            words.append(word.lower())
        else:
            words.append(word.capitalize())
    return " ".join(words)


def _join_title_lines(text: str) -> str:
    """`CHAPTER I.` + `GENERAL REMARKS.` → `CHAPTER I. GENERAL REMARKS.`

    Editions routinely split a heading across two lines.  Left alone, the
    first line becomes the section label and the second becomes body text —
    so citations read "Chapter I" instead of "Chapter I. General Remarks".
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped and _TITLE_LINE.match(stripped):
            # Look ahead past blank lines for ALL-CAPS continuation lines.
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            parts: list[str] = []
            while (
                j < len(lines)
                and lines[j].strip()
                and lines[j].strip().isupper()
                and len(lines[j].strip()) < 80
                and len(parts) < 3
            ):
                parts.append(lines[j].strip())
                j += 1
            if parts:
                out.append(f"{stripped} {' '.join(parts)}")
                i = j
                continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _apply_block_chapters(text: str, template: str, level: int) -> tuple[str, int]:
    """Number-led blocks separated by blank lines become chapter headings.

    Used for Legge's Tao Te Ching, where chapters are delimited only by a
    wider blank-line gap and a leading number — and where the *subsection*
    numbers inside a chapter look identical.  Requiring the number to be the
    next chapter in sequence resolves the ambiguity exactly.
    """
    blocks = re.split(r"\n{3,}", text)
    expected = 1
    out: list[str] = []
    made = 0
    for block in blocks:
        match = re.match(r"^(?:Ch\. )?(\d{1,2})\.", block.strip())
        if match and int(match.group(1)) == expected:
            # The number is now the heading; leaving it inline would make
            # every chapter open with a stray "7." in the quoted text.
            rest = block.strip()[match.end():].lstrip()
            out.append(f"{'#' * level} {template.format(expected)}\n\n{rest}")
            expected += 1
            made += 1
        else:
            out.append(block)
    return "\n\n".join(out), made


def clean(source: Source, raw: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    body = strip_boilerplate(raw).replace("\r\n", "\n").replace("\r", "\n")
    # Trailing whitespace makes every `^HEADING$` pattern fail silently.
    body = "\n".join(line.rstrip() for line in body.split("\n"))

    if source.start:
        pos = _nth(source.start[0], body, source.start[1])
        if pos is None:
            warnings.append(f"start pattern {source.start[0]!r} not found — keeping whole file")
        else:
            body = body[pos:]
    if source.stop:
        pos = _nth(source.stop[0], body, source.stop[1])
        if pos is None:
            warnings.append(f"stop pattern {source.stop[0]!r} not found — keeping to end")
        else:
            body = body[:pos]

    for from_pat, to_pat in source.drop_ranges:
        while True:
            begin = re.search(from_pat, body, re.MULTILINE)
            if not begin:
                break
            end = re.search(to_pat, body[begin.end():], re.MULTILINE)
            body = body[: begin.start()] + (body[begin.end() + end.start():] if end else "")
            if not end:
                break

    # Footnote markers must go: `[3]` in the source would be indistinguishable
    # from this system's own citation markers once the passage reaches the model.
    body = _FOOTNOTE_REF.sub("", body)
    body = _PAGE_MARK.sub("", body)
    body = _ITALIC.sub(r"\1", body)

    if source.join_title_lines:
        body = _join_title_lines(body)

    n_headings = 0
    if source.inline_titles:
        pattern, level = source.inline_titles
        hashes = "#" * level

        def replace(match: re.Match) -> str:
            nonlocal n_headings
            n_headings += 1
            title = title_case(re.sub(r"\s+", " ", match.group(1)).strip())
            return f"\n\n{hashes} {title}\n\n"

        body = re.compile(pattern, re.MULTILINE).sub(replace, body)

    drops = [re.compile(p) for p in source.drop_lines]
    heads = [(re.compile(p), lvl) for p, lvl in source.headings]

    out: list[str] = []
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped and any(d.match(stripped) for d in drops):
            continue
        level = next((lvl for pattern, lvl in heads if stripped and pattern.match(stripped)), None)
        if level is not None:
            out.extend(["", f"{'#' * level} {title_case(stripped)}", ""])
            n_headings += 1
            continue
        out.append(line)
    body = "\n".join(out)

    if source.block_chapters:
        template, level = source.block_chapters
        body, made = _apply_block_chapters(body, template, level)
        n_headings += made

    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if n_headings == 0 and (source.headings or source.inline_titles or source.block_chapters):
        warnings.append("no section headings detected — citations will name the work only")
    return body, warnings


# --------------------------------------------------------------------------
# Write
# --------------------------------------------------------------------------


def front_matter(source: Source) -> str:
    lines = ["---", f"philosopher: {source.philosopher}"]
    if source.philosopher_zh:
        lines.append(f"philosopher_zh: {source.philosopher_zh}")
    lines.append(f"work: {source.work}")
    if source.work_zh:
        lines.append(f"work_zh: {source.work_zh}")
    if source.translator:
        lines.append(f"translator: {source.translator}")
    lines += [
        f"year: {source.year}",
        f"tradition: {source.tradition}",
        f"era: {source.era}",
        "language: en",
        "rights: public-domain",
        f"source: Project Gutenberg #{source.gid}",
        f"source_url: {source.url}",
        f"tags: [{', '.join(source.tags)}]",
        "---",
        "",
    ]
    if source.note:
        lines += [f"<!-- {source.note} -->", ""]
    return "\n".join(lines)


def build(
    source: Source, library_dir: Path, cache_dir: Path, *, force: bool = False
) -> tuple[Path, int, int, list[str]]:
    text, warnings = clean(source, fetch(source, cache_dir, force=force))
    library_dir.mkdir(parents=True, exist_ok=True)
    path = library_dir / f"{source.slug}.md"
    path.write_text(front_matter(source) + text + "\n", encoding="utf-8")
    n_sections = len(re.findall(r"^#{2,3} ", text, re.MULTILINE))
    return path, len(text), n_sections, warnings


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


@dataclass
class WorkFetch:
    slug: str
    philosopher: str
    work: str
    path: Path
    chars: int
    sections: int
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass
class FetchReport:
    library_dir: Path
    works: list[WorkFetch] = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        return sum(1 for w in self.works if w.ok)

    @property
    def total_chars(self) -> int:
        return sum(w.chars for w in self.works)

    @property
    def failures(self) -> list[WorkFetch]:
        return [w for w in self.works if not w.ok]


def select(only: Sequence[str] = ()) -> list[Source]:
    if not only:
        return list(SOURCES)
    needles = [n.lower() for n in only]
    return [
        s for s in SOURCES
        if any(n in s.slug.lower() or n in s.philosopher.lower() for n in needles)
    ]


def fetch_all(
    library_dir: Path,
    cache_dir: Path,
    *,
    only: Sequence[str] = (),
    force: bool = False,
    on_event: Callable[[str, int, int, str], None] | None = None,
) -> FetchReport:
    """Download and clean every configured work into `library_dir`."""
    selected = select(only)
    report = FetchReport(library_dir=Path(library_dir))

    for i, source in enumerate(selected, 1):
        if on_event:
            on_event("fetch", i, len(selected), f"{source.philosopher} · {source.work}")
        try:
            path, size, sections, warnings = build(
                source, Path(library_dir), Path(cache_dir), force=force
            )
        except (urllib.error.URLError, OSError) as exc:
            report.works.append(
                WorkFetch(
                    slug=source.slug, philosopher=source.philosopher, work=source.work,
                    path=Path(library_dir) / f"{source.slug}.md", chars=0, sections=0,
                    error=str(getattr(exc, "reason", exc)),
                )
            )
            continue
        report.works.append(
            WorkFetch(
                slug=source.slug, philosopher=source.philosopher, work=source.work,
                path=path, chars=size, sections=sections, warnings=warnings,
            )
        )
    if on_event:
        on_event("done", len(selected), len(selected), "")
    return report
