"""Reading source texts off disk.

Library files are Markdown with a small front-matter block:

    ---
    philosopher: Marcus Aurelius
    work: Meditations
    translator: George Long
    tradition: Stoicism
    rights: public-domain
    tags: [death, virtue, self-control]
    ---

    ## Book II

    Begin the morning by saying to thyself...

The front matter is what makes citation possible, so it is parsed strictly
enough to catch typos and loosely enough that nobody needs to learn YAML.
Plain `.txt` files without front matter are also accepted — metadata is then
inferred from the filename (`Philosopher - Work.txt`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ..models import Work
from ..util import normalize, slugify

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".text"}

_FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.DOTALL)

# `work` and `title` are accepted for the same field, likewise for the
# philosopher/author pair — people type both.
_ALIASES = {
    "author": "philosopher",
    "author_zh": "philosopher_zh",
    "title": "work",
    "title_zh": "work_zh",
    "book": "work",
    "license": "rights",
    "copyright": "rights",
    "url": "source_url",
    "lang": "language",
    "school": "tradition",
    "period": "era",
    "date": "year",
}

_LIST_FIELDS = {"tags"}


class CorpusError(Exception):
    """A source file that cannot be trusted to produce honest citations."""


@dataclass
class LoadedWork:
    work: Work
    body: str

    @property
    def n_chars(self) -> int:
        return len(self.body)


# --------------------------------------------------------------------------
# Front matter
# --------------------------------------------------------------------------


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Return (metadata, body).

    Supports `key: value`, inline lists `key: [a, b]`, and block lists:

        tags:
          - death
          - virtue
    """
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text

    meta: dict[str, Any] = {}
    pending_list_key: str | None = None

    for raw in match.group(1).splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        stripped = line.lstrip()
        if stripped.startswith("- ") and pending_list_key:
            meta.setdefault(pending_list_key, []).append(_scalar(stripped[2:]))
            continue

        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower().replace("-", "_")
        key = _ALIASES.get(key, key)
        value = value.strip()

        if not value:
            # Either a block list follows, or the field is simply empty.
            pending_list_key = key
            meta.setdefault(key, [])
            continue

        pending_list_key = None
        if value.startswith("[") and value.endswith("]"):
            items = [_scalar(v) for v in value[1:-1].split(",")]
            meta[key] = [i for i in items if i]
        elif key in _LIST_FIELDS:
            meta[key] = [_scalar(v) for v in re.split(r"[,;、]", value) if v.strip()]
        else:
            meta[key] = _scalar(value)

    # A key declared with no items is an empty list only for list fields;
    # otherwise treat it as an empty string.
    for key, value in list(meta.items()):
        if value == [] and key not in _LIST_FIELDS:
            meta[key] = ""

    return meta, text[match.end():]


def _scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value.strip()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def infer_from_filename(path: Path) -> dict[str, str]:
    """`Marcus Aurelius - Meditations.txt` → philosopher + work."""
    stem = path.stem.replace("_", " ").strip()
    for sep in (" - ", " — ", " – ", "--"):
        if sep in stem:
            author, _, title = stem.partition(sep)
            return {"philosopher": author.strip().title(), "work": title.strip()}
    return {"philosopher": "Unknown", "work": stem.title()}


def load_file(path: Path, *, strict: bool = False) -> LoadedWork:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise CorpusError(f"{path.name}: not valid UTF-8 ({exc.reason})") from exc

    meta, body = parse_front_matter(raw)
    if not meta:
        meta = infer_from_filename(path)
        if strict:
            raise CorpusError(
                f"{path.name}: no front matter. Without philosopher/work metadata "
                "the passages cannot be cited."
            )

    philosopher = str(meta.get("philosopher") or "").strip()
    title = str(meta.get("work") or "").strip()
    if not philosopher or not title:
        inferred = infer_from_filename(path)
        philosopher = philosopher or inferred["philosopher"]
        title = title or inferred["work"]

    body = normalize_body(body)
    if not body.strip():
        raise CorpusError(f"{path.name}: file has metadata but no text")

    work = Work(
        id=str(meta.get("id") or "").strip() or slugify(f"{philosopher}-{title}"),
        philosopher=philosopher,
        title=title,
        philosopher_zh=str(meta.get("philosopher_zh") or ""),
        title_zh=str(meta.get("work_zh") or meta.get("title_zh") or ""),
        translator=str(meta.get("translator") or ""),
        year=str(meta.get("year") or ""),
        tradition=str(meta.get("tradition") or ""),
        era=str(meta.get("era") or ""),
        language=str(meta.get("language") or "en"),
        rights=str(meta.get("rights") or "unknown"),
        source=str(meta.get("source") or ""),
        source_url=str(meta.get("source_url") or ""),
        tags=[str(t) for t in (meta.get("tags") or [])],
        path=str(path),
    )
    return LoadedWork(work=work, body=body)


def normalize_body(body: str) -> str:
    """Tidy without destroying structure.

    Blank lines are load-bearing here — they are the paragraph boundaries the
    chunker splits on — so they survive, while trailing whitespace, tabs and
    runs of three or more blank lines do not.
    """
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    body = "\n".join(normalize(line) for line in body.split("\n"))
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def iter_source_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES and not path.name.startswith("."):
            yield path


def load_library(root: Path, *, strict: bool = False) -> tuple[list[LoadedWork], list[str]]:
    """Load every text under `root`.

    Returns (works, errors).  One bad file does not abort the ingest — the
    errors are reported and the rest of the library still indexes.
    """
    works: list[LoadedWork] = []
    errors: list[str] = []
    seen_ids: dict[str, Path] = {}

    if not root.exists():
        return [], [f"library directory not found: {root}"]

    for path in iter_source_files(root):
        try:
            loaded = load_file(path, strict=strict)
        except CorpusError as exc:
            errors.append(str(exc))
            continue
        except Exception as exc:  # pragma: no cover - unexpected IO
            errors.append(f"{path.name}: {exc}")
            continue

        if loaded.work.id in seen_ids:
            # Duplicate ids would make citations ambiguous, so disambiguate
            # rather than silently overwrite.
            other = seen_ids[loaded.work.id]
            loaded.work.id = f"{loaded.work.id}-{slugify(path.stem)[:12]}"
            errors.append(
                f"{path.name}: duplicate work id (also in {other.name}); "
                f"renamed to {loaded.work.id}"
            )
        seen_ids[loaded.work.id] = path
        works.append(loaded)

    return works, errors
