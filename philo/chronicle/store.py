"""An append-only commonplace book.

Three kinds of thing end up here, and they are deliberately one type rather
than three tables:

- **passages** you marked while reading — the commonplace book proper;
- **decisions** you actually faced, with the frameworks the texts supply;
- **questions** you asked, so the record shows what you were circling.

One JSONL file per profile, appended to and never rewritten in place. That is
not laziness about a database: a commonplace book whose value is cumulative
should be impossible to lose to a half-finished write, should survive being
opened in a text editor, and should be greppable by the person whose record
it is. `philo chronicle --path` prints where it lives precisely so it can be
backed up, diffed, or thrown away by hand.

Deletion rewrites the whole file, which is the one operation that can lose
data — so it is the one operation that writes to a temporary file and
renames.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

KINDS = ("passage", "decision", "question")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def day_of(stamp: str) -> str:
    return (stamp or "")[:10]


@dataclass
class Entry:
    id: str = ""
    kind: str = "passage"
    created: str = ""
    # For a passage this is the quoted text; for a decision or question it is
    # what the reader wrote. Either way it is the thing being remembered.
    text: str = ""
    note: str = ""                      # the reader's own words about it
    # Provenance — only meaningful for a saved passage.
    chunk_id: str = ""
    philosopher: str = ""
    work_title: str = ""
    section: str = ""
    tradition: str = ""
    # What the system generated in response (decisions). Kept so the recap can
    # reason over it without a second model call.
    response: str = ""
    # Lightweight citations, not whole chunks: the index already holds the
    # text, and duplicating it here would make the file grow without limit.
    citations: list[dict[str, str]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.created:
            self.created = now_iso()
        if not self.id:
            self.id = make_id(self.kind, self.created, self.text)

    @property
    def day(self) -> str:
        return day_of(self.created)

    @property
    def citation(self) -> str:
        bits = [self.philosopher, self.work_title, self.section]
        return " · ".join(b for b in bits if b)

    def headline(self, limit: int = 72) -> str:
        """One line for a list view."""
        from ..util import truncate

        if self.kind == "passage":
            return truncate(self.text.replace("\n", " "), limit)
        return truncate((self.note or self.text).replace("\n", " "), limit)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Entry":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def make_id(kind: str, created: str, text: str) -> str:
    digest = hashlib.sha1(f"{kind}|{created}|{text}".encode("utf-8")).hexdigest()[:10]
    return f"{kind[:1]}{digest}"


class Chronicle:
    """The record for one profile."""

    def __init__(self, path: Path, entries: Sequence[Entry] = ()) -> None:
        self.path = Path(path)
        self.entries: list[Entry] = list(entries)

    # -- io ---------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "Chronicle":
        """Read the file, skipping any line that is not a whole record.

        A truncated final line is exactly what a crashed append leaves
        behind. Refusing to open the book because of it would punish the
        reader for the writer's failure.
        """
        path = Path(path)
        entries: list[Entry] = []
        if path.is_file():
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(Entry.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, TypeError):
                        continue
        return cls(path, entries)

    @classmethod
    def for_profile(cls, directory: Path, profile: str = "default") -> "Chronicle":
        return cls.load(Path(directory) / f"{profile or 'default'}.jsonl")

    def add(self, entry: Entry) -> Entry:
        """Append one record. The only write that is not a rewrite."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        self.entries.append(entry)
        return entry

    def remove(self, entry_id: str) -> bool:
        """The one lossy operation, so the one that writes atomically."""
        keep = [e for e in self.entries if e.id != entry_id]
        if len(keep) == len(self.entries):
            return False
        self.entries = keep
        self._rewrite()
        return True

    def _rewrite(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for entry in self.entries:
                    fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            os.replace(tmp, self.path)
        finally:
            Path(tmp).unlink(missing_ok=True)

    # -- reading ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterable[Entry]:
        return iter(self.entries)

    def newest_first(self) -> list[Entry]:
        return sorted(self.entries, key=lambda e: e.created, reverse=True)

    def of_kind(self, kind: str) -> list[Entry]:
        return [e for e in self.entries if e.kind == kind]

    def since(self, day: str) -> list[Entry]:
        """Entries from `day` (YYYY-MM-DD) onward, oldest first."""
        return sorted(
            (e for e in self.entries if e.day >= day), key=lambda e: e.created
        )

    def before(self, day: str) -> list[Entry]:
        return sorted((e for e in self.entries if e.day < day), key=lambda e: e.created)

    def has_chunk(self, chunk_id: str) -> bool:
        return any(e.chunk_id == chunk_id for e in self.entries if e.chunk_id)

    def counts(self) -> dict[str, int]:
        out = {kind: 0 for kind in KINDS}
        for entry in self.entries:
            out[entry.kind] = out.get(entry.kind, 0) + 1
        return out

    def days_active(self) -> int:
        return len({e.day for e in self.entries if e.day})
