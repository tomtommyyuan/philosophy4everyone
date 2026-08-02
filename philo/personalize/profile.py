"""User profiles.

A profile is a small JSON file — deliberately, so it is editable by hand and
obvious what the system knows about you.  It feeds two things: which themes
the daily piece draws from, and a short "about this reader" note appended to
the system prompt.

That note may shape *examples, tone and length*.  It may never shape what the
sources are taken to say.  Personalisation that bends the philosophy toward
what the reader wants to hear is the same failure as hallucination, wearing
friendlier clothes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date as _date
from pathlib import Path
from typing import Any

DEFAULT_PROFILE_NAME = "default"

LEVELS = {
    "new": "has never read philosophy; assume no vocabulary at all",
    "curious": "reads popular non-fiction; knows a few names, no technical training",
    "student": "has taken an introductory course; recognises the standard vocabulary",
    "adept": "reads primary texts comfortably; wants precision over hand-holding",
}


@dataclass
class Profile:
    name: str = DEFAULT_PROFILE_NAME
    display_name: str = ""
    language: str = "en"                     # "en" | "zh"
    level: str = "curious"
    interests: list[str] = field(default_factory=list)
    philosophers: list[str] = field(default_factory=list)
    traditions: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    tone: str = ""
    daily_length: str = "short"              # short | medium
    history: list[dict[str, str]] = field(default_factory=list)
    path: Path | None = None

    # ------------------------------------------------------------------
    @classmethod
    def default(cls, name: str = DEFAULT_PROFILE_NAME) -> "Profile":
        return cls(
            name=name,
            display_name="",
            language="en",
            level="curious",
            interests=[
                "how to handle anxiety",
                "what makes a life good",
                "free will and responsibility",
                "how to think about death",
                "knowing what is true",
            ],
            philosophers=[],
            traditions=["Stoicism", "Daoism"],
            avoid=[],
            tone="warm, concrete, a little wry",
        )

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "Profile":
        data = json.loads(Path(path).read_text("utf-8"))
        known = {f for f in cls.__dataclass_fields__} - {"path"}
        profile = cls(**{k: v for k, v in data.items() if k in known})
        profile.path = Path(path)
        if not profile.name:
            profile.name = Path(path).stem
        return profile

    @classmethod
    def load_or_default(cls, profiles_dir: Path, name: str = DEFAULT_PROFILE_NAME) -> "Profile":
        path = Path(profiles_dir) / f"{name}.json"
        if path.is_file():
            return cls.load(path)
        profile = cls.default(name)
        profile.path = path
        return profile

    def save(self, path: Path | None = None) -> Path:
        target = Path(path or self.path or Path(f"{self.name}.json"))
        target.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data.pop("path", None)
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.path = target
        return target

    # ------------------------------------------------------------------
    def reader_note(self) -> str:
        """The prompt-facing description.  Style only — never content."""
        bits: list[str] = []
        who = self.display_name or "the reader"
        bits.append(f"{who}: {LEVELS.get(self.level, LEVELS['curious'])}.")
        if self.interests:
            bits.append("Cares about: " + "; ".join(self.interests[:5]) + ".")
        if self.philosophers:
            bits.append("Drawn to: " + ", ".join(self.philosophers[:5]) + ".")
        if self.traditions:
            bits.append("Traditions they enjoy: " + ", ".join(self.traditions[:4]) + ".")
        if self.avoid:
            bits.append("Prefers you steer clear of: " + ", ".join(self.avoid[:4]) + ".")
        if self.tone:
            bits.append(f"Tone that lands well: {self.tone}.")
        bits.append(
            "Adjust examples, tone and length to fit — never adjust what the sources say, "
            "and do not soften a position because it is uncomfortable."
        )
        return " ".join(bits)

    def bias_terms(self) -> list[str]:
        """Extra query terms that tilt retrieval toward this reader's tastes."""
        return [*self.philosophers, *self.traditions]

    def recent_themes(self, limit: int = 8) -> list[str]:
        return [h.get("theme", "") for h in self.history[-limit:] if h.get("theme")]

    def recent_titles(self, limit: int = 6) -> list[str]:
        return [h.get("title", "") for h in self.history[-limit:] if h.get("title")]

    def record(self, day: str, theme: str, title: str) -> None:
        self.history.append({"date": day, "theme": theme, "title": title})
        self.history = self.history[-60:]

    def already_sent(self, day: str) -> dict[str, str] | None:
        for entry in reversed(self.history):
            if entry.get("date") == day:
                return entry
        return None

    def summary_rows(self) -> list[tuple[str, str]]:
        return [
            ("name", self.name),
            ("reader", self.display_name or "—"),
            ("language", self.language),
            ("level", f"{self.level} — {LEVELS.get(self.level, '')}"),
            ("interests", ", ".join(self.interests) or "—"),
            ("philosophers", ", ".join(self.philosophers) or "—"),
            ("traditions", ", ".join(self.traditions) or "—"),
            ("avoid", ", ".join(self.avoid) or "—"),
            ("tone", self.tone or "—"),
            ("daily sent", str(len(self.history))),
        ]


def list_profiles(profiles_dir: Path) -> list[Path]:
    directory = Path(profiles_dir)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.json") if p.is_file())


def today_str() -> str:
    return _date.today().isoformat()


def as_json(profile: Profile) -> dict[str, Any]:
    data = asdict(profile)
    data.pop("path", None)
    return data
