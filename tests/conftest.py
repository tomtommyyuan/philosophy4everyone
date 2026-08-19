"""Shared fixtures.

Every test runs against a temporary library and index, never the real one, and
always with the offline provider — so the suite is hermetic, needs no key, and
makes no network call.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from philo.config import Settings
from philo.providers.mock import MockProvider

# Two short works with deliberately different shapes: one with book headings
# and ordinary paragraphs, one with very short numbered aphorisms (the case
# that breaks naive chunkers).
STOIC = """\
---
philosopher: Epictetus
philosopher_zh: 爱比克泰德
work: Enchiridion
work_zh: 手册
translator: Elizabeth Carter
year: c. 125 CE
tradition: Stoicism
era: Ancient
rights: public-domain
source: Test fixture
tags: [control, freedom, desire]
---

## Chapter I

Some things are in our control and others not. Things in our control are
opinion, pursuit, desire, aversion, and, in a word, whatever are our own
actions. Things not in our control are body, property, reputation, command,
and, in one word, whatever are not our own actions.

The things in our control are by nature free, unrestrained, unhindered; but
those not in our control are weak, slavish, restrained, belonging to others.
Remember, then, that if you suppose that things by nature slavish are free,
and that what belongs to others is your own, then you will be hindered.

## Chapter V

Men are disturbed not by the things which happen, but by the opinions about
the things. Death, for instance, is not terrible, else it would have appeared
so to Socrates. But the terror consists in our notion of death, that it is
terrible.

When therefore we are hindered, or disturbed, or grieved, let us never impute
it to others, but to ourselves; that is, to our own opinions. It is the action
of an uninstructed person to reproach others for his own misfortunes.
"""

APHORISMS = """\
---
philosopher: Laozi
work: Tao Te Ching
translator: James Legge
tradition: Daoism
era: Ancient
rights: public-domain
tags:
  - wu-wei
  - simplicity
  - water
---

## Chapter 8

The highest excellence is like that of water.

The excellence of water appears in its benefiting all things.

And in its occupying, without striving, the low place which all men dislike.

## Chapter 22

The partial becomes complete.

The crooked becomes straight.

The empty becomes full.

The worn out becomes new.
"""


@pytest.fixture
def library(tmp_path: Path) -> Path:
    directory = tmp_path / "library"
    directory.mkdir()
    (directory / "epictetus-enchiridion.md").write_text(STOIC, encoding="utf-8")
    (directory / "laozi-tao-te-ching.md").write_text(APHORISMS, encoding="utf-8")
    return directory


@pytest.fixture
def settings(tmp_path: Path, library: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    # Every provider prefix, not just the two this fixture configures. A real
    # ANTHROPIC_API_KEY in the developer's shell otherwise leaks in and the
    # suite stops being hermetic — it starts reporting on whoever ran it.
    for name in list(os.environ):
        if name.startswith(
            ("PHILO_", "OPENAI_", "AZURE_OPENAI_", "ANTHROPIC_", "GEMINI_", "GOOGLE_")
        ):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PHILO_PROVIDER", "mock")
    monkeypatch.setenv("PHILO_MOCK_DELAY", "0")
    s = Settings.load(root=tmp_path)
    s.library_dir = library
    s.index_dir = tmp_path / ".philo" / "index"
    s.profiles_dir = tmp_path / "profiles"
    return s


@pytest.fixture
def provider() -> MockProvider:
    return MockProvider()
