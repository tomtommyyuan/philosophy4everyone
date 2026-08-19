"""The check-in: name how you feel, hear what several schools say about it.

The engineering that matters here is not the prompt. It is the **vocabulary
bridge** in `MOODS` below.

The names people give feelings today are not always the names the
translators used, and the gap is uneven. Counted over this library with the
same tokeniser BM25 uses:

    mood         its modern name    its period terms
    worried                    0                2700
    stuck                      0                 437
    lonely                     6                 859
    frustrated                 4                  75
    angry                    138                 161

"Angry" needs no help. **"Worried" does not appear in this library once** —
the passages that are about it say *fear*, *apprehension*, *dread*,
*solicitude*. Ask in the modern word and hybrid retrieval has nothing to
match on lexically and only a weak dense signal, so it returns the least
irrelevant paragraph it owns rather than the one Epictetus wrote about
exactly this.

So each mood carries the words the texts themselves reach for, and the query
is those plus whatever the reader typed. The project's refusal message has
been telling people to do this by hand — "rephrase using words the texts
themselves would use" — and this does it for them, which is the difference
between a feature and a workaround.

Terms are chosen to be *distinctive*, not merely synonymous. An earlier draft
had "worried" widen to "things to come", which tokenises to `things` (1998
occurrences) and `come` (317): BM25's IDF discounts them to nearly nothing,
but they still smear the dense query vector toward no topic in particular.

One completion, not one per school. A daily check-in that costs four model
calls is a daily check-in nobody leaves switched on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..models import ScoredChunk
from ..store.vector_store import Filters
from ..util import detect_language
from .prompts import audit_markers, build_mood_messages, parse_sections


@dataclass(frozen=True)
class Mood:
    key: str
    en: str
    zh: str
    # What the translations call this. Not synonyms of the modern word — the
    # words a 19th-century translator actually reached for.
    terms: tuple[str, ...]

    def label(self, lang: str = "en") -> str:
        return self.zh if lang == "zh" else self.en


MOODS: tuple[Mood, ...] = (
    Mood("angry", "Angry", "生气",
         ("anger", "wrath", "provocation", "revenge", "injury", "insult")),
    Mood("worried", "Worried", "焦虑",
         ("fear", "anxious", "apprehension", "dread", "solicitude", "danger", "uncertainty")),
    Mood("frustrated", "Frustrated", "烦躁",
         ("hindrance", "impediment", "vexation", "thwarted", "obstacle", "impatience")),
    Mood("sad", "Sad", "低落",
         ("grief", "sorrow", "mourning", "loss", "misfortune", "affliction")),
    Mood("lonely", "Lonely", "孤独",
         ("solitude", "alone", "friendship", "company", "society", "retirement")),
    Mood("stuck", "Stuck", "卡住",
         ("irresolution", "doubt", "deliberation", "choice", "will", "beginning")),
    Mood("restless", "Restless", "静不下来",
         ("tranquillity", "disturbance", "desire", "quiet", "wandering", "distraction")),
    Mood("ashamed", "Ashamed", "羞愧",
         ("shame", "disgrace", "blame", "reproach", "fault", "conscience")),
    Mood("grateful", "Grateful", "感激",
         ("gratitude", "benefits", "kindness", "fortune", "thanks", "friendship")),
    Mood("happy", "Happy", "高兴",
         ("happiness", "joy", "delight", "contentment", "blessed", "cheerfulness")),
)

BY_KEY = {m.key: m for m in MOODS}


def resolve(name: str) -> Mood | None:
    """Accept the key, the English label or the Chinese label."""
    needle = (name or "").strip().lower()
    if not needle:
        return None
    for mood in MOODS:
        if needle in (mood.key, mood.en.lower(), mood.zh):
            return mood
    return None


@dataclass
class MoodReading:
    mood: str = ""
    mood_label: str = ""
    reason: str = ""
    feeling: str = ""          # what is happening, in plain words
    schools: str = ""          # two or three schools, each with a real quote
    practice: str = ""         # one thing to try in the next ten minutes
    sources: list[ScoredChunk] = field(default_factory=list)
    raw: str = ""
    grounded: bool = True
    lang: str = "en"
    provider: str = ""
    model: str = ""
    took_ms: int = 0
    invented_markers: set[int] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "mood": self.mood,
            "mood_label": self.mood_label,
            "reason": self.reason,
            "feeling": self.feeling,
            "schools": self.schools,
            "practice": self.practice,
            "grounded": self.grounded,
            "lang": self.lang,
            "provider": self.provider,
            "model": self.model,
            "took_ms": self.took_ms,
            "invented_markers": sorted(self.invented_markers),
            "sources": [s.to_dict() for s in self.sources],
        }


def retrieval_query(mood: Mood, reason: str) -> str:
    """The reader's words plus the texts' words for the same thing.

    Both halves earn their place: the terms make the passage findable at all,
    the reason decides *which* passage about anger is the right one.
    """
    terms = ", ".join(mood.terms)
    return f"{terms} — {reason.strip()}" if reason.strip() else terms


def read_mood(
    engine,
    mood_name: str,
    *,
    reason: str = "",
    k: int = 6,
    lang: str = "",
    reader_note: str = "",
    chat_model: str = "",
    chat_provider: str = "",
    schools: int = 3,
) -> MoodReading:
    from .answerer import AskOptions

    started = time.perf_counter()
    mood = resolve(mood_name)
    if mood is None:
        known = ", ".join(m.key for m in MOODS)
        raise ValueError(f"unknown mood {mood_name!r} — one of: {known}")

    lang = lang or (detect_language(reason) if reason.strip() else "en")
    backend = engine.chat_backend(
        AskOptions(chat_model=chat_model, chat_provider=chat_provider)
    )

    result = engine.retriever.search(
        retrieval_query(mood, reason), k=k, filters=Filters()
    )
    if not result.hits:
        return MoodReading(
            mood=mood.key,
            mood_label=mood.label(lang),
            reason=reason,
            feeling=_nothing_found(mood, lang),
            grounded=False,
            lang=lang,
            provider=backend.name,
            took_ms=int((time.perf_counter() - started) * 1000),
        )

    messages = build_mood_messages(
        mood.label("en"),
        result.hits,
        reason=reason,
        lang=lang,
        reader_note=reader_note,
        schools=schools,
    )
    completion = backend.chat(
        messages,
        temperature=engine.settings.temperature,
        max_tokens=min(engine.settings.max_tokens, 1100),
        task="mood",
        model=chat_model,
    )

    text, invented = audit_markers(completion.text, {h.marker for h in result.hits})
    sections = parse_sections(text)

    return MoodReading(
        mood=mood.key,
        mood_label=mood.label(lang),
        reason=reason,
        feeling=sections.get("FEELING", "").strip() or sections.get("_PREAMBLE", "").strip(),
        schools=sections.get("SCHOOLS", "").strip(),
        practice=sections.get("PRACTICE", "").strip(),
        sources=list(result.hits),
        raw=completion.text,
        grounded=True,
        lang=lang,
        provider=completion.provider or backend.name,
        model=completion.model,
        took_ms=int((time.perf_counter() - started) * 1000),
        invented_markers=invented,
    )


def _nothing_found(mood: Mood, lang: str) -> str:
    if lang == "zh":
        return (
            f"这个library里没有能支撑「{mood.zh}」这件事的原文，所以我不编。"
            "把相关文本放进 library/ 再运行 `philo ingest` 就会有。"
        )
    return (
        f"Nothing in this library speaks closely enough to feeling {mood.en.lower()} "
        "to be worth quoting, so I am not going to invent something. Adding texts to "
        "library/ and re-running `philo ingest` would change that."
    )
