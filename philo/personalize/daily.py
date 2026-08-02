"""Daily Philosophy — one short personalised piece, still fully sourced.

Theme selection is *deterministic given (date, profile)*: the same person on
the same day always gets the same piece, and re-running is free.  Variation
across days comes from the date, not from randomness, which makes the feature
testable and makes "why did I get this today" answerable.

Two ideas keep it from going stale:

- themes come from the reader's stated interests *and* from tags the library
  actually contains, so the feature degrades gracefully when the two do not
  overlap;
- themes used in the last stretch of history are skipped, so a five-interest
  profile does not send the same piece every fifth day.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..config import Settings
from ..generation.answerer import Engine
from ..generation.prompts import audit_markers, build_daily_messages, parse_sections
from ..models import DailyPiece
from ..store.vector_store import Filters
from ..util import detect_language, stable_seed, truncate
from .profile import Profile, today_str

# Fallbacks when a profile lists no interests and the library carries no tags.
GENERIC_THEMES = [
    "what is actually up to me",
    "how to face what I cannot change",
    "what makes a life go well",
    "why we fear death",
    "how to know when I am wrong",
    "what I owe to other people",
    "the value of doing nothing",
    "whether desire can ever be satisfied",
    "how to act when the rules run out",
    "what it means to be free",
]


@dataclass
class DailyResult:
    piece: DailyPiece
    theme_pool: list[str]
    took_ms: int = 0
    grounded: bool = True
    reused: bool = False


def candidate_themes(profile: Profile, engine: Engine, limit: int = 24) -> list[str]:
    """Themes worth writing about for this reader, best first."""
    themes: list[str] = []
    seen: set[str] = set()

    def push(value: str) -> None:
        key = value.strip().lower()
        if key and key not in seen:
            seen.add(key)
            themes.append(value.strip())

    for interest in profile.interests:
        push(interest)

    # Tags the library actually has — keeps the pool honest about what can be
    # sourced, and adds range beyond the reader's stated interests.
    try:
        for tag, count in engine.store.tags():
            if count >= 2:
                push(tag.replace("-", " "))
    except Exception:
        pass

    for theme in GENERIC_THEMES:
        push(theme)

    return themes[:limit]


def pick_theme(profile: Profile, pool: list[str], day: str) -> str:
    if not pool:
        return GENERIC_THEMES[0]
    recent = {t.strip().lower() for t in profile.recent_themes(limit=max(3, len(pool) // 2))}
    fresh = [t for t in pool if t.strip().lower() not in recent] or pool
    seed = stable_seed(day, profile.name, len(fresh))
    return fresh[seed % len(fresh)]


def generate_daily(
    engine: Engine,
    profile: Profile,
    settings: Settings,
    *,
    day: str = "",
    theme: str = "",
    k: int = 5,
    stream_cb=None,
    save: bool = True,
) -> DailyResult:
    started = time.perf_counter()
    day = day or today_str()
    pool = candidate_themes(profile, engine)
    chosen = theme or pick_theme(profile, pool, day)

    lang = profile.language or detect_language(chosen)

    # Soft bias, not a hard filter: appending the reader's favourite names to
    # the *query* tilts retrieval toward them while still allowing a better
    # passage from elsewhere to win. A hard filter would silo the reader.
    query = chosen
    bias = profile.bias_terms()
    if bias:
        query = f"{chosen} — {', '.join(bias[:4])}"

    result = engine.retriever.search(query, k=k, filters=Filters())
    if not result.hits:
        result = engine.retriever.search(chosen, k=k, filters=Filters(), min_score=0.0)

    if not result.hits:
        piece = DailyPiece(
            date=day,
            theme=chosen,
            title="Nothing to send today" if lang != "zh" else "今天没有可靠的内容",
            hook=(
                "The library has no passage that speaks to today's theme, and inventing one "
                "would defeat the point. Add texts to library/ and run `philo ingest`."
                if lang != "zh"
                else "今天的主题在当前library里找不到可引用的原文。与其编造，不如先把相关文本放进 "
                "library/ 再运行 `philo ingest`。"
            ),
            profile_name=profile.name,
            provider=engine.provider.name,
        )
        return DailyResult(
            piece=piece, theme_pool=pool, grounded=False,
            took_ms=int((time.perf_counter() - started) * 1000),
        )

    messages = build_daily_messages(
        chosen,
        result.hits,
        lang=lang,
        reader_note=profile.reader_note(),
        avoid=profile.recent_titles(),
    )
    completion = engine.provider.chat(
        messages,
        temperature=min(0.85, settings.temperature + 0.35),  # a little more life than Q&A
        max_tokens=min(settings.max_tokens, 900),
        stream_cb=stream_cb,
        task="daily",
    )

    valid = {h.marker for h in result.hits}
    text, invented = audit_markers(completion.text, valid)
    sections = parse_sections(text)

    quote = sections.get("QUOTE", "").strip()
    quote_marker = _first_marker(quote)

    piece = DailyPiece(
        date=day,
        theme=chosen,
        title=_clean_title(sections.get("TITLE", "")) or chosen,
        hook=sections.get("HOOK", "").strip() or sections.get("_PREAMBLE", "").strip(),
        quote=quote,
        quote_marker=quote_marker,
        reflection=sections.get("REFLECTION", "").strip(),
        practice=sections.get("PRACTICE", "").strip(),
        sources=list(result.hits),
        raw=completion.text,
        provider=completion.provider or engine.provider.name,
        model=completion.model,
        profile_name=profile.name,
    )

    if save and profile.path is not None:
        profile.record(day, chosen, truncate(piece.title, 90))
        try:
            profile.save()
        except OSError:
            pass  # a read-only profile should never lose the reader their piece

    return DailyResult(
        piece=piece,
        theme_pool=pool,
        grounded=bool(result.hits) and not (invented and not valid),
        took_ms=int((time.perf_counter() - started) * 1000),
    )


def _first_marker(text: str) -> int:
    import re

    m = re.search(r"\[(\d{1,2})\]", text)
    return int(m.group(1)) if m else 0


def _clean_title(title: str) -> str:
    return title.strip().strip("#").strip().strip('"“”').strip()
