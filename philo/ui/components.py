"""Reusable renderables.

Everything the CLI puts on screen is built from these, so the interface stays
consistent: one place decides what a citation looks like, what a score meter
looks like, and how a panel is titled.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from rich.align import Align
from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ..models import Answer, DailyPiece, ScoredChunk
from ..util import char_weight, human_ms, snippet, truncate
from .theme import (
    AMBER,
    PANEL_BOX,
    SLATE,
    TABLE_BOX,
    VIOLET,
    VIOLET_DEEP,
    gradient_text,
    score_bar,
)

# --------------------------------------------------------------------------
# Bilingual labels — the UI speaks whichever language the reader is using.
# --------------------------------------------------------------------------

LABELS: dict[str, tuple[str, str]] = {
    "plain": ("In plain words", "白话讲解"),
    "academic": ("The argument", "论证与文本"),
    "sources": ("Sources", "出处"),
    "passages": ("Passages", "原文段落"),
    "no_sources": ("No sources — nothing was asserted", "无出处 · 未作任何断言"),
    "question": ("Question", "问题"),
    "theme": ("Theme", "主题"),
    "today": ("Daily Philosophy", "每日一哲"),
    "try": ("Try this today", "今天试试看"),
    "reflection": ("Why it is not obvious", "为什么它并不显然"),
    "library": ("Library", "文本库"),
    "philosopher": ("Philosopher", "哲学家"),
    "work": ("Work", "著作"),
    "section": ("Section", "章节"),
    "relevance": ("Relevance", "相关度"),
    "council": ("The council", "众说"),
    "objection": ("The objection", "反对意见"),
    "upshot": ("Where this leaves it", "分歧在哪里"),
    "against": ("against", "针对"),
    "raised_by": ("raised from", "依据"),
    "silent": ("could not speak", "未能发言"),
    "chronicle": ("The chronicle", "记事"),
    "choice": ("What you are choosing", "你在选的是什么"),
    "tests": ("What the texts would ask", "文本会问你什么"),
    "limits": ("What they do not settle", "它们解决不了的"),
    "week": ("The week", "这一周"),
    "thread": ("The thread", "线索"),
    "sit": ("What to sit with", "留给下周"),
    "saved": ("saved", "摘录"),
    "decided": ("decided", "决定"),
    "asked": ("asked", "提问"),
    "echo": ("you kept this", "你之前存过"),
    "tradition": ("Tradition", "传统"),
    "rights": ("Rights", "版权"),
    "chunks": ("Passages", "段落数"),
    "translator": ("Translator", "译者"),
    "grounded": ("grounded in", "依据"),
    "offline": ("offline mock — no API called", "离线模拟 · 未调用任何 API"),
}


def L(key: str, lang: str = "en") -> str:
    en, zh = LABELS.get(key, (key, key))
    return zh if lang == "zh" else en


# --------------------------------------------------------------------------
# Chrome
# --------------------------------------------------------------------------

# The one place the tagline lives, so the terminal and the help screen cannot
# drift apart. The web page carries its own copy — it is a static file with
# nothing to import from.
TAGLINE = "plain words, real books · 让哲学说人话，句句有据"


def banner(*, compact: bool = False, tagline: str = "") -> RenderableType:
    wordmark = gradient_text("P H I L O S O P H Y   F O R   E V E R Y O N E")
    if compact:
        return Padding(wordmark, (0, 0, 0, 1))

    sub = tagline or TAGLINE
    body = Group(
        Text(""),
        Padding(wordmark, (0, 0, 0, 2)),
        Padding(Text(sub, style="muted"), (0, 0, 0, 2)),
        Text(""),
    )
    return body


def rule(title: str = "", *, style: str = "rule.line", align: str = "left") -> Rule:
    return Rule(Text(title, style="subheading") if title else "", style=style, align=align)  # type: ignore[arg-type]


def hint(text: str) -> Text:
    return Text.assemble(("  › ", "hint"), (text, "muted"))


def keyval(rows: Sequence[tuple[str, Any]], *, key_style: str = "muted", value_style: str = "") -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style=key_style, no_wrap=True)
    table.add_column(overflow="fold", style=value_style)
    for key, value in rows:
        table.add_row(Text(str(key)), value if isinstance(value, (Text,)) else Text(str(value)))
    return table


def status_bar(bits: Sequence[tuple[str, str]]) -> Text:
    """`provider mock · model mock-sage-1 · 6 sources · 240ms`"""
    out = Text("  ")
    for i, (label, value) in enumerate(bits):
        if i:
            out.append(" · ", style="dim")
        out.append(f"{label} ", style="dim")
        out.append(value, style="muted")
    return out


# --------------------------------------------------------------------------
# Answers
# --------------------------------------------------------------------------


_MARKER_RE = re.compile(r"(?<!\*)\[(\d{1,2})\](?!\*)")


def _prep_markdown(text: str) -> str:
    """Make citation markers bold so Markdown renders them in amber.

    Rich's Markdown owns the text once it takes over, so there is no hook to
    restyle a substring afterwards. Wrapping `[3]` in `**…**` routes it
    through `markdown.strong`, which the theme paints amber — the markers
    stay visible without giving up real Markdown typography.
    """
    return _MARKER_RE.sub(r"**[\1]**", text)


def prose(text: str) -> Markdown:
    return Markdown(_prep_markdown(text), code_theme="ansi_dark")


def _panel_title(glyph: str, label: str, *, glyph_style: str, label_style: str) -> Text:
    return Text.assemble((f"{glyph} ", glyph_style), (label, label_style))


def answer_panels(answer: Answer, *, lang: str = "en", show_academic: bool = True) -> RenderableType:
    parts: list[RenderableType] = []

    parts.append(
        Panel(
            Padding(prose(answer.plain or "—"), (1, 2)),
            title=_panel_title("◗", L("plain", lang), glyph_style=f"bold {VIOLET}", label_style="heading"),
            title_align="left",
            border_style="frame" if answer.grounded else "warn",
            box=PANEL_BOX,
            padding=0,
        )
    )

    if show_academic and answer.academic.strip():
        parts.append(Text(""))
        parts.append(
            Panel(
                Padding(prose(answer.academic), (1, 2)),
                title=_panel_title("◗", L("academic", lang), glyph_style=f"bold {AMBER}", label_style="subheading"),
                title_align="left",
                border_style="frame.soft",
                box=PANEL_BOX,
                padding=0,
            )
        )
    return Group(*parts)


def sources_table(hits: Sequence[ScoredChunk], *, lang: str = "en", cited: set[int] | None = None) -> Table:
    table = Table(
        box=TABLE_BOX,
        border_style="frame.soft",
        header_style="muted",
        expand=True,
        pad_edge=False,
        show_edge=False,
    )
    table.add_column("#", justify="right", width=2, style="cite", no_wrap=True)
    table.add_column(L("philosopher", lang), style="source.author", no_wrap=True,
                     overflow="ellipsis", max_width=14)
    table.add_column(L("work", lang), style="source.work", no_wrap=True,
                     overflow="ellipsis", max_width=18)
    table.add_column(L("section", lang), style="source.meta", no_wrap=True,
                     overflow="ellipsis", max_width=14)
    table.add_column(L("relevance", lang), justify="left", width=11, no_wrap=True)

    for hit in hits:
        c = hit.chunk
        used = cited is None or hit.marker in cited
        marker = Text(f"{hit.marker}", style="cite" if used else "dim")
        author = Text(c.philosopher or "—", style="source.author" if used else "dim")
        work = Text(c.work_title or "—", style="source.work" if used else "dim")
        # Only the deepest part of the heading trail: the parent sections are
        # already implied by the work, and the full path costs six lines of
        # wrapping for texts like the Discourses.
        section = Text(deepest_section(c.section), style="source.meta")
        meter = Text.assemble((f"{hit.score:.2f} ", "dim"), score_bar(hit.score, 6))
        table.add_row(marker, author, work, section, meter)
    return table


def deepest_section(section: str) -> str:
    return section.split(" · ")[-1].strip() if section else "—"


def source_cards(hits: Sequence[ScoredChunk], *, lang: str = "en", chars: int = 420) -> RenderableType:
    """The retrieved passages themselves — the receipts.

    Shown on demand (`--show-sources`) because the point of a citation is that
    the reader *can* check it, which requires the text to be one keystroke
    away rather than in a file somewhere.
    """
    cards: list[RenderableType] = []
    for hit in hits:
        c = hit.chunk
        header = Text()
        header.append(f" [{hit.marker}] ", style="cite")
        header.append(c.philosopher or "—", style="source.author")
        if c.work_title:
            header.append(" · ", style="dim")
            header.append(c.work_title, style="source.work")
        if c.section:
            header.append(" · ", style="dim")
            header.append(c.section, style="source.meta")


        meta_bits = []
        if c.translator:
            meta_bits.append(f"tr. {c.translator}")
        if c.tradition:
            meta_bits.append(c.tradition)
        if c.rights:
            meta_bits.append(c.rights)
        meta = Text("  ".join(meta_bits), style="dim")

        body = Text(truncate(c.text, chars), style="quote")
        cards.append(
            Panel(
                Group(body, Text(""), meta) if meta_bits else body,
                title=header,
                title_align="left",
                border_style="frame.soft",
                box=PANEL_BOX,
                padding=(1, 2),
            )
        )
        cards.append(Text(""))
    return Group(*cards[:-1]) if cards else Text("—", style="dim")


def answer_view(
    answer: Answer,
    *,
    lang: str = "en",
    show_sources: bool = False,
    show_academic: bool = True,
) -> RenderableType:
    parts: list[RenderableType] = []

    if answer.mode == "direct":
        # An unsourced answer must never be able to pass for a sourced one.
        parts.append(
            Panel(
                Padding(
                    Text(
                        "Answered from the model's own recollection — no passages were "
                        "retrieved, and nothing here is checkable against a text.",
                        style="warn",
                    ),
                    (0, 1),
                ),
                border_style="warn",
                box=PANEL_BOX,
                padding=0,
            )
        )
        parts.append(Text(""))

    parts.append(answer_panels(answer, lang=lang, show_academic=show_academic))

    if answer.mode == "direct":
        return Group(*parts)

    if answer.sources:
        parts.append(Text(""))
        parts.append(rule(f"{L('sources', lang)}  ·  {len(answer.sources)}"))
        parts.append(Text(""))
        parts.append(sources_table(answer.sources, lang=lang, cited=answer.cited_markers))
        if show_sources:
            parts.append(Text(""))
            parts.append(rule(L("passages", lang), style="frame.soft"))
            parts.append(Text(""))
            parts.append(source_cards(answer.sources, lang=lang))
    else:
        parts.append(Text(""))
        parts.append(Text(f"  {L('no_sources', lang)}", style="warn"))

    return Group(*parts)


def answer_footer(answer: Answer, *, offline: bool = False) -> Text:
    provider = answer.provider or "—"
    bits = [
        ("via", f"{provider} (offline)" if offline else provider),
        ("model", answer.model or "—"),
        ("sources", str(len(answer.sources))),
        ("took", human_ms(answer.latency_ms)),
    ]
    if answer.usage.get("total_tokens"):
        bits.append(("tokens", str(answer.usage["total_tokens"])))
    return status_bar(bits)


# --------------------------------------------------------------------------
# Daily
# --------------------------------------------------------------------------


def daily_card(piece: DailyPiece, *, lang: str = "en", show_sources: bool = False) -> RenderableType:
    header = Text()
    header.append("  ✦  ", style=f"bold {AMBER}")
    header.append(L("today", lang), style="heading")
    header.append(f"   {piece.date}", style="dim")
    if piece.theme:
        header.append(f"   ·   {piece.theme}", style="muted")

    body: list[RenderableType] = []
    if piece.title:
        body.append(gradient_text(piece.title))
        body.append(Text(""))
    if piece.hook:
        body.append(prose(piece.hook))
    if piece.quote:
        quoted = "\n".join(f"> {line.strip()}" for line in piece.quote.splitlines() if line.strip())
        body.append(prose(quoted))
    if piece.reflection:
        if piece.quote:
            body.append(Text(""))
        body.append(prose(piece.reflection))
    if piece.practice:
        body.append(Text(""))
        practice = Text()
        practice.append(f"{L('try', lang)}  ", style="ok")
        practice.append(piece.practice.strip(), style="plain")
        body.append(
            Panel(practice, border_style="score.high", box=PANEL_BOX, padding=(0, 2))
        )

    card = Panel(
        Padding(Group(*body), (1, 2)),
        title=header,
        title_align="left",
        border_style="frame",
        box=PANEL_BOX,
        padding=0,
    )

    parts: list[RenderableType] = [card]
    if piece.sources:
        parts.append(Text(""))
        parts.append(rule(f"{L('sources', lang)}  ·  {len(piece.sources)}"))
        parts.append(Text(""))
        parts.append(sources_table(piece.sources, lang=lang))
        if show_sources:
            parts.append(Text(""))
            parts.append(source_cards(piece.sources, lang=lang))
    return Group(*parts)


# --------------------------------------------------------------------------
# Library / retrieval views
# --------------------------------------------------------------------------


def library_table(rows: Sequence[dict[str, Any]], *, lang: str = "en") -> Table:
    table = Table(
        box=TABLE_BOX,
        border_style="frame.soft",
        header_style="muted",
        expand=True,
        show_edge=False,
        pad_edge=False,
    )
    # Five columns fit an 80-column terminal; six did not, and the sixth was
    # `rights` — which is the same value on every row of a public-domain
    # library. It moves to the footer, where a non-public-domain text becomes
    # an exception worth noticing rather than a column of repetition.
    table.add_column(L("philosopher", lang), style="source.author", no_wrap=True,
                     overflow="ellipsis", max_width=14)
    table.add_column(L("work", lang), style="source.work", no_wrap=True,
                     overflow="ellipsis", max_width=24)
    table.add_column(L("tradition", lang), style="source.meta", no_wrap=True,
                     overflow="ellipsis", max_width=13)
    table.add_column(L("translator", lang), style="dim", no_wrap=True,
                     overflow="ellipsis", max_width=12)
    table.add_column(L("chunks", lang), justify="right", style="muted", width=6, no_wrap=True)

    for row in rows:
        restricted = "public" not in (row.get("rights") or "").lower()
        table.add_row(
            Text(row.get("philosopher", "—") or "—", style="warn" if restricted else "source.author"),
            Text(row.get("work_title", "—") or "—"),
            Text(row.get("tradition", "") or "—"),
            Text(row.get("translator", "") or "—"),
            Text(str(row.get("n_chunks", 0))),
        )
    return table


def rights_summary(rows: Sequence[dict[str, Any]]) -> Text:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.get("rights") or "unknown"] = counts.get(row.get("rights") or "unknown", 0) + 1
    out = Text("  rights  ", style="dim")
    for i, (label, count) in enumerate(sorted(counts.items(), key=lambda kv: -kv[1])):
        if i:
            out.append(" · ", style="dim")
        out.append(f"{label} ", style="ok" if "public" in label.lower() else "warn")
        out.append(f"{count}", style="dim")
    return out


def retrieval_table(hits: Sequence[ScoredChunk], *, lang: str = "en") -> Table:
    """Retrieval with the scoring exposed — the debugging view."""
    table = Table(
        box=TABLE_BOX,
        border_style="frame.soft",
        header_style="muted",
        expand=True,
        show_edge=False,
        pad_edge=False,
    )
    table.add_column("#", justify="right", width=2, style="cite", no_wrap=True)
    table.add_column("blend", width=11, no_wrap=True)
    table.add_column("dense", justify="right", width=5, style="dim", no_wrap=True)
    table.add_column("bm25", justify="right", width=5, style="dim", no_wrap=True)
    table.add_column(L("philosopher", lang), style="source.author", no_wrap=True,
                     overflow="ellipsis", max_width=16)
    table.add_column(L("passages", lang), no_wrap=True, overflow="ellipsis", style="muted")

    for hit in hits:
        table.add_row(
            Text(str(hit.marker)),
            Text.assemble((f"{hit.score:.2f} ", "dim"), score_bar(hit.score, 6)),
            Text(f"{hit.dense:.2f}"),
            Text(f"{hit.lexical:.1f}"),
            Text(hit.chunk.philosopher or "—"),
            Text(snippet(hit.chunk.text, 200)),
        )
    return table


def tag_cloud(tags: Sequence[tuple[str, int]], *, limit: int = 40) -> RenderableType:
    if not tags:
        return Text("  no tags in this library", style="dim")
    items = []
    top = max(count for _, count in tags[:limit]) or 1
    for tag, count in tags[:limit]:
        weight = count / top
        style = "cite" if weight > 0.66 else ("muted" if weight > 0.33 else "dim")
        items.append(Text.assemble((f"{tag}", style), (f" {count}", "dim")))
    return Columns(items, padding=(0, 3), equal=False)


def centered(renderable: RenderableType) -> RenderableType:
    return Align.center(renderable)


def bullet_list(items: Iterable[str], *, style: str = "muted", marker: str = "·") -> Text:
    out = Text()
    for item in items:
        out.append(f"  {marker} ", style="dim")
        out.append(item + "\n", style=style)
    return out


# --------------------------------------------------------------------------
# The council
# --------------------------------------------------------------------------


def council_view(
    council: Any,
    *,
    lang: str = "en",
    show_sources: bool = False,
    show_academic: bool = True,
) -> RenderableType:
    """Positions stacked, then the objection.

    Stacked rather than in columns: three philosophical arguments side by side
    in an 80-column terminal become three columns of broken words. Reading
    order carries the comparison well enough, and the seat score in each
    title says which one the library best supports.
    """
    parts: list[RenderableType] = []

    for position in council.positions:
        parts.append(_position_panel(position, lang=lang, show_academic=show_academic))
        if position.answer and show_sources:
            parts.append(Text(""))
            parts.append(Padding(source_cards(position.answer.sources, lang=lang, chars=300), (0, 2)))
        parts.append(Text(""))

    if getattr(council, "objection", None) and council.objection.stands:
        parts.append(_objection_panel(council.objection, lang=lang))
        parts.append(Text(""))
        if show_sources:
            parts.append(Padding(source_cards(council.objection.sources, lang=lang, chars=300), (0, 2)))
            parts.append(Text(""))

    return Group(*parts[:-1]) if parts else Text("—", style="dim")


def _position_panel(position: Any, *, lang: str, show_academic: bool) -> RenderableType:
    seat = position.seat
    title = Text()
    title.append("◗ ", style=f"bold {VIOLET}")
    title.append(seat.tradition, style="heading")
    if seat.philosophers:
        title.append("  " + ", ".join(seat.philosophers[:3]), style="source.author")

    if position.error:
        body: RenderableType = Group(
            Text(f"{L('silent', lang)} — {position.error}", style="warn"),
        )
        return Panel(
            Padding(body, (1, 2)),
            title=title,
            title_align="left",
            subtitle=None,
            border_style="warn",
            box=PANEL_BOX,
            padding=0,
        )

    answer = position.answer
    blocks: list[RenderableType] = [prose(answer.plain or "—")]
    if show_academic and answer.academic.strip():
        blocks.append(Text(""))
        blocks.append(rule(L("academic", lang), style="frame.soft"))
        blocks.append(prose(answer.academic))
    blocks.append(Text(""))
    blocks.append(sources_table(answer.sources, lang=lang, cited=answer.cited_markers))

    return Panel(
        Padding(Group(*blocks), (1, 2)),
        title=title,
        title_align="left",
        subtitle=Text.assemble((f"{seat.score:.2f} ", "dim"), score_bar(seat.score, 6)),
        subtitle_align="right",
        border_style="frame",
        box=PANEL_BOX,
        padding=0,
    )


def _objection_panel(objection: Any, *, lang: str) -> RenderableType:
    title = Text()
    title.append("⟂ ", style=f"bold {AMBER}")
    title.append(L("objection", lang), style="subheading")
    title.append(f"  {L('against', lang)} ", style="dim")
    title.append(objection.against, style="source.author")

    blocks: list[RenderableType] = [prose(objection.text or "—")]
    if objection.upshot.strip():
        blocks.append(Text(""))
        blocks.append(rule(L("upshot", lang), style="frame.soft"))
        blocks.append(prose(objection.upshot))
    blocks.append(Text(""))
    blocks.append(sources_table(objection.sources, lang=lang))

    return Panel(
        Padding(Group(*blocks), (1, 2)),
        title=title,
        title_align="left",
        subtitle=Text(
            f"{L('raised_by', lang)} {', '.join(objection.raised_by)}", style="dim"
        ),
        subtitle_align="right",
        border_style="warn",
        box=PANEL_BOX,
        padding=0,
    )


def council_footer(council: Any, *, lang: str = "en") -> Text:
    bits = [
        ("traditions", str(len(council.spoken))),
        ("model", council.model or "—"),
        ("took", human_ms(council.took_ms)),
    ]
    if council.objection is None and council.held:
        bits.append(("objection", "not raised"))
    return status_bar(bits)


# --------------------------------------------------------------------------
# The Chronicle
# --------------------------------------------------------------------------


_KIND_GLYPH = {"passage": "❝", "decision": "◆", "question": "?"}


def chronicle_table(entries: Sequence[Any], *, lang: str = "en") -> RenderableType:
    """The record as a list, newest first.

    The id column looks like clutter until you want to delete something. It
    is the only handle on an entry, and printing it beside every row is
    cheaper than a second command to look one up.
    """
    if not entries:
        return Text("  nothing recorded yet", style="dim")

    table = Table(
        box=TABLE_BOX, border_style="frame.soft", header_style="muted",
        expand=True, pad_edge=False, show_edge=False,
    )
    table.add_column("", width=1, no_wrap=True)
    table.add_column("date", style="dim", width=10, no_wrap=True)
    table.add_column("kind", style="muted", width=8, no_wrap=True)
    table.add_column(L("passages", lang), overflow="ellipsis")
    table.add_column("id", style="dim", width=11, no_wrap=True)

    for entry in entries:
        glyph = Text(_KIND_GLYPH.get(entry.kind, "·"), style="cite")
        body = Text(entry.headline(96), style="quote" if entry.kind == "passage" else "")
        table.add_row(
            glyph,
            Text(entry.day),
            Text(L(_KIND_LABEL.get(entry.kind, entry.kind), lang)),
            body,
            Text(entry.id),
        )
    return table


_KIND_LABEL = {"passage": "saved", "decision": "decided", "question": "asked"}


def echo_line(echo: Any, *, lang: str = "en", width: int = 80) -> Text:
    """`↺ you kept this 3 months ago — “Men are disturbed not by…”  Mill · …`

    Deliberately one line and never a panel. A resurfaced entry is an aside;
    given a box of its own it would compete with the answer the reader
    actually asked for — and an aside that wraps onto three lines has stopped
    being an aside, so the quote is trimmed to whatever the line has left
    rather than trusting the terminal to do something sensible.
    """
    when = f"{L('echo', lang)} {echo.when}"
    # Philosopher and work only. The section trail is what pushes this over,
    # and the entry itself is one `philo chronicle` away.
    where = " · ".join(b for b in (echo.entry.philosopher, echo.entry.work_title) if b)

    MIN_QUOTE = 20
    fixed = char_weight("  ↺ ") + char_weight(when) + char_weight(" — ")
    # On a narrow terminal something has to go, and the citation goes first:
    # the quote is what makes the reader recognise the entry, and
    # `philo chronicle` is one command away for the rest.
    if width - fixed - char_weight(where) - 2 < MIN_QUOTE:
        where = ""
    spent = fixed + (char_weight(where) + 2 if where else 0)
    body = truncate(echo.entry.headline(200), max(MIN_QUOTE, width - spent))

    out = Text("  ↺ ", style="cite")
    out.append(when, style="muted")
    out.append(" — ", style="dim")
    out.append(body, style="quote")
    if where:
        out.append(f"  {where}", style="source.meta")
    return out


def decision_view(result: Any, *, lang: str = "en", show_sources: bool = False) -> RenderableType:
    parts: list[RenderableType] = []
    for key, text, border in (
        ("choice", result.choice, "frame"),
        ("tests", result.tests, "frame"),
        ("limits", result.limits, "frame.soft"),
    ):
        if not (text or "").strip():
            continue
        parts.append(
            Panel(
                Padding(prose(text), (1, 2)),
                title=_panel_title("◗", L(key, lang),
                                   glyph_style=f"bold {VIOLET}", label_style="heading"),
                title_align="left",
                border_style=border,
                box=PANEL_BOX,
                padding=0,
            )
        )
        parts.append(Text(""))

    if result.sources:
        parts.append(rule(f"{L('sources', lang)}  ·  {len(result.sources)}"))
        parts.append(Text(""))
        parts.append(sources_table(result.sources, lang=lang))
        if show_sources:
            parts.append(Text(""))
            parts.append(source_cards(result.sources, lang=lang))
    return Group(*parts) if parts else Text("—", style="dim")


def recap_view(recap: Any, *, lang: str = "en", width: int = 80) -> RenderableType:
    parts: list[RenderableType] = []
    for key, text, style in (
        ("week", recap.week, "frame"),
        ("thread", recap.thread, "frame"),
        ("sit", recap.sit_with, "frame.soft"),
    ):
        if not (text or "").strip():
            continue
        parts.append(
            Panel(
                Padding(prose(text), (1, 2)),
                title=_panel_title("◗", L(key, lang),
                                   glyph_style=f"bold {VIOLET}", label_style="heading"),
                title_align="left",
                border_style=style,
                box=PANEL_BOX,
                padding=0,
            )
        )
        parts.append(Text(""))

    for echo in recap.echoes:
        parts.append(echo_line(echo, lang=lang, width=width))
        parts.append(Text(""))

    if recap.sources:
        parts.append(rule(f"{L('sources', lang)}  ·  {len(recap.sources)}"))
        parts.append(Text(""))
        parts.append(sources_table(recap.sources, lang=lang))
    return Group(*parts) if parts else Text("—", style="dim")
