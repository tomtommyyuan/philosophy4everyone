"""Composite views: live streaming, ingest progress, diagnostics.

The streaming view is deliberately *transient*.  While the model writes, the
reader sees a soft, unstructured draft; the moment it finishes, that is wiped
and replaced by the finished two-panel answer with its sources.  Streaming
proves the system is alive without leaving a half-formatted mess on screen.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from ..config import Settings, env_report
from ..util import human_count, human_ms
from .components import L, bullet_list, hint, keyval, rule, status_bar
from .theme import AMBER, PANEL_BOX, TABLE_BOX, VIOLET, gradient_text

# Headers the model emits as structure; hidden from the live draft because a
# half-written "## THE ARGU" is noise, not information.
_HEADER_LINE = re.compile(r"^\s{0,3}#{1,6}\s*[A-Z一-鿿].*$")


class StreamView:
    """Live, transient rendering of a streaming completion."""

    def __init__(
        self,
        console: Console,
        *,
        title: str = "thinking",
        subtitle: str = "",
        max_lines: int = 16,
        enabled: bool = True,
    ) -> None:
        self.console = console
        self.title = title
        self.subtitle = subtitle
        self.max_lines = max_lines
        self.enabled = enabled and console.is_terminal
        self.buffer: list[str] = []
        self._live: Live | None = None
        self._since_refresh = 0

    # ------------------------------------------------------------------
    def __enter__(self) -> "StreamView":
        if self.enabled:
            self._live = Live(
                self._render(),
                console=self.console,
                transient=True,
                refresh_per_second=12,
                vertical_overflow="crop",
            )
            self._live.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._live is not None:
            self._live.__exit__(*exc)
            self._live = None

    # ------------------------------------------------------------------
    def feed(self, delta: str) -> None:
        self.buffer.append(delta)
        self._since_refresh += len(delta)
        # Refresh on line breaks or every ~48 characters: enough to look
        # continuous, few enough re-renders to stay cheap.
        if self._live is not None and ("\n" in delta or self._since_refresh > 48):
            self._since_refresh = 0
            self._live.update(self._render())

    @property
    def text(self) -> str:
        return "".join(self.buffer)

    # ------------------------------------------------------------------
    def _render(self) -> RenderableType:
        raw = self.text
        lines = [ln for ln in raw.splitlines() if not _HEADER_LINE.match(ln)]
        tail = lines[-self.max_lines :]
        body = Text("\n".join(tail).strip() or "…", style="muted")

        header = Text()
        header.append("  ", style="")
        header.append("◐ ", style=f"bold {VIOLET}")
        header.append(self.title, style="heading")
        if self.subtitle:
            header.append(f"   {self.subtitle}", style="dim")

        return Panel(
            Padding(body, (1, 2)),
            title=header,
            title_align="left",
            border_style="frame.soft",
            box=PANEL_BOX,
            padding=0,
        )


class Spinner:
    """A one-line transient status for work that has no measurable progress."""

    def __init__(self, console: Console, message: str, *, enabled: bool = True) -> None:
        self.console = console
        self.message = message
        self.enabled = enabled and console.is_terminal
        self._live: Live | None = None

    def __enter__(self) -> "Spinner":
        if self.enabled:
            from rich.spinner import Spinner as RichSpinner

            self._live = Live(
                RichSpinner("dots", text=Text(f" {self.message}", style="muted"), style=VIOLET),
                console=self.console,
                transient=True,
                refresh_per_second=12,
            )
            self._live.__enter__()
        return self

    def update(self, message: str) -> None:
        self.message = message
        if self._live is not None:
            from rich.spinner import Spinner as RichSpinner

            self._live.update(RichSpinner("dots", text=Text(f" {message}", style="muted"), style=VIOLET))

    def __exit__(self, *exc: Any) -> None:
        if self._live is not None:
            self._live.__exit__(*exc)
            self._live = None


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------


def make_ingest_progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(style=VIOLET),
        TextColumn("[muted]{task.description}"),
        BarColumn(bar_width=28, complete_style=AMBER, finished_style="score.high", pulse_style=VIOLET),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


def ingest_summary(report: Any, *, lang: str = "en") -> RenderableType:
    parts: list[RenderableType] = []

    table = Table(
        box=TABLE_BOX,
        border_style="frame.soft",
        header_style="muted",
        expand=True,
        show_edge=False,
        pad_edge=False,
    )
    table.add_column(L("philosopher", lang), style="source.author", no_wrap=True)
    table.add_column(L("work", lang), style="source.work", overflow="ellipsis")
    table.add_column(L("chunks", lang), justify="right", style="muted", width=8)
    table.add_column("chars", justify="right", style="dim", width=9)
    table.add_column("reused", justify="right", style="dim", width=7)

    for work in sorted(report.works, key=lambda w: (w.philosopher, w.title)):
        table.add_row(
            Text(work.philosopher),
            Text(work.title),
            Text(str(work.n_chunks)),
            Text(human_count(work.n_chars)),
            Text(str(work.reused) if work.reused else "—"),
        )
    parts.append(table)
    parts.append(Text(""))

    stats = status_bar(
        [
            ("works", str(report.n_works)),
            ("passages", str(report.n_chunks)),
            ("avg", f"{report.avg_chunk_chars} chars"),
            ("range", f"{report.min_chunk_chars}–{report.max_chunk_chars}"),
            ("embedded", str(report.n_embedded)),
            ("reused", str(report.n_reused)),
            ("dim", str(report.dim)),
            ("took", human_ms(report.took_ms)),
        ]
    )
    parts.append(stats)

    if report.errors:
        parts.append(Text(""))
        parts.append(Text("  warnings", style="warn"))
        parts.append(bullet_list(report.errors[:12], style="dim"))
    return Group(*parts)


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def doctor_view(
    settings: Settings,
    *,
    provider: Any | None = None,
    index_info: dict[str, Any] | None = None,
    provider_status: tuple[bool, str] | None = None,
) -> RenderableType:
    parts: list[RenderableType] = []

    parts.append(rule("configuration"))
    parts.append(Text(""))
    rows: list[tuple[str, Any]] = [
        ("provider", Text(settings.describe_provider(), style="brand")),
        ("chat model", Text(provider.chat_model if provider else settings.chat_model_name)),
        ("embed model", Text(provider.embed_model if provider else settings.embed_model_name)),
        ("library", Text(str(settings.library_dir))),
        ("index", Text(str(settings.index_dir))),
        ("profiles", Text(str(settings.profiles_dir))),
    ]
    parts.append(Padding(keyval(rows), (0, 0, 0, 2)))

    parts.append(Text(""))
    parts.append(rule("environment"))
    parts.append(Text(""))
    env_rows: list[tuple[str, Any]] = []
    for name, value, is_set in env_report():
        env_rows.append((name, Text(value, style="ok" if is_set else "dim")))
    parts.append(Padding(keyval(env_rows), (0, 0, 0, 2)))

    parts.append(Text(""))
    parts.append(rule("checks"))
    parts.append(Text(""))

    checks: list[tuple[str, bool, str]] = []
    lib_ok = settings.library_dir.is_dir()
    n_files = len(list(settings.library_dir.rglob("*.md"))) if lib_ok else 0
    checks.append(("library directory", lib_ok, f"{n_files} markdown files" if lib_ok else "missing"))

    if index_info:
        checks.append(
            (
                "index",
                bool(index_info.get("ok")),
                index_info.get("detail", ""),
            )
        )
    else:
        checks.append(("index", False, "not built — run `philo ingest`"))

    problems = settings.problems()
    checks.append(
        (
            "provider config",
            not problems,
            "ok" if not problems else "; ".join(problems),
        )
    )
    if provider_status is not None:
        ok, detail = provider_status
        checks.append(("provider round-trip", ok, detail))

    check_table = Table.grid(padding=(0, 2))
    check_table.add_column(width=3)
    check_table.add_column(style="muted", no_wrap=True)
    check_table.add_column(style="dim", overflow="fold")
    for name, ok, detail in checks:
        check_table.add_row(
            Text("✓" if ok else "✗", style="ok" if ok else "err"),
            Text(name),
            Text(detail),
        )
    parts.append(Padding(check_table, (0, 0, 0, 2)))

    if settings.is_offline:
        parts.append(Text(""))
        parts.append(
            hint(
                "Running offline. Set OPENAI_API_KEY (or the AZURE_OPENAI_* variables) "
                "in .env, then re-run `philo ingest --rebuild`."
            )
        )
    return Group(*parts)


def error_panel(title: str, message: str, *, hint_text: str = "") -> Panel:
    body: list[RenderableType] = [Text(message, style="plain")]
    if hint_text:
        body.append(Text(""))
        body.append(Text(hint_text, style="hint"))
    return Panel(
        Padding(Group(*body), (1, 2)),
        title=Text(f"  {title}", style="err"),
        title_align="left",
        border_style="err",
        box=PANEL_BOX,
        padding=0,
    )


def help_view(commands: Sequence[tuple[str, str, str]], *, version: str = "") -> RenderableType:
    parts: list[RenderableType] = []
    parts.append(Text(""))
    parts.append(Padding(gradient_text("P H I L O S O P H Y   F O R   E V E R Y O N E"), (0, 0, 0, 2)))
    parts.append(
        Padding(
            Text("every claim carries its source · 每一句都有出处", style="muted"),
            (0, 0, 1, 2),
        )
    )

    table = Table.grid(padding=(0, 3))
    table.add_column(style="brand", no_wrap=True, width=26)
    table.add_column(style="muted", overflow="fold")
    for name, args, description in commands:
        label = Text(f"philo {name}", style="brand")
        if args:
            label.append(f" {args}", style="dim")
        table.add_row(label, Text(description))
    parts.append(Padding(table, (0, 0, 0, 2)))

    parts.append(Text(""))
    parts.append(
        Padding(
            Group(
                Text("getting started", style="subheading"),
                bullet_list(
                    [
                        "philo ingest                   build the index from library/",
                        'philo ask "why do we fear death?"',
                        "philo daily                    today's personalised piece",
                        "philo chat                     an ongoing conversation",
                        "philo doctor                   check configuration and connectivity",
                    ]
                ),
            ),
            (0, 0, 0, 2),
        )
    )
    if version:
        parts.append(Padding(Text(f"v{version}", style="dim"), (1, 0, 0, 2)))
    return Group(*parts)
