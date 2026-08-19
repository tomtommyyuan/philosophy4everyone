"""The command line.

Eight commands, one shape: resolve settings → build what the command needs →
render.  Every failure that a user can realistically cause (no index, no key,
wrong Azure deployment, empty library) is caught and reported as a panel with
a next step, never as a traceback.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console, RenderableType
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from . import __version__
from .config import USER_HOME, ConfigError, Settings, get_settings
from .corpus.gutenberg import SOURCES, fetch_all, select
from .corpus.ingest import ingest, preview_chunks
from .corpus.loader import CorpusError
from .generation.answerer import AskOptions, Conversation, Engine
from .generation.council import DEFAULT_SEATS, MIN_SEATS, hold_council
from .models import Answer
from .personalize.daily import generate_daily
from .personalize.profile import DEFAULT_PROFILE_NAME, LEVELS, Profile, list_profiles
from .providers import get_provider
from .providers.base import ProviderError
from .store.vector_store import Filters, IndexError_, VectorStore
from .ui import (
    L,
    Spinner,
    StreamView,
    answer_footer,
    answer_view,
    banner,
    council_footer,
    council_view,
    daily_card,
    doctor_view,
    error_panel,
    help_view,
    ingest_summary,
    library_table,
    make_console,
    make_ingest_progress,
    retrieval_table,
    rights_summary,
    rule,
    source_cards,
    status_bar,
    tag_cloud,
)
from .ui.components import bullet_list, hint, keyval
from .ui.theme import TABLE_BOX
from .util import detect_language, human_ms, snippet

COMMANDS = [
    ("init", "", "create .env, a profile and a first index"),
    ("fetch", "[--only NAME]", "download the public-domain texts into the library"),
    ("ingest", "[--rebuild]", "read library/, chunk it, embed it, store it"),
    ("ask", '"question"', "answer from the texts, with sources"),
    ("chat", "", "a conversation that remembers the last few turns"),
    ("council", '"question"', "3 traditions answer independently, then argue"),
    ("daily", "", "today's personalised Daily Philosophy"),
    ("search", '"query"', "retrieval only — see what would be sent to the model"),
    ("sources", "", "what is in the library"),
    ("profile", "show|list|set", "who the daily piece is written for"),
    ("serve", "[--port 8000]", "open the web interface at a local URL"),
    ("doctor", "[--probe]", "check configuration, index and connectivity"),
]

EXIT_OK, EXIT_ERROR, EXIT_USAGE = 0, 1, 2


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="philo", add_help=False)
    parser.add_argument("-h", "--help", action="store_true", dest="show_help")
    parser.add_argument("-V", "--version", action="store_true", dest="show_version")
    parser.add_argument("--no-color", action="store_true", help="disable colour output")
    parser.add_argument("--width", type=int, default=None, help="force console width")
    sub = parser.add_subparsers(dest="command")

    def add_filters(p: argparse.ArgumentParser) -> None:
        p.add_argument("--philosopher", "-p", default="", help="restrict to one philosopher")
        p.add_argument("--work", "-w", default="", help="restrict to one work")
        p.add_argument("--tradition", default="", help="restrict to one tradition (Stoicism, Daoism…)")
        p.add_argument("--tag", action="append", default=[], help="restrict to a tag (repeatable)")

    # init
    p_init = sub.add_parser("init", help="first-run setup")
    p_init.add_argument("--force", action="store_true", help="overwrite existing .env / profile")

    # fetch
    p_fetch = sub.add_parser("fetch", help="download the public-domain texts")
    p_fetch.add_argument("--only", nargs="*", default=[], help="philosopher or slug substrings")
    p_fetch.add_argument("--force", action="store_true", help="re-download even if cached")
    p_fetch.add_argument("--list", action="store_true", dest="list_only", help="list available works")
    p_fetch.add_argument("--ingest", action="store_true", help="index immediately afterwards")

    # ingest
    p_ing = sub.add_parser("ingest", help="build the index")
    p_ing.add_argument("--library", default="", help="directory of source texts")
    p_ing.add_argument("--rebuild", action="store_true", help="re-embed everything")
    p_ing.add_argument("--strict", action="store_true", help="fail files that lack front matter")
    p_ing.add_argument("--dry-run", action="store_true", help="chunk only; do not embed or write")
    p_ing.add_argument("--show", type=int, default=0, metavar="N", help="print N sample chunks")

    # ask
    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question", nargs="+")
    p_ask.add_argument("-k", type=int, default=None, help="how many passages to retrieve")
    p_ask.add_argument("--model", default="", help="chat model for this run (embeddings are fixed by the index)")
    p_ask.add_argument("--chat-provider", default="", help="openai | azure | anthropic | gemini")
    p_ask.add_argument("--plain", action="store_true", help="everyday layer only")
    p_ask.add_argument("--no-sources", action="store_true", dest="ungrounded",
                       help="ask the model directly, with no retrieval (answers carry no citations)")
    p_ask.add_argument("--show-sources", "-s", action="store_true", help="print the retrieved passages")
    p_ask.add_argument("--profile", default="", help="write for this reader")
    p_ask.add_argument("--lang", default="", choices=["", "en", "zh"])
    p_ask.add_argument("--no-stream", action="store_true")
    p_ask.add_argument("--json", action="store_true", help="machine-readable output")
    add_filters(p_ask)

    # chat
    p_chat = sub.add_parser("chat", help="interactive conversation")
    p_chat.add_argument("--profile", default="")
    p_chat.add_argument("-k", type=int, default=None)
    p_chat.add_argument("--model", default="", help="chat model for this run (embeddings are fixed by the index)")
    p_chat.add_argument("--chat-provider", default="", help="openai | azure | anthropic | gemini")
    p_chat.add_argument("--plain", action="store_true")
    add_filters(p_chat)

    # council
    p_council = sub.add_parser("council", help="several traditions, answering independently")
    p_council.add_argument("question", nargs="+")
    p_council.add_argument("--seats", type=int, default=DEFAULT_SEATS,
                           help=f"how many traditions to seat (default {DEFAULT_SEATS}, max 4)")
    p_council.add_argument("-k", type=int, default=4, help="passages per tradition")
    p_council.add_argument("--no-objection", action="store_true",
                           help="skip the dialectic pass (saves one completion)")
    p_council.add_argument("--model", default="", help="chat model for this run")
    p_council.add_argument("--chat-provider", default="", help="openai | azure | anthropic | gemini")
    p_council.add_argument("--plain", action="store_true", help="everyday layer only")
    p_council.add_argument("--show-sources", "-s", action="store_true", help="print the passages")
    p_council.add_argument("--profile", default="", help="write for this reader")
    p_council.add_argument("--lang", default="", choices=["", "en", "zh"])
    p_council.add_argument("--json", action="store_true")

    # daily
    p_daily = sub.add_parser("daily", help="today's personalised piece")
    p_daily.add_argument("--profile", default=DEFAULT_PROFILE_NAME)
    p_daily.add_argument("--theme", default="", help="override today's theme")
    p_daily.add_argument("--date", default="", help="generate for a specific date (YYYY-MM-DD)")
    p_daily.add_argument("-k", type=int, default=5)
    p_daily.add_argument("--model", default="", help="chat model for this run")
    p_daily.add_argument("--chat-provider", default="", help="openai | azure | anthropic | gemini")
    p_daily.add_argument("--show-sources", "-s", action="store_true")
    p_daily.add_argument("--no-save", action="store_true", help="do not record it in the profile")
    p_daily.add_argument("--no-stream", action="store_true")
    p_daily.add_argument("--json", action="store_true")

    # search
    p_search = sub.add_parser("search", help="retrieval only")
    p_search.add_argument("query", nargs="+")
    p_search.add_argument("-k", type=int, default=8)
    p_search.add_argument("--full", action="store_true", help="print whole passages")
    p_search.add_argument("--json", action="store_true")
    add_filters(p_search)

    # sources
    p_src = sub.add_parser("sources", help="what is in the library")
    p_src.add_argument("--philosopher", "-p", default="")
    p_src.add_argument("--tags", action="store_true", help="show the tag cloud")
    p_src.add_argument("--json", action="store_true")

    # profile
    p_prof = sub.add_parser("profile", help="manage reader profiles")
    p_prof.add_argument("action", nargs="?", default="show", choices=["show", "list", "set", "init"])
    p_prof.add_argument("--name", default=DEFAULT_PROFILE_NAME)
    p_prof.add_argument("--display-name", default=None)
    p_prof.add_argument("--language", default=None, choices=["en", "zh"])
    p_prof.add_argument("--level", default=None, choices=sorted(LEVELS))
    p_prof.add_argument("--interests", default=None, help="comma separated")
    p_prof.add_argument("--philosophers", default=None, help="comma separated")
    p_prof.add_argument("--traditions", default=None, help="comma separated")
    p_prof.add_argument("--avoid", default=None, help="comma separated")
    p_prof.add_argument("--tone", default=None)

    # serve
    p_serve = sub.add_parser("serve", help="run the web interface")
    p_serve.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    p_serve.add_argument("--open", action="store_true", dest="open_browser", help="open a browser")

    # doctor
    p_doc = sub.add_parser("doctor", help="diagnostics")
    p_doc.add_argument("--probe", action="store_true", help="make a real (tiny) API call")

    sub.add_parser("version", help="print version")
    return parser


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])

    console = make_console(color=not args.no_color, width=args.width)

    if args.show_version:
        console.print(f"philo {__version__}")
        return EXIT_OK
    if args.show_help or not args.command:
        console.print(help_view(COMMANDS, version=__version__))
        return EXIT_OK

    # Loaded leniently: a missing key must not stop `philo doctor` from
    # explaining what is missing.
    settings = get_settings(reload=True, strict=False)
    if args.no_color:
        settings.color = False

    if not settings.ready and args.command not in ("doctor", "version", "init"):
        error = settings.config_error
        console.print(
            error_panel(
                "configuration",
                str(error) if error else "no provider is configured",
                hint_text=getattr(error, "hint", "") or "Run `philo doctor` for the full picture.",
            )
        )
        return EXIT_USAGE

    handlers = {
        "init": cmd_init,
        "fetch": cmd_fetch,
        "ingest": cmd_ingest,
        "ask": cmd_ask,
        "chat": cmd_chat,
        "council": cmd_council,
        "daily": cmd_daily,
        "search": cmd_search,
        "sources": cmd_sources,
        "profile": cmd_profile,
        "serve": cmd_serve,
        "doctor": cmd_doctor,
        "version": lambda a, s, c: (c.print(f"philo {__version__}"), EXIT_OK)[1],
    }

    try:
        return handlers[args.command](args, settings, console)
    except KeyboardInterrupt:
        console.print()
        console.print(Text("  interrupted", style="muted"))
        return 130
    except IndexError_ as exc:
        console.print(error_panel("index", str(exc), hint_text=exc.hint))
        return EXIT_ERROR
    except ProviderError as exc:
        console.print(error_panel(f"{settings.provider} provider", str(exc), hint_text=exc.hint))
        return EXIT_ERROR
    except CorpusError as exc:
        console.print(error_panel("library", str(exc)))
        return EXIT_ERROR
    except ConfigError as exc:
        console.print(error_panel("configuration", str(exc), hint_text=exc.hint))
        return EXIT_USAGE
    except FileNotFoundError as exc:
        console.print(error_panel("not found", str(exc)))
        return EXIT_ERROR
    except OSError as exc:
        console.print(
            error_panel(
                "filesystem",
                f"{exc.strerror or exc}: {getattr(exc, 'filename', '') or settings.index_dir}",
                hint_text=(
                    "philo writes its index to PHILO_INDEX (currently "
                    f"{settings.index_dir}). Set PHILO_HOME or PHILO_INDEX to a writable "
                    "directory, or run from inside a project checkout."
                ),
            )
        )
        return EXIT_ERROR


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _filters(args: argparse.Namespace) -> Filters:
    return Filters(
        philosopher=getattr(args, "philosopher", "") or "",
        work=getattr(args, "work", "") or "",
        tradition=getattr(args, "tradition", "") or "",
        tags=list(getattr(args, "tag", []) or []),
    )


def _load_profile(settings: Settings, name: str) -> Profile | None:
    if not name:
        return None
    return Profile.load_or_default(settings.profiles_dir, name)


def _engine(settings: Settings) -> Engine:
    return Engine(settings, get_provider(settings))


def _header(console: Console, settings: Settings, *, quiet: bool = False) -> None:
    if quiet or not console.is_terminal:
        return
    console.print(banner())


def _provider_line(settings: Settings, provider: Any | None = None) -> Text:
    """Report the models actually in use, not the ones configured.

    Offline, `PHILO_CHAT_MODEL` still says `gpt-4o` while the mock provider is
    answering — printing the configured value would quietly misattribute every
    answer on screen.
    """
    provider = provider or get_provider(settings)
    bits = [("chat", f"{settings.chat_provider} · {provider.chat_model}")]
    if settings.split_providers:
        bits.append(("embed", f"{settings.embed_provider} · {provider.embed_model}"))
    else:
        bits.append(("embed", provider.embed_model))
    return status_bar(bits)


# --------------------------------------------------------------------------
# init
# --------------------------------------------------------------------------


ENV_TEMPLATE = """\
# Philosophy for Everyone — configuration
#
# Leave everything blank to run fully offline against the mock provider.
# The pipeline is identical either way; only the provider changes.

# mock | openai | azure   (auto-detected from the keys below when unset)
PHILO_PROVIDER=

# ---- OpenAI ----------------------------------------------------------
OPENAI_API_KEY=
# OPENAI_BASE_URL=https://api.openai.com/v1

# ---- Azure OpenAI ----------------------------------------------------
# Azure routes by *deployment name*, which is not the model name.
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_CHAT_DEPLOYMENT=
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=

# ---- models (OpenAI only; Azure uses the deployments above) ----------
PHILO_CHAT_MODEL=gpt-4o
PHILO_EMBED_MODEL=text-embedding-3-small

# ---- tuning ----------------------------------------------------------
# PHILO_TOP_K=6
# PHILO_MIN_SCORE=0.12
# PHILO_TEMPERATURE=0.3
"""


def cmd_init(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    _header(console, settings)
    created: list[str] = []
    skipped: list[str] = []

    env_path = settings.root / ".env"
    if env_path.exists() and not args.force:
        skipped.append(str(env_path))
    else:
        env_path.write_text(ENV_TEMPLATE, encoding="utf-8")
        created.append(str(env_path))

    profile_path = settings.profiles_dir / f"{DEFAULT_PROFILE_NAME}.json"
    if profile_path.exists() and not args.force:
        skipped.append(str(profile_path))
    else:
        Profile.default().save(profile_path)
        created.append(str(profile_path))

    console.print(rule("setup"))
    console.print()
    if created:
        console.print(Text("  created", style="ok"))
        console.print(bullet_list(created))
    if skipped:
        console.print(Text("  already present (use --force to overwrite)", style="muted"))
        console.print(bullet_list(skipped, style="dim"))

    console.print()
    if not settings.library_dir.is_dir():
        console.print(
            error_panel(
                "library",
                f"No library directory at {settings.library_dir}.",
                hint_text="Create it and add Markdown texts with front matter, then run `philo ingest`.",
            )
        )
        return EXIT_ERROR

    console.print(rule("building the index"))
    console.print()
    rc = cmd_ingest(argparse.Namespace(library="", rebuild=False, strict=False, dry_run=False, show=0), settings, console)
    if rc != EXIT_OK:
        return rc

    console.print()
    console.print(hint('Try:  philo ask "why do we fear death?"'))
    console.print(hint("Then: philo daily"))
    return EXIT_OK


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------


def cmd_fetch(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    if args.list_only:
        console.print(rule("available works"))
        console.print()
        table = Table.grid(padding=(0, 3))
        table.add_column(style="source.author", no_wrap=True)
        table.add_column(style="source.work", overflow="ellipsis")
        table.add_column(style="dim", no_wrap=True)
        for source in SOURCES:
            table.add_row(
                Text(source.philosopher), Text(source.work), Text(f"PG #{source.gid}")
            )
        console.print(Padding(table, (0, 2)))
        console.print()
        console.print(hint("philo fetch --only kant mill"))
        return EXIT_OK

    selected = select(args.only)
    if not selected:
        console.print(
            error_panel("fetch", f"No work matches {args.only}.",
                        hint_text="`philo fetch --list` shows what is available.")
        )
        return EXIT_USAGE

    cache_dir = settings.root / ".cache" / "gutenberg" if settings.in_project else USER_HOME / "cache"
    console.print(
        status_bar([("into", str(settings.library_dir)), ("works", str(len(selected)))])
    )
    console.print()

    progress = make_ingest_progress(console)
    with progress:
        task = progress.add_task("downloading", total=len(selected))

        def on_event(stage: str, cur: int, total: int, detail: str) -> None:
            progress.update(task, completed=cur, description=f"downloading {detail[:40]}")

        report = fetch_all(
            settings.library_dir, cache_dir,
            only=args.only, force=args.force, on_event=on_event,
        )

    table = Table(box=TABLE_BOX, border_style="frame.soft", header_style="muted",
                  expand=True, show_edge=False, pad_edge=False)
    table.add_column(L("philosopher"), style="source.author", no_wrap=True,
                     overflow="ellipsis", max_width=18)
    table.add_column(L("work"), style="source.work", no_wrap=True,
                     overflow="ellipsis", max_width=28)
    table.add_column("chars", justify="right", style="dim", width=9)
    table.add_column("sections", justify="right", style="muted", width=9)
    for work in report.works:
        table.add_row(
            Text(work.philosopher, style="err" if not work.ok else "source.author"),
            Text(work.work if work.ok else f"failed: {work.error}"),
            Text(f"{work.chars:,d}" if work.ok else "—"),
            Text(str(work.sections) if work.ok else "—"),
        )
    console.print(rule("fetched"))
    console.print()
    console.print(table)
    console.print()
    console.print(
        status_bar([
            ("works", f"{report.n_ok}/{len(report.works)}"),
            ("characters", f"{report.total_chars:,d}"),
            ("library", str(report.library_dir)),
        ])
    )

    warnings = [w for work in report.works for w in work.warnings]
    if warnings:
        console.print()
        console.print(Text("  warnings", style="warn"))
        console.print(bullet_list(warnings[:8], style="dim"))

    if report.failures:
        console.print()
        console.print(
            Text(f"  ⚠ {len(report.failures)} download(s) failed — re-run to retry", style="warn")
        )
        return EXIT_ERROR

    console.print()
    if args.ingest:
        return cmd_ingest(
            argparse.Namespace(library="", rebuild=False, strict=False, dry_run=False, show=0),
            settings, console,
        )
    console.print(hint("next: philo ingest"))
    return EXIT_OK


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    library = Path(args.library) if args.library else settings.library_dir
    provider = get_provider(settings)

    if not library.is_dir():
        console.print(
            error_panel(
                "library",
                f"No such directory: {library}",
                hint_text="Point at one with --library, or set PHILO_LIBRARY in .env.",
            )
        )
        return EXIT_ERROR

    if args.dry_run:
        chunks, errors = preview_chunks(settings, library)
        console.print(rule("dry run — nothing was embedded or written"))
        console.print()
        sizes = [c.n_chars for c in chunks]
        console.print(
            Padding(
                status_bar(
                    [
                        ("works", str(len({c.work_id for c in chunks}))),
                        ("passages", str(len(chunks))),
                        ("avg", f"{sum(sizes)//len(sizes) if sizes else 0} chars"),
                        ("min", str(min(sizes) if sizes else 0)),
                        ("max", str(max(sizes) if sizes else 0)),
                    ]
                ),
                (0, 0, 1, 0),
            )
        )
        for chunk in chunks[: args.show or 3]:
            console.print(rule(f"{chunk.philosopher} · {chunk.locator}", style="frame.soft"))
            console.print(Padding(Text(snippet(chunk.text, 600), style="quote"), (0, 2, 1, 2)))
        if errors:
            console.print(Text("  warnings", style="warn"))
            console.print(bullet_list(errors[:10], style="dim"))
        return EXIT_OK

    console.print(_provider_line(settings, provider))
    console.print()

    progress = make_ingest_progress(console)
    task_ids: dict[str, Any] = {}

    with progress:
        def on_event(stage: str, cur: int, total: int, detail: str) -> None:
            if stage == "load":
                task_ids["load"] = progress.add_task("reading library", total=None)
            elif stage == "chunk":
                if "load" in task_ids:
                    progress.remove_task(task_ids.pop("load"))
                if "chunk" not in task_ids:
                    task_ids["chunk"] = progress.add_task("chunking", total=total)
                progress.update(task_ids["chunk"], completed=cur, description=f"chunking {detail[:38]}")
            elif stage == "embed":
                if "embed" not in task_ids:
                    task_ids["embed"] = progress.add_task(f"embedding · {detail}", total=max(total, 1))
                progress.update(task_ids["embed"], completed=cur, total=max(total, 1))
            elif stage == "store":
                if "store" not in task_ids:
                    task_ids["store"] = progress.add_task("writing index", total=None)

        report = ingest(
            settings,
            provider,
            rebuild=args.rebuild,
            strict=args.strict,
            library=library,
            on_event=on_event,
        )

    if not report.ok:
        console.print(
            error_panel(
                "ingest",
                "No passages were produced."
                + ("\n\n" + "\n".join(report.errors) if report.errors else ""),
                hint_text="Check that library/ holds .md or .txt files with readable text.",
            )
        )
        return EXIT_ERROR

    console.print(rule("indexed"))
    console.print()
    console.print(ingest_summary(report))
    console.print()
    console.print(hint(f"index written to {report.index}"))
    return EXIT_OK


# --------------------------------------------------------------------------
# ask
# --------------------------------------------------------------------------


def cmd_ask(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    question = " ".join(args.question).strip()
    if not question:
        console.print(error_panel("ask", "No question given."))
        return EXIT_USAGE

    profile = _load_profile(settings, args.profile)
    lang = args.lang or (profile.language if profile else "") or detect_language(question)
    engine = _engine(settings)

    options = AskOptions(
        k=args.k,
        filters=_filters(args),
        style="plain" if args.plain else "two-layer",
        reader_note=profile.reader_note() if profile else "",
        lang=lang,
        chat_model=args.model,
        chat_provider=args.chat_provider,
        grounded=not args.ungrounded,
    )

    if not args.json:
        _header(console, settings)
        console.print(rule(f"{L('question', lang)}"))
        console.print(Padding(Text(question, style="question"), (1, 2, 1, 2)))

    stream = settings.stream and not args.no_stream and not args.json
    answer = _run_ask(engine, question, options, console, stream=stream, lang=lang)

    if args.json:
        print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
        return EXIT_OK if answer.grounded else EXIT_ERROR

    console.print(answer_view(answer, lang=lang, show_sources=args.show_sources,
                              show_academic=not args.plain))
    console.print()
    console.print(answer_footer(answer, offline=settings.is_offline))
    _warn_invented(console, answer)
    _warn_cross_language(console, settings, question, answer)
    return EXIT_OK


def _run_ask(
    engine: Engine,
    question: str,
    options: AskOptions,
    console: Console,
    *,
    stream: bool,
    lang: str,
    history: Sequence[Any] = (),
) -> Answer:
    if stream:
        with StreamView(console, title="reading the sources", subtitle=engine.provider.chat_model) as view:
            answer, _ = engine.ask(question, options, history=history, stream_cb=view.feed)
        return answer
    with Spinner(console, "retrieving and generating"):
        answer, _ = engine.ask(question, options, history=history)
    return answer


def _warn_cross_language(console: Console, settings: Settings, question: str, answer: Answer) -> None:
    """Offline retrieval cannot cross languages, and should say so.

    The mock provider's embeddings are hashed bag-of-words: a Chinese query
    and an English passage share no tokens, so the ranking degrades to noise
    rather than failing loudly. Real embedding models are multilingual and do
    not have this problem — so the honest thing is to name the cause.
    """
    if not settings.is_offline or not answer.sources:
        return
    if detect_language(question) == detect_language(answer.sources[0].chunk.text):
        return
    console.print()
    console.print(
        Text(
            "  ⚠ 离线模拟检索按字面匹配，无法跨语言——中文提问在英文原文上排序会退化。"
            "配置 OPENAI_API_KEY 或 AZURE_OPENAI_* 后重新 `philo ingest --rebuild` 即可解决。",
            style="warn",
        )
    )


def _warn_invented(console: Console, answer: Answer) -> None:
    if answer.invented_markers:
        markers = ", ".join(f"[{m}]" for m in sorted(answer.invented_markers))
        console.print()
        console.print(
            Text(
                f"  ⚠ the model produced citation markers with no matching source ({markers}); "
                "they were removed",
                style="warn",
            )
        )
    if answer.truncated:
        console.print(Text("  ⚠ output hit the token limit — raise PHILO_MAX_TOKENS", style="warn"))


# --------------------------------------------------------------------------
# chat
# --------------------------------------------------------------------------


CHAT_HELP = [
    ("/sources", "show the passages behind the last answer"),
    ("/who NAME", "restrict to one philosopher (empty to clear)"),
    ("/new", "forget the conversation so far"),
    ("/help", "this list"),
    ("/exit", "leave"),
]


def cmd_chat(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    profile = _load_profile(settings, args.profile)
    engine = _engine(settings)
    conversation = Conversation()
    filters = _filters(args)
    last: Answer | None = None

    _header(console, settings)
    console.print(_provider_line(settings, engine.provider))
    console.print()
    console.print(hint("/help for commands · /exit to leave"))
    console.print()

    while True:
        try:
            raw = console.input(Text.assemble(("  you ", "brand"), ("› ", "dim")))
        except (EOFError, KeyboardInterrupt):
            console.print()
            return EXIT_OK

        line = raw.strip()
        if not line:
            continue

        if line.startswith("/"):
            cmd, _, rest = line.partition(" ")
            cmd = cmd.lower()
            if cmd in ("/exit", "/quit", "/q"):
                return EXIT_OK
            if cmd == "/help":
                console.print()
                console.print(Padding(keyval([(c, d) for c, d in CHAT_HELP], key_style="kbd"), (0, 2)))
                console.print()
                continue
            if cmd == "/new":
                conversation.clear()
                console.print(Text("  conversation cleared", style="muted"))
                console.print()
                continue
            if cmd == "/who":
                filters.philosopher = rest.strip()
                label = filters.philosopher or "everyone"
                console.print(Text(f"  now reading: {label}", style="muted"))
                console.print()
                continue
            if cmd == "/sources":
                console.print()
                if last and last.sources:
                    console.print(source_cards(last.sources))
                else:
                    console.print(Text("  nothing retrieved yet", style="dim"))
                console.print()
                continue
            console.print(Text(f"  unknown command {cmd}", style="warn"))
            console.print()
            continue

        lang = (profile.language if profile else "") or detect_language(line)
        options = AskOptions(
            k=args.k,
            filters=filters,
            style="plain" if args.plain else "two-layer",
            reader_note=profile.reader_note() if profile else "",
            lang=lang,
            chat_model=args.model,
            chat_provider=args.chat_provider,
        )
        console.print()
        answer = _run_ask(
            engine, line, options, console,
            stream=settings.stream, lang=lang,
            history=conversation.as_messages(),
        )
        conversation.add(line, answer)
        last = answer

        console.print(answer_view(answer, lang=lang, show_academic=not args.plain))
        console.print()
        console.print(answer_footer(answer, offline=settings.is_offline))
        _warn_invented(console, answer)
        console.print()


# --------------------------------------------------------------------------
# council
# --------------------------------------------------------------------------


def cmd_council(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    question = " ".join(args.question).strip()
    if not question:
        console.print(error_panel("council", "No question given."))
        return EXIT_USAGE

    profile = _load_profile(settings, args.profile)
    lang = args.lang or (profile.language if profile else "") or detect_language(question)
    engine = _engine(settings)

    if not args.json:
        _header(console, settings)
        console.print(rule(f"{L('question', lang)}"))
        console.print(Padding(Text(question, style="question"), (1, 2, 1, 2)))

    seats = max(1, min(args.seats, 4))
    with Spinner(console, f"putting the question to {seats} traditions"):
        council = hold_council(
            engine,
            question,
            seats=seats,
            k=args.k,
            lang=lang,
            reader_note=profile.reader_note() if profile else "",
            chat_model=args.model,
            chat_provider=args.chat_provider,
            objection=not args.no_objection,
        )

    if args.json:
        print(json.dumps(council.to_dict(), ensure_ascii=False, indent=2))
        return EXIT_OK if council.held else EXIT_ERROR

    if not council.held:
        console.print(_council_declined(council, settings, lang))
        return EXIT_ERROR

    console.print(rule(f"{L('council', lang)}  ·  {len(council.spoken)}"))
    console.print()
    console.print(council_view(council, lang=lang, show_sources=args.show_sources,
                               show_academic=not args.plain))
    console.print()
    console.print(council_footer(council, lang=lang))
    for position in council.positions:
        if position.answer:
            _warn_invented(console, position.answer)
    return EXIT_OK


def _council_declined(council: Any, settings: Settings, lang: str) -> RenderableType:
    """Why there was no council — which is a fact about the library, not the question.

    Worth distinguishing carefully: nothing retrieved at all is a different
    problem from one tradition monopolising the evidence, and they have
    different fixes.
    """
    spoke = [p.seat.tradition for p in council.spoken]
    if not council.seats:
        body = (
            f"Nothing in this library clears the relevance floor for that question "
            f"— {council.survey_candidates} passages searched, best {council.best_score:.2f} "
            f"against a {settings.min_score:.2f} floor."
        )
        hint_text = "Try `philo search` to see what does match, or add texts to library/."
    else:
        named = ", ".join(spoke) or ", ".join(s.tradition for s in council.seats)
        body = (
            f"Only one tradition here has passages on that question ({named}), so there is "
            f"no council to hold. A debate staged between one voice and itself would be a "
            f"lie about what the library contains."
        )
        hint_text = f"`philo ask` answers it from what is there. {MIN_SEATS} traditions are needed for a council."
    return error_panel("council", body, hint_text=hint_text)


# --------------------------------------------------------------------------
# daily
# --------------------------------------------------------------------------


def cmd_daily(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    profile = Profile.load_or_default(settings.profiles_dir, args.profile)
    engine = _engine(settings)
    lang = profile.language or "en"

    if not args.json:
        _header(console, settings)

    stream = settings.stream and not args.no_stream and not args.json

    def run(cb=None):
        return generate_daily(
            engine,
            profile,
            settings,
            day=args.date,
            theme=args.theme,
            k=args.k,
            stream_cb=cb,
            save=not args.no_save,
            chat_model=args.model,
            chat_provider=args.chat_provider,
        )

    if stream:
        with StreamView(console, title="writing today's piece", subtitle=engine.provider.chat_model) as view:
            result = run(view.feed)
    else:
        with Spinner(console, "composing today's piece"):
            result = run()

    if args.json:
        print(json.dumps(result.piece.to_dict(), ensure_ascii=False, indent=2))
        return EXIT_OK if result.grounded else EXIT_ERROR

    console.print(daily_card(result.piece, lang=lang, show_sources=args.show_sources))
    console.print()
    console.print(
        status_bar(
            [
                ("for", profile.display_name or profile.name),
                ("theme", result.piece.theme),
                ("provider", result.piece.provider or settings.provider),
                ("took", human_ms(result.took_ms)),
            ]
        )
    )
    if not result.grounded:
        console.print()
        console.print(Text("  ⚠ nothing in the library supports this theme", style="warn"))
    elif result.piece.sources:
        _warn_cross_language(
            console, settings, result.piece.theme,
            Answer(question=result.piece.theme, sources=result.piece.sources),
        )
    return EXIT_OK


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------


def cmd_search(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    query = " ".join(args.query).strip()
    engine = _engine(settings)
    result = engine.search(query, AskOptions(k=args.k, filters=_filters(args)))

    if args.json:
        print(json.dumps(
            {
                "query": query,
                "n_candidates": result.n_candidates,
                "best_score": result.best_score,
                "hits": [h.to_dict() for h in result.hits],
            },
            ensure_ascii=False,
            indent=2,
        ))
        return EXIT_OK

    lang = detect_language(query)
    console.print(rule(f"retrieval · {query}"))
    console.print()
    if not result.hits:
        console.print(
            Text(
                f"  nothing above the {settings.min_score:.2f} floor "
                f"(best was {result.best_score:.2f} across {result.n_candidates} passages)",
                style="warn",
            )
        )
        return EXIT_OK

    if args.full:
        console.print(source_cards(result.hits, lang=lang, chars=1200))
    else:
        console.print(retrieval_table(result.hits, lang=lang))
    console.print()
    console.print(
        status_bar(
            [
                ("candidates", str(result.n_candidates)),
                ("returned", str(len(result.hits))),
                ("best cosine", f"{result.best_score:.3f}"),
                ("took", human_ms(result.took_ms)),
            ]
        )
    )
    return EXIT_OK


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------


def cmd_sources(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    provider = get_provider(settings)
    store = VectorStore(settings.index_dir).load(
        expect_model=provider.embed_model, expect_provider=provider.name
    )
    rows = store.works_summary()
    if args.philosopher:
        needle = args.philosopher.lower()
        rows = [r for r in rows if needle in (r["philosopher"] or "").lower()]

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return EXIT_OK

    console.print(rule(f"{L('library')} · {settings.library_dir}"))
    console.print()
    console.print(library_table(rows))
    console.print()
    console.print(
        status_bar(
            [
                ("works", str(len(rows))),
                ("philosophers", str(len({r["philosopher"] for r in rows}))),
                ("passages", str(sum(r["n_chunks"] for r in rows))),
                ("embed", store.manifest.embed_model or "—"),
                ("built", (store.manifest.built_at or "—")[:10]),
            ]
        )
    )
    console.print(rights_summary(rows))
    if args.tags:
        console.print()
        console.print(rule("tags", style="frame.soft"))
        console.print()
        console.print(Padding(tag_cloud(store.tags()), (0, 2)))
    return EXIT_OK


# --------------------------------------------------------------------------
# profile
# --------------------------------------------------------------------------


def cmd_profile(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    if args.action == "list":
        paths = list_profiles(settings.profiles_dir)
        console.print(rule("profiles"))
        console.print()
        if not paths:
            console.print(Text(f"  none in {settings.profiles_dir}", style="dim"))
            console.print(hint("philo profile init --name me"))
            return EXIT_OK
        for path in paths:
            profile = Profile.load(path)
            line = Text()
            line.append(f"  {profile.name}", style="brand")
            line.append(f"   {profile.display_name or '—'}", style="muted")
            line.append(f"   {profile.language} · {profile.level}", style="dim")
            line.append(f"   {len(profile.history)} sent", style="dim")
            console.print(line)
        return EXIT_OK

    profile = Profile.load_or_default(settings.profiles_dir, args.name)

    if args.action in ("set", "init"):
        if args.action == "init" and profile.path and profile.path.exists():
            console.print(Text(f"  {profile.path} already exists — editing it", style="muted"))
        if args.display_name is not None:
            profile.display_name = args.display_name
        if args.language is not None:
            profile.language = args.language
        if args.level is not None:
            profile.level = args.level
        for attr, raw in (
            ("interests", args.interests),
            ("philosophers", args.philosophers),
            ("traditions", args.traditions),
            ("avoid", args.avoid),
        ):
            if raw is not None:
                setattr(profile, attr, [v.strip() for v in raw.split(",") if v.strip()])
        if args.tone is not None:
            profile.tone = args.tone
        path = profile.save(settings.profiles_dir / f"{profile.name}.json")
        console.print(Text(f"  saved {path}", style="ok"))
        console.print()

    console.print(rule(f"profile · {profile.name}"))
    console.print()
    console.print(Padding(keyval(profile.summary_rows()), (0, 2)))
    if profile.history:
        console.print()
        console.print(rule("recent daily pieces", style="frame.soft"))
        console.print()
        console.print(
            bullet_list(
                [f"{h.get('date','')}  {h.get('title','')}" for h in profile.history[-6:]][::-1]
            )
        )
    console.print()
    console.print(hint("philo profile set --interests 'free will, grief, why work' --language zh"))
    return EXIT_OK


# --------------------------------------------------------------------------
# serve
# --------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        console.print(
            error_panel(
                "serve",
                "the web extras are not installed",
                hint_text="pip install 'philo[web]'  (adds fastapi and uvicorn)",
            )
        )
        return EXIT_ERROR

    _header(console, settings)

    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
    console.print(rule("web interface"))
    console.print()
    console.print(Padding(Text.assemble(("  ", ""), (url, "brand")), (0, 0, 1, 0)))
    console.print(_provider_line(settings))

    if args.host not in ("127.0.0.1", "localhost"):
        # Binding beyond localhost puts your API credits behind a public URL.
        console.print()
        if os.environ.get("PHILO_WEB_TOKEN", "").strip():
            console.print(Text("  ✓ PHILO_WEB_TOKEN is set — requests must present it", style="ok"))
        else:
            console.print(
                Text(
                    f"  ⚠ binding to {args.host} exposes this beyond your machine, and every\n"
                    "    request spends your API credits. Set PHILO_WEB_TOKEN to require a token.",
                    style="warn",
                )
            )

    console.print()
    console.print(hint("ctrl-c to stop"))
    console.print()

    if args.open_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        "philo.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )
    return EXIT_OK


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    _header(console, settings)

    index_info: dict[str, Any] | None = None
    provider = get_provider(settings) if settings.ready else None
    try:
        if provider is None:
            raise IndexError_("no provider configured", hint="See the errors above.")
        store = VectorStore(settings.index_dir).load(
            expect_model=provider.embed_model, expect_provider=provider.name
        )
        index_info = {
            "ok": True,
            "detail": (
                f"{len(store)} passages · {store.manifest.n_works} works · "
                f"{store.manifest.dim}-dim · built {store.manifest.built_at or '—'}"
            ),
        }
    except IndexError_ as exc:
        index_info = {"ok": False, "detail": str(exc)}

    provider_status: tuple[bool, str] | None = None
    if args.probe and settings.ready:
        with Spinner(console, "probing the provider"):
            try:
                provider_status = (True, get_provider(settings).healthcheck())
            except ProviderError as exc:
                provider_status = (False, f"{exc}" + (f" — {exc.hint}" if exc.hint else ""))
            except Exception as exc:  # pragma: no cover - unexpected SDK failure
                provider_status = (False, str(exc))

    console.print(doctor_view(settings, provider=provider, index_info=index_info,
                              provider_status=provider_status))
    console.print()
    if not args.probe and settings.ready and not settings.is_offline:
        console.print(hint("philo doctor --probe makes one tiny real API call to verify each key"))
    return EXIT_OK if settings.ready else EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
