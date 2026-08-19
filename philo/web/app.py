"""The HTTP layer.

A plain ASGI app, so the same code serves `philo serve` on localhost and a
serverless deployment on Vercel, Fly or Render without modification.

Three things here are not incidental:

*The engine is a process-level singleton.* Loading the index and building the
BM25 table costs a few hundred milliseconds; doing it per request would
dominate the response time. On a warm serverless instance this is paid once.

*Endpoints are sync `def`.* Starlette runs those in a threadpool, which is
exactly right for the blocking OpenAI SDK — an `async def` calling it would
stall the event loop for every other request.

*A shared-secret gate.* Any public deployment of this app spends *your* API
credits on behalf of whoever finds the URL. `PHILO_WEB_TOKEN` is unset by
default (localhost is your own machine) but the moment the app is reachable
from the internet it should be set.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path
from typing import Any, Iterator

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..config import ConfigError, Settings, get_settings
from ..chronicle import KINDS, Chronicle, Entry, log_decision, weekly_recap
from ..generation.answerer import AskOptions, Conversation, Engine
from ..generation.council import DEFAULT_SEATS, MAX_SEATS, hold_council
from ..personalize.daily import generate_daily
from ..personalize.profile import DEFAULT_PROFILE_NAME, Profile
from ..providers import get_provider
from ..providers.base import ProviderError
from ..store.vector_store import Filters, IndexError_
from ..util import detect_language

PAGE = Path(__file__).with_name("index.html")


def settings_or_error() -> Settings:
    """Settings that never raise.

    A deployment missing its API key must still answer /api/health with the
    reason. Letting ConfigError escape turns the single most common
    misconfiguration into an opaque 500 whose cause is visible only in
    function logs — precisely when the operator has least access to them.
    """
    return get_settings(strict=False)

MAX_K = 12
MAX_QUESTION_CHARS = 600


def max_seats() -> int:
    """How many traditions a visitor may seat.

    A council spends N+1 completions per click, on the operator's key. The
    token gate already decides *who* may spend; this decides how much a
    single click can cost, in the same spirit as PHILO_WEB_MODELS.
    """
    try:
        wanted = int(os.environ.get("PHILO_WEB_MAX_SEATS", "").strip() or MAX_SEATS)
    except ValueError:
        return MAX_SEATS
    return max(0, min(wanted, MAX_SEATS))


# --------------------------------------------------------------------------
# Engine lifecycle
# --------------------------------------------------------------------------

_engine: Engine | None = None
_engine_error: str = ""
_lock = threading.Lock()


def get_engine() -> Engine:
    """Build the engine once per process; report a missing index cleanly."""
    global _engine, _engine_error
    if _engine is not None:
        return _engine
    with _lock:
        if _engine is None:
            settings = settings_or_error()
            if not settings.ready:
                error = settings.config_error
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": str(error) if error else "no provider configured",
                        "hint": getattr(error, "hint", "") or "Set an API key for this deployment.",
                        "code": "unconfigured",
                    },
                )
            engine = Engine(settings, get_provider(settings))
            try:
                engine.store  # forces the index load, and the model-match check
            except IndexError_ as exc:
                _engine_error = f"{exc} — {exc.hint}"
                raise HTTPException(
                    status_code=503,
                    detail={"error": str(exc), "hint": exc.hint, "code": exc.code},
                ) from exc
            _engine = engine
    return _engine


def reset_engine() -> None:
    """Drop the cached engine — used by tests and after a re-ingest."""
    global _engine, _engine_error
    with _lock:
        _engine, _engine_error = None, ""


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------


def require_token(x_philo_token: str | None = Header(default=None)) -> None:
    """Optional shared secret.

    Unset means open, which is correct for `philo serve` on localhost and
    wrong for anything with a public URL — see the module docstring.
    """
    expected = os.environ.get("PHILO_WEB_TOKEN", "").strip()
    if not expected:
        return
    if not x_philo_token or x_philo_token.strip() != expected:
        raise HTTPException(status_code=401, detail={"error": "missing or invalid X-Philo-Token"})


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


MAX_HISTORY_TURNS = 4


class Turn(BaseModel):
    """One earlier exchange, replayed by the client."""

    question: str = Field(default="", max_length=MAX_QUESTION_CHARS)
    answer: str = Field(default="", max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    # Chat only — the embedding model belongs to the index and is not
    # selectable from a web page.
    model: str = Field(default="", max_length=120)
    provider: str = Field(default="", max_length=32)
    k: int | None = Field(default=None, ge=1, le=MAX_K)
    philosopher: str = ""
    work: str = ""
    tradition: str = ""
    plain: bool = False
    lang: str = ""
    profile: str = ""
    # False = ask the model directly, no retrieval, no sources.
    grounded: bool = True
    # Prior turns, oldest first. Only the tail is used; the cap bounds both
    # the prompt size and what a client can push into the context.
    history: list[Turn] = Field(default_factory=list, max_length=40)


class CouncilRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    model: str = Field(default="", max_length=120)
    provider: str = Field(default="", max_length=32)
    seats: int = Field(default=DEFAULT_SEATS, ge=1, le=MAX_SEATS)
    k: int = Field(default=4, ge=1, le=8)
    objection: bool = True
    lang: str = ""
    profile: str = ""


# The browser owns the chronicle and posts the part it wants read back. The
# server keeps nothing: there are no accounts here, a serverless instance has
# no disk worth the name, and one visitor's record must never leak into
# another's. The cap bounds how much context a client can push into a prompt
# the operator pays for.
MAX_ENTRIES = 200


class ChronicleEntry(BaseModel):
    kind: str = "passage"
    created: str = Field(default="", max_length=40)
    text: str = Field(default="", max_length=4000)
    note: str = Field(default="", max_length=1000)
    chunk_id: str = Field(default="", max_length=200)
    philosopher: str = Field(default="", max_length=120)
    work_title: str = Field(default="", max_length=200)
    section: str = Field(default="", max_length=200)

    def entry(self) -> Entry:
        return Entry(
            kind=self.kind if self.kind in KINDS else "question",
            created=self.created,
            text=self.text,
            note=self.note,
            chunk_id=self.chunk_id,
            philosopher=self.philosopher,
            work_title=self.work_title,
            section=self.section,
        )


class DecideRequest(BaseModel):
    situation: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    note: str = Field(default="", max_length=1000)
    model: str = Field(default="", max_length=120)
    provider: str = Field(default="", max_length=32)
    k: int = Field(default=6, ge=1, le=MAX_K)
    lang: str = ""
    profile: str = ""
    entries: list[ChronicleEntry] = Field(default_factory=list, max_length=MAX_ENTRIES)


class RecapRequest(BaseModel):
    entries: list[ChronicleEntry] = Field(default_factory=list, max_length=MAX_ENTRIES)
    days: int = Field(default=7, ge=1, le=90)
    today: str = Field(default="", max_length=10)
    model: str = Field(default="", max_length=120)
    provider: str = Field(default="", max_length=32)
    k: int = Field(default=6, ge=1, le=MAX_K)
    lang: str = ""
    profile: str = ""


def _book(settings: Settings, entries: list[ChronicleEntry]) -> Chronicle:
    """An in-memory record from what the client sent.

    The path is never written to — every call site passes save=False — but
    Chronicle needs one, so it gets a name that says what it is.
    """
    return Chronicle(settings.chronicle_dir / "not-persisted.jsonl",
                     [e.entry() for e in entries])


def _filters(req: AskRequest) -> Filters:
    return Filters(
        philosopher=req.philosopher.strip(),
        work=req.work.strip(),
        tradition=req.tradition.strip(),
    )


def _history(req: AskRequest) -> list:
    pairs = [(t.question, t.answer) for t in req.history if t.question and t.answer]
    return Conversation.from_pairs(pairs, max_turns=MAX_HISTORY_TURNS).as_messages()


def _options(req: AskRequest, settings: Settings) -> AskOptions:
    profile = (
        Profile.load_or_default(settings.profiles_dir, req.profile) if req.profile else None
    )
    return AskOptions(
        k=req.k,
        filters=_filters(req),
        style="plain" if req.plain else "two-layer",
        reader_note=profile.reader_note() if profile else "",
        lang=req.lang or (profile.language if profile else "") or detect_language(req.question),
        chat_model=req.model.strip(),
        chat_provider=req.provider.strip(),
        grounded=req.grounded,
    )


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(
        title="Philosophy for Everyone",
        version=__version__,
        description="Grounded, citation-first philosophy answers.",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    # ---- page ----------------------------------------------------------
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse(PAGE.read_text(encoding="utf-8"))

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        # A 204 must carry no body at all; a JSON `null` here makes the
        # payload longer than the Content-Length the status implies, and
        # uvicorn rejects the response.
        return Response(status_code=204)

    # ---- health --------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        settings = settings_or_error()
        payload: dict[str, Any] = {
            "version": __version__,
            "provider": settings.chat_provider,
            "chat_provider": settings.chat_provider,
            "embed_provider": settings.embed_provider,
            "split_providers": settings.split_providers,
            "provider_label": settings.describe_provider(),
            "offline": settings.is_offline,
            "authenticated": bool(os.environ.get("PHILO_WEB_TOKEN", "").strip()),
            "library_dir": str(settings.library_dir),
            "index_dir": str(settings.index_dir),
            "ok": False,
        }
        try:
            engine = get_engine()
        except HTTPException as exc:
            payload["error"] = exc.detail
            return payload
        payload.update(
            ok=True,
            chat_model=engine.provider.chat_model,
            embed_model=engine.provider.embed_model,
            passages=len(engine.store),
            works=engine.store.manifest.n_works,
            built_at=engine.store.manifest.built_at,
        )
        return payload

    # ---- models --------------------------------------------------------
    @app.get("/api/models")
    def models() -> dict[str, Any]:
        """Chat models this installation can actually reach.

        Embeddings are absent by design: they are fixed by whatever built the
        index, and offering them here would let a dropdown invalidate the
        whole library.
        """
        from ..providers.catalog import available

        settings = settings_or_error()
        if not settings.ready:
            return {"providers": [], "current": {}, "embedding": {}, "restricted": False}
        groups = [g.to_dict() for g in available(settings)]
        allowed = [m.strip() for m in os.environ.get("PHILO_WEB_MODELS", "").split(",") if m.strip()]
        if allowed:
            # A public deployment can cap which models visitors may spend on.
            for group in groups:
                group["models"] = [m for m in group["models"] if m in allowed]
            groups = [g for g in groups if g["models"]]
        return {
            "providers": groups,
            "current": {"provider": settings.chat_provider, "model": settings.chat_model_name},
            "embedding": {"provider": settings.embed_provider, "model": settings.embed_model_name},
            "restricted": bool(allowed),
        }

    # ---- library -------------------------------------------------------
    @app.get("/api/sources")
    def sources(engine: Engine = Depends(get_engine)) -> dict[str, Any]:
        rows = engine.store.works_summary()
        return {
            "works": rows,
            "philosophers": sorted({r["philosopher"] for r in rows if r["philosopher"]}),
            "traditions": sorted({r["tradition"] for r in rows if r["tradition"]}),
            "tags": [{"tag": t, "count": n} for t, n in engine.store.tags()[:40]],
            "passages": sum(r["n_chunks"] for r in rows),
        }

    # ---- retrieval only ------------------------------------------------
    @app.get("/api/search", dependencies=[Depends(require_token)])
    def search(
        q: str = Query(min_length=1, max_length=MAX_QUESTION_CHARS),
        k: int = Query(default=8, ge=1, le=MAX_K),
        philosopher: str = "",
        tradition: str = "",
        engine: Engine = Depends(get_engine),
    ) -> dict[str, Any]:
        result = engine.retriever.search(
            q, k=k, filters=Filters(philosopher=philosopher.strip(), tradition=tradition.strip())
        )
        return {
            "query": q,
            "n_candidates": result.n_candidates,
            "best_score": result.best_score,
            "took_ms": result.took_ms,
            "hits": [h.to_dict() for h in result.hits],
        }

    # ---- ask -----------------------------------------------------------
    @app.post("/api/ask", dependencies=[Depends(require_token)])
    def ask(req: AskRequest, engine: Engine = Depends(get_engine)) -> dict[str, Any]:
        settings = settings_or_error()
        try:
            answer, _ = engine.ask(
                req.question, _options(req, settings), history=_history(req)
            )
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail={"error": str(exc), "hint": exc.hint}) from exc
        return answer.to_dict()

    @app.post("/api/ask/stream", dependencies=[Depends(require_token)])
    def ask_stream(req: AskRequest, engine: Engine = Depends(get_engine)) -> StreamingResponse:
        settings = settings_or_error()
        options = _options(req, settings)
        history = _history(req)

        def events() -> Iterator[str]:
            # The provider pushes deltas through a callback while this
            # generator pulls; a queue plus a worker thread is what bridges
            # the two without an async provider client.
            channel: queue.Queue[tuple[str, Any]] = queue.Queue()

            def run() -> None:
                try:
                    answer, _ = engine.ask(
                        req.question, options, history=history,
                        stream_cb=lambda d: channel.put(("delta", d)),
                    )
                    channel.put(("done", answer.to_dict()))
                except ProviderError as exc:
                    channel.put(("error", {"error": str(exc), "hint": exc.hint}))
                except Exception as exc:  # pragma: no cover - defensive
                    channel.put(("error", {"error": str(exc), "hint": ""}))
                finally:
                    channel.put(("eof", None))

            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            while True:
                kind, payload = channel.get()
                if kind == "eof":
                    break
                yield f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            worker.join(timeout=1.0)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",   # tell nginx-style proxies not to buffer
                "Connection": "keep-alive",
            },
        )

    # ---- council -------------------------------------------------------
    @app.post("/api/council", dependencies=[Depends(require_token)])
    def council(req: CouncilRequest, engine: Engine = Depends(get_engine)) -> dict[str, Any]:
        """Several traditions answering independently, then the objection.

        Not streamed. The positions are generated concurrently, so there is
        no single token stream to follow, and three of them interleaved
        would be unreadable anyway.
        """
        settings = settings_or_error()
        allowed = max_seats()
        if not allowed:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "the council is disabled on this deployment",
                    "hint": "It costs one completion per tradition; PHILO_WEB_MAX_SEATS is 0.",
                },
            )
        profile = (
            Profile.load_or_default(settings.profiles_dir, req.profile) if req.profile else None
        )
        try:
            result = hold_council(
                engine,
                req.question,
                seats=min(req.seats, allowed),
                k=req.k,
                lang=req.lang or (profile.language if profile else "") or detect_language(req.question),
                reader_note=profile.reader_note() if profile else "",
                chat_model=req.model.strip(),
                chat_provider=req.provider.strip(),
                objection=req.objection,
            )
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail={"error": str(exc), "hint": exc.hint}) from exc
        payload = result.to_dict()
        payload["min_score"] = settings.min_score
        return payload

    # ---- the chronicle -------------------------------------------------
    @app.post("/api/decide", dependencies=[Depends(require_token)])
    def decide(req: DecideRequest, engine: Engine = Depends(get_engine)) -> dict[str, Any]:
        """Put the texts to work on one decision. Nothing is stored here."""
        settings = settings_or_error()
        profile = (
            Profile.load_or_default(settings.profiles_dir, req.profile) if req.profile else None
        )
        try:
            result = log_decision(
                engine,
                _book(settings, req.entries),
                req.situation,
                note=req.note,
                lang=req.lang or (profile.language if profile else ""),
                k=req.k,
                reader_note=profile.reader_note() if profile else "",
                chat_model=req.model.strip(),
                chat_provider=req.provider.strip(),
                save=False,
            )
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail={"error": str(exc), "hint": exc.hint}) from exc
        return result.to_dict()

    @app.post("/api/recap", dependencies=[Depends(require_token)])
    def recap(req: RecapRequest, engine: Engine = Depends(get_engine)) -> dict[str, Any]:
        settings = settings_or_error()
        profile = (
            Profile.load_or_default(settings.profiles_dir, req.profile) if req.profile else None
        )
        try:
            result = weekly_recap(
                engine,
                _book(settings, req.entries),
                days=req.days,
                lang=req.lang or (profile.language if profile else ""),
                reader_note=profile.reader_note() if profile else "",
                chat_model=req.model.strip(),
                chat_provider=req.provider.strip(),
                today=req.today.strip(),
                k=req.k,
            )
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail={"error": str(exc), "hint": exc.hint}) from exc
        return result.to_dict()

    # ---- daily ---------------------------------------------------------
    @app.get("/api/daily", dependencies=[Depends(require_token)])
    def daily(
        profile: str = DEFAULT_PROFILE_NAME,
        theme: str = "",
        date: str = "",
        k: int = Query(default=5, ge=1, le=MAX_K),
        model: str = Query(default="", max_length=120),
        provider: str = Query(default="", max_length=32),
        engine: Engine = Depends(get_engine),
    ) -> dict[str, Any]:
        settings = settings_or_error()
        reader = Profile.load_or_default(settings.profiles_dir, profile)
        try:
            # Never persist history from a web request: a shared deployment
            # would otherwise let one visitor's page view rewrite the
            # rotation for everybody.
            result = generate_daily(
                engine, reader, settings, day=date, theme=theme, k=k, save=False,
                chat_model=model.strip(), chat_provider=provider.strip(),
            )
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail={"error": str(exc), "hint": exc.hint}) from exc
        payload = result.piece.to_dict()
        payload["grounded"] = result.grounded
        payload["took_ms"] = result.took_ms
        return payload

    return app


app = create_app()
