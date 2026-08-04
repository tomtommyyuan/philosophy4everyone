"""Google Gemini — chat and embeddings.

The one genuinely interesting thing here is `task_type`. Gemini embeds
documents and queries **asymmetrically**: `RETRIEVAL_DOCUMENT` for passages
going into the index, `RETRIEVAL_QUERY` for the question being asked against
it. The two modes place text in the same space but optimised from opposite
directions, and using the document mode for a query measurably degrades
recall. This is exactly why `EmbeddingBackend` separates `embed` from
`embed_query` — that split exists for providers like this one.

Gemini also reports safety blocks as a *successful* response with no
candidate text, so a blocked answer looks like an empty string unless
`finish_reason` and `prompt_feedback` are checked.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from ..config import Settings
from ..models import Message
from ..util import chunked, l2_normalize
from .base import ChatResult, ProgressCallback, ProviderError, StreamCallback

# Advertised dimensions for the embedding models we name by default; only
# used for display before the first real call, then corrected from the API.
_KNOWN_DIMS = {
    "gemini-embedding-001": 3072,
    "text-embedding-004": 768,
}


class GeminiProvider:
    """Gemini via the official `google-genai` SDK."""

    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Any = None
        self._dim = _KNOWN_DIMS.get(settings.gemini_embed_model, 3072)
        if settings.gemini_embed_dim:
            self._dim = settings.gemini_embed_dim

    # -- identity ---------------------------------------------------------
    @property
    def chat_model(self) -> str:
        return self.settings.gemini_model

    @property
    def embed_model(self) -> str:
        return self.settings.gemini_embed_model

    @property
    def embed_dim(self) -> int:
        return self._dim

    # -- client -----------------------------------------------------------
    @property
    def client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.settings.gemini_api_key:
            raise ProviderError(
                "GEMINI_API_KEY is not set",
                hint="export GEMINI_API_KEY=…  (or GOOGLE_API_KEY; see .env.example)",
            )
        try:
            from google import genai
        except ImportError as exc:
            raise ProviderError(
                "the `google-genai` package is not installed",
                hint="pip install 'philo[gemini]'",
                cause=exc,
            ) from exc

        self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    # -- chat -------------------------------------------------------------
    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1200,
        stream_cb: StreamCallback | None = None,
        task: str = "answer",
    ) -> ChatResult:
        from google.genai import types

        started = time.perf_counter()
        system, contents = _to_contents(messages)

        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        if stream_cb is not None:
            parts: list[str] = []
            final: Any = None
            stream = self._call(
                lambda: self.client.models.generate_content_stream(
                    model=self.chat_model, contents=contents, config=config
                ),
                what="generate_content_stream",
            )
            for event in stream:
                piece = getattr(event, "text", None)
                if piece:
                    parts.append(piece)
                    stream_cb(piece)
                final = event
            text = "".join(parts)
        else:
            final = self._call(
                lambda: self.client.models.generate_content(
                    model=self.chat_model, contents=contents, config=config
                ),
                what="generate_content",
            )
            text = getattr(final, "text", None) or ""

        finish, blocked = _finish_reason(final)
        if blocked and not text.strip():
            # A safety block is an HTTP 200 with no candidate text. Say so
            # instead of rendering a blank answer.
            text = (
                "Gemini's safety filters blocked this response"
                + (f" ({blocked})" if isinstance(blocked, str) else "")
                + "."
            )

        return ChatResult(
            text=text.strip(),
            model=getattr(final, "model_version", "") or self.chat_model,
            provider=self.name,
            usage=_usage(getattr(final, "usage_metadata", None)),
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason=finish,
        )

    # -- embeddings -------------------------------------------------------
    def embed(
        self,
        texts: Sequence[str],
        *,
        on_progress: ProgressCallback | None = None,
    ) -> list[list[float]]:
        return self._embed(list(texts), "RETRIEVAL_DOCUMENT", on_progress=on_progress)

    def embed_query(self, text: str) -> list[float]:
        # RETRIEVAL_QUERY, not RETRIEVAL_DOCUMENT — see the module docstring.
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def _embed(
        self,
        texts: list[str],
        task_type: str,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> list[list[float]]:
        from google.genai import types

        if not texts:
            return []

        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.settings.gemini_embed_dim or None,
        )

        out: list[list[float]] = []
        done = 0
        for batch in chunked(texts, self.settings.embed_batch_size):
            payload = [t if t.strip() else " " for t in batch]
            response = self._call(
                lambda: self.client.models.embed_content(
                    model=self.embed_model, contents=payload, config=config
                ),
                what="embed_content",
            )
            embeddings = getattr(response, "embeddings", None) or []
            if len(embeddings) != len(payload):
                raise ProviderError(
                    f"Gemini returned {len(embeddings)} embeddings for {len(payload)} inputs"
                )
            for item in embeddings:
                # Truncated output dimensions are not unit-length, so
                # normalising here is required, not merely tidy.
                vec = l2_normalize(list(item.values))
                self._dim = len(vec) or self._dim
                out.append(vec)
            done += len(batch)
            if on_progress:
                on_progress(done, len(texts))
        return out

    # -- plumbing ---------------------------------------------------------
    def _call(self, fn, *, what: str):
        from google.genai import errors

        try:
            return fn()
        except errors.ClientError as exc:
            code = getattr(exc, "code", None)
            if code == 401 or code == 403:
                raise ProviderError(
                    f"authentication failed for Gemini: {_msg(exc)}",
                    hint="Check GEMINI_API_KEY (create one at aistudio.google.com/apikey).",
                    cause=exc,
                ) from exc
            if code == 404:
                raise ProviderError(
                    f"{what}: model not found — {_msg(exc)}",
                    hint=(
                        "Set PHILO_GEMINI_MODEL / PHILO_GEMINI_EMBED_MODEL to models your key "
                        "can use. Gemini model names change; `philo doctor --probe` verifies them."
                    ),
                    cause=exc,
                ) from exc
            if code == 429:
                raise ProviderError(
                    f"Gemini rate limit reached: {_msg(exc)}",
                    hint="Free-tier keys have low per-minute limits; wait, or lower PHILO_EMBED_BATCH.",
                    cause=exc,
                ) from exc
            raise ProviderError(f"{what} was rejected: {_msg(exc)}", cause=exc) from exc
        except errors.ServerError as exc:
            raise ProviderError(f"{what} failed on Gemini's side: {_msg(exc)}", cause=exc) from exc
        except errors.APIError as exc:
            raise ProviderError(f"{what} failed: {_msg(exc)}", cause=exc) from exc

    def chat_healthcheck(self) -> str:
        result = self.chat(
            [{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=32
        )
        return f"gemini chat ok · {self.chat_model} ({result.text.strip()[:20] or 'no text'})"

    def embed_healthcheck(self) -> str:
        vec = self.embed_query("ping")
        return f"gemini embed ok · {self.embed_model} ({len(vec)}-dim)"

    def healthcheck(self) -> str:
        return f"{self.chat_healthcheck()} · {self.embed_healthcheck()}"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _to_contents(messages: Sequence[Message]) -> tuple[str, list[dict]]:
    """Convert our message list into Gemini's contents + system_instruction.

    Gemini names the assistant role `model`, and takes the system prompt as a
    separate field rather than a message.
    """
    system_parts: list[str] = []
    contents: list[dict] = []
    for message in messages:
        role = message["role"]
        if role == "system":
            system_parts.append(message["content"])
            continue
        contents.append(
            {
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": message["content"]}],
            }
        )
    return "\n\n".join(system_parts), contents


def _finish_reason(response: Any) -> tuple[str, Any]:
    """(finish_reason, blocked_reason) — blocked is falsy when fine."""
    if response is None:
        return "", None
    feedback = getattr(response, "prompt_feedback", None)
    blocked = getattr(feedback, "block_reason", None) if feedback else None
    candidates = getattr(response, "candidates", None) or []
    finish = ""
    if candidates:
        raw = getattr(candidates[0], "finish_reason", None)
        finish = getattr(raw, "name", None) or (str(raw) if raw else "")
        finish = finish.lower()
        if finish == "safety":
            blocked = blocked or "safety"
        elif finish == "max_tokens":
            finish = "length"
    if blocked:
        return "blocked", getattr(blocked, "name", None) or str(blocked)
    return finish, None


def _usage(meta: Any) -> dict[str, int]:
    if not meta:
        return {}
    prompt = int(getattr(meta, "prompt_token_count", 0) or 0)
    completion = int(getattr(meta, "candidates_token_count", 0) or 0)
    total = int(getattr(meta, "total_token_count", 0) or 0) or (prompt + completion)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def _msg(exc: Exception) -> str:
    message = getattr(exc, "message", None)
    if message:
        return str(message)
    return str(exc)
