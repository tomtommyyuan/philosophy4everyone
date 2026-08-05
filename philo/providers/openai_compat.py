"""OpenAI and Azure OpenAI.

One class covers both: the two APIs differ only in how the client is
constructed and in the fact that Azure addresses *deployments* where OpenAI
addresses *model names*.  `Settings.chat_model_name` already resolves that,
so everything below is shared.

Three pieces of real-world defensiveness live here, and each earns its keep:

1. Parameter drift — the reasoning-model family rejects `max_tokens` (wants
   `max_completion_tokens`) and rejects any `temperature` other than 1.  We
   detect that from the 400 response and retry with adjusted parameters
   instead of handing the user an opaque API error.
2. `stream_options` is not accepted by every API version, so usage reporting
   degrades rather than failing the request.
3. Transient failures (429 / 5xx / connection resets) retry with exponential
   backoff and jitter derived from the attempt number, not a random source.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from ..config import Settings
from ..models import Message
from ..util import chunked, l2_normalize
from .base import ChatResult, ProgressCallback, ProviderError, StreamCallback

# Embedding dimensions for the models we name by default.  Only used for
# display before the first real call.
_KNOWN_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIProvider:
    """Talks to OpenAI or Azure OpenAI through the official `openai` SDK."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # This class covers both OpenAI and Azure, and may be selected for
        # chat, embeddings, or both — take the name from whichever role picked
        # it rather than assuming they agree.
        self.name = settings.chat_provider if settings.chat_provider in ("openai", "azure") else settings.embed_provider
        self._client: Any = None
        self._dim = _KNOWN_DIMS.get(settings.embed_model, 1536)
        # Learned at runtime from 400 responses; see the class docstring.
        self._use_completion_tokens = False
        self._omit_temperature = False
        self._supports_stream_usage = True

    # -- identity ---------------------------------------------------------
    @property
    def chat_model(self) -> str:
        if self.name == "azure":
            return self.settings.azure_chat_deployment or self.settings.chat_model
        return self.settings.chat_model

    @property
    def embed_model(self) -> str:
        if self.name == "azure":
            return self.settings.azure_embed_deployment or self.settings.embed_model
        return self.settings.embed_model

    @property
    def embed_dim(self) -> int:
        return self._dim

    # -- client -----------------------------------------------------------
    @property
    def client(self) -> Any:
        if self._client is not None:
            return self._client

        problems = self.settings.problems_for(self.name)
        if problems:
            raise ProviderError(
                "provider is not configured: " + "; ".join(problems),
                hint="Run `philo doctor` for a full check, or use PHILO_PROVIDER=mock to work offline.",
            )
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - guarded by problems()
            raise ProviderError(
                "the `openai` package is not installed",
                hint="pip install 'philo[openai]'",
                cause=exc,
            ) from exc

        s = self.settings
        try:
            if self.name == "azure":
                self._client = openai.AzureOpenAI(
                    api_key=s.azure_api_key,
                    azure_endpoint=s.azure_endpoint,
                    api_version=s.azure_api_version,
                    timeout=s.request_timeout,
                    max_retries=0,  # we do our own, with better messages
                )
            else:
                kwargs: dict[str, Any] = {
                    "api_key": s.openai_api_key,
                    "timeout": s.request_timeout,
                    "max_retries": 0,
                }
                if s.openai_base_url:
                    kwargs["base_url"] = s.openai_base_url
                if s.openai_org:
                    kwargs["organization"] = s.openai_org
                self._client = openai.OpenAI(**kwargs)
        except Exception as exc:
            raise ProviderError(f"could not create the {self.name} client: {exc}", cause=exc) from exc
        return self._client

    # -- embeddings -------------------------------------------------------
    def embed(
        self,
        texts: Sequence[str],
        *,
        on_progress: ProgressCallback | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        done = 0
        for batch in chunked(list(texts), self.settings.embed_batch_size):
            # The API rejects empty strings; keep positions aligned with a
            # single space and let the zero-ish vector fall out naturally.
            payload = [t if t.strip() else " " for t in batch]
            resp = self._retry(
                lambda: self.client.embeddings.create(model=self.embed_model, input=payload),
                what="embeddings.create",
            )
            rows = sorted(resp.data, key=lambda d: d.index)
            for row in rows:
                vec = l2_normalize(list(row.embedding))
                self._dim = len(vec) or self._dim
                out.append(vec)
            done += len(batch)
            if on_progress:
                on_progress(done, len(texts))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    # -- chat -------------------------------------------------------------
    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1200,
        stream_cb: StreamCallback | None = None,
        task: str = "answer",
        model: str = "",
    ) -> ChatResult:
        started = time.perf_counter()
        payload = [dict(m) for m in messages]
        wanted = model or self.chat_model

        # Two adaptation attempts: one to learn `max_completion_tokens`, one
        # to learn that temperature is fixed. Both are sticky for the process.
        last_error: Exception | None = None
        for _ in range(3):
            kwargs = self._chat_kwargs(payload, temperature, max_tokens,
                                       stream=stream_cb is not None, model=wanted)
            try:
                if stream_cb is not None:
                    text, usage, finish = self._stream(kwargs, stream_cb)
                else:
                    resp = self._retry(
                        lambda: self.client.chat.completions.create(**kwargs),
                        what="chat.completions.create",
                    )
                    choice = resp.choices[0]
                    text = choice.message.content or ""
                    finish = choice.finish_reason or ""
                    usage = _usage_dict(getattr(resp, "usage", None))
                return ChatResult(
                    text=text.strip(),
                    model=wanted,
                    provider=self.name,
                    usage=usage,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    finish_reason=finish,
                )
            except ProviderError as exc:
                if not self._adapt(exc):
                    raise
                last_error = exc

        raise ProviderError(
            f"chat request failed after parameter adaptation: {last_error}",
            hint="Check that the model/deployment name is correct for this endpoint.",
            cause=last_error if isinstance(last_error, Exception) else None,
        )

    def _chat_kwargs(
        self, messages: list[dict], temperature: float, max_tokens: int, *,
        stream: bool, model: str = "",
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": model or self.chat_model, "messages": messages}
        if self._use_completion_tokens:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        if not self._omit_temperature:
            kwargs["temperature"] = temperature
        if stream:
            kwargs["stream"] = True
            if self._supports_stream_usage:
                kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    def _adapt(self, exc: ProviderError) -> bool:
        """Learn from a 400 and report whether it is worth retrying."""
        msg = str(exc).lower()
        if "max_tokens" in msg and "max_completion_tokens" in msg and not self._use_completion_tokens:
            self._use_completion_tokens = True
            return True
        if "temperature" in msg and ("unsupported" in msg or "does not support" in msg or "only the default" in msg):
            if not self._omit_temperature:
                self._omit_temperature = True
                return True
        if "stream_options" in msg and self._supports_stream_usage:
            self._supports_stream_usage = False
            return True
        return False

    def _stream(self, kwargs: dict, stream_cb: StreamCallback) -> tuple[str, dict[str, int], str]:
        parts: list[str] = []
        usage: dict[str, int] = {}
        finish = ""
        stream = self._retry(
            lambda: self.client.chat.completions.create(**kwargs),
            what="chat.completions.create(stream)",
        )
        for event in stream:
            if getattr(event, "usage", None):
                usage = _usage_dict(event.usage)
            for choice in getattr(event, "choices", None) or []:
                delta = getattr(choice, "delta", None)
                piece = getattr(delta, "content", None) if delta else None
                if piece:
                    parts.append(piece)
                    stream_cb(piece)
                if getattr(choice, "finish_reason", None):
                    finish = choice.finish_reason
        return "".join(parts), usage, finish

    # -- plumbing ---------------------------------------------------------
    def _retry(self, call, *, what: str):
        import openai

        delay = 0.8
        last: Exception | None = None
        for attempt in range(1, max(1, self.settings.max_retries) + 1):
            try:
                return call()
            except openai.BadRequestError as exc:
                # Never retried blindly — surfaced so `_adapt` can inspect it.
                raise ProviderError(
                    f"{what} rejected the request: {_msg(exc)}",
                    hint=_hint_for(exc, self.settings),
                    cause=exc,
                ) from exc
            except openai.AuthenticationError as exc:
                raise ProviderError(
                    f"authentication failed for provider '{self.name}': {_msg(exc)}",
                    hint=(
                        "Check AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT."
                        if self.name == "azure"
                        else "Check OPENAI_API_KEY (and OPENAI_BASE_URL if you set one)."
                    ),
                    cause=exc,
                ) from exc
            except openai.NotFoundError as exc:
                raise ProviderError(
                    f"{what} got 404: {_msg(exc)}",
                    hint=(
                        f"Azure deployment '{self.chat_model}' not found on {self.settings.azure_endpoint}. "
                        "The deployment name is not the model name."
                        if self.name == "azure"
                        else f"Model '{self.chat_model}' is not available to this key."
                    ),
                    cause=exc,
                ) from exc
            except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError) as exc:
                last = exc
                if attempt >= self.settings.max_retries:
                    break
                # Deterministic jitter — reproducible runs, still de-synchronised.
                time.sleep(delay * (1 + (attempt % 3) * 0.13))
                delay *= 2
            except openai.APIStatusError as exc:
                raise ProviderError(f"{what} failed ({exc.status_code}): {_msg(exc)}", cause=exc) from exc

        raise ProviderError(
            f"{what} failed after {self.settings.max_retries} attempts: {_msg(last)}",
            hint="The endpoint may be rate-limited or unreachable. `philo doctor` checks connectivity.",
            cause=last,
        )

    def chat_healthcheck(self) -> str:
        result = self.chat(
            [{"role": "user", "content": "Reply with the single word: ready"}],
            temperature=0.0,
            max_tokens=16,
        )
        return f"{self.name} chat ok · {self.chat_model} ({result.text.strip()[:20] or 'no text'})"

    def embed_healthcheck(self) -> str:
        vec = self.embed_query("ping")
        return f"{self.name} embed ok · {self.embed_model} ({len(vec)}-dim)"

    def healthcheck(self) -> str:
        return f"{self.chat_healthcheck()} · {self.embed_healthcheck()}"


def _usage_dict(usage: Any) -> dict[str, int]:
    if not usage:
        return {}
    return {
        k: int(getattr(usage, k))
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        if getattr(usage, k, None) is not None
    }


def _msg(exc: Exception | None) -> str:
    if exc is None:
        return "unknown error"
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    return str(exc)


def _hint_for(exc: Exception, settings: Settings) -> str:
    msg = _msg(exc).lower()
    if "content" in msg and "filter" in msg:
        return "Azure content filtering rejected the request; try rephrasing the question."
    if "deployment" in msg:
        return "Azure routes by deployment name — set AZURE_OPENAI_CHAT_DEPLOYMENT to the deployment, not the model."
    if "context" in msg and "length" in msg:
        return "Too much retrieved context — lower --k or PHILO_CHUNK_MAX."
    return ""
