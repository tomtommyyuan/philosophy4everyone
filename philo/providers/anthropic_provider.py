"""Anthropic (Claude) — chat only.

**Anthropic publishes no embeddings endpoint.** The Messages API is the entire
surface, so this backend implements `ChatBackend` and nothing else; the
registry pairs it with an OpenAI or Gemini embedding backend. That is not a
limitation being worked around — it is the actual shape of the API, and
`CompositeProvider` exists to model it honestly.

Three constraints the current Claude models impose, each of which produces a
400 rather than a warning if ignored:

- **`temperature` is rejected.** The Claude 5 family removed sampling
  parameters entirely. Steering happens through the prompt, so the
  `temperature` the rest of this codebase passes around is dropped here.
- **`max_tokens` covers thinking *and* the answer.** Thinking is on by
  default, so a budget sized for the visible answer alone truncates
  mid-sentence. We raise a floor rather than let that happen silently.
- **`effort` is not universal.** It is rejected by older models, so a 400
  naming it disables it for the process and the request is retried.

A refusal is not an error: the API returns HTTP 200 with
`stop_reason: "refusal"` and possibly empty content. Reading `content[0]`
without checking is how that turns into a confusing blank answer.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from ..config import Settings
from ..models import Message
from .base import ChatResult, ProviderError, StreamCallback

# Thinking shares the output budget with the answer, so a ceiling tuned for
# the visible text alone is not enough on its own.
MIN_MAX_TOKENS = 8192

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


class AnthropicProvider:
    """Claude via the official `anthropic` SDK."""

    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Any = None
        # Learned from 400 responses; sticky for the process.
        self._supports_effort = True

    # -- identity ---------------------------------------------------------
    @property
    def chat_model(self) -> str:
        return self.settings.anthropic_model

    # -- client -----------------------------------------------------------
    @property
    def client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.settings.anthropic_api_key:
            raise ProviderError(
                "ANTHROPIC_API_KEY is not set",
                hint="export ANTHROPIC_API_KEY=sk-ant-…  (see .env.example)",
            )
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError(
                "the `anthropic` package is not installed",
                hint="pip install 'philo[anthropic]'",
                cause=exc,
            ) from exc

        self._client = anthropic.Anthropic(
            api_key=self.settings.anthropic_api_key,
            base_url=self.settings.anthropic_base_url or None,
            timeout=self.settings.request_timeout,
            max_retries=self.settings.max_retries,
        )
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
        model: str = "",
    ) -> ChatResult:
        started = time.perf_counter()
        system, turns = _split_system(messages)

        for _ in range(3):
            kwargs = self._kwargs(system, turns, max_tokens, model or self.chat_model)
            try:
                if stream_cb is not None:
                    text, final = self._stream(kwargs, stream_cb)
                else:
                    final = self._call(lambda: self.client.messages.create(**kwargs))
                    text = _text_of(final)
                return self._result(text, final, started, kwargs["model"])
            except ProviderError as exc:
                if not self._adapt(exc):
                    raise
        raise ProviderError(
            "the request was rejected after parameter adaptation",
            hint=f"Check that '{self.chat_model}' is a valid Claude model for this key.",
        )

    def _kwargs(self, system: str, turns: list[dict], max_tokens: int, model: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            # Never send `temperature`: the Claude 5 family rejects sampling
            # parameters outright.
            "max_tokens": max(max_tokens, MIN_MAX_TOKENS),
            "messages": turns,
        }
        if system:
            kwargs["system"] = system
        if self._supports_effort and self.settings.anthropic_effort:
            kwargs["output_config"] = {"effort": self.settings.anthropic_effort}
        return kwargs

    def _adapt(self, exc: ProviderError) -> bool:
        msg = str(exc).lower()
        if self._supports_effort and ("effort" in msg or "output_config" in msg):
            self._supports_effort = False
            return True
        return False

    def _stream(self, kwargs: dict, stream_cb: StreamCallback) -> tuple[str, Any]:
        parts: list[str] = []

        def run():
            with self.client.messages.stream(**kwargs) as stream:
                # `text_stream` yields visible text only — thinking blocks are
                # excluded, which is exactly what should reach the reader.
                for piece in stream.text_stream:
                    parts.append(piece)
                    stream_cb(piece)
                return stream.get_final_message()

        final = self._call(run)
        return "".join(parts), final

    def _result(self, text: str, final: Any, started: float, model: str) -> ChatResult:
        stop = getattr(final, "stop_reason", "") or ""
        if stop == "refusal":
            details = getattr(final, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            text = text or (
                "This request was declined by the model's safety system"
                + (f" ({category})" if category else "")
                + "."
            )
        usage = getattr(final, "usage", None)
        return ChatResult(
            text=text.strip(),
            model=getattr(final, "model", "") or model,
            provider=self.name,
            usage=_usage(usage),
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason=stop,
        )

    # -- plumbing ---------------------------------------------------------
    def _call(self, fn):
        """Run an SDK call, mapping every failure to a ProviderError.

        The SDK already retries 429/5xx internally (`max_retries`), so this
        only has to translate — no second retry loop.
        """
        import anthropic

        try:
            return fn()
        except anthropic.AuthenticationError as exc:
            raise ProviderError(
                f"authentication failed for Anthropic: {_msg(exc)}",
                hint="Check ANTHROPIC_API_KEY.",
                cause=exc,
            ) from exc
        except anthropic.NotFoundError as exc:
            raise ProviderError(
                f"model '{self.chat_model}' was not found: {_msg(exc)}",
                hint="Set PHILO_ANTHROPIC_MODEL to a model your key can use, e.g. claude-opus-5.",
                cause=exc,
            ) from exc
        except anthropic.BadRequestError as exc:
            # Surfaced rather than retried so `_adapt` can inspect it.
            raise ProviderError(f"Anthropic rejected the request: {_msg(exc)}", cause=exc) from exc
        except anthropic.RateLimitError as exc:
            raise ProviderError(
                f"Anthropic rate limit reached: {_msg(exc)}",
                hint="Wait and retry, or lower PHILO_TOP_K to send less context.",
                cause=exc,
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"could not reach Anthropic: {_msg(exc)}", cause=exc) from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(
                f"Anthropic returned {exc.status_code}: {_msg(exc)}", cause=exc
            ) from exc

    def chat_healthcheck(self) -> str:
        result = self.chat(
            [{"role": "user", "content": "Reply with the single word: ready"}],
            max_tokens=MIN_MAX_TOKENS,
        )
        return f"anthropic chat ok · {self.chat_model} ({result.text.strip()[:20] or 'no text'})"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _split_system(messages: Sequence[Message]) -> tuple[str, list[dict]]:
    """Anthropic takes the system prompt as its own parameter.

    Leaving it in `messages` as a `system` role is a 400 on most models, so it
    is lifted out here rather than at every call site.
    """
    system_parts: list[str] = []
    turns: list[dict] = []
    for message in messages:
        if message["role"] == "system":
            system_parts.append(message["content"])
        else:
            turns.append({"role": message["role"], "content": message["content"]})
    if turns and turns[0]["role"] != "user":
        # The first turn must be `user`; our conversation history always is,
        # but a caller could pass otherwise.
        turns.insert(0, {"role": "user", "content": "(continuing)"})
    return "\n\n".join(system_parts), turns


def _text_of(message: Any) -> str:
    blocks = getattr(message, "content", None) or []
    return "".join(b.text for b in blocks if getattr(b, "type", "") == "text")


def _usage(usage: Any) -> dict[str, int]:
    if not usage:
        return {}
    prompt = int(getattr(usage, "input_tokens", 0) or 0)
    completion = int(getattr(usage, "output_tokens", 0) or 0)
    out = {"prompt_tokens": prompt, "completion_tokens": completion,
           "total_tokens": prompt + completion}
    for name in ("cache_read_input_tokens", "cache_creation_input_tokens"):
        value = getattr(usage, name, None)
        if value:
            out[name] = int(value)
    return out


def _msg(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    return str(exc)
