"""Provider registry.

`get_provider()` is the only place the rest of the codebase learns which
backend it is talking to — and it is a single `if`.  That is the whole point
of the offline-first design.
"""

from __future__ import annotations

from ..config import Settings, get_settings
from .base import ChatResult, Provider, ProviderError, StreamCallback
from .mock import MockProvider

__all__ = [
    "ChatResult",
    "MockProvider",
    "Provider",
    "ProviderError",
    "StreamCallback",
    "get_provider",
]

_cache: dict[str, Provider] = {}


def get_provider(settings: Settings | None = None, *, fresh: bool = False) -> Provider:
    s = settings or get_settings()
    key = f"{s.provider}:{s.chat_model_name}:{s.embed_model_name}"
    if not fresh and key in _cache:
        return _cache[key]

    if s.provider == "mock":
        provider: Provider = MockProvider()
    elif s.provider in ("openai", "azure"):
        from .openai_compat import OpenAIProvider

        provider = OpenAIProvider(s)
    else:  # pragma: no cover - Settings validates this earlier
        raise ProviderError(f"unknown provider {s.provider!r}")

    _cache[key] = provider
    return provider
