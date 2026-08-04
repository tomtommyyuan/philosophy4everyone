"""Provider registry.

`get_provider()` is where a pile of environment variables becomes one object
the rest of the codebase can use. It resolves chat and embeddings separately
and pairs them, because they are not always the same vendor — Anthropic has
no embeddings endpoint, so "use Claude" always means Claude plus somebody
else's index.
"""

from __future__ import annotations

from ..config import Settings, get_settings
from .base import (
    ChatBackend,
    ChatResult,
    CompositeProvider,
    EmbeddingBackend,
    Provider,
    ProviderError,
    StreamCallback,
)
from .mock import MockProvider

__all__ = [
    "ChatResult",
    "CompositeProvider",
    "MockProvider",
    "Provider",
    "ProviderError",
    "StreamCallback",
    "get_backend",
    "get_provider",
]

_cache: dict[str, Provider] = {}
_backends: dict[str, object] = {}


def get_backend(name: str, settings: Settings):
    """Build (and memoise) one vendor backend.

    Memoised per process so that a split configuration — Claude for chat,
    OpenAI for embeddings — never constructs two OpenAI clients, and so a
    single-vendor setup shares one client between both roles.
    """
    key = f"{name}:{id(settings)}"
    if key in _backends:
        return _backends[key]

    if name == "mock":
        backend: object = MockProvider()
    elif name in ("openai", "azure"):
        from .openai_compat import OpenAIProvider

        backend = OpenAIProvider(settings)
    elif name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        backend = AnthropicProvider(settings)
    elif name == "gemini":
        from .gemini import GeminiProvider

        backend = GeminiProvider(settings)
    else:  # pragma: no cover - config validates this earlier
        raise ProviderError(f"unknown provider {name!r}")

    _backends[key] = backend
    return backend


def get_provider(settings: Settings | None = None, *, fresh: bool = False) -> Provider:
    s = settings or get_settings()
    key = f"{s.chat_provider}:{s.chat_model_name}|{s.embed_provider}:{s.embed_model_name}"
    if not fresh and key in _cache:
        return _cache[key]

    chat = get_backend(s.chat_provider, s)
    embed = chat if s.chat_provider == s.embed_provider else get_backend(s.embed_provider, s)

    provider: Provider = CompositeProvider(chat, embed)  # type: ignore[arg-type]
    _cache[key] = provider
    return provider


def reset_providers() -> None:
    """Drop cached clients — used by tests and after a config change."""
    _cache.clear()
    _backends.clear()
