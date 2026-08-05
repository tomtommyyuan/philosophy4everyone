"""Which chat models are actually available to this installation.

Only the **chat** model is selectable. The embedding model is fixed by
whatever built the index — offering it as a choice would let one dropdown
silently invalidate the entire library.

Models are discovered live from each configured provider, because a
hardcoded catalogue goes stale the week after it is written and then quietly
offers models the user's key cannot reach. Discovery is cached for the
process, and every provider falls back to a curated shortlist when its list
endpoint is unavailable — an offline or rate-limited lookup should narrow the
menu, not empty it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..config import CHAT_PROVIDERS, Settings, has_credentials

# Used when live discovery fails, and to order the discovered list so the
# models worth reaching for appear first.
CURATED: dict[str, list[str]] = {
    "openai": ["gpt-5.5", "gpt-5.5-pro", "gpt-5.4-mini", "gpt-4o"],
    "anthropic": ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
    "mock": ["mock-sage-1"],
    # Azure exposes deployments, not models; the configured deployment is the
    # only thing that can be addressed.
    "azure": [],
}

_TTL_SECONDS = 600
_cache: dict[str, tuple[float, list[str]]] = {}


@dataclass
class ProviderModels:
    provider: str
    models: list[str] = field(default_factory=list)
    default: str = ""
    discovered: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "models": self.models,
            "default": self.default,
            "discovered": self.discovered,
            "note": self.note,
        }


def clear_cache() -> None:
    _cache.clear()


def available(settings: Settings) -> list[ProviderModels]:
    """Every chat provider this installation can actually reach."""
    out: list[ProviderModels] = []
    for provider in CHAT_PROVIDERS:
        if not has_credentials(settings, provider):
            continue
        if provider == "mock" and not settings.is_offline:
            # The mock is always "available", but listing it next to real
            # models invites picking it by accident.
            continue
        out.append(_for_provider(settings, provider))
    return out


def _for_provider(settings: Settings, provider: str) -> ProviderModels:
    default = _default_model(settings, provider)

    if provider == "azure":
        return ProviderModels(
            provider=provider,
            models=[default] if default else [],
            default=default,
            note="Azure addresses deployments, not models — set another with AZURE_OPENAI_CHAT_DEPLOYMENT.",
        )

    cached = _cache.get(provider)
    if cached and time.time() - cached[0] < _TTL_SECONDS:
        models, discovered = cached[1], True
    else:
        try:
            models = _discover(settings, provider)
        except Exception:
            models = []
        discovered = bool(models)
        if discovered:
            _cache[provider] = (time.time(), models)

    if not models:
        models = list(CURATED.get(provider, []))

    return ProviderModels(
        provider=provider,
        models=_ordered(provider, models, default),
        default=default,
        discovered=discovered,
        note="" if discovered else "offered from a built-in list — the provider's model list was unavailable",
    )


def _default_model(settings: Settings, provider: str) -> str:
    return {
        "openai": settings.chat_model,
        "azure": settings.azure_chat_deployment,
        "anthropic": settings.anthropic_model,
        "gemini": settings.gemini_model,
        "mock": "mock-sage-1",
    }.get(provider, "")


def _ordered(provider: str, models: list[str], default: str) -> list[str]:
    """Curated models first, then everything else alphabetically."""
    preferred = CURATED.get(provider, [])
    rank = {name: i for i, name in enumerate(preferred)}
    unique = sorted(set(models), key=lambda m: (rank.get(m, len(rank)), m))
    if default and default not in unique:
        unique.insert(0, default)
    return unique


# --------------------------------------------------------------------------
# Live discovery
# --------------------------------------------------------------------------


def _discover(settings: Settings, provider: str) -> list[str]:
    from . import get_backend

    backend = get_backend(provider, settings)
    if provider == "mock":
        return list(CURATED["mock"])
    if provider == "anthropic":
        return [m.id for m in backend.client.models.list(limit=100)]
    if provider == "gemini":
        return [
            (m.name or "").removeprefix("models/")
            for m in backend.client.models.list()
            if "generateContent" in (getattr(m, "supported_actions", None) or [])
        ]
    if provider == "openai":
        # The list is long and mostly not chat models; keep the families that
        # accept the Chat Completions shape.
        return [
            m.id
            for m in backend.client.models.list()
            if m.id.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-"))
            and not any(x in m.id for x in ("audio", "realtime", "transcribe", "tts", "image"))
        ]
    return []
