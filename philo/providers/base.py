"""The provider contract.

Two operations, one interface: turn text into vectors, and turn messages into
an answer.  Everything above this line (chunking, retrieval, prompting,
rendering) is provider-agnostic, which is what lets the whole pipeline run
offline against `MockProvider` and then switch to a real API with one env var.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence, runtime_checkable

from ..models import Message

# Called with each new text delta as it arrives from the model.
StreamCallback = Callable[[str], None]
# Called with (completed, total) during embedding.
ProgressCallback = Callable[[int, int], None]


@dataclass
class ChatResult:
    text: str
    model: str = ""
    provider: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0
    finish_reason: str = ""

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class ProviderError(RuntimeError):
    """Anything that went wrong talking to a model.

    Carries a `hint` so the CLI can tell the user what to actually do about
    it rather than printing a raw traceback.
    """

    def __init__(self, message: str, *, hint: str = "", cause: Exception | None = None):
        super().__init__(message)
        self.hint = hint
        self.cause = cause


@runtime_checkable
class Provider(Protocol):
    name: str

    @property
    def chat_model(self) -> str: ...

    @property
    def embed_model(self) -> str: ...

    @property
    def embed_dim(self) -> int: ...

    def embed(
        self,
        texts: Sequence[str],
        *,
        on_progress: ProgressCallback | None = None,
    ) -> list[list[float]]:
        """Embed a batch of texts.  Vectors come back L2-normalised."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query (some backends use a distinct prefix)."""
        ...

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1200,
        stream_cb: StreamCallback | None = None,
        task: str = "answer",
    ) -> ChatResult:
        """Generate a completion.

        When `stream_cb` is supplied the provider streams and invokes it with
        each delta; the return value is still the complete result, so callers
        get live output *and* usage/latency from one call.

        `task` is a hint ("answer" | "daily" | "plain") that only the offline
        mock provider consults — real providers ignore it.
        """
        ...

    def healthcheck(self) -> str:
        """Cheapest possible round-trip.  Returns a human-readable status."""
        ...
