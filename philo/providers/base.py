"""The provider contract.

Two operations — turn text into vectors, turn messages into an answer — but
deliberately **two separate protocols**, because they are not always the same
vendor.

Anthropic has no embeddings endpoint at all: the Messages API is the whole
surface. So "use Claude" cannot mean "use Claude for everything"; it means
Claude writes the answers while somebody else builds the index. Modelling that
as one monolithic Provider would have forced a fake `embed()` that raises at
the worst possible moment — halfway through ingesting a library.

`CompositeProvider` pairs a chat backend with an embedding backend and
presents the single `Provider` surface the rest of the codebase already uses,
so nothing above this layer knows or cares that two vendors are involved.
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

    @property
    def refused(self) -> bool:
        """The model declined on safety grounds rather than failing.

        Anthropic and Gemini both return a successful response with a refusal
        marker instead of raising, so this has to be checked explicitly or a
        refusal reads as an empty answer.
        """
        return self.finish_reason in {"refusal", "safety", "blocked"}


class ProviderError(RuntimeError):
    """Anything that went wrong talking to a model.

    Carries a `hint` so the CLI can tell the user what to actually do about
    it rather than printing a raw traceback.
    """

    def __init__(self, message: str, *, hint: str = "", cause: Exception | None = None):
        super().__init__(message)
        self.hint = hint
        self.cause = cause


# --------------------------------------------------------------------------
# The two halves
# --------------------------------------------------------------------------


@runtime_checkable
class ChatBackend(Protocol):
    """Turns messages into an answer."""

    name: str

    @property
    def chat_model(self) -> str: ...

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

    def chat_healthcheck(self) -> str: ...


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Turns text into vectors."""

    name: str

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
        """Embed passages for storage.  Vectors come back L2-normalised."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query.

        Separate from `embed` because several providers embed queries and
        documents asymmetrically — Gemini takes an explicit `task_type`, and
        using the document mode for a query measurably degrades retrieval.
        """
        ...

    def embed_healthcheck(self) -> str: ...


@runtime_checkable
class Provider(ChatBackend, EmbeddingBackend, Protocol):
    """Both halves — what the rest of the codebase consumes."""

    def healthcheck(self) -> str: ...


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------


class CompositeProvider:
    """One `Provider` made of two backends.

    `name` deliberately reports the **embedding** backend. That field is what
    the vector store records in its manifest and checks on load, and vector
    compatibility depends only on who produced the vectors — swapping the chat
    model must not invalidate an index, while swapping the embedding model
    must.
    """

    def __init__(self, chat: ChatBackend, embed: EmbeddingBackend) -> None:
        self._chat = chat
        self._embed = embed
        self.name = embed.name

    # -- identity ---------------------------------------------------------
    @property
    def chat_provider(self) -> str:
        return self._chat.name

    @property
    def embed_provider(self) -> str:
        return self._embed.name

    @property
    def chat_model(self) -> str:
        return self._chat.chat_model

    @property
    def embed_model(self) -> str:
        return self._embed.embed_model

    @property
    def embed_dim(self) -> int:
        return self._embed.embed_dim

    @property
    def split(self) -> bool:
        """True when chat and embeddings come from different vendors."""
        return self._chat.name != self._embed.name

    def describe(self) -> str:
        if not self.split:
            return self._chat.name
        return f"{self._chat.name} + {self._embed.name}"

    # -- delegation -------------------------------------------------------
    def chat(self, messages, **kwargs) -> ChatResult:
        return self._chat.chat(messages, **kwargs)

    def embed(self, texts, **kwargs) -> list[list[float]]:
        return self._embed.embed(texts, **kwargs)

    def embed_query(self, text: str) -> list[float]:
        return self._embed.embed_query(text)

    def healthcheck(self) -> str:
        chat = self._chat.chat_healthcheck()
        if not self.split:
            embed = self._embed.embed_healthcheck()
            return f"{chat} · {embed}"
        return f"{chat} · {self._embed.embed_healthcheck()}"

    def chat_healthcheck(self) -> str:
        return self._chat.chat_healthcheck()

    def embed_healthcheck(self) -> str:
        return self._embed.embed_healthcheck()
