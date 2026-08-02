"""The engine: retrieve, prompt, generate, verify.

`Engine` is the single object the CLI talks to.  It owns the store, the
provider and the retriever, and it enforces the two guarantees the project
promises — that nothing is asserted without a retrieved source, and that
every citation marker in the output points at a real passage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from ..config import Settings, get_settings
from ..models import Answer, Message, ScoredChunk, assistant, user
from ..providers import get_provider
from ..providers.base import Provider, StreamCallback
from ..retrieval.retriever import RetrievalResult, Retriever
from ..store.vector_store import Filters, VectorStore
from ..util import detect_language, truncate
from .prompts import audit_markers, build_answer_messages, split_two_layer


@dataclass
class AskOptions:
    k: int | None = None
    filters: Filters = field(default_factory=Filters)
    style: str = "two-layer"        # or "plain"
    reader_note: str = ""
    lang: str = ""                  # "" → detect from the question
    min_score: float | None = None
    temperature: float | None = None


class Engine:
    def __init__(
        self,
        settings: Settings | None = None,
        provider: Provider | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider or get_provider(self.settings)
        self._store = store
        self._retriever: Retriever | None = None

    # ------------------------------------------------------------------
    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = VectorStore(self.settings.index_dir).load(
                expect_model=self.provider.embed_model,
                expect_provider=self.provider.name,
            )
        return self._store

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(self.store, self.provider, self.settings)
        return self._retriever

    # ------------------------------------------------------------------
    def search(self, query: str, options: AskOptions | None = None) -> RetrievalResult:
        opt = options or AskOptions()
        return self.retriever.search(
            query,
            k=opt.k,
            filters=opt.filters,
            min_score=opt.min_score,
        )

    def ask(
        self,
        question: str,
        options: AskOptions | None = None,
        *,
        history: Sequence[Message] = (),
        stream_cb: StreamCallback | None = None,
        retrieval: RetrievalResult | None = None,
    ) -> tuple[Answer, RetrievalResult]:
        opt = options or AskOptions()
        lang = opt.lang or detect_language(question)
        started = time.perf_counter()

        result = retrieval or self.retriever.search(
            _retrieval_query(question, history),
            k=opt.k,
            filters=opt.filters,
            min_score=opt.min_score,
        )

        if not result.hits:
            # No call to the model at all. There is nothing to ground an
            # answer in, and asking anyway invites exactly the confident
            # fabrication this project exists to prevent.
            return self._ungrounded(question, result, lang, started), result

        messages = build_answer_messages(
            question,
            result.hits,
            lang=lang,
            history=list(history),
            style=opt.style,
            reader_note=opt.reader_note,
        )

        completion = self.provider.chat(
            messages,
            temperature=self.settings.temperature if opt.temperature is None else opt.temperature,
            max_tokens=self.settings.max_tokens,
            stream_cb=stream_cb,
            task="answer",
        )

        valid = {h.marker for h in result.hits}
        text, invented = audit_markers(completion.text, valid)
        plain, academic = split_two_layer(text)
        if opt.style == "plain" and not academic:
            plain = plain or text.strip()

        answer = Answer(
            question=question,
            plain=plain,
            academic=academic,
            sources=list(result.hits),
            raw=completion.text,
            grounded=True,
            provider=completion.provider or self.provider.name,
            model=completion.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            usage=completion.usage,
            invented_markers=invented,
            truncated=completion.truncated,
        )
        return answer, result

    # ------------------------------------------------------------------
    def _ungrounded(
        self, question: str, result: RetrievalResult, lang: str, started: float
    ) -> Answer:
        """The honest empty answer, composed locally.

        It is more useful than a model-written apology: it can report the
        actual near-miss score and the size of the filtered pool, which tells
        the user whether to rephrase, widen the filter, or add a text.
        """
        near = result.best_score
        filtered = result.filters is not None and not result.filters.is_empty()

        if lang == "zh":
            lines = [
                f"当前library里没有能支撑这个问题的原文，所以我不作回答。",
                "",
                f"检索范围内共 {result.n_candidates} 个文本块，最接近的一段相关度只有 "
                f"{near:.2f}，低于 {self.settings.min_score:.2f} 的下限。",
            ]
            if filtered:
                lines.append(f"当前过滤条件：{result.filters.describe()}——去掉过滤条件可能会有结果。")
            lines += [
                "",
                "可以试试：",
                "· 换一种问法，用文本里可能出现的词（比如把「焦虑」换成「恐惧」「扰乱」）",
                "· 放宽或去掉 --philosopher / --work 过滤",
                "· 用 `philo sources` 看看library里现在有什么",
                "· 把相关文本放进 library/ 再运行 `philo ingest`",
            ]
        else:
            lines = [
                "Nothing in this library supports an answer to that, so I am not going to give one.",
                "",
                f"I searched {result.n_candidates} passages; the closest scored {near:.2f}, "
                f"below the {self.settings.min_score:.2f} floor.",
            ]
            if filtered:
                lines.append(f"Active filter: {result.filters.describe()} — removing it may help.")
            lines += [
                "",
                "Worth trying:",
                "· rephrase using words the texts themselves would use "
                "(“fear” or “disturbance” rather than “anxiety”)",
                "· relax or drop the --philosopher / --work filter",
                "· run `philo sources` to see what the library actually holds",
                "· drop the relevant text into library/ and run `philo ingest`",
            ]

        return Answer(
            question=question,
            plain="\n".join(lines),
            academic="",
            sources=[],
            grounded=False,
            provider=self.provider.name,
            model=self.provider.chat_model,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


# --------------------------------------------------------------------------
# Conversation
# --------------------------------------------------------------------------


class Conversation:
    """Rolling history for the interactive REPL.

    Only the question and the plain layer are kept, truncated — the academic
    layer and full source blocks would eat the context window within a few
    turns while adding little to follow-up understanding.
    """

    def __init__(self, max_turns: int = 4) -> None:
        self.max_turns = max_turns
        self.turns: list[tuple[str, str]] = []

    def add(self, question: str, answer: Answer) -> None:
        self.turns.append((question, answer.plain))
        self.turns = self.turns[-self.max_turns:]

    def as_messages(self) -> list[Message]:
        out: list[Message] = []
        for q, a in self.turns:
            out.append(user(f"(earlier question) {q}"))
            out.append(assistant(truncate(a, 500)))
        return out

    def last_question(self) -> str:
        return self.turns[-1][0] if self.turns else ""

    def clear(self) -> None:
        self.turns.clear()


def _retrieval_query(question: str, history: Sequence[Message]) -> str:
    """Expand pronoun-only follow-ups with the previous question.

    "Why?" or "他为什么这么说？" carries no retrievable content on its own;
    without this, a follow-up retrieves noise and the answer silently drifts
    off-source.
    """
    if len(question.split()) > 6 or len(question) > 28:
        return question
    previous = ""
    for message in reversed(list(history)):
        if message["role"] == "user":
            previous = message["content"].replace("(earlier question)", "").strip()
            break
    return f"{previous} {question}".strip() if previous else question
