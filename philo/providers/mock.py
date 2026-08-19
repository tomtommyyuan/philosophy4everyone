"""The offline provider.  No network, no key, no cost — and no excuses.

This exists so the entire pipeline (chunk → embed → store → retrieve → prompt
→ generate → render) can be exercised and debugged with zero external moving
parts.  When something breaks in mock mode it is a bug in *our* code; when it
only breaks against a real API it is a key/network/model problem.  Keeping
those two failure classes separable is worth every line below.

Two things are simulated:

*Embeddings* — a hashed bag-of-words projection.  Not semantic, but genuinely
lexical: it really does rank a passage about death above a passage about
rhetoric when you ask about death, so retrieval can be evaluated offline.

*Generation* — an extractive composer.  It never invents philosophy.  It
selects real sentences from the retrieved context and arranges them into the
same two-layer, citation-marked shape a real model is asked to produce, so
the parser and the renderer are exercised against realistic output.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Sequence

from ..models import Message
from ..util import (
    detect_language,
    env_float,
    l2_normalize,
    split_sentences,
    tokenize,
    truncate,
)
from .base import ChatResult, ProgressCallback, StreamCallback

MOCK_DIM = 384


class MockProvider:
    """Deterministic, dependency-free stand-in for a real model."""

    name = "mock"

    def __init__(self, dim: int = MOCK_DIM, chat_model: str = "mock-sage-1") -> None:
        self._dim = dim
        self._chat_model = chat_model
        self._delay = env_float("PHILO_MOCK_DELAY", 0.006)

    # -- identity ---------------------------------------------------------
    @property
    def chat_model(self) -> str:
        return self._chat_model

    @property
    def embed_model(self) -> str:
        return f"mock-embed-{self._dim}"

    @property
    def embed_dim(self) -> int:
        return self._dim

    # -- embeddings -------------------------------------------------------
    def embed(
        self,
        texts: Sequence[str],
        *,
        on_progress: ProgressCallback | None = None,
    ) -> list[list[float]]:
        out: list[list[float]] = []
        total = len(texts)
        for i, text in enumerate(texts, 1):
            out.append(self._hash_embed(text))
            if on_progress and (i % 16 == 0 or i == total):
                on_progress(i, total)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._hash_embed(text)

    def _hash_embed(self, text: str) -> list[float]:
        """Feature-hash tokens and adjacent token pairs into a fixed vector.

        Sublinear term frequency (1 + log tf) keeps a word repeated twenty
        times from swamping the vector, which matters a lot for the
        incantatory repetition in texts like the Daodejing.
        """
        vec = [0.0] * self._dim
        tokens = tokenize(text)
        if not tokens:
            return vec

        counts = Counter(tokens)
        # Bigrams give the vector a little word-order sensitivity, so
        # "fear of death" and "death of fear" are not identical.
        counts.update(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))

        for term, tf in counts.items():
            weight = 1.0 + math.log(tf)
            if "_" in term:
                weight *= 0.6  # bigrams are supporting evidence, not primary
            # Two independent hashes per term reduce collision damage; the
            # sign hash keeps unrelated collisions cancelling rather than
            # accumulating.
            h = _fnv1a(term)
            idx = h % self._dim
            sign = 1.0 if (h >> 17) & 1 else -1.0
            vec[idx] += sign * weight
            h2 = _fnv1a("~" + term)
            vec[h2 % self._dim] += (1.0 if (h2 >> 17) & 1 else -1.0) * weight * 0.5

        return l2_normalize(vec)

    # -- generation -------------------------------------------------------
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
        prompt = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
        sources = _parse_sources(prompt)
        question = _parse_question(prompt)
        lang = detect_language(question or prompt)

        if task == "direct":
            text = _compose_direct(question, lang)
        elif task == "daily":
            text = _compose_daily(question, sources, lang)
        elif not sources and task not in ("themes", "paper"):
            # themes never has sources; paper legitimately has none when the
            # philosopher is not in the library.
            text = _compose_refusal(question, lang)
        elif task == "decision":
            text = _compose_decision(_parse_fenced(prompt) or question, sources, lang)
        elif task == "recap":
            text = _compose_recap(_parse_record(prompt), sources, lang)
        elif task == "objection":
            text = _compose_objection(question, sources, lang)
        elif task == "mood":
            text = _compose_mood(prompt, sources, lang)
        elif task == "themes":
            text = _compose_themes(prompt, lang)
        elif task == "paper":
            text = _compose_paper(prompt, sources, lang)
        else:
            # answer, council — both want the two-layer shape.
            text = _compose_answer(question, sources, lang)

        if stream_cb:
            for piece in _tokenize_for_stream(text):
                stream_cb(piece)
                if self._delay:
                    time.sleep(self._delay)

        return ChatResult(
            text=text,
            model=model or self._chat_model,
            provider=self.name,
            usage={
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": len(text) // 4,
                "total_tokens": (len(prompt) + len(text)) // 4,
            },
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason="stop",
        )

    def chat_healthcheck(self) -> str:
        return f"mock chat ready · {self._chat_model}"

    def embed_healthcheck(self) -> str:
        vec = self.embed_query("the unexamined life is not worth living")
        assert abs(sum(x * x for x in vec) - 1.0) < 1e-6
        return f"mock embed ready · {self._dim}-dim hashed"

    def healthcheck(self) -> str:
        return f"{self.chat_healthcheck()} · {self.embed_healthcheck()}"


# --------------------------------------------------------------------------
# Prompt parsing — the mock reads the same context block the real model sees
# --------------------------------------------------------------------------

_SOURCE_RE = re.compile(
    r"^\[(\d{1,2})\]\s*(.+?)\n\"\"\"\n(.*?)\n\"\"\"",
    re.MULTILINE | re.DOTALL,
)


class _Source:
    __slots__ = ("marker", "citation", "text", "philosopher", "work")

    def __init__(self, marker: int, citation: str, text: str) -> None:
        self.marker = marker
        self.citation = citation.strip()
        self.text = text.strip()
        bits = [b.strip() for b in re.split(r"·|,", citation)]
        self.philosopher = bits[0] if bits else ""
        self.work = bits[1] if len(bits) > 1 else ""


def _parse_sources(prompt: str) -> list[_Source]:
    return [
        _Source(int(m.group(1)), m.group(2), m.group(3))
        for m in _SOURCE_RE.finditer(prompt)
    ]


def _parse_question(prompt: str) -> str:
    m = re.search(r"(?:QUESTION|问题|THEME|主题)\s*:\s*(.+)", prompt)
    if m:
        return m.group(1).strip()
    # Fall back to the last non-empty line that isn't part of a source block.
    for line in reversed(prompt.splitlines()):
        line = line.strip()
        if line and not line.startswith(("[", '"""', "-", "#")):
            return line
    return ""


def _parse_fenced(prompt: str) -> str:
    """The block a decision prompt fences off as the reader's own words."""
    m = re.search(r'THE DECISION[^\n]*\n"""\n(.*?)\n"""', prompt, re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_record(prompt: str) -> list[str]:
    """The `- [2026-02-01] decided: …` lines a recap prompt carries."""
    return [
        line.strip("- ").strip()
        for line in prompt.splitlines()
        if line.strip().startswith("- [")
    ]


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------


def _key_sentences(src: _Source, query: str, limit: int = 2) -> list[str]:
    """Pick the sentences that actually earn their place in the answer.

    Scored by overlap with the query, with a mild preference for
    quotable-length sentences — very short fragments carry no argument and
    very long ones stop being quotable.
    """
    qterms = set(tokenize(query))
    scored: list[tuple[float, str]] = []
    for sent in split_sentences(src.text):
        clean = sent.strip()
        if len(clean) < 25:
            continue
        terms = set(tokenize(clean))
        if not terms:
            continue
        overlap = len(qterms & terms) / max(1, len(qterms)) if qterms else 0.0
        length_fit = 1.0 - abs(len(clean) - 160) / 400.0
        scored.append((overlap * 2.0 + max(0.0, length_fit), clean))
    if not scored:
        return [truncate(src.text, 200)]
    scored.sort(key=lambda p: p[0], reverse=True)
    return [s for _, s in scored[:limit]]


def _compose_answer(question: str, sources: list[_Source], lang: str) -> str:
    top = sources[:3]
    names = _oxford([s.philosopher for s in top if s.philosopher], lang)

    plain_lines: list[str] = []
    academic_lines: list[str] = []

    if lang == "zh":
        plain_lines.append(
            f"关于「{question}」，检索到的原文里，{names}谈得最直接。他们的说法可以这样理解："
        )
        for s in top:
            sent = _key_sentences(s, question, 1)[0]
            plain_lines.append(f"\n- {s.philosopher}：“{truncate(sent, 180)}” [{s.marker}]")
        plain_lines.append(
            "\n以上都是原文的直接摘录。离线模式不作解释与推衍——切换到真实模型后，"
            "同样这几段原文会被展开成完整的白话讲解。"
        )

        academic_lines.append("检索到的段落按论证角色排列如下：\n")
        for s in top:
            sents = _key_sentences(s, question, 2)
            academic_lines.append(f"**[{s.marker}] {s.citation}**")
            for sent in sents:
                academic_lines.append(f"> {truncate(sent, 300)}")
            academic_lines.append("")
        academic_lines.append(
            f"以上引文均直接取自检索到的{len(sources)}段原文，未作推衍。"
            "若要判断这些立场彼此是否相容，需要回到各自的完整语境。"
        )
    else:
        plain_lines.append(
            f"On “{question}”, the retrieved passages put {names} closest to the question. "
            "Here is what they actually say, in ordinary language:"
        )
        for s in top:
            sent = _key_sentences(s, question, 1)[0]
            plain_lines.append(f"\n- {s.philosopher or 'The text'}: “{truncate(sent, 180)}” [{s.marker}]")
        plain_lines.append(
            "\nEach line above is lifted straight from the retrieved passages. Offline mode "
            "selects and arranges; it does not interpret. Point this at a real model and the "
            "same passages come back as a worked explanation."
        )

        academic_lines.append("The retrieved passages, arranged by argumentative role:\n")
        for s in top:
            sents = _key_sentences(s, question, 2)
            academic_lines.append(f"**[{s.marker}] {s.citation}**")
            for sent in sents:
                academic_lines.append(f"> {truncate(sent, 300)}")
            academic_lines.append("")
        academic_lines.append(
            f"Every line above is quoted directly from the {len(sources)} retrieved passages; "
            "nothing has been extrapolated. Whether these positions are mutually consistent "
            "requires reading each in its full context."
        )

    head_plain = "## IN PLAIN WORDS"
    head_acad = "## THE ARGUMENT"
    return f"{head_plain}\n" + "\n".join(plain_lines) + f"\n\n{head_acad}\n" + "\n".join(academic_lines)


def _compose_direct(question: str, lang: str) -> str:
    """The offline provider has no knowledge to recall.

    Unsourced mode asks the model to answer from memory. A hashed-embedding
    mock has no memory, and pretending otherwise would misrepresent what the
    comparison shows.
    """
    if lang == "zh":
        return (
            "## IN PLAIN WORDS\n"
            f"「{question}」——离线模拟没有可供回忆的知识，所以这里没有答案。\n\n"
            "## THE ARGUMENT\n"
            "无出处模式要求模型凭自身知识作答；mock provider 只会检索与摘录，"
            "本身不具备任何哲学知识。配置真实 API key 后，这个模式才有意义——"
            "也才能和有出处的模式做对照。"
        )
    return (
        "## IN PLAIN WORDS\n"
        f"There is no answer here for “{question}”. The offline mock has no "
        "knowledge to recall.\n\n"
        "## THE ARGUMENT\n"
        "Unsourced mode asks the model to answer from its own memory. The mock "
        "provider only selects and quotes retrieved text — it holds no philosophy "
        "of its own, so there is nothing for it to recall. Point this at a real "
        "model and the comparison between the two modes becomes meaningful."
    )


def _compose_refusal(question: str, lang: str) -> str:
    if lang == "zh":
        return (
            "## IN PLAIN WORDS\n"
            f"关于「{question}」，我在当前library里没有检索到相关的原文段落，所以不能回答。\n\n"
            "## THE ARGUMENT\n"
            "本系统只根据检索到的原文作答。既然没有可引用的段落，任何回答都会是凭记忆编造，"
            "这正是本项目要避免的。可以尝试：换一个说法提问、放宽 --philosopher 过滤条件，"
            "或用 `philo ingest` 把相关文本加入library。"
        )
    return (
        "## IN PLAIN WORDS\n"
        f"I could not find anything in the current library that speaks to “{question}”, "
        "so I am not going to answer it.\n\n"
        "## THE ARGUMENT\n"
        "This system answers only from retrieved primary text. With no passage to cite, any "
        "answer would be reconstructed from memory — exactly the failure mode this project "
        "exists to prevent. Try rephrasing, relaxing the --philosopher filter, or adding the "
        "relevant text with `philo ingest`."
    )


def _compose_daily(theme: str, sources: list[_Source], lang: str) -> str:
    if not sources:
        return _compose_refusal(theme, lang)
    s = sources[0]
    quote = _key_sentences(s, theme, 1)[0]
    others = sources[1:3]

    if lang == "zh":
        return (
            f"## TITLE\n今天，先把{theme}放回它的原文里\n\n"
            f"## HOOK\n你大概已经把「{theme}」当成一个常识了。"
            f"但{s.philosopher}在《{s.work}》里谈到它时，语气要冷静得多——"
            "他不是在安慰谁，而是在做一个区分。\n\n"
            f"## QUOTE\n{truncate(quote, 260)} [{s.marker}]\n\n"
            f"## REFLECTION\n这句话之所以值得停一下，是因为它把一个习以为常的说法重新变成了问题。"
            + (f"{others[0].philosopher}在另一处也谈到相近的地方 [{others[0].marker}]。" if others else "")
            + "离线模式只负责把原文摆到你面前，判断留给你自己。\n\n"
            f"## PRACTICE\n今天遇到一件让你烦躁的事时，先花十秒钟问一句："
            "这件事里，究竟哪一部分在我手里？然后只处理那一部分。"
        )
    return (
        f"## TITLE\nPutting “{theme}” back in its original words\n\n"
        f"## HOOK\nYou have probably filed “{theme}” away as common sense. "
        f"When {s.philosopher} takes it up in {s.work}, the tone is far cooler — "
        "less consolation, more distinction-drawing.\n\n"
        f"## QUOTE\n{truncate(quote, 260)} [{s.marker}]\n\n"
        f"## REFLECTION\nThe line is worth pausing on because it turns something familiar back "
        "into a question. "
        + (f"{others[0].philosopher} presses on neighbouring ground [{others[0].marker}]. " if others else "")
        + "Offline mode puts the passage in front of you; the judgement stays yours.\n\n"
        f"## PRACTICE\nThe next time something today irritates you, take ten seconds first: "
        "which part of this is actually up to me? Then work only on that part."
    )


def _oxford(names: list[str], lang: str) -> str:
    names = [n for n in dict.fromkeys(names) if n]
    if not names:
        return "the retrieved passages" if lang != "zh" else "这些段落"
    if lang == "zh":
        return "、".join(names)
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _tokenize_for_stream(text: str, size: int = 18) -> list[str]:
    """Chop into word-ish pieces so streaming looks like a real model."""
    pieces, buf = [], ""
    for word in re.split(r"(\s+)", text):
        buf += word
        if len(buf) >= size:
            pieces.append(buf)
            buf = ""
    if buf:
        pieces.append(buf)
    return pieces


def _fnv1a(s: str) -> int:
    """FNV-1a — stable across processes, unlike Python's salted hash()."""
    h = 0xCBF29CE484222325
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _compose_decision(situation: str, sources: list["_Source"], lang: str) -> str:
    """Offline, a decision reading is exactly what the mock can honestly give:
    the passages that matched, arranged as questions rather than as advice."""
    top = sources[:3]
    if lang == "zh":
        lines = [
            "## THE CHOICE",
            f"你写下的是：「{truncate(situation, 200)}」。离线模式不作解读，"
            "只能把检索到的原文按它们提出的问题排列出来。",
            "",
            "## WHAT THE TEXTS WOULD ASK",
        ]
        for s in top:
            lines.append(f"- {s.philosopher}：“{truncate(_key_sentences(s, situation, 1)[0], 200)}” [{s.marker}]")
        lines += [
            "",
            "## WHAT THEY DO NOT SETTLE",
            f"以上{len(top)}段都是直接摘录，没有一句是针对你的情况写的——"
            "把它们套到你的具体处境上，正是真实模型要做的事。",
        ]
        return "\n".join(lines)

    lines = [
        "## THE CHOICE",
        f"What you wrote down: “{truncate(situation, 200)}”. Offline mode does not "
        "interpret; it can only lay out the passages that matched, as the questions "
        "they raise.",
        "",
        "## WHAT THE TEXTS WOULD ASK",
    ]
    for s in top:
        lines.append(
            f"- {s.philosopher or 'The text'}: “{truncate(_key_sentences(s, situation, 1)[0], 200)}” [{s.marker}]"
        )
    lines += [
        "",
        "## WHAT THEY DO NOT SETTLE",
        f"All {len(top)} lines above are quoted directly; none of them was written "
        "about your situation. Applying them to your specifics is precisely the part "
        "a real model does.",
    ]
    return "\n".join(lines)


def _compose_recap(record: list[str], sources: list["_Source"], lang: str) -> str:
    """A recap the mock can make honestly: the record restated, no pattern claimed.

    Inventing a thread is the failure the real prompt forbids, and the mock
    has no way to find a real one — so it says so rather than guessing.
    """
    shown = record[:6]
    if lang == "zh":
        lines = ["## THE WEEK", "这一周你记下了："]
        lines += [f"- {truncate(item, 160)}" for item in shown] or ["- （无）"]
        lines += [
            "",
            "## THE THREAD",
            "离线模式无法判断这些条目之间是否真有一条线索，所以不作声称——"
            "编造一个模式比没有模式更糟。相关原文：",
        ]
        lines += [f"> {truncate(_key_sentences(s, ' '.join(shown), 1)[0], 200)} [{s.marker}]" for s in sources[:2]]
        lines += ["", "## WHAT TO SIT WITH", "这些条目里，哪一条你其实已经决定了，只是还没承认？"]
        return "\n".join(lines)

    lines = ["## THE WEEK", "What you put down this week:"]
    lines += [f"- {truncate(item, 160)}" for item in shown] or ["- (nothing)"]
    lines += [
        "",
        "## THE THREAD",
        "Offline mode cannot tell whether these entries share a thread, so it does "
        "not claim one — a manufactured pattern is worse than none. The passages "
        "behind them:",
    ]
    lines += [
        f"> {truncate(_key_sentences(s, ' '.join(shown), 1)[0], 200)} [{s.marker}]"
        for s in sources[:2]
    ]
    lines += [
        "",
        "## WHAT TO SIT WITH",
        "Which of these had you already decided before you wrote it down?",
    ]
    return "\n".join(lines)


def _compose_objection(question: str, sources: list["_Source"], lang: str) -> str:
    top = sources[:3]
    if lang == "zh":
        lines = ["## THE OBJECTION", "离线模式不作论辩，只能列出其他传统在这个问题上的原文："]
        lines += [f"- {s.philosopher}：“{truncate(_key_sentences(s, question, 1)[0], 200)}” [{s.marker}]" for s in top]
        lines += ["", "## WHERE THIS LEAVES IT", "这些段落是否真的构成反驳，需要真实模型来判断。"]
        return "\n".join(lines)

    lines = [
        "## THE OBJECTION",
        "Offline mode does not argue. What it can do is put the other traditions' "
        "passages on the table:",
    ]
    lines += [
        f"- {s.philosopher or 'The text'}: “{truncate(_key_sentences(s, question, 1)[0], 200)}” [{s.marker}]"
        for s in top
    ]
    lines += [
        "",
        "## WHERE THIS LEAVES IT",
        "Whether these passages actually contradict the position is the judgement a "
        "real model makes; the mock only shows what would be judged.",
    ]
    return "\n".join(lines)


def _compose_mood(prompt: str, sources: list["_Source"], lang: str) -> str:
    """Offline, the honest check-in is the passages themselves, by tradition."""
    import re as _re

    m = _re.search(r"TODAY THEY FEEL:\s*(.+)", prompt)
    mood = m.group(1).strip() if m else "this"
    reason = _parse_fenced_any(prompt)
    seen: dict[str, "_Source"] = {}
    for s in sources:
        seen.setdefault(s.philosopher or "The text", s)
    picked = list(seen.values())[:3]
    query = reason or mood

    if lang == "zh":
        lines = [
            "## WHAT IS HAPPENING",
            f"你说今天感觉「{mood}」。" + (f"你写的是：{truncate(reason, 160)}" if reason else "")
            + "离线模式不作解读，只把检索到的原文按出处列出来。",
            "",
            "## THE SCHOOLS",
        ]
        for s in picked:
            lines += [f"**{s.philosopher or '原文'}。**「{truncate(_key_sentences(s, query, 1)[0], 200)}」[{s.marker}]", ""]
        lines += ["## ONE THING", "把上面任意一句读两遍——一遍看它说了什么，一遍看它没说什么。"]
        return "\n".join(lines)

    lines = [
        "## WHAT IS HAPPENING",
        f"You said you feel {mood} today."
        + (f" What you wrote: {truncate(reason, 160)}" if reason else "")
        + " Offline mode does not interpret; it lays out the passages that matched.",
        "",
        "## THE SCHOOLS",
    ]
    for s in picked:
        lines += [
            f"**{s.philosopher or 'The text'}.** “{truncate(_key_sentences(s, query, 1)[0], 200)}” [{s.marker}]",
            "",
        ]
    lines += [
        "## ONE THING",
        "Read any one line above twice — once for what it says, once for what it does not.",
    ]
    return "\n".join(lines)


def _parse_fenced_any(prompt: str) -> str:
    """The reader's own words, from whichever fenced block a prompt used."""
    import re as _re

    m = _re.search(r'(?:IN THEIR WORDS|THE DECISION[^\n]*)\n"""\n(.*?)\n"""', prompt, _re.DOTALL)
    return m.group(1).strip() if m else ""


def _paper_body(prompt: str) -> str:
    import re as _re

    m = _re.search(r"THE PAPER\n=+\n'''\n(.*?)\n'''", prompt, _re.DOTALL)
    return m.group(1).strip() if m else ""


def _compose_themes(prompt: str, lang: str) -> str:
    """Offline, the honest reading of a paper is its own opening sentences."""
    body = _paper_body(prompt)
    lead = " ".join(split_sentences(body)[:3]) or truncate(body, 300)
    # The paper's own words lead. This block is also the retrieval query, and
    # a preamble in front of it dilutes the embedding with text about the mock
    # rather than text about the paper.
    if lang == "zh":
        return (
            "## WHAT THE PAPER CLAIMS\n"
            f"{truncate(lead, 600)}\n\n"
            "（以上是论文开头的原文。把它改写成传统自己的词汇，是真实模型要做的事。）"
        )
    return (
        "## WHAT THE PAPER CLAIMS\n"
        f"{truncate(lead, 600)}\n\n"
        "(That is the paper's own opening, quoted. Rewriting it into the "
        "tradition's vocabulary is the part a real model does.)"
    )


def _compose_paper(prompt: str, sources: list["_Source"], lang: str) -> str:
    top = sources[:3]
    if lang == "zh":
        lines = ["## WHERE THEY WOULD AGREE",
                 "离线模式不作推断。以下是检索到的这位哲学家的原文：" if top
                 else "这个library里没有这位哲学家的文本，离线模式也没有可凭借的知识。"]
        lines += [f"- 「{truncate(_key_sentences(s, '', 1)[0], 200)}」[{s.marker}]" for s in top]
        lines += ["", "## WHERE THEY WOULD OBJECT",
                  "他会反对什么，需要真实模型把上面的立场应用到这篇论文上。",
                  "", "## THE QUESTION THEY WOULD PUT",
                  "这一段同样需要推断，离线模式不作推断。"]
        return "\n".join(lines)

    lines = [
        "## WHERE THEY WOULD AGREE",
        "Offline mode does not infer. What it can show is the retrieved text:" if top
        else "This library holds nothing by that philosopher, and the mock has no "
             "knowledge of its own to fall back on.",
    ]
    lines += [f"- “{truncate(_key_sentences(s, '', 1)[0], 200)}” [{s.marker}]" for s in top]
    lines += [
        "",
        "## WHERE THEY WOULD OBJECT",
        "Applying those positions to this paper is the inference a real model makes.",
        "",
        "## THE QUESTION THEY WOULD PUT",
        "Also an inference, and also not something the mock will pretend to.",
    ]
    return "\n".join(lines)
