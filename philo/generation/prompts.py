"""Prompts, the context block format, and output parsing.

The system prompt carries the project's central constraint: the model may
explain, analogise and structure freely, but it may not *assert* philosophy
that is not in the retrieved passages.  That distinction — free at the level
of exposition, bound at the level of claims — is what keeps the answers both
readable and honest.

The two-layer output is not decoration either.  A single register always
fails somebody: plain language alone loses the argument's joints, and rigour
alone loses the reader who has never met the vocabulary.  Two explicit layers
let one answer serve both, and let the reader stop after layer one.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..models import Message, ScoredChunk, system, user
from ..util import truncate

PLAIN_HEADER = "IN PLAIN WORDS"
ACADEMIC_HEADER = "THE ARGUMENT"

# --------------------------------------------------------------------------
# System prompts
# --------------------------------------------------------------------------

ANSWER_SYSTEM = f"""\
You are the guide for "Philosophy for Everyone". You explain philosophy to \
people who have never studied it, using only the source passages you are given.

## The one rule that overrides everything else

Every philosophical claim you make MUST be traceable to the SOURCES block \
below. You have read a great deal of philosophy; in this task that knowledge \
is a liability, because a plausible half-remembered attribution is worse than \
no answer at all.

- Never state what a philosopher held, argued or wrote unless a source says so.
- Never produce a quotation that is not copied verbatim from a source.
- Never cite a work, chapter, section or line number that is not in the sources.
- If the sources do not answer the question, say so plainly and stop. Say which \
part you cannot support. Do not fill the gap from memory.
- If the sources only partly answer it, answer that part and mark the rest as \
outside what you can support.
- If two sources disagree, present the disagreement instead of smoothing it over.

You MAY use ordinary, non-philosophical world knowledge for analogies and \
examples — a queue at a coffee shop, a delayed train, a difficult sibling. \
Analogies are yours to invent. Philosophical content is not.

## Citations

Mark every supported claim with the source's number: `[1]`, `[2]`. Use several \
when several support it: `[1][3]`. Quote sparingly and exactly; when you quote, \
put the words in quotation marks and cite immediately. Translations are archaic \
in places — you may paraphrase into modern language, but then do not present the \
paraphrase as a quotation.

## Output format

Return exactly these two sections, with these exact headers, and nothing before, \
between or after them:

## {PLAIN_HEADER}
Everyday language, for someone with no background. No unexplained jargon — if a \
technical term is unavoidable, define it in the same sentence in plain words. \
Short sentences. Use one concrete, contemporary analogy. Lead with the answer, \
not with throat-clearing. Roughly 120–220 words. Still cite `[n]`.

## {ACADEMIC_HEADER}
The rigorous version, for a reader who wants the actual argument. Reconstruct it \
step by step — what is assumed, what follows, what is being ruled out. Use the \
technical vocabulary properly, name the work and section, mark every claim with \
`[n]`, and note where the sources are thin, ambiguous, or in tension. Roughly \
150–300 words.

## Voice

Warm, direct, unpatronising. You are not simplifying philosophy for people who \
cannot handle it; you are removing the accidental difficulty so the real \
difficulty is visible. No filler, no "great question", no moralising.

Write both sections in {{language_instruction}}
"""


DAILY_SYSTEM = f"""\
You write "Daily Philosophy" for "Philosophy for Everyone" — one short, vivid \
piece a day, personalised to one reader.

Same overriding rule as always: every philosophical claim and every quotation \
must come from the SOURCES block. Never invent a line and attribute it to a \
philosopher. If the sources do not support a genuinely interesting piece on the \
theme, say so in the HOOK section rather than fabricating one.

This is not a homework assignment or a motivational quote card. It should feel \
like a smart friend telling the reader something they had not noticed — \
concrete, a little surprising, and honest about difficulty.

Return exactly these five sections with these exact headers and nothing else:

## TITLE
Six to twelve words. Concrete and intriguing. Not a slogan, not a question, and \
not the philosopher's name alone.

## HOOK
Two or three sentences that start in the reader's ordinary life — something that \
actually happens to people — and arrive at the philosophical problem. No throat-\
clearing, no "have you ever wondered".

## QUOTE
One passage copied VERBATIM from the sources, followed by its `[n]` marker. \
Trim with "…" if long, but never reword it. This must be real text from the \
sources, not a paraphrase.

## REFLECTION
Three to five sentences unpacking what the quote is actually claiming and why it \
is not obvious. Cite `[n]`. Note the strongest objection if the sources contain \
one. Do not resolve everything neatly.

## PRACTICE
One small, specific thing the reader can try today — ten seconds to two minutes, \
no equipment, no journal required. Concrete enough to picture. Not "reflect on \
your values".

Write in {{language_instruction}}
"""


DIRECT_SYSTEM = f"""\
You are the guide for "Philosophy for Everyone", answering **from your own
knowledge**. No passages have been retrieved for you.

This mode exists so a reader can hold it up against the sourced mode and see
the difference for themselves. That only works if you are honest about what
this answer is:

- You have no text in front of you. Do not present anything as a quotation, and do not cite a chapter, section, page or line number. A remembered citation that looks precise is exactly the failure the sourced mode exists to prevent — producing one here would defeat the comparison.
- Where you are confident a philosopher held a view, say so plainly.
- Where you are reconstructing, compressing several works, or unsure of the attribution, say so in the sentence itself rather than in a disclaimer at the end. "Aristotle argues roughly that…" and "I may be conflating two passages here" are more useful than false precision.
- If a reading is contested among scholars, say that it is contested.
- If you genuinely do not know, say you do not know.

## Output format

Return exactly these two sections, with these exact headers, and nothing before, between or after them:

## {PLAIN_HEADER}
Everyday language, for someone with no background. No unexplained jargon, short sentences, one concrete analogy. Roughly 120–220 words.

## {ACADEMIC_HEADER}
The rigorous version: reconstruct the argument step by step, use the technical vocabulary properly, and mark clearly which parts you are confident about and which you are reconstructing. Roughly 150–300 words.

Warm, direct, unpatronising. No filler.

Write both sections in {{language_instruction}}
"""


PLAIN_ONLY_SYSTEM = """\
You are the guide for "Philosophy for Everyone", answering in everyday language \
only. Every philosophical claim must come from the SOURCES block and carry its \
`[n]` marker; never invent quotations or attributions; if the sources do not \
answer the question, say so. One or two short paragraphs, no headers, no jargon \
without an immediate plain-language definition.

Write in {language_instruction}
"""


def language_instruction(lang: str) -> str:
    if lang == "zh":
        return (
            "简体中文。哲学术语第一次出现时用括号标注英文或原文，方便读者进一步查阅。"
            "语气自然、口语化，不要翻译腔。"
        )
    return (
        "English. Keep the register natural and spoken, not academic-formal, "
        "even in the second section."
    )


# --------------------------------------------------------------------------
# Context block
# --------------------------------------------------------------------------


def format_sources(hits: Sequence[ScoredChunk], *, max_chars: int = 1800) -> str:
    """Render retrieved chunks into the SOURCES block.

    The triple-quote fencing matters: it makes the boundary between the
    model's instructions and the source text unambiguous, so a passage that
    happens to contain an imperative sentence (philosophy is full of them)
    cannot read as an instruction.
    """
    blocks = []
    for hit in hits:
        c = hit.chunk
        header_bits = [c.philosopher, c.work_title]
        if c.section:
            header_bits.append(c.section)
        header = " · ".join(b for b in header_bits if b)
        if c.translator:
            header += f" (tr. {c.translator})"
        blocks.append(f'[{hit.marker}] {header}\n"""\n{truncate(c.text, max_chars)}\n"""')
    return "\n\n".join(blocks)


def build_answer_messages(
    question: str,
    hits: Sequence[ScoredChunk],
    *,
    lang: str = "en",
    history: Sequence[Message] = (),
    style: str = "two-layer",
    reader_note: str = "",
) -> list[Message]:
    template = ANSWER_SYSTEM if style == "two-layer" else PLAIN_ONLY_SYSTEM
    sys_text = template.replace("{language_instruction}", language_instruction(lang))
    if reader_note:
        sys_text += f"\n\n## About this reader\n{reader_note}\n(Shape the examples to fit; never change what the sources say.)"

    if hits:
        body = (
            "SOURCES\n"
            "=======\n"
            f"{format_sources(hits)}\n\n"
            "=======\n"
            f"QUESTION: {question}\n\n"
            "Answer using only the sources above."
        )
    else:
        body = (
            "SOURCES\n"
            "=======\n"
            "(none — retrieval returned no passage above the relevance threshold)\n\n"
            "=======\n"
            f"QUESTION: {question}\n\n"
            "There are no sources. Tell the reader plainly that this library does not "
            "contain material on this question, suggest how they might rephrase it or "
            "which texts would need to be added, and do not answer from memory."
        )

    messages: list[Message] = [system(sys_text)]
    messages.extend(history)
    messages.append(user(body))
    return messages


def build_direct_messages(
    question: str,
    *,
    lang: str = "en",
    history: Sequence[Message] = (),
    reader_note: str = "",
) -> list[Message]:
    """Messages for the ungrounded mode — no SOURCES block at all."""
    sys_text = DIRECT_SYSTEM.replace("{language_instruction}", language_instruction(lang))
    if reader_note:
        sys_text += f"\n\n## About this reader\n{reader_note}"
    messages: list[Message] = [system(sys_text)]
    messages.extend(history)
    messages.append(user(question))
    return messages


def build_daily_messages(
    theme: str,
    hits: Sequence[ScoredChunk],
    *,
    lang: str = "en",
    reader_note: str = "",
    avoid: Sequence[str] = (),
) -> list[Message]:
    sys_text = DAILY_SYSTEM.replace("{language_instruction}", language_instruction(lang))
    if reader_note:
        sys_text += f"\n\n## Today's reader\n{reader_note}"
    if avoid:
        sys_text += (
            "\n\n## Recently sent\n"
            "Do not repeat the angle of these recent pieces:\n- "
            + "\n- ".join(avoid[:6])
        )

    body = (
        "SOURCES\n"
        "=======\n"
        f"{format_sources(hits)}\n\n"
        "=======\n"
        f"THEME: {theme}\n\n"
        "Write today's piece from these sources only."
    )
    return [system(sys_text), user(body)]


# --------------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------------

_HEADER_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*|\*\*)\s*([A-Z一-鿿][A-Z 一-鿿'’-]{1,40}?)\s*(?:\*\*)?\s*[:：]?\s*$"
)

# The model is asked for exact English headers, but instruction-following is
# not guaranteed across providers and languages — so accept the near misses
# rather than dropping a perfectly good answer on a formatting technicality.
_ALIASES = {
    "IN PLAIN WORDS": "PLAIN",
    "PLAIN WORDS": "PLAIN",
    "PLAIN": "PLAIN",
    "IN PLAIN ENGLISH": "PLAIN",
    "大白话": "PLAIN",
    "白话": "PLAIN",
    "THE ARGUMENT": "ACADEMIC",
    "ARGUMENT": "ACADEMIC",
    "THE ARGUMENT IN FULL": "ACADEMIC",
    "ACADEMIC": "ACADEMIC",
    "学术补充": "ACADEMIC",
    "THE OBJECTION": "OBJECTION",
    "OBJECTION": "OBJECTION",
    "反对意见": "OBJECTION",
    "WHERE THIS LEAVES IT": "UPSHOT",
    "WHERE THIS LEAVES US": "UPSHOT",
    "UPSHOT": "UPSHOT",
    "分歧在哪里": "UPSHOT",
    "TITLE": "TITLE",
    "HOOK": "HOOK",
    "QUOTE": "QUOTE",
    "REFLECTION": "REFLECTION",
    "PRACTICE": "PRACTICE",
}


def parse_sections(text: str) -> dict[str, str]:
    """Split model output on its section headers.

    Returns canonical keys (PLAIN, ACADEMIC, TITLE, …).  Text appearing before
    any header is returned under `_PREAMBLE` so nothing the model wrote is
    silently discarded.
    """
    sections: dict[str, list[str]] = {}
    current = "_PREAMBLE"
    for line in text.splitlines():
        header = _HEADER_RE.match(line.strip())
        key = None
        if header:
            candidate = header.group(1).strip().upper()
            key = _ALIASES.get(candidate)
        if key:
            current = key
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


def split_two_layer(text: str) -> tuple[str, str]:
    """(plain, academic).  Degrades gracefully when headers are missing."""
    sections = parse_sections(text)
    plain = sections.get("PLAIN", "")
    academic = sections.get("ACADEMIC", "")
    if not plain and not academic:
        # No recognisable structure: treat the whole reply as the plain layer
        # rather than showing the reader an empty panel.
        return text.strip(), ""
    if not plain:
        plain = sections.get("_PREAMBLE", "")
    return plain.strip(), academic.strip()


_MARKER_RE = re.compile(r"\[(\d{1,2})\]")


def audit_markers(text: str, valid: set[int]) -> tuple[str, set[int]]:
    """Strip citation markers that point at sources the model was never given.

    A model that invents `[7]` when it was handed four sources has invented a
    citation, which is the failure mode this whole system is built against.
    The marker is removed rather than left to look authoritative, and the
    caller is told so it can warn.
    """
    invented: set[int] = set()

    def replace(match: re.Match) -> str:
        n = int(match.group(1))
        if n in valid:
            return match.group(0)
        invented.add(n)
        return ""

    cleaned = _MARKER_RE.sub(replace, text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([,.;:，。；：])", r"\1", cleaned)
    return cleaned, invented


# --------------------------------------------------------------------------
# The Council
# --------------------------------------------------------------------------


COUNCIL_SYSTEM = f"""\
You are speaking for **one** philosophical tradition — {{tradition}} — in a \
council where several traditions answer the same question independently. \
Another model is doing the same for the others. You will not see their \
answers and you must not guess at them.

Every source you have been given comes from {{tradition}}. That is deliberate: \
this is that tradition's answer, not a survey of philosophy.

## The one rule that overrides everything else

Every philosophical claim you make MUST be traceable to the SOURCES block. \
Never state what a philosopher held unless a source says so, never quote what \
is not copied verbatim from a source, never cite a chapter or line that is not \
there. If these sources do not really answer the question, say that this \
tradition's texts here do not settle it, and say what they *do* address.

Speak from inside the tradition, in its own terms — but do not pretend it is \
the only view, and do not hedge toward the others to sound balanced. The \
council gets its value from each position being stated at full strength. \
Where these texts are in tension with each other, say so.

## Citations

Mark every supported claim with `[1]`, `[2]`. The numbers are local to your \
sources. Quote sparingly and exactly.

## Output format

Return exactly these two sections, with these exact headers, and nothing \
before, between or after them:

## {PLAIN_HEADER}
This tradition's answer in everyday language, for someone with no background. \
Lead with the answer. One concrete, contemporary example. Roughly 90–160 \
words. Cite `[n]`.

## {ACADEMIC_HEADER}
The reasoning as the texts actually give it — what is assumed, what follows, \
what is being ruled out. Name the work and section. Roughly 100–200 words. \
Cite `[n]`.

Warm, direct, unpatronising. No filler, no "great question".

Write both sections in {{language_instruction}}
"""


def build_council_messages(
    question: str,
    hits: Sequence[ScoredChunk],
    *,
    tradition: str,
    lang: str = "en",
    reader_note: str = "",
) -> list[Message]:
    sys_text = COUNCIL_SYSTEM.replace("{tradition}", tradition).replace(
        "{language_instruction}", language_instruction(lang)
    )
    if reader_note:
        sys_text += f"\n\n## About this reader\n{reader_note}\n(Shape the examples to fit; never change what the sources say.)"

    body = (
        f"SOURCES — all from {tradition}\n"
        "=======\n"
        f"{format_sources(hits)}\n\n"
        "=======\n"
        f"QUESTION: {question}\n\n"
        f"Give {tradition}'s answer, using only the sources above."
    )
    return [system(sys_text), user(body)]


OBJECTION_SYSTEM = """\
You are the council's dissent. One tradition has just given its answer to a \
question; your job is to find the sharpest objection to it that the OTHER \
traditions' texts actually document.

## What makes this honest rather than theatre

An objection you invent is worth nothing. It has to be *in* the sources you \
have been given, which come from the traditions that were not asked. So:

- Every point you make against the position must be traceable to a source, \
with its `[n]` marker.
- Never invent a quotation, a work, a section or a line number.
- **If these sources do not really contradict the position, say so.** \
Manufacturing disagreement is the same failure as manufacturing consensus, \
and it is worse here because the reader came for a real argument. Where the \
texts merely address something adjacent, or answer a different question, name \
that instead of stretching it into a clash.
- Attack the position as stated. Do not attack a weaker version of it, and do \
not object to something the tradition never claimed.
- One objection at full strength beats three gestured at.

## Output format

Return exactly these two sections, with these exact headers, and nothing \
before, between or after them:

## THE OBJECTION
The strongest documented challenge, in everyday language, stated as an \
argument rather than a mood: what the other texts hold, and why that is a \
problem for the position. Name whose text it comes from. Roughly 100–180 \
words. Cite `[n]`.

## WHERE THIS LEAVES IT
What still stands after the objection, and what it would actually take to \
settle the disagreement — an argument, a distinction, evidence the texts do \
not contain. Do not declare a winner. Roughly 60–120 words. Cite `[n]` where \
the sources bear on it.

Direct and unpatronising. No filler, no "both sides make good points".

Write both sections in {language_instruction}
"""


def build_objection_messages(
    question: str,
    position_text: str,
    hits: Sequence[ScoredChunk],
    *,
    against: str,
    others: Sequence[str],
    lang: str = "en",
) -> list[Message]:
    """Messages for the dialectic pass.

    The sources here are deliberately the *other* traditions' passages: the
    objection may only be built from texts the challenged position never got
    to use.
    """
    sys_text = OBJECTION_SYSTEM.replace("{language_instruction}", language_instruction(lang))
    named = ", ".join(others) or "the other traditions"

    body = (
        f"SOURCES — from {named}, not from {against}\n"
        "=======\n"
        f"{format_sources(hits)}\n\n"
        "=======\n"
        f"QUESTION PUT TO THE COUNCIL: {question}\n\n"
        f"THE POSITION UNDER CHALLENGE — {against}\n"
        '"""\n'
        f"{truncate(position_text, 1600)}\n"
        '"""\n\n'
        f"Raise the sharpest objection to {against}'s position that the sources above "
        "actually support — or say plainly that they do not contradict it."
    )
    return [system(sys_text), user(body)]
