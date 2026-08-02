"""Chunking that respects arguments.

Fixed-width chunking is the default everywhere and it is wrong for
philosophy.  Cut at character 800 and you routinely land between a premise
and its conclusion — the retrieved passage then says "therefore the soul is
immortal" with the reasoning stranded in a neighbouring chunk that never gets
retrieved.  The model fills the gap from memory, and you have manufactured
exactly the hallucination this project exists to prevent.

So the unit here is the *argument*, approximated as follows:

1. Split on section headings (`## Book II`) — never merge across them, since
   a heading is the author's own boundary marker.
2. Inside a section, paragraphs are the atoms.  Accumulate whole paragraphs
   up to a target size.
3. A paragraph too large to keep whole is split *between sentences*, never
   inside one, with one sentence of overlap so a conclusion keeps its premise.
4. Fragments below the minimum are merged into a neighbour, because a
   two-line chunk retrieves well and explains nothing.

Length is measured with `char_weight`, which counts a CJK character as more
than a Latin one — otherwise Chinese chunks silently carry three times the
content of English ones at the same nominal size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import Settings
from ..models import Chunk, Work
from ..util import char_weight, split_sentences

# Markdown ATX headings become section labels.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
# Common in-text section markers used by public-domain editions.
_INLINE_SECTION_RE = re.compile(
    r"^(?:BOOK|CHAPTER|PART|SECTION|Book|Chapter|Part|Section)\s+"
    r"([IVXLCDM]+|\d+|[A-Z][a-z]+)\.?\s*$"
)


@dataclass
class ChunkStats:
    n_chunks: int = 0
    n_chars: int = 0
    min_chars: int = 0
    max_chars: int = 0

    @property
    def avg_chars(self) -> int:
        return self.n_chars // self.n_chunks if self.n_chunks else 0


@dataclass
class _Block:
    """A run of paragraphs sharing one section heading."""

    section: str
    paragraphs: list[str]


def split_blocks(body: str) -> list[_Block]:
    blocks: list[_Block] = []
    current = _Block(section="", paragraphs=[])
    # Heading levels are tracked so `### The Good` nests under `## Book I`.
    trail: dict[int, str] = {}

    for raw_para in re.split(r"\n\s*\n", body):
        para = raw_para.strip()
        if not para:
            continue

        heading = _as_heading(para)
        if heading is not None:
            level, title = heading
            if current.paragraphs:
                blocks.append(current)
            trail = {lvl: t for lvl, t in trail.items() if lvl < level}
            trail[level] = title
            section = " · ".join(trail[lvl] for lvl in sorted(trail))
            current = _Block(section=section, paragraphs=[])
            continue

        current.paragraphs.append(_clean_paragraph(para))

    if current.paragraphs:
        blocks.append(current)
    return [b for b in blocks if any(p.strip() for p in b.paragraphs)]


def _as_heading(para: str) -> tuple[int, str] | None:
    """A paragraph is a heading only if it is a *single* heading line."""
    lines = [ln for ln in para.splitlines() if ln.strip()]
    if len(lines) != 1:
        return None
    line = lines[0].strip()
    m = _HEADING_RE.match(line)
    if m:
        return len(m.group(1)), m.group(2).strip()
    m = _INLINE_SECTION_RE.match(line)
    if m:
        return 2, line.rstrip(".")
    return None


def _clean_paragraph(para: str) -> str:
    """Unwrap hard line breaks; keep list and verse structure intact.

    Public-domain texts are usually hard-wrapped at ~70 columns, which would
    otherwise make every line look like its own paragraph.  Lines that begin
    with a list or verse marker keep their break.
    """
    lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
    if not lines:
        return ""
    out = [lines[0]]
    for line in lines[1:]:
        if re.match(r"^([-*+•]|\d+[.)]|>)\s+", line):
            out.append("\n" + line)
        else:
            prev = out[-1]
            # No space before CJK, which does not use inter-word spacing.
            joiner = "" if prev and "一" <= prev[-1] <= "鿿" else " "
            out[-1] = prev + joiner + line if not prev.endswith("\n") else prev + line
    return "".join(out).strip()


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------


def pack_paragraphs(paragraphs: list[str], settings: Settings) -> list[str]:
    units: list[str] = []
    buf: list[str] = []
    buf_w = 0

    for para in paragraphs:
        if not para.strip():
            continue
        w = char_weight(para)

        if w > settings.chunk_max_chars:
            if buf:
                units.append("\n\n".join(buf))
                buf, buf_w = [], 0
            units.extend(split_long(para, settings))
            continue

        if buf_w and buf_w + w > settings.chunk_target_chars:
            units.append("\n\n".join(buf))
            buf, buf_w = [para], w
        else:
            buf.append(para)
            buf_w += w

    if buf:
        units.append("\n\n".join(buf))
    return merge_small(units, settings)


def split_long(paragraph: str, settings: Settings) -> list[str]:
    """Split one oversized paragraph on sentence boundaries, with overlap."""
    sentences = split_sentences(paragraph)
    if len(sentences) <= 1:
        return hard_split(paragraph, settings.chunk_max_chars)

    pieces: list[str] = []
    buf: list[str] = []
    buf_w = 0

    for sent in sentences:
        w = char_weight(sent)
        if w > settings.chunk_max_chars:
            # A single sentence longer than the ceiling (unpunctuated classical
            # text, mostly). Flush, then break it on whitespace as a last resort.
            if buf:
                pieces.append(" ".join(buf))
                buf, buf_w = [], 0
            pieces.extend(hard_split(sent, settings.chunk_max_chars))
            continue

        if buf_w and buf_w + w > settings.chunk_target_chars:
            pieces.append(" ".join(buf))
            # Carry the tail sentence forward so a conclusion keeps its
            # premise — but only when it is genuinely a tail. Authors like
            # Nietzsche and Mill write 400-character sentences, and blindly
            # duplicating one inflates the index by a third and can push the
            # next piece past the hard ceiling.
            overlap = _overlap_for(buf, w, settings)
            buf = [*overlap, sent]
            buf_w = sum(char_weight(s) for s in buf)
        else:
            buf.append(sent)
            buf_w += w

    if buf:
        tail = " ".join(buf)
        # Overlap can leave a final piece that is nothing but the overlap.
        if pieces and tail and pieces[-1].endswith(tail):
            pass
        else:
            pieces.append(tail)
    return [p.strip() for p in pieces if p.strip()]


def _overlap_for(buf: list[str], next_weight: int, settings: Settings) -> list[str]:
    """Which trailing sentences to repeat into the next piece.

    Two limits, both load-bearing: an overlap may not exceed a quarter of the
    target size (or the index balloons with duplicated text), and it may not
    push the next piece over the hard ceiling (or `chunk_max_chars` silently
    stops meaning anything).
    """
    if not settings.chunk_overlap_sentences or not buf:
        return []
    budget = min(
        settings.chunk_target_chars // 4,
        max(0, settings.chunk_max_chars - next_weight),
    )
    overlap: list[str] = []
    total = 0
    for sentence in reversed(buf[-settings.chunk_overlap_sentences:]):
        weight = char_weight(sentence)
        if total + weight > budget:
            break
        overlap.insert(0, sentence)
        total += weight
    return overlap


def hard_split(text: str, limit: int) -> list[str]:
    """Absolute last resort: break on whitespace near the limit."""
    out: list[str] = []
    remaining = text.strip()
    while char_weight(remaining) > limit:
        window = remaining[:limit]
        cut = max(window.rfind(" "), window.rfind("，"), window.rfind("、"))
        if cut < limit // 2:
            cut = limit
        out.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        out.append(remaining)
    return out


def merge_small(units: list[str], settings: Settings) -> list[str]:
    """Fold undersized units into whichever neighbour has room.

    Runs to a fixed point so a chain of one-line aphorisms (the Analects, the
    Daodejing) collapses into properly sized chunks instead of only pairing up.
    """
    if not units:
        return []

    changed = True
    while changed and len(units) > 1:
        changed = False
        out: list[str] = []
        i = 0
        while i < len(units):
            unit = units[i]
            if char_weight(unit) >= settings.chunk_min_chars:
                out.append(unit)
                i += 1
                continue

            # +2 for the blank line the merge inserts, so a merge can never
            # produce a chunk one or two characters over the stated ceiling.
            joined = char_weight(unit) + 2
            prev_ok = bool(out) and char_weight(out[-1]) + joined <= settings.chunk_max_chars
            nxt = units[i + 1] if i + 1 < len(units) else None
            next_ok = nxt is not None and char_weight(nxt) + joined <= settings.chunk_max_chars

            if prev_ok and (not next_ok or char_weight(out[-1]) <= char_weight(nxt or "")):
                out[-1] = out[-1] + "\n\n" + unit
                changed = True
                i += 1
            elif next_ok:
                units[i + 1] = unit + "\n\n" + nxt  # type: ignore[operator]
                changed = True
                i += 1
            else:
                # Too small to keep, too big to merge anywhere: keep it whole.
                out.append(unit)
                i += 1
        units = out
    return units


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def chunk_work(work: Work, body: str, settings: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0
    for block in split_blocks(body):
        for text in pack_paragraphs(block.paragraphs, settings):
            text = text.strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    id=Chunk.make_id(work.id, index, text),
                    work_id=work.id,
                    text=text,
                    section=block.section,
                    chunk_index=index,
                    philosopher=work.philosopher,
                    philosopher_zh=work.philosopher_zh,
                    work_title=work.title,
                    work_title_zh=work.title_zh,
                    translator=work.translator,
                    tradition=work.tradition,
                    era=work.era,
                    rights=work.rights,
                    source=work.source,
                    tags=list(work.tags),
                )
            )
            index += 1
    return chunks


def stats_for(chunks: list[Chunk]) -> ChunkStats:
    if not chunks:
        return ChunkStats()
    sizes = [c.n_chars for c in chunks]
    return ChunkStats(
        n_chunks=len(chunks),
        n_chars=sum(sizes),
        min_chars=min(sizes),
        max_chars=max(sizes),
    )
