"""Visual language: ink and candlelight.

Deep violet for structure, warm amber for anything the reader should actually
look at, cool slate for machinery.  Two rules hold the whole interface
together:

- Amber means *provenance*.  Citation markers, source numbers and quotes are
  amber and nothing else is, so the eye learns in one screen that gold marks
  the parts that are traceable to a real book.
- Violet means *us*.  Frames, headers, the model's own voice.

Every colour is chosen to stay legible on both dark and light terminals —
mid-tone hues rather than the near-black or near-white extremes — and the
whole palette collapses cleanly to bold/dim when NO_COLOR is set.
"""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.theme import Theme

# -- palette ---------------------------------------------------------------
VIOLET = "#a78bfa"
VIOLET_DEEP = "#7c3aed"
VIOLET_SOFT = "#c4b5fd"
AMBER = "#f0b429"
AMBER_SOFT = "#fcd34d"
SLATE = "#8b93a7"
SLATE_DIM = "#6b7280"
TEAL = "#5eead4"
GREEN = "#34d399"
RED = "#f87171"
INK = "#e2e8f0"

# Gradient used for the wordmark: violet drifting into amber, like a page
# lit from one side.
GRADIENT = [
    "#8b5cf6", "#9370f7", "#a78bfa", "#b9a0f0",
    "#d2b48c", "#e8c06a", "#f0b429", "#f5a623",
]

PHILO_THEME = Theme(
    {
        # structure
        "brand": f"bold {VIOLET}",
        "frame": VIOLET_DEEP,
        "frame.soft": SLATE_DIM,
        "heading": f"bold {VIOLET}",
        "subheading": f"bold {VIOLET_SOFT}",
        "rule.line": VIOLET_DEEP,
        # provenance — amber, and only amber
        "cite": f"bold {AMBER}",
        "cite.dim": AMBER,
        "quote": f"italic {AMBER_SOFT}",
        "source.work": f"bold {INK}",
        "source.author": f"bold {AMBER}",
        "source.meta": SLATE,
        # content
        "plain": INK,
        "academic": INK,
        "question": f"bold {INK}",
        "muted": SLATE,
        "dim": SLATE_DIM,
        "kbd": f"bold {TEAL}",
        # state
        "ok": f"bold {GREEN}",
        "warn": f"bold {AMBER}",
        "err": f"bold {RED}",
        "hint": TEAL,
        "offline": f"bold {SLATE}",
        # score bars
        "score.high": GREEN,
        "score.mid": AMBER,
        "score.low": SLATE_DIM,
        # Rich's Markdown renderer looks these up by name. Overriding them is
        # what keeps model-authored prose inside our palette instead of
        # Rich's defaults — and it is how `[n]` markers stay amber even after
        # Markdown has taken over the text (see `_prep_markdown`).
        "markdown.paragraph": INK,
        "markdown.text": INK,
        "markdown.strong": f"bold {AMBER}",
        "markdown.em": f"italic {VIOLET_SOFT}",
        "markdown.item": INK,
        "markdown.item.bullet": f"bold {VIOLET}",
        "markdown.item.number": f"bold {VIOLET}",
        "markdown.block_quote": f"italic {AMBER_SOFT}",
        "markdown.code": TEAL,
        "markdown.h1": f"bold {VIOLET}",
        "markdown.h2": f"bold {VIOLET}",
        "markdown.h3": f"bold {VIOLET_SOFT}",
        "markdown.h4": f"bold {VIOLET_SOFT}",
        "markdown.hr": VIOLET_DEEP,
        "markdown.link": TEAL,
    }
)

# Frames should be present without shouting. Rounded corners everywhere for
# content; a heavier head only where a table needs a column boundary.
PANEL_BOX = box.ROUNDED
TABLE_BOX = box.SIMPLE_HEAVY
QUOTE_BOX = box.HEAVY


def make_console(*, color: bool = True, width: int | None = None, stderr: bool = False) -> Console:
    return Console(
        theme=PHILO_THEME,
        no_color=not color,
        width=width,
        stderr=stderr,
        highlight=False,       # we do our own emphasis; Rich's guesses fight the palette
        soft_wrap=False,
        emoji=False,
    )


def gradient_text(text: str, *, colors: list[str] | None = None, bold: bool = True):
    """Spread a colour ramp across the characters of a string."""
    from rich.text import Text

    ramp = colors or GRADIENT
    out = Text()
    printable = [i for i, ch in enumerate(text) if ch.strip()]
    total = max(1, len(printable) - 1)
    seen = 0
    for ch in text:
        if not ch.strip():
            out.append(ch)
            continue
        idx = int(seen / total * (len(ramp) - 1))
        out.append(ch, style=f"bold {ramp[idx]}" if bold else ramp[idx])
        seen += 1
    return out


def score_style(score: float) -> str:
    if score >= 0.66:
        return "score.high"
    if score >= 0.33:
        return "score.mid"
    return "score.low"


def score_bar(score: float, width: int = 10):
    """A small meter.  Eighth-blocks so low scores still render something."""
    from rich.text import Text

    score = max(0.0, min(1.0, score))
    filled = score * width
    whole = int(filled)
    remainder = filled - whole
    partials = " ▏▎▍▌▋▊▉"
    bar = "█" * whole
    if whole < width and remainder > 0.06:
        bar += partials[min(len(partials) - 1, int(remainder * 8))]
    bar = bar.ljust(width, "·")
    return Text(bar, style=score_style(score))
