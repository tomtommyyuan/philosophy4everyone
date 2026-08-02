"""Small shared helpers: env loading, tokenisation, vector math.

Deliberately dependency-light.  numpy is used when present but every routine
here has a pure-Python fallback, so `pip install rich` is enough to run the
whole offline pipeline.
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

try:  # optional, ~50x faster similarity search on large indexes
    import numpy as _np
except Exception:  # pragma: no cover - exercised only on numpy-less installs
    _np = None

HAS_NUMPY = _np is not None


# --------------------------------------------------------------------------
# .env
# --------------------------------------------------------------------------


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> dict[str, str]:
    """Minimal `.env` reader so we don't need python-dotenv.

    Supports `KEY=value`, `export KEY=value`, `#` comments, and single/double
    quoted values.  Missing file is not an error.
    """
    p = Path(path)
    loaded: dict[str, str] = {}
    if not p.is_file():
        return loaded
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key:
            continue
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def env_str(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------

# Latin words, plus every CJK ideograph as its own token.  Philosophy corpora
# are routinely bilingual (Laozi, Zhuangzi, the Analects) and a naive
# whitespace split throws away all Chinese signal.
_TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿]")

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "of", "to", "in", "on", "for", "with", "as", "by", "at",
    "from", "is", "are", "was", "were", "be", "been", "being", "it", "its",
    "he", "she", "they", "them", "his", "her", "their", "we", "us", "our",
    "you", "your", "i", "me", "my", "not", "no", "do", "does", "did", "so",
    "what", "which", "who", "whom", "when", "where", "how", "why", "all",
    "any", "some", "such", "can", "will", "would", "should", "could", "may",
    "might", "must", "shall", "there", "here", "one", "also", "into", "upon",
    "about", "have", "has", "had", "thou", "thy", "thee", "hath", "doth",
    "unto", "shalt", "himself", "itself", "themselves", "more", "most",
    "other", "others", "same", "own", "just", "only", "very", "much",
}


def normalize(text: str) -> str:
    """NFKC + collapse whitespace.  Keeps CJK punctuation intact."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in _STOPWORDS]
    return toks


# Sentence boundaries for English *and* Chinese.  Used by the chunker when a
# paragraph is too long to keep whole — we split between sentences, never
# mid-clause, so an argument never gets guillotined.
_SENT_RE = re.compile(
    r"""(?<=[.!?])["')\]]*\s+      # english terminator (+ closing quote), then space
      | (?<=[。！？；])["')\]』」]*  # chinese terminator (+ closing quote)
    """,
    re.VERBOSE,
)


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_RE.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def char_weight(text: str) -> int:
    """Length measure that treats one CJK character as ~2.5 latin characters.

    Chinese is far denser per character, so raw `len()` would let Chinese
    chunks carry several times more content than English ones at the same
    nominal size.
    """
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    return len(text) + int(cjk * 1.5)


def truncate(text: str, limit: int, ellipsis: str = "…") -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(ellipsis))].rstrip() + ellipsis


def snippet(text: str, limit: int = 220) -> str:
    """One-line preview used in tables."""
    return truncate(re.sub(r"\s+", " ", text), limit)


def slugify(text: str, max_len: int = 60) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:max_len].strip("-") or "untitled"


def detect_language(text: str) -> str:
    """`zh` if the text is meaningfully Chinese, else `en`.

    Used to make the model answer in the language the user asked in.
    """
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    letters = sum(1 for c in text if c.isalpha())
    if letters == 0:
        return "en"
    return "zh" if cjk / max(1, letters) > 0.15 else "en"


# --------------------------------------------------------------------------
# Vector math
# --------------------------------------------------------------------------


def l2_normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return list(vec)
    return [v / norm for v in vec]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def cosine_batch(query: Sequence[float], matrix: Sequence[Sequence[float]]) -> list[float]:
    """Cosine of `query` against every row of `matrix`.

    Rows are assumed L2-normalised (the store guarantees this on write), so
    this reduces to a dot product against a normalised query.
    """
    q = l2_normalize(query)
    if HAS_NUMPY and len(matrix):
        m = matrix if isinstance(matrix, _np.ndarray) else _np.asarray(matrix, dtype="float32")
        return (m @ _np.asarray(q, dtype="float32")).tolist()
    return [dot(q, row) for row in matrix]


def minmax(values: Sequence[float]) -> list[float]:
    """Scale to 0..1.  All-equal input maps to all-1.0 (nothing to prefer)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if math.isclose(hi, lo):
        return [1.0] * len(values)
    span = hi - lo
    return [(v - lo) / span for v in values]


def stable_seed(*parts: object) -> int:
    """Deterministic integer seed from any inputs (no hash randomisation)."""
    import hashlib

    joined = "\x1f".join(str(p) for p in parts)
    return int(hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12], 16)


def chunked(items: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def human_count(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k".replace(".0k", "k")
    return f"{n / 1_000_000:.1f}M".replace(".0M", "M")


def human_ms(ms: float) -> str:
    if ms < 1000:
        return f"{int(ms)}ms"
    return f"{ms / 1000:.1f}s"
