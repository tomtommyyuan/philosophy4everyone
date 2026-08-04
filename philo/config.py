"""Configuration: paths, provider selection, tunables.

Provider resolution is the important bit.  `PHILO_PROVIDER` wins if set;
otherwise we auto-detect from whichever credentials exist, and fall back to
the fully offline `mock` provider.  That fallback is the architectural
promise of this project: the pipeline always runs, with or without a network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .util import env_bool, env_float, env_int, env_str, load_dotenv

PROVIDERS = ("mock", "openai", "azure")

# User-level home for an installed CLI that is not sitting in a checkout.
USER_HOME = Path(env_str("PHILO_HOME", str(Path.home() / ".philo")))


def _has_exact_dir(parent: Path, name: str) -> bool:
    """`(parent / name).is_dir()` that is not fooled by a case-insensitive FS.

    macOS and Windows match `library` against `/Library`, so a naive upward
    search for a project root walks all the way to `/`, "finds" the system
    Library folder and decides the project root is the filesystem root — then
    tries to write `/.philo`. Comparing against the real directory listing is
    the only reliable check.
    """
    try:
        if not (parent / name).is_dir():
            return False
        return any(entry.name == name for entry in parent.iterdir())
    except OSError:
        return False


def _is_project_root(path: Path) -> bool:
    if _has_exact_dir(path, ".philo"):
        return True
    # A `library/` alone is not enough — it has to hold actual source texts,
    # or any directory that merely contains one would capture the search.
    if _has_exact_dir(path, "library"):
        library = path / "library"
        try:
            return any(
                entry.suffix.lower() in {".md", ".markdown", ".txt", ".text"}
                for entry in library.iterdir()
                if entry.is_file()
            )
        except OSError:
            return False
    return False


def _find_root(start: Path | None = None) -> Path:
    """Nearest ancestor that is genuinely a philo project, else the cwd.

    The search stops at the home directory and at the filesystem root: a
    project root above either is never what the user meant, and returning one
    puts the index somewhere unwritable.
    """
    cur = (start or Path.cwd()).resolve()
    home = Path.home().resolve()
    for candidate in [cur, *cur.parents]:
        if candidate == candidate.parent:      # filesystem root
            break
        if _is_project_root(candidate):
            return candidate
        if candidate == home:
            break
    return cur


def _is_writable(path: Path) -> bool:
    probe = path if path.is_dir() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return os.access(probe, os.W_OK)


@dataclass
class Settings:
    # ---- paths -----------------------------------------------------------
    root: Path = field(default_factory=_find_root)
    library_dir: Path = field(init=False)
    index_dir: Path = field(init=False)
    profiles_dir: Path = field(init=False)
    # True when running inside a checkout; False when installed, in which
    # case everything lives under ~/.philo.
    in_project: bool = field(init=False, default=False)

    # ---- provider --------------------------------------------------------
    provider: str = "mock"
    chat_model: str = "gpt-4o"
    embed_model: str = "text-embedding-3-small"

    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_org: str = ""

    azure_api_key: str = ""
    azure_endpoint: str = ""
    azure_api_version: str = "2024-10-21"
    azure_chat_deployment: str = ""
    azure_embed_deployment: str = ""

    # ---- generation ------------------------------------------------------
    temperature: float = 0.3
    max_tokens: int = 1400
    request_timeout: float = 60.0
    max_retries: int = 3
    stream: bool = True

    # ---- chunking --------------------------------------------------------
    chunk_min_chars: int = 320     # merge anything smaller into a neighbour
    chunk_target_chars: int = 900  # the size we aim for
    chunk_max_chars: int = 1500    # hard ceiling; split on sentence boundaries
    chunk_overlap_sentences: int = 1

    # ---- retrieval -------------------------------------------------------
    top_k: int = 6
    candidate_k: int = 24          # fetched before re-ranking / diversifying
    hybrid_alpha: float = 0.72     # weight on dense vs lexical score
    mmr_lambda: float = 0.6        # 1.0 = pure relevance, 0.0 = pure diversity
    min_score: float = 0.12        # below this we declare "not in the library"
    max_per_work: int = 3          # stops one book monopolising the answer

    # ---- misc ------------------------------------------------------------
    embed_batch_size: int = 64
    color: bool = True

    def __post_init__(self) -> None:
        # Inside a checkout everything lives beside the code. Installed via
        # pip there is no checkout, so the CLI keeps its texts, index and
        # profiles under ~/.philo instead of scattering them across whatever
        # directory the user happened to be standing in.
        self.in_project = _is_project_root(self.root) and _is_writable(self.root)
        base = self.root if self.in_project else USER_HOME
        default_index = base / ".philo" / "index" if self.in_project else base / "index"
        self.library_dir = Path(env_str("PHILO_LIBRARY", str(base / "library")))
        self.index_dir = Path(env_str("PHILO_INDEX", str(default_index)))
        self.profiles_dir = Path(env_str("PHILO_PROFILES", str(base / "profiles")))

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, root: Path | None = None, **overrides) -> "Settings":
        root = _find_root(root)
        load_dotenv(root / ".env")

        s = cls(root=root)

        s.openai_api_key = env_str("OPENAI_API_KEY")
        s.openai_base_url = env_str("OPENAI_BASE_URL") or env_str("OPENAI_API_BASE")
        s.openai_org = env_str("OPENAI_ORG_ID") or env_str("OPENAI_ORGANIZATION")

        s.azure_api_key = env_str("AZURE_OPENAI_API_KEY")
        s.azure_endpoint = env_str("AZURE_OPENAI_ENDPOINT").rstrip("/")
        s.azure_api_version = env_str("AZURE_OPENAI_API_VERSION", s.azure_api_version)
        s.azure_chat_deployment = env_str("AZURE_OPENAI_CHAT_DEPLOYMENT")
        s.azure_embed_deployment = env_str("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or env_str(
            "AZURE_OPENAI_EMBED_DEPLOYMENT"
        )

        s.provider = _resolve_provider(s)
        s.chat_model = env_str("PHILO_CHAT_MODEL", s.chat_model)
        s.embed_model = env_str("PHILO_EMBED_MODEL", s.embed_model)

        s.temperature = env_float("PHILO_TEMPERATURE", s.temperature)
        s.max_tokens = env_int("PHILO_MAX_TOKENS", s.max_tokens)
        s.request_timeout = env_float("PHILO_TIMEOUT", s.request_timeout)
        s.max_retries = env_int("PHILO_MAX_RETRIES", s.max_retries)
        s.stream = env_bool("PHILO_STREAM", s.stream)

        s.chunk_min_chars = env_int("PHILO_CHUNK_MIN", s.chunk_min_chars)
        s.chunk_target_chars = env_int("PHILO_CHUNK_TARGET", s.chunk_target_chars)
        s.chunk_max_chars = env_int("PHILO_CHUNK_MAX", s.chunk_max_chars)

        s.top_k = env_int("PHILO_TOP_K", s.top_k)
        s.candidate_k = env_int("PHILO_CANDIDATE_K", s.candidate_k)
        s.hybrid_alpha = env_float("PHILO_HYBRID_ALPHA", s.hybrid_alpha)
        s.mmr_lambda = env_float("PHILO_MMR_LAMBDA", s.mmr_lambda)
        s.min_score = env_float("PHILO_MIN_SCORE", s.min_score)

        s.color = not env_bool("NO_COLOR", False)

        for key, value in overrides.items():
            if value is not None and hasattr(s, key):
                setattr(s, key, value)

        # `__post_init__` ran before overrides; re-derive path fields if the
        # caller moved the root.
        if "root" in overrides and overrides["root"]:
            s.root = Path(overrides["root"])
            s.__post_init__()
        return s

    # ------------------------------------------------------------------
    @property
    def is_offline(self) -> bool:
        return self.provider == "mock"

    @property
    def chat_model_name(self) -> str:
        """What to actually put on the wire — Azure addresses deployments."""
        if self.provider == "azure":
            return self.azure_chat_deployment or self.chat_model
        return self.chat_model

    @property
    def embed_model_name(self) -> str:
        if self.provider == "azure":
            return self.azure_embed_deployment or self.embed_model
        return self.embed_model

    def describe_provider(self) -> str:
        if self.provider == "mock":
            return "mock · offline"
        if self.provider == "azure":
            host = self.azure_endpoint.replace("https://", "").split(".")[0] or "azure"
            return f"azure · {host}"
        if self.openai_base_url:
            host = self.openai_base_url.replace("https://", "").split("/")[0]
            return f"openai · {host}"
        return "openai"

    def problems(self) -> list[str]:
        """Configuration errors that would make a real API call fail.

        Surfaced by `philo doctor` and checked before any live request, so the
        failure message is about config rather than an opaque HTTP 401.
        """
        issues: list[str] = []
        if self.provider == "openai":
            if not self.openai_api_key:
                issues.append("OPENAI_API_KEY is not set")
        elif self.provider == "azure":
            if not self.azure_api_key:
                issues.append("AZURE_OPENAI_API_KEY is not set")
            if not self.azure_endpoint:
                issues.append("AZURE_OPENAI_ENDPOINT is not set")
            if not self.azure_chat_deployment:
                issues.append("AZURE_OPENAI_CHAT_DEPLOYMENT is not set (Azure addresses deployments, not model names)")
            if not self.azure_embed_deployment:
                issues.append("AZURE_OPENAI_EMBEDDING_DEPLOYMENT is not set")
        if self.provider != "mock":
            try:
                import openai  # noqa: F401
            except ImportError:
                issues.append("the `openai` package is not installed — run: pip install 'philo[openai]'")
        return issues


def _resolve_provider(s: Settings) -> str:
    """Explicit env wins; otherwise pick whatever is actually usable."""
    requested = env_str("PHILO_PROVIDER").lower()
    if requested in PROVIDERS:
        return requested
    if requested:
        raise ValueError(f"PHILO_PROVIDER={requested!r} is not one of {PROVIDERS}")
    if s.azure_api_key and s.azure_endpoint:
        return "azure"
    if s.openai_api_key:
        return "openai"
    return "mock"


_settings: Settings | None = None


def get_settings(reload: bool = False, **overrides) -> Settings:
    global _settings
    if _settings is None or reload or overrides:
        _settings = Settings.load(**overrides)
    return _settings


def env_report() -> list[tuple[str, str, bool]]:
    """(name, displayed value, is_set) for `philo doctor`.  Secrets masked."""
    names = [
        ("PHILO_PROVIDER", False),
        ("PHILO_CHAT_MODEL", False),
        ("PHILO_EMBED_MODEL", False),
        ("OPENAI_API_KEY", True),
        ("OPENAI_BASE_URL", False),
        ("AZURE_OPENAI_API_KEY", True),
        ("AZURE_OPENAI_ENDPOINT", False),
        ("AZURE_OPENAI_API_VERSION", False),
        ("AZURE_OPENAI_CHAT_DEPLOYMENT", False),
        ("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", False),
    ]
    out = []
    for name, secret in names:
        raw = os.environ.get(name, "")
        if not raw:
            out.append((name, "—", False))
        elif secret:
            out.append((name, f"{raw[:3]}…{raw[-4:]} ({len(raw)} chars)", True))
        else:
            out.append((name, raw, True))
    return out
