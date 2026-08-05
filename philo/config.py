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

# Who can do what. Anthropic publishes no embeddings endpoint, so it is a
# chat provider only — see philo/providers/base.py for why that shapes the
# whole layer.
CHAT_PROVIDERS = ("openai", "azure", "anthropic", "gemini", "mock")
EMBED_PROVIDERS = ("openai", "azure", "gemini", "mock")
PROVIDERS = tuple(dict.fromkeys(CHAT_PROVIDERS + EMBED_PROVIDERS))

# Order used when several sets of credentials are present. Azure comes first
# because configuring it (endpoint plus two deployment names) is a deliberate
# act, never something that happens by accident.
CHAT_PREFERENCE = ("azure", "openai", "anthropic", "gemini")
EMBED_PREFERENCE = ("azure", "openai", "gemini")

# User-level home for an installed CLI that is not sitting in a checkout.
USER_HOME = Path(env_str("PHILO_HOME", str(Path.home() / ".philo")))

# The directory containing the `philo` package — the repo root in a checkout,
# and the bundle root in a serverless deployment.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _anchor(value: str, default: Path) -> Path:
    """Resolve a configured path, tolerating a relative one.

    A relative `PHILO_INDEX` is normal in a serverless deployment, where the
    index ships inside the bundle — but the runtime's working directory is
    not guaranteed to be the bundle root, and resolving against the wrong one
    makes every request fail with "no index" for a file that is right there.
    Try the working directory first, then the package's own location.
    """
    if not value:
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.exists():
        return from_cwd
    from_package = (PACKAGE_ROOT / path).resolve()
    return from_package if from_package.exists() else from_cwd


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

    # ---- providers -------------------------------------------------------
    # Chat and embeddings are resolved independently; they are frequently the
    # same vendor and occasionally must not be (Anthropic cannot embed).
    chat_provider: str = "mock"
    embed_provider: str = "mock"

    chat_model: str = "gpt-5.5"
    embed_model: str = "text-embedding-3-small"

    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_org: str = ""

    azure_api_key: str = ""
    azure_endpoint: str = ""
    azure_api_version: str = "2024-10-21"
    azure_chat_deployment: str = ""
    azure_embed_deployment: str = ""

    anthropic_api_key: str = ""
    anthropic_base_url: str = ""
    anthropic_model: str = "claude-opus-5"
    # Claude controls reasoning depth with `effort`, not `temperature`
    # (sampling parameters are rejected outright by the Claude 5 family).
    anthropic_effort: str = "medium"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_embed_model: str = "gemini-embedding-001"
    # 3072 by default; 768/1536 trade a little accuracy for a much smaller index.
    gemini_embed_dim: int = 0

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
    # Set instead of raising when loaded non-strictly, so `philo doctor` can
    # still run and explain the problem. Refusing to start *and* refusing to
    # diagnose would be the worst of both.
    config_error: Exception | None = None

    def __post_init__(self) -> None:
        # Inside a checkout everything lives beside the code. Installed via
        # pip there is no checkout, so the CLI keeps its texts, index and
        # profiles under ~/.philo instead of scattering them across whatever
        # directory the user happened to be standing in.
        self.in_project = _is_project_root(self.root) and _is_writable(self.root)
        base = self.root if self.in_project else USER_HOME
        default_index = base / ".philo" / "index" if self.in_project else base / "index"
        self.library_dir = _anchor(env_str("PHILO_LIBRARY"), base / "library")
        self.index_dir = _anchor(env_str("PHILO_INDEX"), default_index)
        self.profiles_dir = _anchor(env_str("PHILO_PROFILES"), base / "profiles")

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, root: Path | None = None, *, strict: bool = True, **overrides) -> "Settings":
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

        s.anthropic_api_key = env_str("ANTHROPIC_API_KEY")
        s.anthropic_base_url = env_str("ANTHROPIC_BASE_URL")
        s.anthropic_model = env_str("PHILO_ANTHROPIC_MODEL", s.anthropic_model)
        s.anthropic_effort = env_str("PHILO_ANTHROPIC_EFFORT", s.anthropic_effort)

        # GOOGLE_API_KEY is what the SDK itself reads, so accept both names.
        s.gemini_api_key = env_str("GEMINI_API_KEY") or env_str("GOOGLE_API_KEY")
        s.gemini_model = env_str("PHILO_GEMINI_MODEL", s.gemini_model)
        s.gemini_embed_model = env_str("PHILO_GEMINI_EMBED_MODEL", s.gemini_embed_model)
        s.gemini_embed_dim = env_int("PHILO_GEMINI_EMBED_DIM", s.gemini_embed_dim)

        try:
            s.chat_provider, s.embed_provider = resolve_providers(s)
        except ConfigError as exc:
            if strict:
                raise
            s.chat_provider = s.embed_provider = ""
            s.config_error = exc
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
    def provider(self) -> str:
        """The provider a user would name — the one writing the answers."""
        return self.chat_provider

    @property
    def is_offline(self) -> bool:
        return self.chat_provider == "mock" and self.embed_provider == "mock"

    @property
    def split_providers(self) -> bool:
        return self.chat_provider != self.embed_provider

    @property
    def chat_model_name(self) -> str:
        """What to actually put on the wire, per provider."""
        if self.chat_provider == "azure":
            return self.azure_chat_deployment or self.chat_model
        if self.chat_provider == "anthropic":
            return self.anthropic_model
        if self.chat_provider == "gemini":
            return self.gemini_model
        if self.chat_provider == "mock":
            return "mock-sage-1"
        return self.chat_model

    @property
    def embed_model_name(self) -> str:
        if self.embed_provider == "azure":
            return self.azure_embed_deployment or self.embed_model
        if self.embed_provider == "gemini":
            return self.gemini_embed_model
        if self.embed_provider == "mock":
            return "mock-embed-384"
        return self.embed_model

    def describe_provider(self) -> str:
        chat = self._describe_one(self.chat_provider)
        if not self.split_providers:
            return chat
        # Say both, because which one embedded the index determines whether
        # that index can still be read.
        return f"{chat} + {self._describe_one(self.embed_provider)} (embeddings)"

    def _describe_one(self, provider: str) -> str:
        if provider == "mock":
            return "mock · offline"
        if provider == "azure":
            host = self.azure_endpoint.replace("https://", "").split(".")[0] or "azure"
            return f"azure · {host}"
        if provider == "openai" and self.openai_base_url:
            host = self.openai_base_url.replace("https://", "").split("/")[0]
            return f"openai · {host}"
        return provider

    @property
    def ready(self) -> bool:
        return bool(self.chat_provider and self.embed_provider) and self.config_error is None

    def problems(self) -> list[str]:
        """Configuration errors that would make a real API call fail.

        Surfaced by `philo doctor` and checked before any live request, so the
        failure message is about config rather than an opaque HTTP 401.
        """
        if self.config_error is not None:
            return [str(self.config_error)]
        issues: list[str] = []
        for role, provider in (("chat", self.chat_provider), ("embedding", self.embed_provider)):
            issues.extend(self._problems_for(role, provider))
        return list(dict.fromkeys(issues))

    def problems_for(self, provider: str) -> list[str]:
        """Config errors for one vendor only.

        A backend must not refuse to start because a *different* provider in
        a split configuration is misconfigured.
        """
        issues: list[str] = []
        for role in ("chat", "embedding"):
            if getattr(self, f"{role.replace('embedding', 'embed')}_provider") == provider:
                issues.extend(self._problems_for(role, provider))
        return list(dict.fromkeys(issues))

    def _problems_for(self, role: str, provider: str) -> list[str]:
        issues: list[str] = []
        if provider == "openai" and not self.openai_api_key:
            issues.append(f"OPENAI_API_KEY is not set (needed for {role})")
        elif provider == "anthropic" and not self.anthropic_api_key:
            issues.append("ANTHROPIC_API_KEY is not set")
        elif provider == "gemini" and not self.gemini_api_key:
            issues.append(f"GEMINI_API_KEY is not set (needed for {role})")
        elif provider == "azure":
            if not self.azure_api_key:
                issues.append("AZURE_OPENAI_API_KEY is not set")
            if not self.azure_endpoint:
                issues.append("AZURE_OPENAI_ENDPOINT is not set")
            if role == "chat" and not self.azure_chat_deployment:
                issues.append(
                    "AZURE_OPENAI_CHAT_DEPLOYMENT is not set "
                    "(Azure addresses deployments, not model names)"
                )
            if role == "embedding" and not self.azure_embed_deployment:
                issues.append("AZURE_OPENAI_EMBEDDING_DEPLOYMENT is not set")

        package = {"openai": "openai", "azure": "openai",
                   "anthropic": "anthropic", "gemini": "google-genai"}.get(provider)
        extra = {"openai": "openai", "azure": "openai",
                 "anthropic": "anthropic", "gemini": "gemini"}.get(provider)
        if package:
            try:
                __import__(package.replace("-", "_") if package != "google-genai" else "google.genai")
            except ImportError:
                issues.append(
                    f"the `{package}` package is not installed — run: pip install 'philo[{extra}]'"
                )
        return issues


class ConfigError(RuntimeError):
    """The provider configuration cannot produce a working system."""

    def __init__(self, message: str, *, hint: str = ""):
        super().__init__(message)
        self.hint = hint


SETUP_HELP = """Set one of these and re-run:

  export OPENAI_API_KEY=sk-...                 # chat + embeddings
  export ANTHROPIC_API_KEY=sk-ant-...          # chat only, needs an embedding key too
  export GEMINI_API_KEY=...                    # chat + embeddings
  export AZURE_OPENAI_API_KEY=...  AZURE_OPENAI_ENDPOINT=https://<res>.openai.azure.com \
         AZURE_OPENAI_CHAT_DEPLOYMENT=...  AZURE_OPENAI_EMBEDDING_DEPLOYMENT=...

Or run entirely offline with no key at all:

  export PHILO_PROVIDER=mock"""


def has_credentials(s: Settings, provider: str) -> bool:
    """Whether `provider` has everything it needs to make a call."""
    if provider == "mock":
        return True
    if provider == "azure":
        return bool(s.azure_api_key and s.azure_endpoint)
    if provider == "openai":
        return bool(s.openai_api_key)
    if provider == "anthropic":
        return bool(s.anthropic_api_key)
    if provider == "gemini":
        return bool(s.gemini_api_key)
    return False


def _requested(name: str, allowed: tuple[str, ...]) -> str:
    value = env_str(name).lower()
    if not value:
        return ""
    if value not in allowed:
        raise ConfigError(
            f"{name}={value!r} is not one of {allowed}",
            hint=(
                "Anthropic has no embeddings endpoint, so it cannot be an embedding provider."
                if value == "anthropic" and "anthropic" not in allowed
                else f"Valid values: {', '.join(allowed)}."
            ),
        )
    return value


def resolve_providers(s: Settings) -> tuple[str, str]:
    """Decide who writes answers and who builds vectors.

    Explicit settings win; otherwise we pick the first provider in a fixed
    preference order that actually has credentials. An API key is required —
    with nothing configured this raises rather than silently falling back to
    the offline mock, because a system that quietly stops calling real models
    is worse than one that refuses to start.
    """
    both = _requested("PHILO_PROVIDER", PROVIDERS)
    chat = _requested("PHILO_CHAT_PROVIDER", CHAT_PROVIDERS)
    embed = _requested("PHILO_EMBED_PROVIDER", EMBED_PROVIDERS)

    if both and not chat:
        chat = both
    # `PHILO_PROVIDER=anthropic` cannot set the embedding side; it falls
    # through to auto-detection below.
    if both and not embed and both in EMBED_PROVIDERS:
        embed = both

    if not chat:
        chat = next((p for p in CHAT_PREFERENCE if has_credentials(s, p)), "")
    if not embed:
        embed = next((p for p in EMBED_PREFERENCE if has_credentials(s, p)), "")

    if not chat and not embed:
        raise ConfigError("no API key found for any provider", hint=SETUP_HELP)
    if not chat:
        raise ConfigError(
            "no chat provider is configured", hint=SETUP_HELP
        )
    if not embed:
        raise ConfigError(
            f"'{chat}' is configured for chat, but nothing can produce embeddings"
            + (" — Anthropic has no embeddings endpoint" if chat == "anthropic" else ""),
            hint=(
                "Add an embedding provider alongside it:\n\n"
                "  export OPENAI_API_KEY=sk-...      # text-embedding-3-small\n"
                "  export GEMINI_API_KEY=...         # gemini-embedding-001\n\n"
                "Or select one explicitly with PHILO_EMBED_PROVIDER."
            ),
        )

    for role, provider in (("chat", chat), ("embedding", embed)):
        if not has_credentials(s, provider):
            raise ConfigError(
                f"'{provider}' was selected for {role} but its credentials are missing",
                hint=SETUP_HELP,
            )
    return chat, embed


_settings: Settings | None = None


def get_settings(reload: bool = False, *, strict: bool = True, **overrides) -> Settings:
    global _settings
    if _settings is None or reload or overrides:
        _settings = Settings.load(strict=strict, **overrides)
    return _settings


def env_report() -> list[tuple[str, str, bool]]:
    """(name, displayed value, is_set) for `philo doctor`.  Secrets masked."""
    names = [
        ("PHILO_PROVIDER", False),
        ("PHILO_CHAT_PROVIDER", False),
        ("PHILO_EMBED_PROVIDER", False),
        ("OPENAI_API_KEY", True),
        ("OPENAI_BASE_URL", False),
        ("PHILO_CHAT_MODEL", False),
        ("PHILO_EMBED_MODEL", False),
        ("ANTHROPIC_API_KEY", True),
        ("PHILO_ANTHROPIC_MODEL", False),
        ("PHILO_ANTHROPIC_EFFORT", False),
        ("GEMINI_API_KEY", True),
        ("PHILO_GEMINI_MODEL", False),
        ("PHILO_GEMINI_EMBED_MODEL", False),
        ("AZURE_OPENAI_API_KEY", True),
        ("AZURE_OPENAI_ENDPOINT", False),
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
