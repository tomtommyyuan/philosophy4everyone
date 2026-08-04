"""Provider resolution and the three real backends.

No network and no keys: the SDK clients are replaced with fakes, which is the
only way to assert the things that actually matter about an adapter — that it
sends the right parameters, and that it turns each vendor's idiosyncratic
"successful failure" (a refusal, a safety block) into something a reader can
understand.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from philo.config import ConfigError, Settings, has_credentials, resolve_providers
from philo.providers import get_provider, reset_providers
from philo.providers.base import CompositeProvider


@pytest.fixture(autouse=True)
def env(monkeypatch):
    """A blank provider environment for every test in this module.

    Writes go through monkeypatch, never `os.environ` directly — a direct
    write survives the test and silently reconfigures whatever module pytest
    runs next.
    """
    import os

    for name in list(os.environ):
        if name.startswith(("PHILO_", "OPENAI_", "AZURE_OPENAI_", "ANTHROPIC_", "GEMINI_", "GOOGLE_")):
            monkeypatch.delenv(name, raising=False)
    reset_providers()
    yield monkeypatch
    reset_providers()


@pytest.fixture
def make(env):
    def _make(tmp_path: Path, **values) -> Settings:
        for key, value in values.items():
            env.setenv(key, value)
        return Settings.load(root=tmp_path)

    return _make


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def test_an_api_key_is_now_required(make, tmp_path: Path):
    """No credentials must fail loudly, not silently answer from a mock."""
    with pytest.raises(ConfigError) as excinfo:
        make(tmp_path)
    assert "no API key" in str(excinfo.value)
    # The hint has to be copy-pasteable, not a pointer to the docs.
    assert "export OPENAI_API_KEY" in excinfo.value.hint
    assert "export ANTHROPIC_API_KEY" in excinfo.value.hint
    assert "export GEMINI_API_KEY" in excinfo.value.hint
    assert "PHILO_PROVIDER=mock" in excinfo.value.hint


def test_offline_mode_still_available_when_asked_for(make, tmp_path: Path):
    s = make(tmp_path, PHILO_PROVIDER="mock")
    assert (s.chat_provider, s.embed_provider) == ("mock", "mock")
    assert s.is_offline


@pytest.mark.parametrize(
    "keys,expected",
    [
        ({"OPENAI_API_KEY": "sk-x"}, ("openai", "openai")),
        ({"GEMINI_API_KEY": "g"}, ("gemini", "gemini")),
        (
            {"AZURE_OPENAI_API_KEY": "a", "AZURE_OPENAI_ENDPOINT": "https://x.openai.azure.com"},
            ("azure", "azure"),
        ),
    ],
)
def test_a_single_key_selects_that_provider_for_both_roles(make, tmp_path: Path, keys, expected):
    s = make(tmp_path, **keys)
    assert (s.chat_provider, s.embed_provider) == expected


def test_anthropic_alone_is_not_enough_because_it_cannot_embed(make, tmp_path: Path):
    with pytest.raises(ConfigError) as excinfo:
        make(tmp_path, ANTHROPIC_API_KEY="sk-ant-x")
    assert "no embeddings endpoint" in str(excinfo.value)
    assert "GEMINI_API_KEY" in excinfo.value.hint


def test_anthropic_pairs_with_whatever_can_embed(make, tmp_path: Path):
    s = make(tmp_path, ANTHROPIC_API_KEY="sk-ant-x", GEMINI_API_KEY="g",
             PHILO_CHAT_PROVIDER="anthropic")
    assert s.chat_provider == "anthropic"
    assert s.embed_provider == "gemini"
    assert s.split_providers
    assert "anthropic" in s.describe_provider() and "gemini" in s.describe_provider()


def test_anthropic_is_rejected_as_an_embedding_provider(make, tmp_path: Path):
    with pytest.raises(ConfigError) as excinfo:
        make(tmp_path, ANTHROPIC_API_KEY="k", PHILO_EMBED_PROVIDER="anthropic")
    assert "no embeddings endpoint" in excinfo.value.hint


def test_explicit_settings_beat_credential_detection(make, tmp_path: Path):
    s = make(tmp_path, OPENAI_API_KEY="sk-x", GEMINI_API_KEY="g",
             PHILO_CHAT_PROVIDER="gemini", PHILO_EMBED_PROVIDER="openai")
    assert (s.chat_provider, s.embed_provider) == ("gemini", "openai")


def test_selecting_a_provider_without_its_key_is_an_error(make, tmp_path: Path):
    with pytest.raises(ConfigError) as excinfo:
        make(tmp_path, OPENAI_API_KEY="sk-x", PHILO_CHAT_PROVIDER="anthropic")
    assert "credentials are missing" in str(excinfo.value)


def test_unknown_provider_names_are_rejected(make, tmp_path: Path):
    with pytest.raises(ConfigError):
        make(tmp_path, OPENAI_API_KEY="sk-x", PHILO_PROVIDER="llama")


def test_has_credentials_requires_both_azure_halves(make, tmp_path: Path):
    s = make(tmp_path, PHILO_PROVIDER="mock", AZURE_OPENAI_API_KEY="k")
    assert has_credentials(s, "azure") is False
    s.azure_endpoint = "https://x.openai.azure.com"
    assert has_credentials(s, "azure") is True


def test_composite_reports_the_embedding_provider_as_its_identity(make, tmp_path: Path):
    """The store's compatibility check keys off `.name`.

    Vectors are comparable iff the *embedding* side matches, so swapping the
    chat model must not invalidate an index and swapping the embedder must.
    """
    s = make(tmp_path, ANTHROPIC_API_KEY="k", OPENAI_API_KEY="sk-x",
             PHILO_CHAT_PROVIDER="anthropic")
    provider = get_provider(s)
    assert isinstance(provider, CompositeProvider)
    assert provider.name == "openai"
    assert provider.chat_provider == "anthropic"
    assert provider.split


def test_single_vendor_shares_one_backend_between_roles(make, tmp_path: Path):
    s = make(tmp_path, OPENAI_API_KEY="sk-x")
    provider = get_provider(s)
    assert provider._chat is provider._embed
    assert not provider.split


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------


class _FakeAnthropicMessage:
    def __init__(self, text="ok", stop_reason="end_turn", stop_details=None):
        self.content = [type("B", (), {"type": "text", "text": text})()]
        self.stop_reason = stop_reason
        self.stop_details = stop_details
        self.model = "claude-opus-5"
        self.usage = type("U", (), {"input_tokens": 11, "output_tokens": 7})()


@pytest.fixture
def anthropic_provider(make, tmp_path: Path):
    from philo.providers.anthropic_provider import AnthropicProvider

    s = make(tmp_path, ANTHROPIC_API_KEY="sk-ant-x", OPENAI_API_KEY="sk-x",
             PHILO_CHAT_PROVIDER="anthropic")
    provider = AnthropicProvider(s)
    captured: dict = {}

    class Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeAnthropicMessage()

    provider._client = type("C", (), {"messages": Messages()})()
    provider.captured = captured  # type: ignore[attr-defined]
    return provider


def test_anthropic_never_sends_temperature(anthropic_provider):
    """The Claude 5 family rejects sampling parameters with a 400."""
    anthropic_provider.chat(
        [{"role": "user", "content": "hi"}], temperature=0.9, max_tokens=100
    )
    assert "temperature" not in anthropic_provider.captured
    assert "top_p" not in anthropic_provider.captured
    assert "top_k" not in anthropic_provider.captured


def test_anthropic_raises_max_tokens_above_the_thinking_floor(anthropic_provider):
    """Thinking shares the output budget; a small ceiling truncates answers."""
    anthropic_provider.chat([{"role": "user", "content": "hi"}], max_tokens=200)
    assert anthropic_provider.captured["max_tokens"] >= 8192


def test_anthropic_lifts_the_system_prompt_out_of_messages(anthropic_provider):
    anthropic_provider.chat(
        [
            {"role": "system", "content": "You are grounded."},
            {"role": "user", "content": "why?"},
        ]
    )
    assert anthropic_provider.captured["system"] == "You are grounded."
    assert [m["role"] for m in anthropic_provider.captured["messages"]] == ["user"]


def test_anthropic_sends_effort_not_temperature(anthropic_provider):
    anthropic_provider.chat([{"role": "user", "content": "hi"}])
    assert anthropic_provider.captured["output_config"] == {"effort": "medium"}


def test_anthropic_refusal_becomes_readable_text(make, tmp_path: Path):
    """A refusal is HTTP 200 with empty content — not an exception."""
    from philo.providers.anthropic_provider import AnthropicProvider

    s = make(tmp_path, ANTHROPIC_API_KEY="k", OPENAI_API_KEY="sk-x",
             PHILO_CHAT_PROVIDER="anthropic")
    provider = AnthropicProvider(s)
    refusal = _FakeAnthropicMessage(
        text="", stop_reason="refusal",
        stop_details=type("D", (), {"category": "cyber"})(),
    )
    provider._client = type(
        "C", (), {"messages": type("M", (), {"create": lambda self, **k: refusal})()}
    )()

    result = provider.chat([{"role": "user", "content": "..."}])
    assert result.refused
    assert "declined" in result.text.lower()
    assert "cyber" in result.text


def test_anthropic_missing_key_is_actionable(make, tmp_path: Path):
    from philo.providers.anthropic_provider import AnthropicProvider
    from philo.providers.base import ProviderError

    s = make(tmp_path, PHILO_PROVIDER="mock")
    with pytest.raises(ProviderError) as excinfo:
        AnthropicProvider(s).client
    assert "ANTHROPIC_API_KEY" in excinfo.value.hint


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------


@pytest.fixture
def gemini_provider(make, tmp_path: Path):
    from philo.providers.gemini import GeminiProvider

    s = make(tmp_path, GEMINI_API_KEY="g")
    provider = GeminiProvider(s)
    captured: dict = {}

    class Models:
        def embed_content(self, model, contents, config):
            captured["embed"] = {"model": model, "contents": contents, "config": config}
            values = [0.6, 0.8] + [0.0] * 6
            return type("R", (), {"embeddings": [type("E", (), {"values": values})() for _ in contents]})()

        def generate_content(self, model, contents, config):
            captured["generate"] = {"model": model, "contents": contents, "config": config}
            return type(
                "R", (),
                {
                    "text": "hello",
                    "candidates": [type("C", (), {"finish_reason": "STOP"})()],
                    "prompt_feedback": None,
                    "usage_metadata": None,
                    "model_version": "gemini-2.5-flash",
                },
            )()

    provider._client = type("C", (), {"models": Models()})()
    provider.captured = captured  # type: ignore[attr-defined]
    return provider


def test_gemini_embeds_documents_and_queries_asymmetrically(gemini_provider):
    """Using the document task type for a query measurably hurts recall."""
    gemini_provider.embed(["a passage"])
    assert gemini_provider.captured["embed"]["config"].task_type == "RETRIEVAL_DOCUMENT"

    gemini_provider.embed_query("a question")
    assert gemini_provider.captured["embed"]["config"].task_type == "RETRIEVAL_QUERY"


def test_gemini_normalises_embeddings(gemini_provider):
    """Truncated output dimensions are not unit-length as returned."""
    vec = gemini_provider.embed_query("x")
    assert abs(sum(v * v for v in vec) - 1.0) < 1e-9


def test_gemini_lifts_system_and_renames_the_assistant_role(gemini_provider):
    gemini_provider.chat(
        [
            {"role": "system", "content": "Be grounded."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
        ]
    )
    call = gemini_provider.captured["generate"]
    assert call["config"].system_instruction == "Be grounded."
    assert [c["role"] for c in call["contents"]] == ["user", "model", "user"]


def test_gemini_safety_block_becomes_readable_text(make, tmp_path: Path):
    """A blocked response is a 200 with no candidate text."""
    from philo.providers.gemini import GeminiProvider

    s = make(tmp_path, GEMINI_API_KEY="g")
    provider = GeminiProvider(s)

    class Models:
        def generate_content(self, model, contents, config):
            return type(
                "R", (),
                {
                    "text": "",
                    "candidates": [],
                    "prompt_feedback": type("F", (), {"block_reason": "SAFETY"})(),
                    "usage_metadata": None,
                    "model_version": "gemini-2.5-flash",
                },
            )()

    provider._client = type("C", (), {"models": Models()})()
    result = provider.chat([{"role": "user", "content": "..."}])
    assert result.refused
    assert "blocked" in result.text.lower()


def test_gemini_missing_key_is_actionable(make, tmp_path: Path):
    from philo.providers.base import ProviderError
    from philo.providers.gemini import GeminiProvider

    s = make(tmp_path, PHILO_PROVIDER="mock")
    with pytest.raises(ProviderError) as excinfo:
        GeminiProvider(s).client
    assert "GEMINI_API_KEY" in excinfo.value.hint


# --------------------------------------------------------------------------
# Cross-provider wiring
# --------------------------------------------------------------------------


def test_a_split_setup_routes_each_call_to_the_right_vendor(make, tmp_path: Path):
    from philo.providers.base import CompositeProvider

    calls: list[str] = []

    class Chat:
        name = "anthropic"
        chat_model = "claude-opus-5"

        def chat(self, messages, **kw):
            calls.append("chat")
            from philo.providers.base import ChatResult

            return ChatResult(text="answer", provider="anthropic")

        def chat_healthcheck(self):
            return "chat ok"

    class Embed:
        name = "openai"
        embed_model = "text-embedding-3-small"
        embed_dim = 1536

        def embed(self, texts, **kw):
            calls.append("embed")
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text):
            calls.append("embed_query")
            return [1.0, 0.0]

        def embed_healthcheck(self):
            return "embed ok"

    provider = CompositeProvider(Chat(), Embed())
    provider.chat([{"role": "user", "content": "hi"}])
    provider.embed(["a"])
    provider.embed_query("q")

    assert calls == ["chat", "embed", "embed_query"]
    assert provider.chat_model == "claude-opus-5"
    assert provider.embed_model == "text-embedding-3-small"
    assert provider.describe() == "anthropic + openai"
