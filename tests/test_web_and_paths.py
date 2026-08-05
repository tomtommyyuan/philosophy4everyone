"""The HTTP interface, and the path resolution it depends on.

The path tests exist because of a real bug: on a case-insensitive filesystem
`(candidate / "library").is_dir()` matches `/Library`, so the upward search
for a project root walked to `/` and the CLI tried to write `/.philo`. Every
installed use failed. The regression test below fails on the old code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from philo.config import Settings, USER_HOME, _find_root, _has_exact_dir, _is_project_root
from philo.corpus.ingest import ingest
from philo.providers.mock import MockProvider

fastapi = pytest.importorskip("fastapi", reason="web extras not installed")
from fastapi.testclient import TestClient  # noqa: E402


# --------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------


def test_case_insensitive_match_does_not_count_as_a_directory(tmp_path: Path):
    (tmp_path / "Library").mkdir()
    assert _has_exact_dir(tmp_path, "Library") is True
    # The whole point: on a case-insensitive filesystem the naive check says
    # yes here, and only a listing comparison knows the real name.
    assert _has_exact_dir(tmp_path, "library") is False
    if (tmp_path / "library").is_dir():
        assert _has_exact_dir(tmp_path, "library") is not (tmp_path / "library").is_dir()


def test_a_bare_library_directory_is_not_a_project_root(tmp_path: Path):
    (tmp_path / "library").mkdir()
    # No texts inside — this is somebody's music library, not ours.
    assert _is_project_root(tmp_path) is False
    (tmp_path / "library" / "plato.md").write_text("---\nphilosopher: Plato\nwork: Ion\n---\n\nText.", "utf-8")
    assert _is_project_root(tmp_path) is True


def test_find_root_never_escapes_to_the_filesystem_root(tmp_path: Path, monkeypatch):
    """The regression test for the `/Library` bug."""
    monkeypatch.chdir(tmp_path)
    root = _find_root()
    assert root != Path("/")
    assert root == tmp_path.resolve()


def test_index_falls_back_to_the_user_home_outside_a_project(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PHILO_PROVIDER", "mock")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PHILO_INDEX", raising=False)
    monkeypatch.delenv("PHILO_LIBRARY", raising=False)
    s = Settings.load()
    assert s.in_project is False
    # Never the cwd, and never anywhere unwritable.
    assert USER_HOME in s.index_dir.parents or s.index_dir.is_relative_to(USER_HOME)


def test_inside_a_checkout_paths_stay_local(tmp_path: Path, library: Path, monkeypatch):
    monkeypatch.setenv("PHILO_PROVIDER", "mock")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PHILO_INDEX", raising=False)
    monkeypatch.delenv("PHILO_LIBRARY", raising=False)
    s = Settings.load()
    assert s.in_project is True
    assert s.index_dir == tmp_path.resolve() / ".philo" / "index"


def test_env_overrides_win(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PHILO_PROVIDER", "mock")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PHILO_INDEX", str(tmp_path / "custom"))
    assert Settings.load().index_dir == tmp_path / "custom"


# --------------------------------------------------------------------------
# Web app
# --------------------------------------------------------------------------


@pytest.fixture
def client(settings: Settings, provider: MockProvider, monkeypatch):
    # `philo.web` re-exports the FastAPI instance as `app`, so importing the
    # module by that name would shadow it.
    from philo import config as config_mod
    from philo.web.app import app as fastapi_app, reset_engine

    ingest(settings, provider)
    monkeypatch.setattr(config_mod, "_settings", settings)
    reset_engine()
    with TestClient(fastapi_app) as c:
        yield c
    reset_engine()


def test_page_is_served_and_self_contained(client):
    res = client.get("/")
    assert res.status_code == 200
    body = res.text
    assert "PHILOSOPHY FOR EVERYONE" in body
    # A strict deployment blocks external requests, and the page must work
    # offline; nothing may be loaded from another host.
    for forbidden in ("http://", "cdn.", "<script src=", "<link rel=\"stylesheet\""):
        assert forbidden not in body.replace("http://localhost", "")


def test_health_reports_the_index(client):
    d = client.get("/api/health").json()
    assert d["ok"] is True
    assert d["provider"] == "mock"
    assert d["offline"] is True
    assert d["passages"] > 0
    assert d["works"] == 2


def test_ask_returns_a_grounded_answer_with_sources(client):
    res = client.post("/api/ask", json={"question": "what is in my control?"})
    assert res.status_code == 200
    a = res.json()
    assert a["grounded"] is True
    assert a["sources"]
    assert a["sources"][0]["chunk"]["philosopher"]
    assert a["plain"].strip()


def test_ask_refuses_when_nothing_is_relevant(client):
    res = client.post("/api/ask", json={"question": "lattice gauge renormalisation in QCD"})
    a = res.json()
    # Either grounded from a weak match or an explicit refusal, but never a
    # fabricated citation.
    if not a["grounded"]:
        assert a["sources"] == []
        assert "library" in a["plain"] or "philo ingest" in a["plain"]


def test_ask_honours_a_philosopher_filter(client):
    a = client.post("/api/ask", json={"question": "water", "philosopher": "Laozi"}).json()
    assert a["sources"]
    assert all(s["chunk"]["philosopher"] == "Laozi" for s in a["sources"])


def test_ask_validates_input(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "hi", "k": 99}).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 5000}).status_code == 422


def test_stream_emits_deltas_then_a_final_answer(client):
    with client.stream("POST", "/api/ask/stream", json={"question": "what is in my control?"}) as res:
        assert res.status_code == 200
        body = "".join(res.iter_text())
    assert body.count("event: delta") > 3
    assert "event: done" in body
    final = body.rsplit("event: done\ndata: ", 1)[1].split("\n\n")[0]
    payload = json.loads(final)
    assert payload["grounded"] is True
    assert payload["sources"]


def test_sources_endpoint_describes_the_library(client):
    d = client.get("/api/sources").json()
    assert {w["philosopher"] for w in d["works"]} == {"Epictetus", "Laozi"}
    assert d["philosophers"] and d["traditions"]


def test_search_endpoint_is_retrieval_only(client):
    d = client.get("/api/search", params={"q": "water", "k": 3}).json()
    assert len(d["hits"]) <= 3
    assert d["n_candidates"] > 0


def test_daily_endpoint_never_writes_to_the_profile(client, settings: Settings):
    """A shared deployment must not let one visitor rewrite the rotation."""
    from philo.personalize.profile import Profile

    profile = Profile.default("default")
    profile.save(settings.profiles_dir / "default.json")
    before = json.loads((settings.profiles_dir / "default.json").read_text("utf-8"))

    d = client.get("/api/daily", params={"date": "2026-01-01"}).json()
    assert d["theme"]

    after = json.loads((settings.profiles_dir / "default.json").read_text("utf-8"))
    assert after["history"] == before["history"] == []


def test_favicon_has_no_body(client):
    res = client.get("/favicon.ico")
    assert res.status_code == 204
    # A 204 carrying a body makes the payload longer than its own
    # Content-Length, which uvicorn rejects outright.
    assert res.content == b""


# --------------------------------------------------------------------------
# Token gate
# --------------------------------------------------------------------------


def test_token_gate_blocks_and_allows(client, monkeypatch):
    monkeypatch.setenv("PHILO_WEB_TOKEN", "s3cret")

    assert client.post("/api/ask", json={"question": "hi"}).status_code == 401
    assert client.post(
        "/api/ask", json={"question": "hi"}, headers={"X-Philo-Token": "wrong"}
    ).status_code == 401
    ok = client.post(
        "/api/ask", json={"question": "what is in my control?"},
        headers={"X-Philo-Token": "s3cret"},
    )
    assert ok.status_code == 200
    # Health stays open so a monitor can see the service without the secret.
    assert client.get("/api/health").json()["authenticated"] is True


def test_no_token_configured_means_open(client):
    assert client.post("/api/ask", json={"question": "what is in my control?"}).status_code == 200


# --------------------------------------------------------------------------
# Index failure reporting
# --------------------------------------------------------------------------


def test_index_errors_are_classified(settings: Settings, provider: MockProvider):
    """A UI must tell "missing" from "mismatch" without matching on prose."""
    from philo.store.vector_store import IndexError_, VectorStore

    with pytest.raises(IndexError_) as missing:
        VectorStore(settings.index_dir / "nowhere").load()
    assert missing.value.code == "missing"

    ingest(settings, provider)
    with pytest.raises(IndexError_) as mismatch:
        VectorStore(settings.index_dir).load(expect_model="text-embedding-3-small")
    assert mismatch.value.code == "mismatch"
    # The two need different remedies: a rebuild, not a fetch.
    assert "--rebuild" in mismatch.value.hint
    assert "--rebuild" not in missing.value.hint


def test_health_reports_the_failure_code_and_hint(settings: Settings, provider: MockProvider, monkeypatch):
    """The page renders the API's remedy; it must actually be sent one."""
    from philo import config as config_mod
    from philo.web.app import app as fastapi_app, reset_engine

    ingest(settings, provider)
    # Same shape as the real report: index built offline, provider now real.
    settings.chat_provider = settings.embed_provider = "openai"
    settings.openai_api_key = "sk-test"
    monkeypatch.setattr(config_mod, "_settings", settings)
    reset_engine()
    try:
        with TestClient(fastapi_app) as client:
            payload = client.get("/api/health").json()
    finally:
        reset_engine()

    assert payload["ok"] is False
    error = payload["error"]
    assert error["code"] == "mismatch"
    assert "not comparable" in error["error"]
    assert "philo ingest --rebuild" in error["hint"]


def test_the_page_uses_the_code_rather_than_a_fixed_command():
    """`philo fetch` is the wrong advice after a provider switch."""
    from philo.web.app import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "index mismatch" in page
    assert "philo ingest --rebuild" in page
    assert "d.hint" in page


# --------------------------------------------------------------------------
# Model selection from the page
# --------------------------------------------------------------------------


def test_models_endpoint_offers_chat_only(client):
    d = client.get("/api/models").json()
    assert d["providers"] and d["providers"][0]["provider"] == "mock"
    assert d["current"]["model"]
    # Embeddings are reported for display but never as a choice.
    assert d["embedding"]["model"]
    assert all("embed" not in m for g in d["providers"] for m in g["models"])


def test_ask_honours_the_selected_model(client):
    a = client.post(
        "/api/ask", json={"question": "what is in my control?", "model": "picked-model"}
    ).json()
    assert a["model"] == "picked-model"
    assert a["grounded"] is True


def test_an_unknown_provider_is_refused(client):
    res = client.post(
        "/api/ask", json={"question": "hi", "provider": "nonsense", "model": "x"}
    )
    assert res.status_code == 502
    assert "unknown chat provider" in res.json()["detail"]["error"]


def test_web_models_allowlist_caps_what_visitors_can_spend_on(client, monkeypatch):
    monkeypatch.setenv("PHILO_WEB_MODELS", "mock-sage-1")
    d = client.get("/api/models").json()
    assert d["restricted"] is True
    assert [m for g in d["providers"] for m in g["models"]] == ["mock-sage-1"]

    monkeypatch.setenv("PHILO_WEB_MODELS", "something-else")
    assert client.get("/api/models").json()["providers"] == []


def test_the_page_wires_the_selector(client):
    page = client.get("/").text
    assert 'id="f-model"' in page
    assert "/api/models" in page
    assert "chosenModel()" in page


def test_a_relative_index_path_does_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    """Serverless runtimes do not guarantee cwd is the bundle root.

    `vercel.json` ships `PHILO_INDEX=deploy/index`. Resolved against the
    wrong directory, every request fails with "no index" for a file that is
    sitting right there in the bundle.
    """
    from philo.config import PACKAGE_ROOT, _anchor

    bundled = PACKAGE_ROOT / "deploy" / "index"
    bundled.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)          # a cwd with no deploy/ at all
    assert _anchor("deploy/index", tmp_path / "fallback") == bundled.resolve()

    # An absolute path is always honoured verbatim.
    assert _anchor(str(tmp_path / "custom"), tmp_path) == tmp_path / "custom"
