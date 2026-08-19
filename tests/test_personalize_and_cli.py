"""Profiles, the daily piece, providers, and the command line."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from philo.config import Settings
from philo.corpus.ingest import ingest
from philo.generation.answerer import Engine
from philo.personalize.daily import candidate_themes, generate_daily, pick_theme
from philo.personalize.profile import Profile
from philo.providers.mock import MockProvider
from philo.util import cosine, detect_language, l2_normalize, load_dotenv


@pytest.fixture
def engine(settings: Settings, provider: MockProvider) -> Engine:
    ingest(settings, provider)
    return Engine(settings, provider)


@pytest.fixture
def profile(settings: Settings) -> Profile:
    p = Profile.default("tester")
    p.interests = ["what is in my control", "how to face what I cannot change"]
    p.philosophers = ["Epictetus"]
    p.path = settings.profiles_dir / "tester.json"
    return p


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------


def test_profile_round_trips(settings: Settings, profile: Profile):
    path = profile.save()
    reloaded = Profile.load(path)
    assert reloaded.interests == profile.interests
    assert reloaded.philosophers == ["Epictetus"]


def test_reader_note_shapes_style_but_never_content(profile: Profile):
    note = profile.reader_note()
    assert "what is in my control" in note
    # The note must always carry the guard against personalising the
    # philosophy itself, not just the presentation.
    assert "never adjust what the sources say" in note.lower()


def test_missing_profile_falls_back_to_a_default(settings: Settings):
    p = Profile.load_or_default(settings.profiles_dir, "nobody")
    assert p.name == "nobody"
    assert p.interests


# --------------------------------------------------------------------------
# Daily
# --------------------------------------------------------------------------


def test_daily_is_deterministic_for_a_given_day(engine: Engine, settings: Settings, profile: Profile):
    a = generate_daily(engine, profile, settings, day="2026-01-01", save=False)
    b = generate_daily(engine, profile, settings, day="2026-01-01", save=False)
    assert a.piece.theme == b.piece.theme
    assert a.piece.title == b.piece.title


def test_daily_varies_across_days(engine: Engine, settings: Settings, profile: Profile):
    themes = {
        generate_daily(engine, profile, settings, day=f"2026-01-{d:02d}", save=False).piece.theme
        for d in range(1, 8)
    }
    assert len(themes) > 1


def test_daily_is_grounded_and_quotes_a_real_passage(engine: Engine, settings: Settings, profile: Profile):
    result = generate_daily(engine, profile, settings, day="2026-03-03", save=False)
    piece = result.piece
    assert piece.sources
    assert piece.quote.strip()
    assert piece.practice.strip()
    # The quote must be lifted from a passage actually retrieved.
    corpus = " ".join(s.chunk.text for s in piece.sources)
    fragment = piece.quote.split("[")[0].strip().strip("…").strip()[:40]
    assert fragment and fragment in corpus


def test_daily_records_history_and_avoids_repeating_themes(engine: Engine, settings: Settings, profile: Profile):
    profile.save()
    first = generate_daily(engine, profile, settings, day="2026-02-01")
    assert profile.history and profile.history[-1]["date"] == "2026-02-01"
    pool = candidate_themes(profile, engine)
    assert pick_theme(profile, pool, "2026-02-02") != first.piece.theme


def test_explicit_theme_overrides_selection(engine: Engine, settings: Settings, profile: Profile):
    result = generate_daily(engine, profile, settings, day="2026-04-04", theme="water", save=False)
    assert result.piece.theme == "water"


# --------------------------------------------------------------------------
# Provider + config
# --------------------------------------------------------------------------


def test_mock_embeddings_are_deterministic_and_normalised(provider: MockProvider):
    a = provider.embed_query("the unexamined life is not worth living")
    b = provider.embed_query("the unexamined life is not worth living")
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9


def test_mock_embeddings_carry_lexical_signal(provider: MockProvider):
    query = provider.embed_query("fear of death")
    near = provider.embed_query("death is not to be feared by the wise")
    far = provider.embed_query("the rules of grammar and musical melody")
    assert cosine(query, near) > cosine(query, far)


def test_mock_never_invents_when_given_no_sources(provider: MockProvider):
    result = provider.chat([{"role": "user", "content": "QUESTION: what is virtue?"}])
    assert "not" in result.text.lower()
    assert "[1]" not in result.text


def test_dotenv_parsing(tmp_path: Path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        '# comment\nexport OPENAI_API_KEY="sk-quoted"\nPHILO_TOP_K=9\nEMPTY=\n', encoding="utf-8"
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    loaded = load_dotenv(env)
    assert loaded["OPENAI_API_KEY"] == "sk-quoted"
    assert loaded["PHILO_TOP_K"] == "9"


def test_language_detection():
    assert detect_language("我该怎么面对焦虑？") == "zh"
    assert detect_language("how should I face anxiety?") == "en"


def test_l2_normalize_handles_zero_vector():
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


@pytest.fixture
def cwd(tmp_path: Path, settings: Settings, provider: MockProvider, monkeypatch):
    ingest(settings, provider)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def run(argv, capsys):
    from philo.cli import main

    code = main(argv)
    return code, capsys.readouterr().out


def test_cli_help_lists_every_command(capsys):
    code, out = run(["--help"], capsys)
    assert code == 0
    for command in ("ingest", "ask", "chat", "council", "daily", "save", "decide",
                    "chronicle", "recap", "search", "sources", "profile", "doctor"):
        assert command in out


def test_cli_ask_json_is_machine_readable(cwd, capsys):
    code, out = run(["ask", "what is in my control?", "--json", "--no-stream"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert payload["grounded"] is True
    assert payload["sources"]
    assert payload["sources"][0]["chunk"]["work_title"]


def test_cli_search_json(cwd, capsys):
    code, out = run(["search", "water", "--json"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert payload["hits"]
    assert payload["hits"][0]["chunk"]["philosopher"]


def test_cli_sources_json_lists_the_library(cwd, capsys):
    code, out = run(["sources", "--json"], capsys)
    assert code == 0
    rows = json.loads(out)
    assert {r["philosopher"] for r in rows} == {"Epictetus", "Laozi"}


def test_cli_daily_json(cwd, capsys):
    code, out = run(["daily", "--json", "--no-stream", "--no-save", "--date", "2026-05-05"], capsys)
    assert code == 0
    piece = json.loads(out)
    assert piece["theme"]
    assert piece["sources"]


def test_cli_council_json_carries_every_position(cwd, capsys):
    code, out = run(
        ["council", "what is excellence and what is in our control?", "--json", "--seats", "2"],
        capsys,
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["held"] is True
    traditions = {p["tradition"] for p in payload["positions"]}
    assert traditions == {"Stoicism", "Daoism"}
    for position in payload["positions"]:
        cited = {s["chunk"]["tradition"] for s in position["answer"]["sources"]}
        assert cited == {position["tradition"]}


def test_cli_council_declines_rather_than_staging_a_debate(cwd, capsys, monkeypatch):
    """One tradition is not a council, and the exit code should say so."""
    monkeypatch.setenv("PHILO_MIN_SCORE", "0.99")
    code, out = run(["council", "what is in our control?"], capsys)
    assert code == 1
    assert "Traceback" not in out
    assert "relevance floor" in out


# --------------------------------------------------------------------------
# The chronicle, end to end
# --------------------------------------------------------------------------


def test_cli_save_needs_something_to_have_been_retrieved_first(cwd, capsys):
    code, out = run(["save", "1"], capsys)
    assert code == 2
    assert "philo search" in out


def test_cli_search_then_save_then_chronicle(cwd, capsys):
    assert run(["search", "control", "-k", "3"], capsys)[0] == 0
    code, out = run(["save", "1", "--note", "worth returning to"], capsys)
    assert code == 0
    assert "kept 1" in out

    rows = json.loads(run(["chronicle", "--json"], capsys)[1])
    assert len(rows) == 1
    assert rows[0]["kind"] == "passage"
    assert rows[0]["note"] == "worth returning to"
    assert rows[0]["chunk_id"]


def test_cli_save_does_not_keep_the_same_passage_twice(cwd, capsys):
    run(["search", "control", "-k", "3"], capsys)
    run(["save", "1"], capsys)
    code, out = run(["save", "1"], capsys)
    assert code == 0
    assert "already there" in out
    assert len(json.loads(run(["chronicle", "--json"], capsys)[1])) == 1


def test_cli_save_reports_a_marker_that_was_never_offered(cwd, capsys):
    run(["search", "control", "-k", "2"], capsys)
    code, out = run(["save", "9"], capsys)
    assert code == 2
    assert "nothing matches" in out


def test_cli_decide_records_and_can_be_forgotten(cwd, capsys):
    code, out = run(["decide", "someone disturbed me; is my reaction in my control?",
                     "--json"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert payload["sources"]
    entry_id = payload["id"]

    assert run(["chronicle", "--forget", entry_id], capsys)[0] == 0
    assert json.loads(run(["chronicle", "--json"], capsys)[1]) == []


def test_cli_recap_says_so_rather_than_inventing_a_week(cwd, capsys):
    code, out = run(["recap"], capsys)
    assert code == 1
    assert "Nothing recorded" in out
    assert "Traceback" not in out


def test_cli_recap_reads_back_what_was_recorded(cwd, capsys):
    run(["search", "control", "-k", "3"], capsys)
    run(["save", "1"], capsys)
    code, out = run(["recap", "--json"], capsys)
    assert code == 0
    recap = json.loads(out)
    assert recap["n_entries"] == 1
    # The reader's own save is source [1], ahead of anything freshly retrieved.
    assert recap["sources"][0]["marker"] == 1


def test_cli_chronicle_prints_where_the_file_lives(cwd, capsys):
    code, out = run(["chronicle", "--path"], capsys)
    assert code == 0
    assert out.strip().endswith("default.jsonl")


def test_cli_reports_a_missing_index_without_a_traceback(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("PHILO_PROVIDER", "mock")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "library").mkdir()
    code, out = run(["ask", "anything at all"], capsys)
    assert code == 1
    assert "philo ingest" in out
    assert "Traceback" not in out


def test_cli_version(capsys):
    code, out = run(["--version"], capsys)
    assert code == 0
    assert "philo" in out
