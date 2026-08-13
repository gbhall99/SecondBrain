"""CLI ergonomics: clean errors instead of tracebacks, human output, --version."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from secondbrain import __version__, cli

runner = CliRunner()


@pytest.fixture
def cli_settings(settings, monkeypatch):
    """Point every command at the hermetic test settings."""
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    return settings


# --- global config guard ------------------------------------------------------


def test_config_validation_error_is_one_line_exit_2(monkeypatch):
    from pydantic import ValidationError

    from secondbrain.config import LLMConfig

    try:
        LLMConfig(request_timeout_s=0)
    except ValidationError as exc:
        captured = exc
    else:  # pragma: no cover
        raise AssertionError("expected ValidationError")

    def boom():
        raise captured

    monkeypatch.setattr(cli, "get_settings", boom)
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 2
    assert "Config error:" in result.output
    assert "request_timeout_s" in result.output
    assert "config.local.toml" in result.output
    assert "Traceback" not in result.output


def test_config_toml_parse_error_is_one_line_exit_2(monkeypatch):
    import tomllib

    def boom():
        tomllib.loads("this is [not toml")
        raise AssertionError("unreachable")  # pragma: no cover

    monkeypatch.setattr(cli, "get_settings", boom)
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 2
    assert "Config error: could not parse TOML" in result.output
    assert "Traceback" not in result.output


# --- uninitialised DB ---------------------------------------------------------


@pytest.mark.parametrize("args", [["status"], ["stats"], ["search", "x"],
                                  ["show", "2026-01-01"], ["queue"], ["projects"]])
def test_read_commands_on_uninitialised_db(cli_settings, args):
    # No `sb init`: the DB file doesn't exist.
    assert not cli_settings.db_path.exists()
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 2, result.output
    assert "Not initialised — run `sb init`" in result.output
    # and no empty DB file was created as a side effect
    assert not cli_settings.db_path.exists()


# --- date + enum validation ---------------------------------------------------


@pytest.mark.parametrize("args", [
    ["search", "x", "--since", "not-a-date"],
    ["search", "x", "--until", "2026-13-45"],
    ["show", "yesterdayish"],
    ["timeline", "01/02/2026"],
    ["forget", "day", "not-a-date", "--yes"],
    ["forget", "range", "2026-01-01", "garbage", "--yes"],
    ["export", "--since", "2026/01/01"],
])
def test_bad_date_arguments_exit_2(cli_settings, args):
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 2, result.output
    assert "YYYY-MM-DD" in result.output
    assert "Traceback" not in result.output


def test_bad_enum_options_exit_2_with_choices(cli_settings):
    result = runner.invoke(cli.app, ["export", "--format", "xml"])
    assert result.exit_code == 2
    assert "json, md, markdown, both" in result.output

    result = runner.invoke(cli.app, ["search", "x", "--mode", "psychic"])
    assert result.exit_code == 2
    assert "auto, fulltext, semantic" in result.output


def test_export_writes_files_for_empty_corpus(cli_settings, conn, tmp_path):
    out = tmp_path / "exports"
    result = runner.invoke(cli.app, ["export", "--format", "json", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "Exported" in result.output  # something was reported, never silent


# --- devices without the audio extra -----------------------------------------


def test_devices_without_audio_extra(cli_settings, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "sounddevice":
            raise ImportError("No module named 'sounddevice'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = runner.invoke(cli.app, ["devices"])
    assert result.exit_code == 1
    assert "pip install -e '.[audio]'" in result.output


# --- LLM failure messages -----------------------------------------------------


def test_decompose_llm_connect_error_is_friendly(cli_settings, conn, monkeypatch):
    import httpx

    from secondbrain.query import service

    def boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(service, "decompose_goal", boom)
    result = runner.invoke(cli.app, ["decompose", "1"])
    assert result.exit_code == 1, result.output
    assert "is Ollama running?" in result.output
    assert "Traceback" not in result.output


def test_digest_llm_timeout_is_friendly(cli_settings, conn, monkeypatch):
    import httpx

    from secondbrain.query import service

    def boom(*a, **k):
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(service, "generate_digest", boom)
    result = runner.invoke(cli.app, ["digest"])
    assert result.exit_code == 1, result.output
    assert "didn't answer within" in result.output


def test_task_research_llm_failure_is_friendly(cli_settings, conn, monkeypatch):
    import httpx

    from secondbrain.query import service

    def boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(service, "task_research", boom)
    result = runner.invoke(cli.app, ["task", "research", "1"])
    assert result.exit_code == 1, result.output
    assert "is Ollama running?" in result.output


# --- --version ----------------------------------------------------------------


def test_version_flag():
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert f"sb {__version__}" in result.output


# --- human-readable status / stats / queue ------------------------------------


def test_status_human_and_json(cli_settings, conn):
    human = runner.invoke(cli.app, ["status"])
    assert human.exit_code == 0, human.output
    assert "Recording:" in human.output
    assert "Disk free:" in human.output
    assert "{" not in human.output  # not a JSON dump

    machine = runner.invoke(cli.app, ["status", "--json"])
    assert machine.exit_code == 0
    data = json.loads(machine.output)
    assert {"recording", "paused", "jobs", "segments_total"} <= set(data)


def test_stats_human_and_json(cli_settings, conn):
    human = runner.invoke(cli.app, ["stats"])
    assert human.exit_code == 0, human.output
    assert "Segments:" in human.output
    machine = runner.invoke(cli.app, ["stats", "--json"])
    assert json.loads(machine.output)["segments"] == 0


def test_queue_human_and_json(cli_settings, conn):
    from secondbrain.pipeline import queue as q

    q.enqueue(conn, "transcribe", {"audio_file_id": 1}, max_attempts=1)
    q.fail(conn, q.claim_next(conn), "boom")
    human = runner.invoke(cli.app, ["queue"])
    assert human.exit_code == 0, human.output
    assert "failed" in human.output
    assert "Recent failures:" in human.output
    assert "boom" in human.output

    machine = runner.invoke(cli.app, ["queue", "--json"])
    data = json.loads(machine.output)
    assert data["counts"].get("failed") == 1
    assert data["recent_failures"]


# --- sb logs ------------------------------------------------------------------


def test_logs_prints_path_and_tail(cli_settings):
    from secondbrain.logging_setup import log_file_path

    path = log_file_path(cli_settings)
    result = runner.invoke(cli.app, ["logs"])
    assert result.exit_code == 0, result.output
    assert str(path) in result.output
    assert "no log file yet" in result.output

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"line {i}\n" for i in range(100)))
    result = runner.invoke(cli.app, ["logs", "-n", "3"])
    assert result.exit_code == 0
    assert "line 99" in result.output
    assert "line 96" not in result.output


# --- serve proxy posture ------------------------------------------------------


def test_serve_passes_explicit_proxy_settings(cli_settings, conn, monkeypatch):
    import uvicorn

    captured = {}

    def fake_run(app, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    result = runner.invoke(cli.app, ["serve"])
    assert result.exit_code == 0, result.output
    assert captured["proxy_headers"] is True
    assert captured["forwarded_allow_ips"] == "127.0.0.1"


# --- config set-hf-token ------------------------------------------------------


def test_set_hf_token_writes_repo_local_config_not_cwd(cli_settings, monkeypatch, tmp_path):
    import tomllib

    import secondbrain.config as config_mod

    monkeypatch.setattr(config_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(config_mod, "reload_settings", lambda: cli_settings)
    monkeypatch.chdir(tmp_path / ".." if (tmp_path / "..").exists() else tmp_path)
    result = runner.invoke(cli.app, ["config", "set-hf-token", "hf_test_token"])
    assert result.exit_code == 0, result.output
    written = tmp_path / "config.local.toml"
    assert written.exists()  # repo root, not the CWD
    assert tomllib.loads(written.read_text())["diarization"]["hf_token"] == "hf_test_token"
    assert (written.stat().st_mode & 0o777) == 0o600
    assert "Restart the daemon/server" in result.output
