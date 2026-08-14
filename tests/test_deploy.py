"""Deploy automation: plist rendering + launchd install.

Runs on Linux/CI — `launchctl` is never invoked (the runner is mocked).
"""

from __future__ import annotations

import plistlib

import pytest

from secondbrain import deploy


def test_templates_render_with_no_placeholders_and_valid_xml():
    src = deploy.deploy_dir()
    for fname in {**deploy.ALWAYS_AGENTS, **deploy.MENUBAR_AGENT}.values():
        text = (src / fname).read_text()
        rendered = deploy.render_plist(text, repo="/Users/me/SecondBrain", python="/Users/me/SecondBrain/.venv/bin/python")
        assert deploy.REPO_PLACEHOLDER not in rendered
        assert deploy.PYTHON_PLACEHOLDER not in rendered
        parsed = plistlib.loads(rendered.encode())  # raises if malformed
        assert parsed["ProgramArguments"][0] == "/Users/me/SecondBrain/.venv/bin/python"
        assert parsed["WorkingDirectory"] == "/Users/me/SecondBrain"
        # launchd's minimal PATH omits Homebrew; agents must add it back so tools
        # like ffmpeg resolve (otherwise transcription fails under launchd).
        assert "/opt/homebrew/bin" in parsed["EnvironmentVariables"]["PATH"]


def test_install_writes_daemon_and_web_by_default(tmp_path):
    dest = tmp_path / "LaunchAgents"
    written = deploy.install_launchd(
        repo=tmp_path / "repo",
        python="/venv/bin/python",
        launch_agents_dir=dest,
        runner=lambda *a, **k: None,
    )
    names = {p.name for p in written}
    assert names == {"com.secondbrain.daemon.plist", "com.secondbrain.web.plist"}
    assert "com.secondbrain.menubar.plist" not in {p.name for p in dest.iterdir()}
    body = (dest / "com.secondbrain.web.plist").read_text()
    assert "/venv/bin/python" in body
    assert str(tmp_path / "repo") in body


def test_include_menubar_adds_third_agent(tmp_path):
    dest = tmp_path / "LaunchAgents"
    written = deploy.install_launchd(
        launch_agents_dir=dest, include_menubar=True, runner=lambda *a, **k: None
    )
    assert {p.name for p in written} == {
        "com.secondbrain.daemon.plist",
        "com.secondbrain.web.plist",
        "com.secondbrain.menubar.plist",
    }


def test_load_invokes_launchctl(tmp_path):
    calls: list[list[str]] = []
    deploy.install_launchd(
        launch_agents_dir=tmp_path / "LA",
        load=True,
        runner=lambda cmd, **k: calls.append(cmd),
    )
    # Each of the two default agents is unloaded (reload) then loaded -w.
    assert calls[1][:3] == ["launchctl", "load", "-w"]
    assert sum(1 for c in calls if c[:2] == ["launchctl", "load"]) == 2


def test_unload_calls_launchctl_unload_for_existing(tmp_path):
    dest = tmp_path / "LA"
    deploy.install_launchd(launch_agents_dir=dest, runner=lambda *a, **k: None)  # write first
    calls: list[list[str]] = []
    written = deploy.install_launchd(
        launch_agents_dir=dest, unload=True, runner=lambda cmd, **k: calls.append(cmd)
    )
    assert written == []
    assert all(c[:2] == ["launchctl", "unload"] for c in calls)
    assert len(calls) == 2


@pytest.mark.parametrize("include_menubar", [False, True])
def test_agents_selection(include_menubar):
    a = deploy.agents(include_menubar=include_menubar)
    assert ("com.secondbrain.menubar" in a) == include_menubar
    assert "com.secondbrain.daemon" in a and "com.secondbrain.web" in a


# --- batch 3: load reporting, status, stale detection --------------------------


class _Proc:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr


def test_load_agents_reports_per_agent_success_and_failure(tmp_path):
    written = deploy.install_launchd(
        launch_agents_dir=tmp_path / "LA", runner=lambda *a, **k: _Proc()
    )

    def runner(cmd, **kwargs):
        assert kwargs.get("capture_output") is True
        if cmd[:2] == ["launchctl", "unload"]:
            return _Proc(returncode=1, stderr="not loaded")  # ignored
        if "daemon" in cmd[-1]:
            return _Proc(returncode=0)
        return _Proc(returncode=5, stderr="Load failed: 5: Input/output error")

    results = deploy.load_agents(written, runner=runner)
    by_label = {r.label: r for r in results}
    assert by_label["com.secondbrain.daemon"].ok
    assert not by_label["com.secondbrain.web"].ok
    assert "Input/output error" in by_label["com.secondbrain.web"].stderr


def test_agent_status_reports_installed_loaded_and_logs(tmp_path):
    dest = tmp_path / "LA"
    deploy.install_launchd(launch_agents_dir=dest, runner=lambda *a, **k: _Proc())

    def runner(cmd, **kwargs):
        assert cmd[:2] == ["launchctl", "list"]
        return _Proc(returncode=0 if cmd[2] == "com.secondbrain.daemon" else 1)

    statuses = deploy.agent_status(
        launch_agents_dir=dest, runner=runner, have_launchctl=True, include_menubar=True
    )
    by = {s.label: s for s in statuses}
    assert by["com.secondbrain.daemon"].installed and by["com.secondbrain.daemon"].loaded
    assert by["com.secondbrain.web"].installed and not by["com.secondbrain.web"].loaded
    assert not by["com.secondbrain.menubar"].installed  # never written
    # log paths parsed from the rendered plist
    assert any("daemon" in p for p in by["com.secondbrain.daemon"].log_paths)


def test_agent_status_degrades_without_launchctl(tmp_path):
    statuses = deploy.agent_status(
        launch_agents_dir=tmp_path / "LA", have_launchctl=False,
        runner=lambda *a, **k: pytest.fail("launchctl must not be invoked"),
    )
    assert all(s.loaded is None for s in statuses)
    assert all(not s.installed for s in statuses)


def test_stale_plists_detects_python_and_repo_mismatch(tmp_path):
    dest = tmp_path / "LA"
    deploy.install_launchd(
        repo=tmp_path / "old-repo", python="/old/venv/bin/python",
        launch_agents_dir=dest, runner=lambda *a, **k: _Proc(),
    )
    problems = deploy.stale_plists(
        launch_agents_dir=dest, repo=tmp_path / "new-repo", python="/new/venv/bin/python"
    )
    assert problems
    assert any("/old/venv/bin/python" in p for p in problems)
    assert any("old-repo" in p for p in problems)
    # matching environment → clean
    assert deploy.stale_plists(
        launch_agents_dir=dest, repo=tmp_path / "old-repo", python="/old/venv/bin/python"
    ) == []
    # nothing installed → clean
    assert deploy.stale_plists(launch_agents_dir=tmp_path / "empty") == []


def test_cli_deploy_launchd_refuses_non_macos(monkeypatch, settings):
    from typer.testing import CliRunner

    from secondbrain import cli

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr("sys.platform", "linux")
    result = CliRunner().invoke(cli.app, ["deploy", "launchd"])
    assert result.exit_code == 1
    assert "macOS" in result.output


def test_cli_deploy_status_degrades_off_macos(monkeypatch, settings, tmp_path):
    from typer.testing import CliRunner

    from secondbrain import cli

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    real_agent_status = deploy.agent_status
    monkeypatch.setattr(
        deploy, "agent_status",
        lambda **kw: real_agent_status(
            launch_agents_dir=tmp_path / "LA", have_launchctl=False,
            runner=lambda *a, **k: None, **kw,
        ),
    )
    result = CliRunner().invoke(cli.app, ["deploy", "status"])
    assert result.exit_code == 0, result.output
    assert "com.secondbrain.daemon" in result.output
    assert "not installed" in result.output
    assert "launchctl unavailable" in result.output
