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
