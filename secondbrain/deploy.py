"""Render and install the always-on ``launchd`` agents under ``deploy/``.

The plist templates use ``__REPO__`` and ``__PYTHON__`` placeholders; this module
fills them with the repo root and the running interpreter and (optionally) loads
them via ``launchctl``. Pure-Python rendering is deliberately separated from the
side-effecting install so it stays testable on Linux/CI (no macOS required).
"""

from __future__ import annotations

import plistlib
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_PLACEHOLDER = "__REPO__"
PYTHON_PLACEHOLDER = "__PYTHON__"

# launchd label -> template filename in deploy/. Daemon + web are always installed;
# the menu bar agent is opt-in (needs a GUI session + the `mac` extra).
ALWAYS_AGENTS: dict[str, str] = {
    "com.secondbrain.daemon": "com.secondbrain.daemon.plist",
    "com.secondbrain.web": "com.secondbrain.web.plist",
}
MENUBAR_AGENT: dict[str, str] = {
    "com.secondbrain.menubar": "com.secondbrain.menubar.plist",
}


def repo_root() -> Path:
    """Repo root = the parent of the ``secondbrain`` package directory."""
    return Path(__file__).resolve().parent.parent


def deploy_dir() -> Path:
    return repo_root() / "deploy"


def render_plist(template_text: str, *, repo: Path | str, python: Path | str) -> str:
    """Substitute the ``__REPO__`` / ``__PYTHON__`` placeholders in a template."""
    return template_text.replace(REPO_PLACEHOLDER, str(repo)).replace(
        PYTHON_PLACEHOLDER, str(python)
    )


def agents(*, include_menubar: bool) -> dict[str, str]:
    out = dict(ALWAYS_AGENTS)
    if include_menubar:
        out.update(MENUBAR_AGENT)
    return out


def install_launchd(
    *,
    repo: Path | None = None,
    python: str | None = None,
    include_menubar: bool = False,
    launch_agents_dir: Path | None = None,
    load: bool = False,
    unload: bool = False,
    src_dir: Path | None = None,
    runner=subprocess.run,
) -> list[Path]:
    """Render each agent's plist into ``launch_agents_dir`` and optionally
    (un)load it via ``launchctl``.

    Returns the plist paths written (empty when ``unload=True``). ``runner`` is
    injectable so tests can assert the ``launchctl`` calls without a real macOS.
    """
    repo = Path(repo) if repo is not None else repo_root()
    python = python or sys.executable
    src = src_dir if src_dir is not None else deploy_dir()
    dest_dir = launch_agents_dir or (Path.home() / "Library" / "LaunchAgents")
    dest_dir.mkdir(parents=True, exist_ok=True)

    selected = agents(include_menubar=include_menubar)
    if unload:
        for label in selected:
            dest = dest_dir / f"{label}.plist"
            if dest.exists():
                runner(["launchctl", "unload", str(dest)], check=False)
        return []

    written: list[Path] = []
    for label, fname in selected.items():
        dest = dest_dir / f"{label}.plist"
        dest.write_text(render_plist((src / fname).read_text(), repo=repo, python=python))
        written.append(dest)
        if load:
            # Reload idempotently: unload (ignore failure) then load -w.
            runner(["launchctl", "unload", str(dest)], check=False)
            runner(["launchctl", "load", "-w", str(dest)], check=False)
    return written


@dataclass
class LoadResult:
    """Outcome of one ``launchctl load`` invocation."""

    label: str
    ok: bool
    stderr: str = ""


def load_agents(paths: list[Path], runner=subprocess.run) -> list[LoadResult]:
    """(Re)load each written plist via launchctl, capturing per-agent outcomes.

    Unlike the fire-and-forget ``install_launchd(load=True)`` path, this reports
    success/failure + stderr for every agent so the CLI can say which one broke
    instead of an unconditional "Loaded".
    """
    results: list[LoadResult] = []
    for dest in paths:
        # Reload idempotently: unload (ignore failure) then load -w.
        runner(["launchctl", "unload", str(dest)],
               check=False, capture_output=True, text=True)
        proc = runner(["launchctl", "load", "-w", str(dest)],
                      check=False, capture_output=True, text=True)
        ok = getattr(proc, "returncode", 0) == 0
        stderr = ((getattr(proc, "stderr", "") or "").strip())
        results.append(LoadResult(label=dest.stem, ok=ok, stderr=stderr))
    return results


@dataclass
class AgentStatus:
    """Install/load state of one launchd agent for ``sb deploy status``."""

    label: str
    installed: bool
    loaded: bool | None  # None = launchctl unavailable (non-macOS)
    plist_path: Path
    log_paths: list[str] = field(default_factory=list)


def agent_status(
    *,
    include_menubar: bool = True,
    launch_agents_dir: Path | None = None,
    runner=subprocess.run,
    have_launchctl: bool | None = None,
) -> list[AgentStatus]:
    """Per-agent install + load status; degrades gracefully off-macOS."""
    import shutil

    dest_dir = launch_agents_dir or (Path.home() / "Library" / "LaunchAgents")
    if have_launchctl is None:
        have_launchctl = shutil.which("launchctl") is not None
    out: list[AgentStatus] = []
    for label in agents(include_menubar=include_menubar):
        dest = dest_dir / f"{label}.plist"
        installed = dest.exists()
        loaded: bool | None = None
        if have_launchctl:
            proc = runner(["launchctl", "list", label],
                          check=False, capture_output=True, text=True)
            loaded = getattr(proc, "returncode", 1) == 0
        logs: list[str] = []
        if installed:
            try:
                data = plistlib.loads(dest.read_bytes())
                for key in ("StandardOutPath", "StandardErrorPath"):
                    if data.get(key):
                        logs.append(str(data[key]))
            except Exception:  # noqa: BLE001 - a corrupt plist still reports installed
                pass
        out.append(AgentStatus(label=label, installed=installed, loaded=loaded,
                               plist_path=dest, log_paths=logs))
    return out


def stale_plists(
    *,
    launch_agents_dir: Path | None = None,
    repo: Path | None = None,
    python: str | None = None,
) -> list[str]:
    """Installed plists whose python path / repo directory no longer match this
    environment (venv moved, repo relocated) — the agents would run old code.

    Returns human-readable mismatch descriptions; empty when everything (or
    nothing) is installed consistently.
    """
    dest_dir = launch_agents_dir or (Path.home() / "Library" / "LaunchAgents")
    repo = Path(repo) if repo is not None else repo_root()
    python = python or sys.executable
    problems: list[str] = []
    for label in agents(include_menubar=True):
        dest = dest_dir / f"{label}.plist"
        if not dest.exists():
            continue
        try:
            data = plistlib.loads(dest.read_bytes())
        except Exception:  # noqa: BLE001
            problems.append(f"{label}: unreadable plist at {dest}")
            continue
        args = data.get("ProgramArguments") or []
        if args and args[0] != python:
            problems.append(f"{label}: runs {args[0]}, current interpreter is {python}")
        wd = data.get("WorkingDirectory")
        if wd and Path(wd) != repo:
            problems.append(f"{label}: WorkingDirectory {wd}, current repo is {repo}")
    return problems
