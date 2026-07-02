"""Render and install the always-on ``launchd`` agents under ``deploy/``.

The plist templates use ``__REPO__`` and ``__PYTHON__`` placeholders; this module
fills them with the repo root and the running interpreter and (optionally) loads
them via ``launchctl``. Pure-Python rendering is deliberately separated from the
side-effecting install so it stays testable on Linux/CI (no macOS required).
"""

from __future__ import annotations

import subprocess
import sys
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
