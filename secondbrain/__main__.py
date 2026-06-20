"""Run the SecondBrain CLI as a module: ``python -m secondbrain ...``.

Mirrors the ``sb`` console script (``pyproject.toml`` ``[project.scripts]``) so the
launchd agents in ``deploy/`` can invoke a single interpreter without depending on
the venv's ``bin`` directory being on ``PATH``.
"""

from secondbrain.cli import app

if __name__ == "__main__":
    app()
