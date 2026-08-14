"""Structured logging setup shared by the daemon and the web server."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from secondbrain.config import Settings, get_settings

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Rotated file log: 10 MB × 5 backups under <data>/logs/.
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024
LOG_FILE_BACKUPS = 5


def log_file_path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.data_path / "logs" / "secondbrain.log"


def configure_logging(settings: Settings | None = None) -> None:
    """Configure root logging: stdout + a rotating file under <data>/logs.

    ``force=True`` so a library (e.g. uvicorn's import-time defaults) that
    already touched the root logger can't silently swallow our handlers.
    """
    settings = settings or get_settings()
    level = getattr(logging, settings.logging.level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if settings.logging.file_enabled:
        path = log_file_path(settings)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(
                RotatingFileHandler(
                    path, maxBytes=LOG_FILE_MAX_BYTES, backupCount=LOG_FILE_BACKUPS
                )
            )
        except OSError:  # unwritable data dir must not kill the process
            logging.getLogger(__name__).warning("could not open log file %s", path)
    logging.basicConfig(level=level, format=_FORMAT, handlers=handlers, force=True)
