"""Structured logging setup shared by the daemon and the web server."""

from __future__ import annotations

import logging

from secondbrain.config import Settings, get_settings

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    level = getattr(logging, settings.logging.level.upper(), logging.INFO)
    logging.basicConfig(level=level, format=_FORMAT)
