"""Logging configuration: rotating file handler + force reconfiguration."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from secondbrain import logging_setup


def _cleanup_root():
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    logging.basicConfig(level=logging.WARNING, force=True)


def test_configure_logging_adds_rotating_file_handler(settings):
    try:
        logging_setup.configure_logging(settings)
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
        assert len(file_handlers) == 1
        h = file_handlers[0]
        assert h.baseFilename == str(logging_setup.log_file_path(settings))
        assert h.maxBytes == 10 * 1024 * 1024
        assert h.backupCount == 5
        logging.getLogger("secondbrain.test").info("hello file")
        h.flush()
        assert "hello file" in logging_setup.log_file_path(settings).read_text()
    finally:
        _cleanup_root()


def test_configure_logging_respects_file_enabled_false(settings):
    try:
        settings.logging.file_enabled = False
        logging_setup.configure_logging(settings)
        root = logging.getLogger()
        assert not [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
        assert not logging_setup.log_file_path(settings).exists()
    finally:
        _cleanup_root()


def test_configure_logging_forces_over_prior_config(settings):
    try:
        # A library configured logging first — configure_logging must still win.
        logging.basicConfig(level=logging.CRITICAL, force=True)
        logging_setup.configure_logging(settings)
        assert logging.getLogger().level == logging.INFO  # settings default
        assert any(isinstance(h, RotatingFileHandler)
                   for h in logging.getLogger().handlers)
    finally:
        _cleanup_root()


def test_log_file_path_under_data_dir(settings):
    p = logging_setup.log_file_path(settings)
    assert p == settings.data_path / "logs" / "secondbrain.log"
