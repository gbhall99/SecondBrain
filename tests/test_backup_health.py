"""Backup-freshness health check."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from secondbrain import health


def _check(conn, settings):
    return next(c for c in health.run_checks(conn, settings) if c.name == "backups")


def test_no_backups_is_ok_with_hint(conn, settings):
    c = _check(conn, settings)
    assert c.ok is True
    assert "run `sb backup`" in c.detail


def test_recent_backup_is_ok(conn, settings):
    bdir = settings.data_path / "backups"
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "secondbrain-20260101-000000.db").write_bytes(b"x")
    c = _check(conn, settings)
    assert c.ok is True
    assert "snapshot(s)" in c.detail


def test_stale_backup_flagged(conn, settings, monkeypatch):
    bdir = settings.data_path / "backups"
    bdir.mkdir(parents=True, exist_ok=True)
    f = bdir / "secondbrain-20000101-000000.db"
    f.write_bytes(b"x")
    old = (datetime.now(UTC) - timedelta(days=60)).timestamp()
    import os

    os.utime(f, (old, old))
    c = _check(conn, settings)
    assert c.ok is False
    assert "ago" in c.detail
