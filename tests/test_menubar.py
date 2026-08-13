"""Menu bar formatting logic — pure functions, no rumps/macOS required."""

from __future__ import annotations

from datetime import UTC, datetime

from secondbrain.menubar import app as menubar


def test_title_recording_states():
    assert menubar.title_for({"recording": True}) == menubar.REC_ON
    assert menubar.title_for({"recording": False, "paused": True}) == menubar.REC_OFF


def test_title_warns_on_capture_stale():
    t = menubar.title_for(
        {"recording": True, "capture_stale": True, "capture_stale_for": "12m"}
    )
    assert t == "⚠️ SB — not capturing (12m)"
    # Not recording → staleness is expected, no warning.
    t = menubar.title_for(
        {"recording": False, "capture_stale": True, "capture_stale_for": "3h"}
    )
    assert t == menubar.REC_OFF


def test_status_line_includes_counts_and_briefs():
    st = {"segments_today": 42, "jobs": {"pending": 3}, "disk_free_gb": 120.5,
          "digest_count_today": 2}
    line = menubar.status_line(st)
    assert "Today: 42" in line and "Queue: 3" in line and "120.5 GB free" in line
    assert "2 briefs" in line
    st["digest_count_today"] = 0
    assert "briefs" not in menubar.status_line(st)


def test_base_url_rewrites_non_loopback_hosts(settings):
    assert menubar.base_url(settings) == "http://127.0.0.1:8765"
    settings.api.host = "0.0.0.0"
    assert menubar.base_url(settings) == "http://127.0.0.1:8765"
    settings.api.host = "100.64.0.7"  # tailnet bind: still open locally
    assert menubar.base_url(settings) == "http://127.0.0.1:8765"
    settings.api.host = "localhost"
    assert menubar.base_url(settings) == "http://localhost:8765"


def test_pause_until_iso():
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=UTC)
    assert menubar.pause_until_iso(None) is None
    iso = menubar.pause_until_iso(15, now=now)
    assert iso is not None and iso.startswith("2026-08-13T12:15:00")


def test_pause_choices_cover_timed_and_indefinite():
    minutes = [m for _, m in menubar.PAUSE_CHOICES]
    assert minutes == [15, 30, 60, None]


def test_refresh_interval_is_10s():
    assert menubar.REFRESH_INTERVAL_S == 10
