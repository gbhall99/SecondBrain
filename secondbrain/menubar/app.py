"""Menu bar app — the always-visible recording indicator + one-tap pause.

This is a key consent control: the user can always see whether SecondBrain is
listening and stop it instantly. Requires the ``mac`` extra (rumps); macOS only.
Talks to the shared SQLite DB directly (the daemon runs as a separate process).

The title/menu formatting and URL selection are pure module-level functions so
they can be unit-tested without rumps/macOS.
"""

from __future__ import annotations

import webbrowser
from datetime import UTC, datetime, timedelta

from secondbrain.config import Settings, get_settings
from secondbrain.query import service
from secondbrain.storage import state
from secondbrain.storage.db import db_session
from secondbrain.storage.models import iso_from_dt

REC_ON = "🔴 SecondBrain"
REC_OFF = "⏸ SecondBrain"
ERROR_TITLE = "SecondBrain — status unavailable"

# Poll cadence for the status refresh timer (seconds). 10s keeps the indicator
# honest without hammering the shared SQLite DB from a second process.
REFRESH_INTERVAL_S = 10

# Timed-pause menu choices: (menu label, minutes; None = until resumed).
PAUSE_CHOICES: tuple[tuple[str, int | None], ...] = (
    ("Pause 15 min", 15),
    ("Pause 30 min", 30),
    ("Pause 60 min", 60),
    ("Pause until resumed", None),
)


def title_for(status: dict) -> str:
    """Menu bar title for a status payload.

    A stale capture (recording claims on but no audio arriving) outranks the
    red recording dot — the dot would be a lie.
    """
    if status.get("recording") and status.get("capture_stale"):
        span = status.get("capture_stale_for") or "?"
        return f"⚠️ SB — not capturing ({span})"
    return REC_ON if status.get("recording") else REC_OFF


def status_line(status: dict) -> str:
    """The one-line summary shown as the first (disabled) menu item."""
    briefs = status.get("digest_count_today", 0)
    line = (
        f"Today: {status['segments_today']} · Queue: "
        f"{status['jobs'].get('pending', 0)} · {status['disk_free_gb']} GB free"
    )
    if briefs:
        line += f" · {briefs} briefs"
    return line


def base_url(settings: Settings) -> str:
    """Dashboard URL that works from this machine.

    ``api.host`` may be a bind address (0.0.0.0) or a tailnet IP; the menu bar
    always runs on the same machine as the server, so loopback always works —
    and 0.0.0.0 isn't a connectable address at all.
    """
    from secondbrain.security.auth import is_loopback

    host = settings.api.host
    if not is_loopback(host):
        host = "127.0.0.1"
    return f"http://{host}:{settings.api.port}"


def pause_until_iso(minutes: int | None, *, now: datetime | None = None) -> str | None:
    """UTC ISO expiry for a timed pause; None for "until resumed"."""
    if minutes is None:
        return None
    return iso_from_dt((now or datetime.now(UTC)) + timedelta(minutes=minutes))


def run() -> None:
    import rumps  # lazy: macOS only

    settings = get_settings()

    class SecondBrainBar(rumps.App):
        def __init__(self):
            super().__init__(REC_OFF, quit_button="Quit")
            self.status_item = rumps.MenuItem("Status: …", callback=None)
            self.resume_item = rumps.MenuItem("Resume recording", callback=self.on_resume)
            pause_items = [
                rumps.MenuItem(label, callback=self._pause_callback(minutes))
                for label, minutes in PAUSE_CHOICES
            ]
            self.menu = [
                self.status_item,
                None,
                *pause_items,
                self.resume_item,
                None,
                rumps.MenuItem("Today's brief", callback=self.on_brief),
                rumps.MenuItem("Open dashboard", callback=self.on_open),
            ]

        def _status(self) -> dict:
            with db_session(settings=settings) as conn:
                return service.status(conn, settings)

        @rumps.timer(REFRESH_INTERVAL_S)
        def refresh(self, _=None):
            try:
                st = self._status()
            except Exception:  # noqa: BLE001 - a DB hiccup must not kill the app
                self.title = ERROR_TITLE
                self.status_item.title = "Status unavailable"
                return
            self.title = title_for(st)
            self.status_item.title = status_line(st)

        def _pause_callback(self, minutes: int | None):
            def _cb(_):
                with db_session(settings=settings) as conn:
                    state.set_paused(conn, True, until_iso=pause_until_iso(minutes))
                self.refresh()

            return _cb

        def on_resume(self, _):
            with db_session(settings=settings) as conn:
                state.set_paused(conn, False)
            self.refresh()

        def on_open(self, _):
            webbrowser.open(f"{base_url(settings)}/")

        def on_brief(self, _):
            webbrowser.open(f"{base_url(settings)}/brief")

    SecondBrainBar().run()


if __name__ == "__main__":
    run()
