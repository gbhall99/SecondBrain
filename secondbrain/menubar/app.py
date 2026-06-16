"""Menu bar app — the always-visible recording indicator + one-tap pause.

This is a key consent control: the user can always see whether SecondBrain is
listening and stop it instantly. Requires the ``mac`` extra (rumps); macOS only.
Talks to the shared SQLite DB directly (the daemon runs as a separate process).
"""

from __future__ import annotations

import webbrowser

from secondbrain.config import get_settings
from secondbrain.query import service
from secondbrain.storage import state
from secondbrain.storage.db import db_session

REC_ON = "🔴 SecondBrain"
REC_OFF = "⏸ SecondBrain"


def run() -> None:
    import rumps  # lazy: macOS only

    settings = get_settings()

    class SecondBrainBar(rumps.App):
        def __init__(self):
            super().__init__(REC_OFF, quit_button="Quit")
            self.toggle_item = rumps.MenuItem("Pause recording", callback=self.on_toggle)
            self.status_item = rumps.MenuItem("Status: …", callback=None)
            self.menu = [
                self.status_item,
                None,
                self.toggle_item,
                rumps.MenuItem("Open dashboard", callback=self.on_open),
            ]

        def _status(self) -> dict:
            with db_session(settings=settings) as conn:
                return service.status(conn, settings)

        @rumps.timer(3)
        def refresh(self, _=None):
            st = self._status()
            recording = st["recording"]
            self.title = REC_ON if recording else REC_OFF
            self.toggle_item.title = "Pause recording" if recording else "Resume recording"
            self.status_item.title = (
                f"Today: {st['segments_today']} · Queue: "
                f"{st['jobs'].get('pending', 0)} · {st['disk_free_gb']} GB free"
            )

        def on_toggle(self, _):
            with db_session(settings=settings) as conn:
                paused = state.is_paused(conn, default=settings.consent.paused)
                state.set_paused(conn, not paused)
            self.refresh()

        def on_open(self, _):
            webbrowser.open(f"http://{settings.api.host}:{settings.api.port}/")

    SecondBrainBar().run()


if __name__ == "__main__":
    run()
