"""Always-on supervisor (launchd entrypoint).

Runs three cooperating loops, each with its own SQLite connection:
  1. capture  — the rolling recorder (room audio -> FLAC chunks -> queue)
  2. worker   — drains transcription jobs (VAD -> transcribe -> store -> index)
  3. maintenance — periodic raw-audio retention sweep

The local web API is run separately via ``sb serve`` (or its own launchd job) so
capture keeps running even if the UI is restarted.
"""

from __future__ import annotations

import logging
import signal
import threading
import time

from secondbrain.capture.recorder import Recorder
from secondbrain.config import Settings, get_settings
from secondbrain.pipeline import worker
from secondbrain.storage import retention
from secondbrain.storage.db import init_db

log = logging.getLogger("secondbrain.daemon")

WORKER_IDLE_SLEEP = 2.0
RETENTION_INTERVAL_S = 3600


class Daemon:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._recorder: Recorder | None = None

    # --- loops ---------------------------------------------------------------

    def _capture_loop(self) -> None:
        conn = init_db(settings=self.settings)
        self._recorder = Recorder(conn, self.settings)
        try:
            self._recorder.run()
        except Exception:  # noqa: BLE001
            log.exception("capture loop crashed")
        finally:
            conn.close()

    def _worker_loop(self) -> None:
        conn = init_db(settings=self.settings)
        try:
            while not self._stop.is_set():
                try:
                    ran = worker.run_once(conn, settings=self.settings)
                except Exception:  # noqa: BLE001
                    log.exception("worker iteration failed")
                    ran = False
                if not ran:
                    self._stop.wait(WORKER_IDLE_SLEEP)
        finally:
            conn.close()

    def _maintenance_loop(self) -> None:
        conn = init_db(settings=self.settings)
        try:
            while not self._stop.is_set():
                try:
                    n = retention.sweep_expired_audio(conn, self.settings)
                    if n:
                        log.info("retention: deleted %d expired raw-audio files", n)
                except Exception:  # noqa: BLE001
                    log.exception("retention sweep failed")
                self._stop.wait(RETENTION_INTERVAL_S)
        finally:
            conn.close()

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self.settings.ensure_dirs()
        init_db(settings=self.settings).close()  # create schema once up front
        for target in (self._capture_loop, self._worker_loop, self._maintenance_loop):
            t = threading.Thread(target=target, name=target.__name__, daemon=True)
            t.start()
            self._threads.append(t)
        log.info("SecondBrain daemon started (%d loops)", len(self._threads))

    def stop(self) -> None:
        log.info("stopping daemon…")
        self._stop.set()
        if self._recorder:
            self._recorder.stop()

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    daemon = Daemon()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: daemon.stop())
    daemon.run_forever()


if __name__ == "__main__":
    main()
