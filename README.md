# SecondBrain

An always-on, **fully local** personal "second brain" for your Mac mini. It
continuously records ambient room audio, transcribes it on-device, and gives you
a private, searchable record of everything that was said — with **nothing ever
leaving your machine**.

> **Phase 1 (this release): capture + transcript.** Speaker identification, a
> personal knowledge graph, proactive assistance, and goals are planned in later
> phases — see [the roadmap](docs/ROADMAP.md).

## What it does today

```
room mic ─► capture daemon ─► rolling FLAC chunks ─► VAD (drop silence)
        ─► on-device transcription (MLX) ─► SQLite (timestamped, searchable)
        ─► full-text + semantic search ─► web UI · menu bar · CLI
```

- **Always-on capture** to short FLAC chunks; survives reboots via `launchd`.
- **On-device transcription** (Parakeet or Whisper via Apple MLX) — no cloud.
- **Silence skipped** with voice-activity detection so you don't transcribe (or
  store) dead air.
- **Searchable transcript log** — full-text (FTS5) + optional local semantic
  search (sqlite-vec + a local embedding model).
- **Privacy controls built in:** always-visible recording indicator + one-tap
  pause (menu bar), raw-audio auto-deletion after a retention window
  (transcripts kept), and disk-space guardrails.

## ⚠️ Recording consent — read this first

Recording conversations is **legally regulated and varies by jurisdiction**.
Many US states and most of the EU require **all-party consent**. You are
responsible for complying with the law where you are and for informing the people
around you. SecondBrain gives you the tools (visible indicator, instant pause,
short raw-audio retention, per-speaker opt-out hooks) — but **using them lawfully
is on you**. Verify your local law before deploying.

## Install (Apple Silicon Mac mini)

```bash
git clone <repo> SecondBrain && cd SecondBrain
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[audio,ml,mac]"     # PortAudio: `brew install portaudio` if needed
sb init                              # create data dirs + database (or: alembic upgrade head)
sb devices                           # find your room mic, then set it in config.toml
```

Edit `config.toml` (or `config.local.toml`) — at minimum set
`[capture].input_device`, and review `[consent]` retention settings.

## Run

```bash
sb start          # always-on capture + transcription daemon (foreground)
sb serve          # local web UI at http://127.0.0.1:8765
python -m secondbrain.menubar.app   # menu bar indicator + pause control
```

For an always-on setup, install the `launchd` agent in
[`deploy/com.secondbrain.daemon.plist`](deploy/com.secondbrain.daemon.plist)
(edit the paths first). macOS will prompt for **Microphone** permission on first
run.

## Use

```bash
sb status                                   # recording state, queue, disk
sb search "what did we decide about pricing"
sb show 2026-06-16                          # everything from a day
sb pause   /   sb resume                    # toggle recording live
sb drain                                    # transcribe pending chunks now (off-peak)
sb sweep                                    # delete expired raw audio now
```

## How it's organised

| Area | Module |
|---|---|
| Config | `secondbrain/config.py`, `config.toml` |
| Capture | `secondbrain/capture/` |
| Pipeline (queue, VAD, transcription, worker) | `secondbrain/pipeline/` |
| Storage (schema, models, retention, state) | `secondbrain/storage/` |
| Search (full-text, semantic, fusion) | `secondbrain/search/` |
| Web UI / API | `secondbrain/query/`, `secondbrain/web/` |
| Menu bar | `secondbrain/menubar/` |
| Daemon | `secondbrain/daemon.py` |
| CLI | `secondbrain/cli.py` |

## Design notes

- **Local-first / offline.** Audio, transcripts, models, and (in later phases)
  the LLM all stay on the Mac.
- **Provenance everywhere.** Every transcript segment is traceable back to its
  audio file + time offset, so we can re-transcribe with better models later —
  and so the future knowledge graph can cite its sources.
- **Pluggable, testable backends.** Apple-specific pieces (CoreAudio, MLX, rumps)
  sit behind interfaces with mock/CI implementations, so the platform-agnostic
  logic is unit-tested on any OS.

## Development

```bash
pip install -e ".[dev]"
pytest          # unit tests (mock audio + transcriber; no MLX/CoreAudio needed)
ruff check .
```

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full phased plan (diarization &
voice profiles → local-LLM knowledge graph → proactive assistance & goals).

## License

MIT — see [LICENSE](LICENSE).
