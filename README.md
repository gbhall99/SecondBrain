# SecondBrain

An always-on, **fully local** personal "second brain" for your Mac mini. It
continuously records ambient room audio, transcribes it on-device, and gives you
a private, searchable record of everything that was said — with **nothing ever
leaving your machine**.

> **Phases 1–7 shipped:** capture + transcript, diarization & voice profiles, a
> local-LLM knowledge graph with grounded Q&A, proactive assistance + goals,
> hardening (auth, encryption, health), goals-as-an-advanced-to-do-list (AI
> decomposition, prioritisation matrix, daily plan, task research), and
> speaker-quality self-correction (exemplar matching, re-attribution, a correction
> loop). See [the roadmap](docs/ROADMAP.md).

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
- **Speaker diarization & voice profiles** (Phase 2) — conversations are diarized
  as a whole (pyannote 3.1), each line attributed to a global speaker with a
  confidence, your own voice enrolled explicitly, and recurring unknown voices
  clustered over time so you can name them once and relabel all history.
- **Knowledge graph + grounded Q&A** (Phase 3) — a local LLM (Ollama) reads each
  diarized conversation and extracts people, projects, facts, decisions, and
  action items into a SQLite knowledge graph (with provenance + fact versioning);
  `sb ask` / the web chat answer questions grounded in your data **with
  citations** (and clearly-labeled general knowledge when helpful). Nothing leaves
  the machine.
- **Proactivity + goals** (Phase 4) — set goals, and get a curated **morning
  brief** + **weekly review**: commitments you owe (before due) and ones owed to
  you (overdue), goal progress, cross-conversation connections, and opt-in candid
  coaching — all ranked with noise control (daily cap, snooze, thumbs-up/down,
  confidence floor) and cited. Web `/brief` + `/goals`, menu-bar count, real-time
  nudges for urgent commitments, `sb digest` / `sb goals`.
- **Privacy controls built in:** always-visible recording indicator + one-tap
  pause (menu bar), raw-audio auto-deletion after a retention window
  (transcripts kept; deferred until a conversation is diarized), disk-space
  guardrails, and per-speaker opt-out (redacts that person's words).

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

For an always-on setup (survives reboot), install the `launchd` agents with one
command — it fills in the paths and loads them for you:

```bash
sb deploy launchd --load --include-menubar
```

macOS will prompt for **Microphone** permission on first run. The full Mac mini
deployment guide — install script, config, optional Ollama/pyannote, Tailscale
remote access, and verification — is in **[docs/DEPLOY.md](docs/DEPLOY.md)**.

## Use

```bash
sb status                                   # recording state, queue, disk
sb search "what did we decide about pricing"
sb show 2026-06-16                          # everything from a day
sb pause   /   sb resume                    # toggle recording live
sb drain                                    # process pending jobs now (off-peak)
sb sweep                                    # delete expired raw audio now
```

### Speakers (Phase 2)

```bash
sb speaker setup                  # one-time: download/authorize pyannote models
sb speaker enroll-owner           # guided recording of your own voice
sb speaker unknowns               # voices awaiting a name
sb speaker name 7 "Alice"         # name a voice (relabels all their history)
sb speaker merge 9 7              # 9 is actually 7 → merge + relabel
sb speaker cluster                # merge recurring unknown voices now
sb speaker opt-out 4              # redact a person's past + future words
```

Or use the web **Speakers** page (`/speakers`) to play sample clips and label
voices with the "Who is this?" queue.

**Improving accuracy over time (Phase 7):** matching is exemplar-aware (a voice is
compared to a person's centroid *and* their nearest stored samples). Fix a wrong
line on the **`/day`** page (or `sb speaker reassign <segment> <speaker>`) — it
locks that line and adds a confirmed sample so future matching improves. As
profiles improve, `sb speaker reattribute` (also nightly) relabels past
low-confidence lines; `sb speaker recompute`/`prune`/`quality` manage profiles.
Overlapping-speech lines are flagged low-confidence; pyannote overlap params are
exposed in `[diarization]` for on-device tuning.

### Knowledge graph & Q&A (Phase 3)

```bash
sb ask "what did Dana commit to this week?"   # grounded answer + sources
```

Web: **Ask** (`/chat`) for grounded Q&A with citation chips, and **Graph**
(`/graph`) to browse people/projects/decisions and their sources.

### Goals & the proactive brief (Phase 4)

```bash
sb goals add "Ship pricing v2" -d "revamp tiers" -p 1 --target-date 2026-09-30
sb goals list
sb digest                          # today's morning brief + ranked items
sb digest --weekly                 # the weekly review
sb digest-action 3 done            # done | dismiss | snooze | up | down
```

Web: **Brief** (`/brief`) shows the morning brief with per-item actions; **Goals**
(`/goals`) manages goals. The brief is generated nightly at `[proactive].digest_hour`
(weekly review on `weekly_review_weekday`); enable with `[proactive].enabled = true`
(and `coaching_enabled` / `event_triggers` as desired). It reuses the same local
Ollama model as Phase 3.

**One-time local-LLM setup (fully offline):** install [Ollama](https://ollama.com),
`ollama serve`, and `ollama pull llama3.1:8b-instruct` (or your preferred instruct
model). Then set `[llm].backend = "ollama"` (and `[llm].model`) and
`[extraction].enabled = true` in `config.local.toml`. Extraction runs as a queued
job after each conversation is diarized; drain it off-peak with `sb drain`.

**One-time pyannote setup (gated models, local thereafter):** create a read token
at <https://huggingface.co/settings/tokens>, accept the conditions on
`pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`, then put the
token in `config.local.toml` (`[diarization].hf_token`) or the `HF_TOKEN` env var,
set `[diarization].enabled = true`, and run `sb speaker setup`. After download,
diarization runs fully offline.

## How it's organised

| Area | Module |
|---|---|
| Config | `secondbrain/config.py`, `config.toml` |
| Capture | `secondbrain/capture/` |
| Pipeline (queue, VAD, transcription, diarization, conversations, worker) | `secondbrain/pipeline/` |
| Speakers (registry, matching, clustering, enrollment, attribution) | `secondbrain/speaker/` |
| Local LLM (Ollama / mock client) | `secondbrain/llm/` |
| Knowledge (extraction, entity resolution, graph store, chat) | `secondbrain/knowledge/` |
| Goals (store, auto-linking) | `secondbrain/goals/` |
| Proactive (detectors, ranking/noise-control, digest engine) | `secondbrain/proactive/` |
| Security (auth) + health + logging | `secondbrain/security/`, `secondbrain/health.py` |
| Tasks (decompose, prioritise, plan, research) | `secondbrain/tasks/` |
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

### Hardening: remote access, encryption, health (Phase 5)

```bash
sb doctor                          # preflight: config, disk, migrations, backends
sb auth set-password               # set the web UI username/password
```

- **Health:** `GET /health` (no auth) reports migration version, disk, counts, and
  backend reachability; `sb doctor` is the CLI preflight.
- **Secure remote access:** auth is OFF by default (local-only). To reach the UI
  from your phone/laptop, either keep `127.0.0.1` and run
  `tailscale serve https / 127.0.0.1:8765`, or set `[api].host` to your tailnet IP
  **and** `[security].require_auth = true` (the server *refuses* a non-loopback
  bind without auth). Loopback requests never need a login.
- **Encryption at rest:** `pip install -e ".[secure]"`, set
  `[security].encrypt_db = true` + a passphrase (in `config.local.toml` or
  `SB_SECURITY__DB_PASSPHRASE`) to open the SQLite DB via SQLCipher. Note this does
  **not** cover the WAL sidecar files or the raw audio on disk — use **FileVault**
  for true full-disk at-rest protection. See [`docs/OPERATIONS.md`](docs/OPERATIONS.md)
  for security caveats, troubleshooting, and tuning.

### Goals as an advanced to-do list (Phase 6)

```bash
sb goals add "Launch newsletter" -p 1
sb decompose <goal_id> --accept     # AI breaks the goal into a task tree (you approve)
sb plan --capacity 240              # propose today's prioritised plan (Eisenhower + score)
sb plan --accept                    # commit it to your day
sb task list ; sb task done <id>
sb task research <id>               # local graph-RAG research (--web for opt-in web)
```

Web: **Tasks** (`/tasks`) shows Today + backlog with one-tap done/research; the
goal page can decompose a goal into a plan. Action items detected in your
conversations can be promoted to tasks. Prioritisation is an Eisenhower matrix
(urgent×important) plus a weighted score; research is **local-first** (your own
knowledge graph) with **opt-in web research** (`[tasks].web_research_enabled` +
a search endpoint).

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full phased plan. Phases 1–6 are
shipped; remaining ideas: backup/export, data "forget", calendar time-blocking,
and diarization/profile quality tuning.

## License

MIT — see [LICENSE](LICENSE).
