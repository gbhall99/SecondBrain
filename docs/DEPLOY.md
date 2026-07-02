# Deploying SecondBrain on a Mac mini

This is the canonical, end-to-end guide for running SecondBrain always-on on an
Apple Silicon Mac. Everything stays local — audio, transcripts, models, and the
LLM all run on the machine.

> **Legal:** recording-consent law is jurisdiction-dependent (many US states and
> the EU require all-party consent). You are responsible for compliance — verify
> your local law before deploying.

## 0. Prerequisites
- Apple Silicon Mac (M-series) running macOS.
- [Homebrew](https://brew.sh) (for PortAudio and, optionally, Ollama).
- Python 3.11+ (`brew install python@3.12`). The macOS **system** `python3` (3.9)
  is too old — the installer auto-detects a `python3.11`/`3.12`/`3.13` on your PATH,
  and tells you to `brew install python@3.12` if none is found.

## 1. One-command install
```bash
git clone <repo> SecondBrain && cd SecondBrain
./deploy/install.sh
```
The installer is idempotent (safe to re-run). It creates `.venv`, installs
PortAudio, installs SecondBrain with the `[audio,ml,mac]` extras, runs `sb init`,
and prints the remaining steps. To do it by hand instead:
```bash
python3.12 -m venv .venv && source .venv/bin/activate
brew install portaudio
pip install -e ".[audio,ml,mac]" -c constraints.txt
sb init
```

## 2. Configure
Put machine-specific settings and all secrets in `config.local.toml` (gitignored),
which overrides the committed `config.toml`.
```bash
sb devices            # find your room mic
```
`install.sh` seeds `config.local.toml` from `config.local.toml.example`. Edit it —
at minimum set the input device and review consent/retention:
```toml
[capture]
input_device = "Your Room Mic"      # name (or substring) from `sb devices`

[consent]
raw_audio_retention_hours = 168     # raw FLAC auto-deleted after this (transcripts kept)
```
Check the effective config (secrets redacted): `sb config show`.

## 2a. Grant Microphone permission (required)
macOS gates microphone access (TCC). The **first time** the daemon opens the mic,
macOS shows a Microphone permission prompt — **allow it**. If the daemon is started
headless by launchd you may not see a prompt; in that case grant it manually:
**System Settings → Privacy & Security → Microphone** and enable the entry for your
terminal / Python. Without this, capture silently fails (the daemon retries
forever and records nothing). `sb doctor` includes a **microphone** check that
flags a missing or unconfigured input device.

## 2b. First run downloads models (network needed once)
On the first capture, the transcription model (Parakeet) and the VAD model (Silero)
auto-download from HuggingFace (a few hundred MB) into the HuggingFace cache;
semantic-search embeddings (`bge-small`) download on first index. After that
everything runs **fully offline**. Ensure outbound access to HuggingFace for this
one-time fetch (or pre-seed the cache).

## What's on vs off by default
The seeded `config.local.toml` enables the AI features, and `install.sh` sets up
their prerequisites (Ollama + HF token). To run **capture-only**, set
`enabled = false` under `[diarization]`/`[extraction]`/`[proactive]` (or
`SB_SKIP_AI=1 ./deploy/install.sh`).

| Feature | Default (local) | Prerequisite |
|---------|-----------------|--------------|
| Capture + transcription + VAD | **ON** | set the mic; grant permission |
| Full-text + semantic search | **ON** | included in the `ml` extra |
| Speaker diarization (who-spoke) | **ON** | HF token (install.sh prompts; gated pyannote terms) |
| Knowledge graph / Q&A / proactive brief | **ON** | Ollama running + model pulled (install.sh does this) |
| Remote access · at-rest encryption | OFF | Tailscale + auth · `[secure]` extra + passphrase |

> **The chain:** diarization (who-spoke) → extraction (knowledge graph) → proactive
> brief. Extraction only runs on diarized conversations, so **a HF token is required
> for the knowledge graph to populate.** No token / no Ollama ⇒ those jobs stay idle
> (capture + search still work).

## 3. AI backends (set up automatically by install.sh, still fully offline)
`install.sh` installs/starts Ollama, pulls the model, and prompts for the HF token.
To do it by hand (or on a re-run where `config.local.toml` already exists):
- **Local LLM** (Q&A, morning brief, knowledge extraction):
  ```bash
  brew install ollama && brew services start ollama
  ollama pull llama3.1:8b-instruct
  ```
  ensure `[llm].backend = "ollama"` and `[extraction].enabled = true` (default in
  the seeded local config).
- **Speaker diarization** (pyannote): create a HuggingFace read token, accept the
  gated-model terms, then:
  ```bash
  export HF_TOKEN=...        # or put it in config.local.toml
  sb speaker setup
  sb speaker enroll-owner    # ~30s guided recording of your voice
  ```

## 4. Run always-on (launchd)
`sb deploy` fills the `deploy/*.plist` templates with this venv's Python and the
repo path, installs them to `~/Library/LaunchAgents/`, and loads them — no manual
editing. It runs a `sb doctor` preflight first and refuses on a broken setup.
```bash
sb deploy launchd --load --include-menubar
```
This installs three agents:

| Agent | What it runs | Notes |
|-------|--------------|-------|
| `com.secondbrain.daemon` | capture + transcribe + maintenance | background |
| `com.secondbrain.web` | `sb serve` (web UI/API) | background, `127.0.0.1:8765` |
| `com.secondbrain.menubar` | menu bar indicator + pause | needs a logged-in GUI session |

macOS prompts for **Microphone** permission on first run. To stop/remove them:
```bash
sb deploy launchd --unload --include-menubar
```
For a foreground run while testing, skip launchd and use `sb start`, `sb serve`,
and `python -m secondbrain.menubar.app` in separate terminals.

## 5. Remote access (optional, Tailscale)
The web UI binds to `127.0.0.1` only. To reach it from your phone/laptop:
- **Recommended:** `tailscale serve https / 127.0.0.1:8765` — TLS, tailnet-only,
  no app changes; the session cookie is marked `Secure` over HTTPS.
- **Alternative:** bind to your tailnet IP and require auth:
  ```toml
  [api]
  host = "100.x.y.z"          # your tailnet IP / MagicDNS name
  [security]
  require_auth = true
  ```
  ```bash
  sb auth set-password
  ```
The server **fail-closes**: it refuses to bind off-loopback without
`require_auth` + a password. Avoid `tailscale funnel` (exposes publicly).

## 6. Verify (on-device checklist)
Development and CI run on Linux with mock backends, so the macOS-only paths
(CoreAudio capture, MLX transcription, pyannote, `rumps`, `launchctl`, the mic TCC
prompt) are **only verified here, on the Mac.** Run through this once:
```bash
sb status            # recording state, queue depth, disk
sb doctor            # migrations, disk, DB, backends, microphone, daemon heartbeat
open http://127.0.0.1:8765
```
1. `sb doctor` is all green (note the **microphone** check — see caveat below).
2. Speak for a few minutes → `sb show <today>` shows transcripts; the menu bar
   shows "recording".
3. Hold a short multi-person conversation → after it closes it diarizes
   (`sb timeline <today>` attributes speakers); then `sb projects` / `/graph`
   show extracted entities and `sb ask "…"` returns a cited answer.
4. Reboot → all three launchd agents relaunch.

> **Microphone-check caveat:** `sb doctor`'s `microphone` check is a *presence*
> proxy — it catches a missing or misconfigured input device, but macOS may still
> *list* the mic while denying access (TCC), which it cannot detect. If capture
> records nothing, grant access in **System Settings → Privacy & Security →
> Microphone** even when the check passes.

## 7. Troubleshooting
- **First stop — self-heal:** `sb repair` (or `sb doctor --fix`). It safely and
  idempotently recreates missing data dirs, brings the schema to head, seeds
  `config.local.toml`, re-queues jobs a crashed worker left mid-run, and
  checkpoints the WAL. It never deletes data; genuine DB corruption is reported so
  you can `sb restore` a backup. Re-running `./deploy/install.sh` is equally safe.
  The daemon also runs this repair automatically every time it (re)starts.
- **Logs:** `data/daemon.{out,err}.log`, `data/web.{out,err}.log`,
  `data/menubar.{out,err}.log`.
- **Stuck pipeline:** `sb status` (queue depth), `sb queue` (inspect/reclaim),
  `sb drain --max-jobs 1` (process one job with a full traceback).
- **`/health` daemon check stale:** a loop died — launchd relaunches it; jobs
  stuck in `running` auto-reclaim after 30 min.
- See [docs/OPERATIONS.md](OPERATIONS.md) for security caveats and tuning knobs.
