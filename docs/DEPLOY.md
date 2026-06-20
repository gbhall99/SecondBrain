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
- Python 3.11+ (`brew install python@3.12`).

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
At minimum set the input device and review consent/retention:
```toml
[capture]
input_device = "Your Room Mic"      # name from `sb devices`

[consent]
raw_audio_retention_hours = 168     # raw FLAC auto-deleted after this (transcripts kept)
```
Check the effective config (secrets redacted): `sb config show`.

## 3. Optional backends (still fully offline)
- **Local LLM** (Q&A, morning brief, knowledge extraction):
  ```bash
  brew install ollama && ollama serve &
  ollama pull llama3.1:8b-instruct
  ```
  then set `[llm].backend = "ollama"` and `[extraction].enabled = true`.
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

## 6. Verify
```bash
sb status            # recording state, queue depth, disk
sb doctor            # migrations, disk, DB, backends, daemon heartbeat
open http://127.0.0.1:8765
```
Speak for a few minutes, then confirm transcripts appear (`sb show <today>`), the
menu bar shows "recording", and a reboot relaunches the agents.

## 7. Troubleshooting
- **Logs:** `data/daemon.{out,err}.log`, `data/web.{out,err}.log`,
  `data/menubar.{out,err}.log`.
- **Stuck pipeline:** `sb status` (queue depth), `sb queue` (inspect/reclaim),
  `sb drain --max-jobs 1` (process one job with a full traceback).
- **`/health` daemon check stale:** a loop died — launchd relaunches it; jobs
  stuck in `running` auto-reclaim after 30 min.
- See [docs/OPERATIONS.md](OPERATIONS.md) for security caveats and tuning knobs.
