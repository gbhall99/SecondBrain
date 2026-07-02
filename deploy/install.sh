#!/usr/bin/env bash
#
# One-command bring-up for SecondBrain on an Apple Silicon Mac mini.
# Idempotent: safe to re-run. See docs/DEPLOY.md for the full guide.
#
#   curl-free usage:  git clone <repo> SecondBrain && cd SecondBrain
#                     ./deploy/install.sh
#
set -euo pipefail

# Resolve the repo root (parent of this script's deploy/ directory).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!  \033[0m %s\n' "$*"; }

# --- 1. Platform check ------------------------------------------------------
if [[ "$(uname -s)" != "Darwin" ]]; then
  warn "This installer targets macOS; on $(uname -s) it will likely fail. Continuing anyway."
elif [[ "$(uname -m)" != "arm64" ]]; then
  warn "Not Apple Silicon (uname -m=$(uname -m)); MLX transcription needs arm64."
fi

# --- 2. Python venv ---------------------------------------------------------
# SecondBrain needs Python >= 3.11. macOS ships an older system python3 (e.g.
# 3.9), so pick a suitable interpreter rather than assuming `python3` is new enough.
py_ok() { "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)' 2>/dev/null; }

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 || ! py_ok "${PYTHON_BIN}"; then
  found=""
  for cand in python3.13 python3.12 python3.11; do
    if command -v "${cand}" >/dev/null 2>&1 && py_ok "${cand}"; then found="${cand}"; break; fi
  done
  if [[ -n "${found}" ]]; then
    PYTHON_BIN="${found}"
  else
    have="$(command -v "${PYTHON_BIN}" >/dev/null 2>&1 && "${PYTHON_BIN}" -V 2>&1 || echo 'none found')"
    warn "SecondBrain needs Python 3.11+ (have: ${have})."
    warn "Install it and re-run:  brew install python@3.12  &&  ./deploy/install.sh"
    exit 1
  fi
fi
PYVER="$(${PYTHON_BIN} -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
say "Using ${PYTHON_BIN} (Python ${PYVER})"

# Recreate the venv if it's missing or was built with an unsuitable Python (self-heal
# a stale .venv left by an earlier run on the system python).
if [[ -d .venv ]] && ! py_ok .venv/bin/python; then
  warn "Existing .venv uses an unsupported Python; recreating with ${PYTHON_BIN}."
  rm -rf .venv
fi
if [[ ! -d .venv ]]; then
  say "Creating virtualenv at .venv"
  "${PYTHON_BIN}" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# --- 3. PortAudio (capture backend) ----------------------------------------
if ! "${PYTHON_BIN}" -c 'import ctypes.util,sys; sys.exit(0 if ctypes.util.find_library("portaudio") else 1)' 2>/dev/null; then
  if command -v brew >/dev/null 2>&1; then
    say "Installing PortAudio via Homebrew"
    brew install portaudio
  else
    warn "PortAudio not found and Homebrew is unavailable. Install PortAudio, then re-run."
  fi
fi

# --- 4. Python dependencies -------------------------------------------------
say "Installing SecondBrain (.[audio,ml,mac])"
CONSTRAINTS=()
[[ -f constraints.txt ]] && CONSTRAINTS=(-c constraints.txt)
pip install --upgrade pip >/dev/null
pip install -e ".[audio,ml,mac]" "${CONSTRAINTS[@]}"

# --- 5. Initialise data dir + database -------------------------------------
say "Initialising data directory and database"
sb init
sb repair || warn "repair reported an issue (see above)"  # self-heal; non-fatal

# Seed a local override file from the template so the user edits a real file.
# The template enables the AI features (diarization, LLM extraction, proactive);
# the steps below set up their prerequisites so first run is turnkey.
if [[ ! -f config.local.toml && -f config.local.toml.example ]]; then
  cp config.local.toml.example config.local.toml
  say "Created config.local.toml (AI features enabled; set your mic in [capture])"
fi

# --- 6. AI feature setup (Ollama + HuggingFace) ----------------------------
# Skippable for a capture-only install: SB_SKIP_AI=1 ./deploy/install.sh
AI_MODEL="${SB_OLLAMA_MODEL:-llama3.1:8b-instruct}"
if [[ "${SB_SKIP_AI:-0}" == "1" ]]; then
  warn "SB_SKIP_AI=1 — skipping Ollama + HuggingFace setup. Disable AI features in "
  warn "config.local.toml ([diarization]/[extraction]/[proactive] enabled=false) if unused."
else
  # Ollama: local LLM for the knowledge graph, Q&A, and the proactive brief.
  if ! command -v ollama >/dev/null 2>&1; then
    if command -v brew >/dev/null 2>&1; then
      say "Installing Ollama via Homebrew"
      brew install ollama || warn "Ollama install failed — install it manually from https://ollama.com"
    else
      warn "Ollama not found and Homebrew unavailable. Install from https://ollama.com, then re-run."
    fi
  fi
  if command -v ollama >/dev/null 2>&1; then
    # Start Ollama as a background service (persists across reboots via brew) so the
    # daemon's extraction jobs can reach it.
    if command -v brew >/dev/null 2>&1; then
      brew services start ollama >/dev/null 2>&1 || true
    fi
    if ! ollama list >/dev/null 2>&1; then
      nohup ollama serve >/dev/null 2>&1 &
    fi
    # Wait (up to ~30s) for the server to accept connections before pulling, so the
    # first-run pull doesn't race the server start.
    ready=0
    for _ in $(seq 1 30); do
      if ollama list >/dev/null 2>&1; then ready=1; break; fi
      sleep 1
    done
    if [[ "${ready}" == "1" ]]; then
      say "Pulling Ollama model ${AI_MODEL} (one-time download)"
      ollama pull "${AI_MODEL}" || warn "Model pull failed — run \`ollama pull ${AI_MODEL}\` later."
    else
      warn "Ollama did not become ready — run \`ollama serve\`, then \`ollama pull ${AI_MODEL}\`."
    fi
  fi

  # HuggingFace token: required for diarization (gated pyannote models), which in
  # turn feeds the knowledge graph. Skip if already set; prompt only when interactive.
  if [[ -f config.local.toml ]] \
     && grep -Eq '^[[:space:]]*hf_token[[:space:]]*=' config.local.toml \
     && ! grep -Eq '^[[:space:]]*hf_token[[:space:]]*=[[:space:]]*""[[:space:]]*$' config.local.toml; then
    say "HuggingFace token already set in config.local.toml — skipping prompt."
  else
    if [[ -t 0 ]]; then
      say "Diarization needs a HuggingFace token (accept terms at"
      say "  https://huggingface.co/pyannote/speaker-diarization-3.1 first)."
      printf '    Paste your HF read token (or press Enter to skip): '
      read -rs HF_TOKEN_INPUT || HF_TOKEN_INPUT=""
      printf '\n'
    else
      HF_TOKEN_INPUT="${HF_TOKEN:-}"
    fi
    if [[ -n "${HF_TOKEN_INPUT:-}" ]]; then
      sb config set-hf-token "${HF_TOKEN_INPUT}"   # safe, tested TOML edit
      say "Downloading diarization models (sb speaker setup)…"
      sb speaker setup || warn "sb speaker setup failed — check the token + accepted model terms."
    else
      warn "No HF token provided — diarization and the knowledge graph stay idle until you set"
      warn "[diarization].hf_token in config.local.toml (then run \`sb speaker setup\`)."
    fi
  fi
fi

# --- 7. Next steps ----------------------------------------------------------
cat <<'EOF'

==> Almost there. Finish configuring SecondBrain:

  1. Pick your room mic:
       sb devices
     then set [capture].input_device in config.local.toml (gitignored).
     Review [consent].raw_audio_retention_hours and [security] while you're there.

  2. Enroll your voice (so the knowledge graph knows which speaker is you):
       sb speaker enroll-owner

  3. Install the always-on launchd agents (survives reboot):
       sb deploy launchd --load --include-menubar
     macOS will prompt for Microphone permission on first run.

  4. Verify (and self-heal anything fixable):
       sb doctor --fix    # repairs, then checks mic, Ollama, migrations, disk…
       open http://127.0.0.1:8765

If anything ever looks off later, just re-run `./deploy/install.sh` (idempotent)
or `sb repair` — both are safe and self-healing.

AI features (diarization, knowledge graph, proactive brief) are enabled in
config.local.toml; this script set up Ollama + the HF token where possible.
Full guide: docs/DEPLOY.md
EOF

say "Running preflight (sb doctor --fix)…"
sb doctor --fix || warn "Some checks failed — see docs/DEPLOY.md (this is expected before you set a mic)."
