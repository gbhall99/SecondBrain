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
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  warn "python3 not found. Install Python 3.11+ (e.g. \`brew install python@3.12\`) and re-run."
  exit 1
fi
PYVER="$(${PYTHON_BIN} -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
say "Using ${PYTHON_BIN} (Python ${PYVER})"

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
    # Start it as a background service (persists across reboots via brew) so the
    # daemon's extraction jobs can reach it.
    if command -v brew >/dev/null 2>&1; then
      brew services start ollama >/dev/null 2>&1 || ollama serve >/dev/null 2>&1 &
    else
      ollama serve >/dev/null 2>&1 &
    fi
    say "Pulling Ollama model ${AI_MODEL} (one-time download)"
    ollama pull "${AI_MODEL}" || warn "Model pull failed — run \`ollama pull ${AI_MODEL}\` later."
  fi

  # HuggingFace token: required for diarization (gated pyannote models), which in
  # turn feeds the knowledge graph. Prompt only when interactive.
  if [[ -t 0 ]]; then
    say "Diarization needs a HuggingFace token (accept terms on "
    say "  https://huggingface.co/pyannote/speaker-diarization-3.1 first)."
    printf '    Paste your HF read token (or press Enter to skip): '
    read -rs HF_TOKEN_INPUT || HF_TOKEN_INPUT=""
    printf '\n'
  else
    HF_TOKEN_INPUT="${HF_TOKEN:-}"
  fi
  if [[ -n "${HF_TOKEN_INPUT}" ]]; then
    # Safe single replacement of the placeholder line in config.local.toml.
    HF_TOKEN_INPUT="${HF_TOKEN_INPUT}" "${PYTHON_BIN}" - <<'PY'
import os, pathlib
p = pathlib.Path("config.local.toml")
text = p.read_text()
tok = os.environ["HF_TOKEN_INPUT"]
if 'hf_token = ""' in text:
    p.write_text(text.replace('hf_token = ""', f'hf_token = "{tok}"', 1))
    print("    wrote hf_token to config.local.toml")
else:
    print("    note: could not find hf_token placeholder; set it manually")
PY
    say "Downloading diarization models (sb speaker setup)…"
    sb speaker setup || warn "sb speaker setup failed — check the token + that you accepted the model terms."
  else
    warn "No HF token provided — diarization and the knowledge graph stay idle until you set"
    warn "[diarization].hf_token in config.local.toml (then run \`sb speaker setup\`)."
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

  4. Verify:
       sb doctor          # checks mic, Ollama, migrations, disk…
       open http://127.0.0.1:8765

AI features (diarization, knowledge graph, proactive brief) are enabled in
config.local.toml; this script set up Ollama + the HF token where possible.
Full guide: docs/DEPLOY.md
EOF

say "Running preflight (sb doctor)…"
sb doctor || warn "Some checks failed — see docs/DEPLOY.md (this is expected before you set a mic)."
