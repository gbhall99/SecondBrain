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

# --- 6. Next steps ----------------------------------------------------------
cat <<'EOF'

==> Almost there. Finish configuring SecondBrain:

  1. Pick your room mic:
       sb devices
     then set [capture].input_device in config.local.toml (gitignored).
     Review [consent].raw_audio_retention_hours and [security] while you're there.

  2. (Optional, fully offline) Local LLM for Q&A, briefs and extraction:
       brew install ollama && ollama serve &
       ollama pull llama3.1:8b-instruct
     then enable [llm] and [extraction] in config.

  3. (Optional) Speaker diarization needs a HuggingFace token:
       export HF_TOKEN=... ; sb speaker setup

  4. Install the always-on launchd agents (survives reboot):
       sb deploy launchd --load --include-menubar
     macOS will prompt for Microphone permission on first run.

  5. Verify:
       sb doctor
       open http://127.0.0.1:8765

Full guide: docs/DEPLOY.md
EOF

say "Running preflight (sb doctor)…"
sb doctor || warn "Some checks failed — see docs/DEPLOY.md (this is expected before you set a mic)."
