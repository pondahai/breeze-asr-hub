#!/usr/bin/env bash
# One command to go from a fresh clone to a runnable install.
#
#   scripts/install.sh                 full install
#   scripts/install.sh --skip-deps     skip apt (no sudo available)
#   scripts/install.sh --model PATH    also place a model from a local file
#
# Every step is idempotent: re-running after a failure resumes rather than
# rebuilding from scratch.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SKIP_DEPS=false
MODEL_SOURCE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-deps) SKIP_DEPS=true; shift ;;
    --model) MODEL_SOURCE="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

step() { printf '\n\033[1m[%s/5] %s\033[0m\n' "$1" "$2"; }

# --- 1. System packages -----------------------------------------------------
step 1 "System dependencies"
if [ "${SKIP_DEPS}" = true ]; then
  echo "  skipped (--skip-deps)"
elif command -v apt-get >/dev/null 2>&1; then
  PACKAGES="build-essential cmake git ffmpeg alsa-utils python3-pip libopenblas-dev"
  MISSING=""
  for pkg in ${PACKAGES}; do
    dpkg -s "${pkg}" >/dev/null 2>&1 || MISSING="${MISSING} ${pkg}"
  done
  if [ -n "${MISSING}" ]; then
    echo "  installing:${MISSING}"
    sudo apt-get update -qq
    # shellcheck disable=SC2086
    sudo apt-get install -y ${MISSING}
  else
    echo "  all present"
  fi
else
  echo "  no apt-get -- install manually: cmake git ffmpeg alsa-utils python3-pip"
fi

# --- 2. Configuration -------------------------------------------------------
step 2 "Configuration"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "  created .env from .env.example"
else
  echo "  .env already exists, leaving it alone"
fi

# --- 3. Hardware probe ------------------------------------------------------
step 3 "Hardware probe"
bash scripts/probe_hardware.sh

# --- 4. Python + engine -----------------------------------------------------
step 4 "Python packages and inference engine"
python3 -m pip install --user -q -r requirements.txt || \
  echo "  WARNING: some Python packages failed; check 'pip3 install -r requirements.txt'"
bash scripts/setup_engine.sh

# --- 5. Model ---------------------------------------------------------------
step 5 "ASR model"
if [ -n "${MODEL_SOURCE}" ]; then
  bash scripts/fetch_model.sh "${MODEL_SOURCE}"
else
  bash scripts/fetch_model.sh || echo "  (continuing without a model -- see the message above)"
fi

cat <<'EOF'

============================================================
Install complete.

Next steps
  1. Calibrate the microphone for this room and mic:
       python3 -m breeze_hub.calibrate --write
  2. Run in the foreground to check it works:
       scripts/run.sh realtime
       scripts/run.sh batch
  3. Install as boot services once happy:
       scripts/service.sh install
       scripts/service.sh status
============================================================
EOF
