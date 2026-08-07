#!/usr/bin/env bash
# Put a Breeze ASR ggml model in place.
#
# Deliberately does not hard-code a download URL: nobody publishes a ready-made
# ggml build of Breeze ASR, and inventing a plausible-looking Hugging Face link
# that 404s is worse than saying so. Three ways to get one:
#
#     scripts/fetch_model.sh /path/to/ggml-breeze-asr-26.bin   local file
#     MODEL_URL= in .env, then scripts/fetch_model.sh          your own mirror
#     scripts/fetch_model.sh --convert                         build from HF
#
# --convert hands off to scripts/convert_model.sh, which downloads the upstream
# checkpoint and converts it. That needs torch and a few GB of scratch space,
# which is why it is opt-in rather than the default.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "${REPO_ROOT}/.env" ] && set -a && . "${REPO_ROOT}/.env" && set +a

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/ggml-breeze-asr-26.bin}"
case "${MODEL_PATH}" in
  /*) ;;
  *) MODEL_PATH="${REPO_ROOT}/${MODEL_PATH}" ;;
esac
MODEL_URL="${MODEL_URL:-}"
MODEL_SHA256="${MODEL_SHA256:-}"
SOURCE="${1:-}"

usage() {
  cat <<EOF
Usage: scripts/fetch_model.sh [PATH | --convert [ARGS...] | --help]

  PATH        copy an existing ggml model into place
  --convert   build one from the upstream Hugging Face checkpoint
              (remaining arguments are passed to scripts/convert_model.sh)
  --help      show this message

With no arguments, uses MODEL_URL from .env if set, or reports what is
missing. The model lands at MODEL_PATH:

  ${MODEL_PATH}
EOF
}

case "${SOURCE}" in
  --convert)
    shift
    exec bash "${REPO_ROOT}/scripts/convert_model.sh" "$@"
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  -*)
    echo "unknown option: ${SOURCE}" >&2
    echo >&2
    usage >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "${MODEL_PATH}")"

verify() {
  [ -z "${MODEL_SHA256}" ] && return 0
  command -v sha256sum >/dev/null 2>&1 || { echo "  sha256sum unavailable, skipping checksum"; return 0; }
  echo "  verifying checksum"
  local actual
  actual="$(sha256sum "${MODEL_PATH}" | awk '{print $1}')"
  if [ "${actual}" != "${MODEL_SHA256}" ]; then
    echo "checksum mismatch:" >&2
    echo "  expected ${MODEL_SHA256}" >&2
    echo "  actual   ${actual}" >&2
    exit 1
  fi
}

if [ -f "${MODEL_PATH}" ] && [ -z "${SOURCE}" ]; then
  SIZE="$(du -h "${MODEL_PATH}" | awk '{print $1}')"
  echo "==> Model already present: ${MODEL_PATH} (${SIZE})"
  verify
  exit 0
fi

if [ -n "${SOURCE}" ]; then
  echo "==> Copying model from ${SOURCE}"
  cp "${SOURCE}" "${MODEL_PATH}"
elif [ -n "${MODEL_URL}" ]; then
  echo "==> Downloading model from ${MODEL_URL}"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --progress-bar -o "${MODEL_PATH}.part" "${MODEL_URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${MODEL_PATH}.part" "${MODEL_URL}"
  else
    echo "neither curl nor wget is available" >&2
    exit 1
  fi
  mv "${MODEL_PATH}.part" "${MODEL_PATH}"
else
  cat >&2 <<EOF

No model available and no source given.

  expected at : ${MODEL_PATH}

Provide one of:
  1. a local file   ->  scripts/fetch_model.sh /path/to/ggml-breeze-asr-26.bin
  2. a download URL ->  set MODEL_URL= in .env, then re-run
  3. build your own ->  scripts/fetch_model.sh --convert

Option 3 downloads ${MODEL_HF_REPO:-MediaTek-Research/Breeze-ASR-25} from Hugging Face and converts
it to ggml. It needs torch, transformers and a few GB of scratch space (see
requirements-convert.txt), and it does not have to run on this machine -- build
the .bin anywhere and come back to option 1.

Any whisper.cpp-compatible ggml model works; Breeze ASR is simply the one tuned
for Taiwanese-accented Mandarin. For a quick smoke test you can point MODEL_PATH
at a stock model from whisper.cpp's models/download-ggml-model.sh instead.

EOF
  exit 2
fi

verify
echo "==> Model ready: ${MODEL_PATH} ($(du -h "${MODEL_PATH}" | awk '{print $1}'))"
