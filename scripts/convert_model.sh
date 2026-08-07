#!/usr/bin/env bash
# Build a whisper.cpp ggml model from a Hugging Face Whisper checkpoint.
#
# This is the other half of scripts/fetch_model.sh: that script *places* a model
# you already have, this one *produces* one from upstream weights. They are
# separate on purpose -- conversion needs torch and transformers, a few
# gigabytes of scratch disk and a real network connection, none of which a
# runtime install should be made to carry.
#
#   scripts/convert_model.sh                    convert MODEL_HF_REPO -> MODEL_PATH
#   scripts/convert_model.sh --variant 25       Taiwanese Mandarin / code-switching
#   scripts/convert_model.sh --variant 26       Taiwanese Hokkien (Taigi)
#   scripts/convert_model.sh --repo openai/whisper-small
#   scripts/convert_model.sh --src ./Breeze-ASR-25   use a checkout you already have
#   scripts/convert_model.sh --quantize q5_0    shrink it for an edge device
#   scripts/convert_model.sh --keep-src         keep the download for a re-run
#
# Run it anywhere with disk and bandwidth -- it does not have to be the machine
# that serves inference. Copy the resulting .bin over and use fetch_model.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "${REPO_ROOT}/.env" ] && set -a && . "${REPO_ROOT}/.env" && set +a

ENGINE_DIR="${REPO_ROOT}/engine/whisper.cpp"
CONVERTER="${ENGINE_DIR}/models/convert-h5-to-ggml.py"
SRC_ROOT="${REPO_ROOT}/var/model-src"

HF_REPO="${MODEL_HF_REPO:-MediaTek-Research/Breeze-ASR-25}"
HF_REVISION="${MODEL_HF_REVISION:-main}"
# convert-h5-to-ggml.py needs exactly one file from the openai/whisper tree --
# the precomputed mel filterbank -- so fetch that rather than cloning the repo.
MEL_FILTERS_URL="${MEL_FILTERS_URL:-https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz}"

QUANTIZE=""
KEEP_SRC=false
FORCE=false
USE_F32=false
LOCAL_SRC=""
VARIANT=""
REPO_EXPLICIT=false
OUT_EXPLICIT=false

# Keep this table in step with MODEL_VARIANTS in breeze_hub/config.py -- the
# services resolve a variant name to the same filename this writes.
variant_repo() {
  case "$1" in
    25) echo "MediaTek-Research/Breeze-ASR-25" ;;
    26) echo "MediaTek-Research/Breeze-ASR-26" ;;
    *)  return 1 ;;
  esac
}
variant_filename() {
  case "$1" in
    25) echo "models/ggml-breeze-asr-25.bin" ;;
    26) echo "models/ggml-breeze-asr-26.bin" ;;
    *)  return 1 ;;
  esac
}

while [ $# -gt 0 ]; do
  case "$1" in
    --variant) VARIANT="${2:?--variant needs a value, e.g. 25}"; shift 2 ;;
    --repo) HF_REPO="${2:?--repo needs a value}"; REPO_EXPLICIT=true; shift 2 ;;
    --revision) HF_REVISION="${2:?--revision needs a value}"; shift 2 ;;
    --src) LOCAL_SRC="${2:?--src needs a path}"; shift 2 ;;
    --out) MODEL_PATH="${2:?--out needs a value}"; OUT_EXPLICIT=true; shift 2 ;;
    --quantize) QUANTIZE="${2:?--quantize needs a type, e.g. q5_0}"; shift 2 ;;
    --keep-src) KEEP_SRC=true; shift ;;
    --force) FORCE=true; shift ;;
    --f32) USE_F32=true; shift ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

# --variant is shorthand for the matching --repo and --out, so that the file on
# disk always says which upstream model it came from. Explicit flags still win.
if [ -n "${VARIANT}" ]; then
  VARIANT_REPO="$(variant_repo "${VARIANT}")" || {
    echo "unknown --variant ${VARIANT} (known: 25, 26)" >&2; exit 1; }
  [ "${REPO_EXPLICIT}" = true ] || HF_REPO="${VARIANT_REPO}"
  [ "${OUT_EXPLICIT}" = true ] || MODEL_PATH="$(variant_filename "${VARIANT}")"
fi

MODEL_PATH="${MODEL_PATH:-models/ggml-breeze-asr-${MODEL_VARIANT:-25}.bin}"
case "${MODEL_PATH}" in
  /*) ;;
  *) MODEL_PATH="${REPO_ROOT}/${MODEL_PATH}" ;;
esac

if [ -n "${LOCAL_SRC}" ]; then
  [ -d "${LOCAL_SRC}" ] || { echo "--src ${LOCAL_SRC} is not a directory" >&2; exit 1; }
  HF_SRC="$(cd "${LOCAL_SRC}" && pwd)"
  # Never delete a directory the user pointed us at.
  KEEP_SRC=true
else
  HF_SRC="${SRC_ROOT}/$(echo "${HF_REPO}" | tr '/' '_')"
fi
WHISPER_SRC="${SRC_ROOT}/openai-whisper"
BUILD_DIR="${SRC_ROOT}/build"

# --- 1. Preflight -----------------------------------------------------------
# Everything that can be known to fail is checked before the multi-gigabyte
# download starts, not after it.

echo "==> Checking prerequisites"

if [ -f "${MODEL_PATH}" ] && [ "${FORCE}" != true ]; then
  echo "Model already exists: ${MODEL_PATH}" >&2
  echo "Pass --force to rebuild it, or --out PATH to write elsewhere." >&2
  exit 1
fi

command -v python3 >/dev/null 2>&1 || { echo "python3 not found" >&2; exit 1; }

if [ ! -f "${CONVERTER}" ]; then
  cat >&2 <<EOF
whisper.cpp checkout missing -- expected the converter at:
  ${CONVERTER}

Run scripts/setup_engine.sh first (it clones whisper.cpp), or point this script
at an existing checkout by symlinking it to engine/whisper.cpp.
EOF
  exit 1
fi

# huggingface_hub is only needed for the download, so --src stays usable on a
# machine that has torch but no hub.
NEEDED="torch transformers numpy"
[ -n "${LOCAL_SRC}" ] || NEEDED="${NEEDED} huggingface_hub"

MISSING_PY="$(python3 - ${NEEDED} <<'PY'
import sys

missing = []
for mod in sys.argv[1:]:
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
print(" ".join(missing))
PY
)"

if [ -n "${MISSING_PY}" ]; then
  cat >&2 <<EOF
Conversion needs Python packages that are not installed: ${MISSING_PY}

  python3 -m pip install --user -r requirements-convert.txt

These are build-time only -- the services themselves never import torch, so
there is no need to install them on the machine that runs inference. On a
Jetson, install the NVIDIA-built torch wheel for your JetPack rather than the
one from PyPI, or just convert on a desktop and copy the .bin across.
EOF
  exit 3
fi

if [ -n "${LOCAL_SRC}" ]; then
  echo "  source   : ${HF_SRC} (local)"
else
  echo "  source   : ${HF_REPO} (${HF_REVISION})"
fi
echo "  output   : ${MODEL_PATH}"
echo "  scratch  : ${SRC_ROOT}"

mkdir -p "${SRC_ROOT}" "${BUILD_DIR}" "$(dirname "${MODEL_PATH}")"

# --- 2. Download the checkpoint ---------------------------------------------

if [ -n "${LOCAL_SRC}" ]; then
  echo "==> Using existing checkout, skipping download"
  [ -f "${HF_SRC}/config.json" ] || { echo "${HF_SRC} has no config.json -- not a Whisper checkout" >&2; exit 1; }
else
echo "==> Downloading ${HF_REPO}"
python3 - "${HF_REPO}" "${HF_REVISION}" "${HF_SRC}" <<'PY'
import os
import sys

from huggingface_hub import list_repo_files, snapshot_download

repo, revision, dest = sys.argv[1:4]
token = os.environ.get("HF_TOKEN") or None

files = list_repo_files(repo, revision=revision, token=token)

# Repos often ship safetensors *and* a legacy pytorch_model.bin. Taking both
# doubles a multi-gigabyte download for no benefit, so pick one.
weights = "*.safetensors" if any(f.endswith(".safetensors") for f in files) else "*.bin"
allow = ["*.json", "*.txt", "*.model", weights]
ignore = ["optimizer.bin", "training_args.bin", "*.msgpack", "*.h5", "*.onnx"]

print("  weight format: {}".format(weights))
snapshot_download(
    repo_id=repo,
    revision=revision,
    local_dir=dest,
    token=token,
    allow_patterns=allow,
    ignore_patterns=ignore,
)
PY
fi

# --- 3. Repair what the converter assumes but repos do not always ship -------
# convert-h5-to-ggml.py reads vocab.json and added_tokens.json directly. Newer
# exports often carry only the consolidated tokenizer.json, and the converter
# crashes on the missing file rather than falling back. Note that this writes
# into the source directory, including one passed with --src.

python3 - "${HF_SRC}" <<'PY'
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1])

if not (src / "vocab.json").exists():
    # Read the BPE vocabulary straight out of tokenizer.json rather than going
    # through transformers: its slow-tokenizer save path was dropped in v5, and
    # tokenizer.save_pretrained() there writes tokenizer.json again and reports
    # success while leaving vocab.json absent. `model.vocab` is exactly the base
    # vocabulary vocab.json holds -- added tokens live in a separate key, which
    # is what we want, since counting them here would shift every token id.
    print("  vocab.json absent, reconstructing it from tokenizer.json")
    spec_path = src / "tokenizer.json"
    if not spec_path.exists():
        sys.exit("neither vocab.json nor tokenizer.json is present in {}".format(src))
    vocab = json.loads(spec_path.read_text(encoding="utf-8")).get("model", {}).get("vocab")
    if not isinstance(vocab, dict):
        sys.exit("tokenizer.json in {} has no BPE vocabulary to recover".format(src))
    (src / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")

if not (src / "added_tokens.json").exists():
    # The converter loads this file and then never uses the result, so an empty
    # object satisfies it without affecting the output.
    print("  added_tokens.json absent, writing an empty one")
    (src / "added_tokens.json").write_text("{}", encoding="utf-8")

for required in ("config.json", "vocab.json"):
    if not (src / required).exists():
        sys.exit("{} is missing from the checkout and cannot be reconstructed".format(required))
PY

# --- 4. Mel filterbank ------------------------------------------------------

ASSETS="${WHISPER_SRC}/whisper/assets"
mkdir -p "${ASSETS}"
if [ ! -f "${ASSETS}/mel_filters.npz" ]; then
  echo "==> Fetching mel filterbank"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --progress-bar -o "${ASSETS}/mel_filters.npz.part" "${MEL_FILTERS_URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${ASSETS}/mel_filters.npz.part" "${MEL_FILTERS_URL}"
  else
    echo "neither curl nor wget is available" >&2
    exit 1
  fi
  mv "${ASSETS}/mel_filters.npz.part" "${ASSETS}/mel_filters.npz"
fi

# A proxy or captive portal happily returns an HTML error page with status 200.
# Loading the archive here turns that into a clear failure instead of a
# traceback thirty seconds into the conversion.
python3 - "${ASSETS}/mel_filters.npz" <<'PY'
import sys

import numpy as np

path = sys.argv[1]
try:
    with np.load(path) as f:
        keys = sorted(f.keys())
except Exception as exc:  # noqa: BLE001 -- any failure means the file is not usable
    sys.exit("{} is not a readable .npz ({}); delete it and re-run".format(path, exc))
if not any(k.startswith("mel_") for k in keys):
    sys.exit("{} has no mel_* array (found {})".format(path, keys))
print("  mel filters: {}".format(", ".join(keys)))
PY

# --- 5. Convert -------------------------------------------------------------

echo "==> Converting to ggml (this takes a few minutes and a lot of RAM)"
rm -f "${BUILD_DIR}/ggml-model.bin" "${BUILD_DIR}/ggml-model-f32.bin"
if [ "${USE_F32}" = true ]; then
  python3 "${CONVERTER}" "${HF_SRC}" "${WHISPER_SRC}" "${BUILD_DIR}" f32
  CONVERTED="${BUILD_DIR}/ggml-model-f32.bin"
else
  python3 "${CONVERTER}" "${HF_SRC}" "${WHISPER_SRC}" "${BUILD_DIR}"
  CONVERTED="${BUILD_DIR}/ggml-model.bin"
fi

[ -f "${CONVERTED}" ] || { echo "converter finished but ${CONVERTED} is missing" >&2; exit 1; }

# --- 6. Quantize (optional) -------------------------------------------------

if [ -n "${QUANTIZE}" ]; then
  # whisper.cpp renamed the tool to whisper-quantize; older checkouts still
  # call it quantize. Look for either, in both the modern bin/ layout and the
  # flat one older builds produced.
  QUANT_NAMES="whisper-quantize quantize"
  find_quant_bin() {
    local n
    for n in ${QUANT_NAMES}; do
      if [ -x "${ENGINE_DIR}/build/bin/${n}" ]; then echo "${ENGINE_DIR}/build/bin/${n}"; return 0; fi
      if [ -x "${ENGINE_DIR}/build/${n}" ]; then echo "${ENGINE_DIR}/build/${n}"; return 0; fi
    done
    return 1
  }
  QUANT_BIN="$(find_quant_bin || true)"
  if [ -z "${QUANT_BIN}" ]; then
    command -v cmake >/dev/null 2>&1 || { echo "cmake not found, needed to build the quantize tool" >&2; exit 1; }
    # Converting does not require a *built* engine, only the checkout, so the
    # build directory may never have been configured. Quantizing is CPU-only
    # and single-purpose, so configure a plain one rather than reaching for
    # the accelerator flags setup_engine.sh picks.
    if [ ! -f "${ENGINE_DIR}/build/CMakeCache.txt" ]; then
      echo "==> Configuring whisper.cpp (build directory not set up yet)"
      cmake -S "${ENGINE_DIR}" -B "${ENGINE_DIR}/build" \
        -DCMAKE_BUILD_TYPE=Release -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON
    fi
    echo "==> Building whisper.cpp's quantize tool"
    BUILT=false
    for TARGET in ${QUANT_NAMES}; do
      if cmake --build "${ENGINE_DIR}/build" --config Release --target "${TARGET}" 2>/dev/null; then
        BUILT=true
        break
      fi
    done
    [ "${BUILT}" = true ] || {
      echo "could not build a quantize target (tried: ${QUANT_NAMES})" >&2
      exit 1
    }
    QUANT_BIN="$(find_quant_bin || true)"
  fi
  [ -n "${QUANT_BIN}" ] && [ -x "${QUANT_BIN}" ] || { echo "quantize tool not found after building" >&2; exit 1; }

  echo "==> Quantizing to ${QUANTIZE}"
  QUANTIZED="${BUILD_DIR}/ggml-model-${QUANTIZE}.bin"
  "${QUANT_BIN}" "${CONVERTED}" "${QUANTIZED}" "${QUANTIZE}"
  rm -f "${CONVERTED}"
  CONVERTED="${QUANTIZED}"
fi

# --- 7. Install -------------------------------------------------------------

mv "${CONVERTED}" "${MODEL_PATH}"
rm -rf "${BUILD_DIR}"

if [ -n "${LOCAL_SRC}" ]; then
  :  # --src points at someone else's directory; leave it exactly where it is.
elif [ "${KEEP_SRC}" = true ]; then
  echo "==> Keeping the download at ${HF_SRC} (--keep-src)"
else
  rm -rf "${HF_SRC}"
fi

SIZE="$(du -h "${MODEL_PATH}" | awk '{print $1}')"
echo "==> Model ready: ${MODEL_PATH} (${SIZE})"

if command -v sha256sum >/dev/null 2>&1; then
  SHA="$(sha256sum "${MODEL_PATH}" | awk '{print $1}')"
  cat <<EOF

Pin the result so fetch_model.sh can verify future copies of it -- add to .env:

  MODEL_PATH=${MODEL_PATH#"${REPO_ROOT}/"}
  MODEL_SHA256=${SHA}
EOF
fi
