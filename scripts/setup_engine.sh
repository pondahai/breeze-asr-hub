#!/usr/bin/env bash
# Fetch and build whisper.cpp with flags chosen from hardware.json.
#
# CMake only. whisper.cpp deprecated its Makefile, and the old
# `make || cmake -B build && cmake --build build` idiom is a trap: `||` and `&&`
# bind left-to-right, so a *successful* make still ran the final build step
# against a directory that was never configured.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_DIR="${REPO_ROOT}/engine/whisper.cpp"
HARDWARE_JSON="${REPO_ROOT}/hardware.json"

WHISPER_REPO="${WHISPER_REPO:-https://github.com/ggml-org/whisper.cpp.git}"
WHISPER_REF="${WHISPER_REF:-master}"

if [ ! -f "${HARDWARE_JSON}" ]; then
  echo "hardware.json missing -- run scripts/probe_hardware.sh first" >&2
  exit 1
fi

read_json() {
  python3 -c "import json,sys;d=json.load(open(sys.argv[1]));
k=sys.argv[2].split('.')
for part in k: d=d.get(part, '') if isinstance(d, dict) else ''
print(d if d not in (None, False) else ('' if d is not True else 'true'))" "${HARDWARE_JSON}" "$1"
}

ACCELERATOR="$(read_json accelerator)"
CUDA_ARCH="$(read_json cuda.arch)"
CPU_CORES="$(read_json cpu_cores)"
HAS_BLAS="$(read_json has_openblas)"
JOBS="${JOBS:-${CPU_CORES:-4}}"

echo "==> Setting up whisper.cpp (accelerator: ${ACCELERATOR}, jobs: ${JOBS})"

mkdir -p "${REPO_ROOT}/engine"
if [ ! -d "${ENGINE_DIR}/.git" ]; then
  echo "  cloning ${WHISPER_REPO}"
  git clone --depth 1 --branch "${WHISPER_REF}" "${WHISPER_REPO}" "${ENGINE_DIR}"
else
  echo "  reusing existing checkout at ${ENGINE_DIR}"
fi

cd "${ENGINE_DIR}"

CMAKE_ARGS=(-B build -DCMAKE_BUILD_TYPE=Release -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON)

if [ "${ACCELERATOR}" = "cuda" ]; then
  CMAKE_ARGS+=(-DGGML_CUDA=ON)
  if [ -n "${CUDA_ARCH}" ]; then
    CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}")
  fi
  # Xavier-class parts OOM during multimodal inference when ggml uses CUDA
  # virtual memory management; disabling it costs nothing measurable here.
  if [ "${CUDA_ARCH}" = "72" ]; then
    CMAKE_ARGS+=(-DGGML_CUDA_NO_VMM=ON)
  fi
  echo "  building with CUDA${CUDA_ARCH:+ (sm_${CUDA_ARCH})}"
else
  if [ "${HAS_BLAS}" = "true" ]; then
    CMAKE_ARGS+=(-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS)
    echo "  building for CPU with OpenBLAS"
  else
    echo "  building for CPU (no OpenBLAS found; install libopenblas-dev to speed this up)"
  fi
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build build --config Release -j "${JOBS}" --target whisper-cli

BINARY="${ENGINE_DIR}/build/bin/whisper-cli"
if [ ! -x "${BINARY}" ]; then
  # Some generators drop binaries directly in build/ instead of build/bin/.
  ALT="${ENGINE_DIR}/build/whisper-cli"
  if [ -x "${ALT}" ]; then
    mkdir -p "$(dirname "${BINARY}")"
    ln -sf "${ALT}" "${BINARY}"
  else
    echo "build finished but ${BINARY} is missing" >&2
    exit 1
  fi
fi

echo "==> whisper-cli ready: ${BINARY}"
