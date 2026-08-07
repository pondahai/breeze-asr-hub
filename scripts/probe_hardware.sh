#!/usr/bin/env bash
# Detect what this machine can do and write hardware.json.
#
# Everything downstream -- the build flags, the systemd units, the Python
# services, the web UI's feature toggles -- reads that one file. Re-run this
# after changing hardware and the rest of the project follows.
#
# Detection order matters. Jetson is checked BEFORE nvidia-smi because JetPack
# ships no nvidia-smi at all: probing for it first and giving up would misreport
# every Jetson as a CPU-only box.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${REPO_ROOT}/hardware.json"

# nvcc lives outside the default PATH on Jetson, and non-interactive SSH does
# not source the profile that would add it. Without this line CUDA detection
# fails on exactly the machines that need it most.
export PATH="${PATH}:/usr/local/cuda/bin"

say() { printf '  %s\n' "$*"; }

echo "==> Probing hardware"

# --- OS / CPU ---------------------------------------------------------------
ARCH="$(uname -m)"
OS_NAME="$(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME}")"
[ -z "${OS_NAME}" ] && OS_NAME="$(uname -s)"
CPU_CORES="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
MEM_KB="$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
MEM_GB=$(( MEM_KB / 1024 / 1024 ))
[ "${MEM_GB}" -lt 1 ] && MEM_GB=1

say "os        : ${OS_NAME} (${ARCH})"
say "cpu       : ${CPU_CORES} cores"
say "memory    : ~${MEM_GB} GB"

# --- Jetson -----------------------------------------------------------------
IS_JETSON=false
PLATFORM="${OS_NAME}"
CUDA_ARCH=""

if [ -f /etc/nv_tegra_release ] || [ -f /sys/firmware/devicetree/base/model ]; then
  MODEL="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || true)"
  if [ -f /etc/nv_tegra_release ]; then
    IS_JETSON=true
    L4T="$(awk '{print $2 " " $5}' /etc/nv_tegra_release 2>/dev/null | tr -d ',')"
    PLATFORM="${MODEL:-NVIDIA Jetson} (L4T ${L4T})"
    say "platform  : ${PLATFORM}"

    # CUDA compute capability per Tegra SoC. Passing this explicitly avoids a
    # cmake probe that guesses wrong (or refuses to guess) when cross-checking
    # an integrated GPU.
    COMPAT="$(tr -d '\0' < /proc/device-tree/compatible 2>/dev/null || true)"
    case "${COMPAT}${MODEL}" in
      *t234*|*Orin*|*orin*)     CUDA_ARCH=87 ;;
      *t194*|*Xavier*|*xavier*) CUDA_ARCH=72 ;;
      *t186*)                   CUDA_ARCH=62 ;;
      *t210*|*Nano*|*nano*)     CUDA_ARCH=53 ;;
    esac
    [ -n "${CUDA_ARCH}" ] && say "cuda arch : sm_${CUDA_ARCH}"
  fi
fi

# --- CUDA -------------------------------------------------------------------
HAS_CUDA=false
CUDA_VERSION=""
NVCC_PATH=""
GPU_NAME=""
GPU_VRAM_GB=0

if command -v nvcc >/dev/null 2>&1; then
  NVCC_PATH="$(command -v nvcc)"
elif [ -x /usr/local/cuda/bin/nvcc ]; then
  NVCC_PATH=/usr/local/cuda/bin/nvcc
fi

if [ -n "${NVCC_PATH}" ]; then
  HAS_CUDA=true
  CUDA_VERSION="$("${NVCC_PATH}" --version 2>/dev/null | awk '/release/ {print $5}' | tr -d ',')"
  say "cuda      : ${CUDA_VERSION} (${NVCC_PATH})"
fi

if [ "${IS_JETSON}" = true ]; then
  # Unified memory: the GPU shares system RAM, and nvidia-smi does not exist.
  GPU_NAME="${MODEL:-Jetson integrated GPU}"
  GPU_VRAM_GB="${MEM_GB}"
  say "gpu       : ${GPU_NAME} (shared ${GPU_VRAM_GB} GB)"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  HAS_CUDA=true
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
  VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)"
  GPU_VRAM_GB=$(( VRAM_MB / 1024 ))
  say "gpu       : ${GPU_NAME} (${GPU_VRAM_GB} GB VRAM)"
fi

if [ "${HAS_CUDA}" = true ]; then
  ACCELERATOR="cuda"
else
  ACCELERATOR="cpu"
  say "gpu       : none detected -- CPU inference (OpenBLAS / AVX2 / NEON)"
fi

# --- Toolchain --------------------------------------------------------------
HAS_FFMPEG=false
command -v ffmpeg >/dev/null 2>&1 && HAS_FFMPEG=true
say "ffmpeg    : ${HAS_FFMPEG}"

HAS_BLAS=false
if ls /usr/lib/*/libopenblas.so* >/dev/null 2>&1 || ls /usr/lib/libopenblas.so* >/dev/null 2>&1; then
  HAS_BLAS=true
fi
[ "${ACCELERATOR}" = "cpu" ] && say "openblas  : ${HAS_BLAS}"

# --- Microphone -------------------------------------------------------------
# Prefer a PCM matching MIC_KEYWORD, then any non-onboard sysdefault, then
# "default". On Jetson the onboard APE card enumerates but has nothing attached,
# so it is skipped -- picking it yields a meter that never moves.
MIC_KEYWORD="${MIC_KEYWORD:-C525}"
MIC_DEVICE=""

if command -v arecord >/dev/null 2>&1; then
  PCMS="$(arecord -L 2>/dev/null | grep -v '^ ' || true)"
  MIC_DEVICE="$(echo "${PCMS}" | grep -i "${MIC_KEYWORD}" | grep '^sysdefault' | head -n1 || true)"
  if [ -z "${MIC_DEVICE}" ]; then
    MIC_DEVICE="$(echo "${PCMS}" | grep '^sysdefault:CARD=' | grep -vi 'APE' | head -n1 || true)"
  fi
  [ -z "${MIC_DEVICE}" ] && MIC_DEVICE="default"
  say "mic       : ${MIC_DEVICE}"
else
  say "mic       : arecord not found (install alsa-utils for the realtime service)"
fi

# --- Capability matrix ------------------------------------------------------
# Breeze ASR is unconditionally true: whisper.cpp always has a CPU path, so the
# core product works everywhere. Everything else is earned by hardware.
CAP_BREEZE=true
CAP_WHISPERX=false
CAP_LLM=false
CAP_REALTIME=false

[ -n "${MIC_DEVICE}" ] && CAP_REALTIME=true
if [ "${HAS_CUDA}" = true ] && [ "${GPU_VRAM_GB}" -ge 7 ]; then
  CAP_WHISPERX=true
  CAP_LLM=true
fi

WHISPER_CLI="${REPO_ROOT}/engine/whisper.cpp/build/bin/whisper-cli"

cat > "${OUT}" <<JSON
{
  "generated_at": "$(date -Iseconds)",
  "platform": "${PLATFORM}",
  "os": "${OS_NAME}",
  "arch": "${ARCH}",
  "cpu_cores": ${CPU_CORES},
  "total_memory_gb": ${MEM_GB},
  "is_jetson": ${IS_JETSON},
  "accelerator": "${ACCELERATOR}",
  "cuda": {
    "available": ${HAS_CUDA},
    "version": "${CUDA_VERSION}",
    "nvcc": "${NVCC_PATH}",
    "arch": "${CUDA_ARCH}"
  },
  "gpu_name": "${GPU_NAME}",
  "gpu_vram_gb": ${GPU_VRAM_GB},
  "has_ffmpeg": ${HAS_FFMPEG},
  "has_openblas": ${HAS_BLAS},
  "mic_device": "${MIC_DEVICE}",
  "whisper_cli": "${WHISPER_CLI}",
  "capabilities": {
    "breeze_asr": ${CAP_BREEZE},
    "realtime_vad": ${CAP_REALTIME},
    "whisperx_diarization": ${CAP_WHISPERX},
    "gemma_e2b_multimodal": ${CAP_LLM}
  }
}
JSON

echo "==> Wrote ${OUT}"
