"""Single source of truth for runtime configuration.

Every tunable in this project resolves through here, in this order:

    1. process environment (highest priority -- systemd, docker, `FOO=1 python ...`)
    2. the `.env` file at the repository root
    3. the defaults in this module (lowest priority)

Hardware-derived values (CUDA, microphone device, CPU cores) are *not* configured
here -- they come from `hardware.json`, written by `scripts/probe_hardware.sh`.
Anything the probe detects can still be overridden by an explicit env var, so a
user who knows better than the probe always wins.

No third-party imports: this must work on a bare Jetson with Python 3.8.
"""

import json
import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

ENV_FILE = REPO_ROOT / ".env"
HARDWARE_FILE = REPO_ROOT / "hardware.json"


def _load_env_file(path=ENV_FILE):
    """Populate os.environ from a KEY=VALUE file without clobbering real env vars."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_env_file()


def _hardware():
    if not HARDWARE_FILE.exists():
        return {}
    try:
        return json.loads(HARDWARE_FILE.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


HARDWARE = _hardware()


def _str(key, default=""):
    value = os.environ.get(key, "")
    return value if value != "" else default


def _int(key, default):
    try:
        return int(_str(key, str(default)))
    except ValueError:
        return default


def _float(key, default):
    try:
        return float(_str(key, str(default)))
    except ValueError:
        return default


def _bool(key, default=False):
    value = _str(key, "1" if default else "0").lower()
    return value in ("1", "true", "yes", "on")


def _path(key, default):
    """Resolve a path setting, treating relative values as repo-relative."""
    value = _str(key, "")
    candidate = pathlib.Path(value) if value else pathlib.Path(default)
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate)


# --- Engine -----------------------------------------------------------------
# The probe records where it built whisper.cpp; env vars override for people who
# already have a system-wide build they would rather reuse.

WHISPER_CLI = _path("WHISPER_CLI", HARDWARE.get("whisper_cli") or "engine/whisper.cpp/build/bin/whisper-cli")

# --- Models -----------------------------------------------------------------
# MediaTek publishes two Breeze ASR models tuned for different things, and which
# one is "better" depends entirely on what is being said. Keep both and let the
# caller choose per request -- whisper-cli is spawned per job, so switching
# costs nothing but a different -m argument.
#
# Filenames deliberately match the upstream repo name. An earlier layout shipped
# a single ggml-breeze-asr-26.bin that was fetched from whichever repo happened
# to be configured, which made it impossible to tell what a given .bin actually
# contained.

MODEL_VARIANTS = {
    "25": {
        "repo": "MediaTek-Research/Breeze-ASR-25",
        "filename": "ggml-breeze-asr-25.bin",
        "summary": "Taiwanese Mandarin, plus Mandarin-English code-switching",
    },
    "26": {
        "repo": "MediaTek-Research/Breeze-ASR-26",
        "filename": "ggml-breeze-asr-26.bin",
        "summary": "Taiwanese Hokkien (Taigi), transcribed as Chinese characters",
    },
}

MODEL_DIR = _path("MODEL_DIR", "models")
MODEL_VARIANT = _str("MODEL_VARIANT", "25")


def model_path(variant=None):
    """Path to a variant's .bin. Falls back to the default variant.

    An explicit MODEL_PATH overrides the default variant only -- asking for a
    named variant always resolves through MODEL_DIR, otherwise a machine that
    pins MODEL_PATH could never reach the other model.
    """
    name = str(variant or MODEL_VARIANT)
    if name not in MODEL_VARIANTS:
        raise KeyError("unknown model variant: {} (have: {})".format(
            name, ", ".join(sorted(MODEL_VARIANTS))))
    if variant is None or name == MODEL_VARIANT:
        override = _str("MODEL_PATH")
        if override:
            return _path("MODEL_PATH", override)
    return MODEL_DIR / MODEL_VARIANTS[name]["filename"]


def available_models():
    """Variants whose .bin is actually on disk, for the UI to offer."""
    found = []
    for name in sorted(MODEL_VARIANTS):
        path = MODEL_DIR / MODEL_VARIANTS[name]["filename"]
        if name == MODEL_VARIANT:
            path = model_path(name)
        if path.exists():
            found.append({
                "variant": name,
                "path": str(path),
                "summary": MODEL_VARIANTS[name]["summary"],
                "default": name == MODEL_VARIANT,
            })
    return found


MODEL_PATH = model_path()
MODEL_URL = _str("MODEL_URL")
MODEL_SHA256 = _str("MODEL_SHA256")
# Only read by scripts/convert_model.sh, which builds a ggml model from the
# upstream checkpoint. Kept here so every setting still has one home.
MODEL_HF_REPO = _str("MODEL_HF_REPO", MODEL_VARIANTS[MODEL_VARIANT]["repo"]
                     if MODEL_VARIANT in MODEL_VARIANTS else "MediaTek-Research/Breeze-ASR-25")
MODEL_HF_REVISION = _str("MODEL_HF_REVISION", "main")
ASR_LANGUAGE = _str("ASR_LANGUAGE", "zh")
ASR_THREADS = _int("ASR_THREADS", HARDWARE.get("cpu_cores") or 4)

# --- Network ----------------------------------------------------------------

HOST = _str("HOST", "0.0.0.0")
BATCH_PORT = _int("BATCH_PORT", 8013)
REALTIME_HTTP_PORT = _int("REALTIME_HTTP_PORT", 8015)
REALTIME_WS_PORT = _int("REALTIME_WS_PORT", 8016)

# --- Audio capture ----------------------------------------------------------
# MIC_DEVICE empty means "ask breeze_hub.audio to pick one at startup", which is
# what makes this project portable across machines with different sound cards.

MIC_DEVICE = _str("MIC_DEVICE", HARDWARE.get("mic_device") or "")
MIC_KEYWORD = _str("MIC_KEYWORD", "C525")
SAMPLE_RATE = _int("SAMPLE_RATE", 16000)
BYTES_PER_SAMPLE = 2
FRAME_DURATION_SEC = _float("FRAME_DURATION_SEC", 0.1)

# --- VAD --------------------------------------------------------------------
# Threshold is a raw int16 RMS value (0-32767), matching what the meter in the
# web console displays. Run `python3 -m breeze_hub.calibrate` to derive it for a
# new microphone rather than guessing.

VAD_NOISE_THRESHOLD = _int("VAD_NOISE_THRESHOLD", HARDWARE.get("vad_noise_threshold") or 1200)
VAD_SILENCE_DURATION_SEC = _float("VAD_SILENCE_DURATION_SEC", 0.6)
VAD_MIN_SPEECH_SEC = _float("VAD_MIN_SPEECH_SEC", 0.8)
VAD_MAX_SPEECH_SEC = _float("VAD_MAX_SPEECH_SEC", 10.0)
VAD_MAX_HISTORY = _int("VAD_MAX_HISTORY", 200)

# --- Camera -----------------------------------------------------------------

CAMERA_ENABLED = _bool("CAMERA_ENABLED", True)
CAMERA_INDEX = _int("CAMERA_INDEX", 0)
CAMERA_WIDTH = _int("CAMERA_WIDTH", 640)
CAMERA_HEIGHT = _int("CAMERA_HEIGHT", 360)
CAMERA_JPEG_QUALITY = _int("CAMERA_JPEG_QUALITY", 70)

# --- Batch service ----------------------------------------------------------

UPLOAD_DIR = _path("UPLOAD_DIR", "var/uploads")
RESULT_DIR = _path("RESULT_DIR", "var/results")
LOG_DIR = _path("LOG_DIR", "var/logs")
WORK_DIR = _path("WORK_DIR", "var/tmp")

# --- Optional downstream engines --------------------------------------------
# All optional. Each is probed for liveness at runtime; when one is absent the
# UI degrades instead of erroring.

WHISPERX_API_URL = _str("WHISPERX_API_URL", "http://127.0.0.1:8088")
LLM_API_URL = _str("LLM_API_URL", "http://127.0.0.1:18082")
LLM_MODEL_NAME = _str("LLM_MODEL_NAME", "")
HF_TOKEN = _str("HF_TOKEN")

FRAME_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * FRAME_DURATION_SEC)


def capability(name, default=False):
    """Look up a probed capability flag, e.g. capability('whisperx_diarization')."""
    return bool(HARDWARE.get("capabilities", {}).get(name, default))


def ensure_dirs():
    for directory in (UPLOAD_DIR, RESULT_DIR, LOG_DIR, WORK_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def describe():
    """Human-readable startup banner -- printed by both services."""
    return "\n".join([
        "  repo root    : {}".format(REPO_ROOT),
        "  whisper-cli  : {} {}".format(WHISPER_CLI, "" if WHISPER_CLI.exists() else "(MISSING)"),
        "  model        : {} {}".format(MODEL_PATH, "" if MODEL_PATH.exists() else "(MISSING)"),
        "  variants     : {}".format(
            ", ".join("{}{}".format(m["variant"], " (default)" if m["default"] else "")
                      for m in available_models()) or "none on disk"),
        "  accelerator  : {}".format(HARDWARE.get("accelerator", "unknown")),
        "  platform     : {}".format(HARDWARE.get("platform", "unknown")),
    ])
