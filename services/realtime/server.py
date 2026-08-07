#!/usr/bin/env python3
"""Realtime VAD + ASR console (HTTP console + WebSocket stream).

Three cooperating threads:

    vad_loop()    reads 100 ms frames from the microphone, broadcasts the RMS
                  meter and raw PCM at ~10 fps, and cuts a speech segment once
                  the level drops below the threshold for long enough.
    asr_worker()  drains the segment queue through whisper.cpp, one at a time,
                  so concurrent batch jobs still have GPU headroom.
    ws_server()   owns the asyncio loop the other two broadcast into.

The HTTP console (static/index.html) is deliberately dependency-free: no CDN, no
build step, so it works on an air-gapped edge box.
"""

import asyncio
import json
import os
import pathlib
import queue
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import websockets

from breeze_hub import audio, config

try:
    import cv2
except ImportError:  # OpenCV is optional -- the console just hides the webcam panel.
    cv2 = None

STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"

state = {
    "status": "STARTING",
    "volume": 0,
    "device": "",
    "last_transcription": "",
    "transcriptions": [],
}

asr_queue = queue.Queue()
ws_clients = set()
ws_loop = None


# --------------------------------------------------------------------------
# Broadcast plumbing
# --------------------------------------------------------------------------

def broadcast(payload):
    """Fire a JSON message at every connected console. Safe to call from any thread."""
    if ws_loop is None or not ws_clients:
        return
    message = json.dumps(payload, ensure_ascii=False)
    asyncio.run_coroutine_threadsafe(_broadcast_async(message), ws_loop)


async def _broadcast_async(message):
    if not ws_clients:
        return
    await asyncio.gather(
        *[client.send(message) for client in list(ws_clients)],
        return_exceptions=True,
    )


def set_status(status):
    state["status"] = status
    broadcast({"type": "status", "status": status, "volume": state["volume"]})


# --------------------------------------------------------------------------
# ASR worker
# --------------------------------------------------------------------------

# Which Breeze variant the next utterance will go through. Switchable while the
# service runs -- whisper-cli is spawned per utterance, so a change costs
# nothing and takes effect on the next one.
_model_lock = threading.Lock()
_current_variant = config.MODEL_VARIANT


def current_model():
    with _model_lock:
        return _current_variant, config.model_path(_current_variant)


def set_current_model(variant):
    """Switch variants. Raises KeyError for an unknown name, IOError if absent."""
    path = config.model_path(variant)
    if not path.exists():
        raise IOError("model not converted yet: {}".format(path))
    global _current_variant
    with _model_lock:
        _current_variant = variant
    return path


def asr_worker():
    while True:
        item = asr_queue.get()
        if item is None:
            break
        wav_path, duration = item

        set_status("TRANSCRIBING")
        started = time.time()
        _, model = current_model()
        cmd = [
            str(config.WHISPER_CLI),
            "-m", str(model),
            "-f", str(wav_path),
            "-l", config.ASR_LANGUAGE,
            "-t", str(config.ASR_THREADS),
            "--no-timestamps",
            "-nt",
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            text = result.stdout.decode("utf-8", "replace").strip()
            if result.returncode != 0 and not text:
                text = ""
                print("[asr] whisper-cli exited {}: {}".format(
                    result.returncode,
                    result.stderr.decode("utf-8", "replace").strip()[-300:],
                ))
        except OSError as exc:
            print("[asr] cannot run {}: {}".format(config.WHISPER_CLI, exc))
            text = ""

        elapsed = time.time() - started

        if text:
            entry = {
                "type": "transcription",
                "id": int(time.time() * 1000),
                "time": time.strftime("%H:%M:%S"),
                "text": text,
                "duration": round(duration, 1),
                "cuda_time": round(elapsed, 2),
            }
            state["transcriptions"].insert(0, entry)
            del state["transcriptions"][config.VAD_MAX_HISTORY:]
            state["last_transcription"] = text
            broadcast(entry)

        set_status("LISTENING")

        try:
            os.remove(wav_path)
        except OSError:
            pass

        asr_queue.task_done()


# --------------------------------------------------------------------------
# Capture + VAD
# --------------------------------------------------------------------------

def vad_loop():
    import base64

    proc, device = audio.open_capture()
    state["device"] = device
    print("[vad] capturing from {}".format(device))
    set_status("LISTENING")

    speaking = False
    speech = bytearray()
    silent_frames = 0
    max_silent_frames = int(config.VAD_SILENCE_DURATION_SEC / config.FRAME_DURATION_SEC)
    counter = 0

    while True:
        frame = proc.stdout.read(config.FRAME_BYTES)
        if not frame or len(frame) < config.FRAME_BYTES:
            if proc.poll() is not None:
                print("[vad] capture process died, restarting")
                proc, device = audio.open_capture()
                state["device"] = device
                continue
            time.sleep(0.01)
            continue

        rms = audio.calculate_rms(frame)
        state["volume"] = int(rms)

        broadcast({
            "type": "frame",
            "volume": int(rms),
            "status": state["status"],
            "pcm": base64.b64encode(frame).decode("ascii"),
        })

        if rms > config.VAD_NOISE_THRESHOLD:
            if not speaking:
                speaking = True
                state["status"] = "RECORDING"
            speech.extend(frame)
            silent_frames = 0
            continue

        if not speaking:
            continue

        speech.extend(frame)
        silent_frames += 1
        duration = len(speech) / float(config.SAMPLE_RATE * config.BYTES_PER_SAMPLE)

        if silent_frames >= max_silent_frames or duration >= config.VAD_MAX_SPEECH_SEC:
            speaking = False
            if duration >= config.VAD_MIN_SPEECH_SEC:
                counter += 1
                wav_path = config.WORK_DIR / "segment_{}_{}.wav".format(counter, int(time.time()))
                audio.write_wav(wav_path, speech)
                asr_queue.put((wav_path, duration))
            else:
                state["status"] = "LISTENING"
            speech = bytearray()
            silent_frames = 0


# --------------------------------------------------------------------------
# Camera (optional, off by default until the user opts in from the console)
# --------------------------------------------------------------------------

class CameraManager:
    def __init__(self):
        self.cap = None
        self.active = False
        self.lock = threading.Lock()

    @property
    def available(self):
        return cv2 is not None and config.CAMERA_ENABLED

    def start(self):
        if not self.available:
            return
        with self.lock:
            if self.active:
                return
            cap = cv2.VideoCapture(config.CAMERA_INDEX)
            if not cap.isOpened():
                print("[camera] index {} unavailable".format(config.CAMERA_INDEX))
                cap.release()
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.cap = cap
            self.active = True

    def stop(self):
        with self.lock:
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            self.active = False

    def frame(self):
        with self.lock:
            if not self.active or self.cap is None:
                return None
            ok, frame = self.cap.read()
            if not ok:
                return None
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.CAMERA_JPEG_QUALITY])
            return jpeg.tobytes() if ok else None


camera = CameraManager()


# --------------------------------------------------------------------------
# HTTP console
# --------------------------------------------------------------------------

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class ConsoleHandler(BaseHTTPRequestHandler):
    def _send(self, code, content_type, body=b""):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_HEAD(self):
        self._send(200, "text/html; charset=utf-8")

    def _json(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(code, "application/json; charset=utf-8", body)

    def do_POST(self):
        path = self.path.split("?", 1)[0]

        if path == "/api/model":
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                variant = str(json.loads(raw.decode("utf-8") or "{}").get("model", "")).strip()
            except ValueError:
                self._json(400, {"ok": False, "error": "malformed JSON"})
                return
            try:
                path_ = set_current_model(variant)
            except KeyError as exc:
                self._json(400, {"ok": False, "error": str(exc)})
                return
            except IOError as exc:
                self._json(409, {"ok": False, "error": str(exc)})
                return
            self._json(200, {"ok": True, "model": variant, "path": str(path_)})

        else:
            self._send(404, "text/plain")

    def do_GET(self):
        path = self.path.split("?", 1)[0]

        if path == "/":
            html = (STATIC_DIR / "index.html").read_bytes()
            self._send(200, "text/html; charset=utf-8", html)

        elif path == "/api/config":
            # The console asks the server what this machine can do, rather than
            # assuming a webcam and a fixed WebSocket port exist.
            body = json.dumps({
                "ws_port": config.REALTIME_WS_PORT,
                "camera": camera.available,
                "device": state["device"],
                "threshold": config.VAD_NOISE_THRESHOLD,
                "sample_rate": config.SAMPLE_RATE,
                "accelerator": config.HARDWARE.get("accelerator", "unknown"),
                "model": current_model()[0],
                "models": config.available_models(),
            }).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)

        elif path == "/api/models":
            body = json.dumps({
                "current": current_model()[0],
                "models": config.available_models(),
            }, ensure_ascii=False).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)

        elif path == "/api/transcriptions":
            body = json.dumps(state["transcriptions"], ensure_ascii=False).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)

        elif path == "/video_frame":
            jpeg = camera.frame()
            if jpeg:
                self._send(200, "image/jpeg", jpeg)
            else:
                self._send(404, "text/plain")

        else:
            self._send(404, "text/plain")

    def log_message(self, fmt, *args):
        pass


# --------------------------------------------------------------------------
# WebSocket server
# --------------------------------------------------------------------------

async def ws_handler(websocket, path=None):
    """Signature covers both websockets<14 (passes path) and >=14 (does not)."""
    ws_clients.add(websocket)
    try:
        await websocket.send(json.dumps({
            "type": "status",
            "status": state["status"],
            "volume": state["volume"],
        }))
        async for message in websocket:
            try:
                data = json.loads(message)
            except ValueError:
                continue
            if data.get("type") == "camera":
                if data.get("action") == "start":
                    camera.start()
                else:
                    camera.stop()
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        ws_clients.discard(websocket)


def ws_server():
    global ws_loop
    ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ws_loop)

    async def run():
        async with websockets.serve(ws_handler, config.HOST, config.REALTIME_WS_PORT):
            await asyncio.Future()

    ws_loop.run_until_complete(run())


# --------------------------------------------------------------------------

def main():
    config.ensure_dirs()

    missing = [str(p) for p in (config.WHISPER_CLI, current_model()[1]) if not p.exists()]
    if missing:
        sys.exit("Missing required files:\n  {}\nRun scripts/setup_engine.sh and "
                 "scripts/fetch_model.sh --convert first.".format("\n  ".join(missing)))

    threading.Thread(target=asr_worker, daemon=True).start()
    threading.Thread(target=ws_server, daemon=True).start()
    threading.Thread(target=vad_loop, daemon=True).start()

    print("=" * 60)
    print("breeze-asr-hub :: realtime console")
    print(config.describe())
    print("  console      : http://{}:{}".format(config.HOST, config.REALTIME_HTTP_PORT))
    print("  websocket    : ws://{}:{}".format(config.HOST, config.REALTIME_WS_PORT))
    print("  vad threshold: {} (raw int16 RMS)".format(config.VAD_NOISE_THRESHOLD))
    print("=" * 60)

    ThreadedHTTPServer((config.HOST, config.REALTIME_HTTP_PORT), ConsoleHandler).serve_forever()


if __name__ == "__main__":
    main()
