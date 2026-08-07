#!/usr/bin/env python3
from flask import Flask, request, render_template, jsonify, send_file, Response, stream_with_context
from werkzeug.utils import secure_filename
import subprocess, time, uuid, pathlib, threading, os, signal, json, requests

# Utilities for smart splitting
from audio_utils import get_audio_duration, find_silences, calculate_splits, split_audio, merge_subtitles, merge_texts

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from breeze_hub import config

APP_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = config.REPO_ROOT
UPLOAD_DIR = config.UPLOAD_DIR
RESULT_DIR = config.RESULT_DIR
LOG_DIR = config.LOG_DIR
config.ensure_dirs()

CAPABILITIES_FILE = config.HARDWARE_FILE

WHISPER = config.WHISPER_CLI
MODEL = config.MODEL_PATH
WHISPERX_API_URL = config.WHISPERX_API_URL
LLM_API_URL = config.LLM_API_URL
ALLOWED = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}

app = Flask(__name__)
jobs = {}
lock = threading.Lock()

def tail_text(path, n=40):
    p = pathlib.Path(path)
    if not p.exists():
        return ''
    lines = p.read_text(errors='ignore').splitlines()
    return '\n'.join(lines[-n:])

def ensure_wav(in_path: pathlib.Path, wav_path: pathlib.Path):
    if in_path.suffix.lower() == '.wav':
        return in_path
    cmd = ['ffmpeg', '-y', '-i', str(in_path), '-ac', '1', '-ar', '16000', str(wav_path)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0 or not wav_path.exists():
        raise RuntimeError('ffmpeg convert failed: ' + (p.stderr or p.stdout)[-500:])
    return wav_path

def run_whisperx_job(job_id, in_path, out_base, log_path, language, min_speakers, max_speakers, hf_token):
    with lock:
        j = jobs.get(job_id)
        
    try:
        with open(log_path, 'a', encoding='utf-8') as logf:
            logf.write("=== 離線 WhisperX 聲紋發言人標記服務開始 ===\n")
            logf.write(f"音訊路徑: {in_path}\n")
            logf.flush()
            
            wav_path = UPLOAD_DIR / f"{job_id}.wav"
            try:
                audio_path = ensure_wav(pathlib.Path(in_path), wav_path)
                if audio_path != pathlib.Path(in_path):
                    logf.write(f"Converted to WAV: {audio_path.name}\n")
                    with lock:
                        j['audio_path'] = str(audio_path)
            except Exception as e:
                logf.write(f"Audio conversion failed: {e}\n")
                audio_path = pathlib.Path(in_path)

            logf.write("正在傳送請求給離線聲紋微服務 (Port 8088)...(預估耗時較長，請耐心等候)\n")
            logf.flush()
            
            payload = {
                'file_path': str(audio_path),
                'language': language,
                'log_path': str(log_path),
            }
            if min_speakers is not None: payload['min_speakers'] = min_speakers
            if max_speakers is not None: payload['max_speakers'] = max_speakers
            if hf_token: payload['hf_token'] = hf_token
            
            r = requests.post(f"{WHISPERX_API_URL}/api/diarize", data=payload, timeout=1800)
            if r.status_code != 200:
                raise RuntimeError(f"FastAPI 服務回傳錯誤 ({r.status_code}): {r.text}")
                
            res = r.json()
            logf.write("轉錄與聲紋分割完成！開始格式化輸出...\n")
            logf.flush()
            
            segments = res.get('segments', [])
            
            # 1. TXT
            txt_path = pathlib.Path(f"{out_base}.txt")
            txt_lines = []
            for seg in segments:
                spk = seg.get('speaker', '未知講者')
                spk_label = spk.replace("SPEAKER_", "講者 ")
                txt_lines.append(f"【{spk_label}】 {seg.get('text', '').strip()}")
            txt_path.write_text("\n".join(txt_lines), encoding='utf-8')
            
            # 2. SRT
            srt_path = pathlib.Path(f"{out_base}.srt")
            srt_lines = []
            for idx, seg in enumerate(segments, 1):
                start_s = seg.get('start', 0.0)
                end_s = seg.get('end', 0.0)
                
                def format_srt_time(s):
                    hrs = int(s // 3600)
                    mins = int((s % 3600) // 60)
                    secs = int(s % 60)
                    ms = int(round((s % 1) * 1000))
                    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"
                
                spk = seg.get('speaker', '未知講者')
                spk_label = spk.replace("SPEAKER_", "講者 ")
                
                srt_lines.append(str(idx))
                srt_lines.append(f"{format_srt_time(start_s)} --> {format_srt_time(end_s)}")
                srt_lines.append(f"【{spk_label}】{seg.get('text', '').strip()}")
                srt_lines.append("")
            srt_path.write_text("\n".join(srt_lines), encoding='utf-8')

            # 3. VTT
            vtt_path = pathlib.Path(f"{out_base}.vtt")
            vtt_lines = ["WEBVTT", ""]
            for idx, seg in enumerate(segments, 1):
                start_s = seg.get('start', 0.0)
                end_s = seg.get('end', 0.0)
                
                def format_vtt_time(s):
                    hrs = int(s // 3600)
                    mins = int((s % 3600) // 60)
                    secs = int(s % 60)
                    ms = int(round((s % 1) * 1000))
                    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"
                
                spk = seg.get('speaker', '未知講者')
                spk_label = spk.replace("SPEAKER_", "講者 ")
                
                vtt_lines.append(f"{format_vtt_time(start_s)} --> {format_vtt_time(end_s)}")
                vtt_lines.append(f"【{spk_label}】{seg.get('text', '').strip()}")
                vtt_lines.append("")
            vtt_path.write_text("\n".join(vtt_lines), encoding='utf-8')
            
            # Cleanup main converted wav if it was created
            if audio_path != pathlib.Path(in_path):
                audio_path.unlink(missing_ok=True)
            
            logf.write("=== 所有輸出檔案寫入完成，任務結束！ ===\n")
            logf.flush()
            with lock:
                if j['status'] == 'running':
                    j['status'] = 'done'
                    j['returncode'] = 0
                    
    except Exception as e:
        with open(log_path, 'a', encoding='utf-8') as logf:
            logf.write(f"\n[錯誤] 聲紋處理失敗: {e}\n")
            import traceback
            traceback.print_exc(file=logf)
            logf.flush()
        with lock:
            if j['status'] == 'running':
                j['status'] = 'failed'
                j['returncode'] = 1

def process_job_thread(job_id, in_path, out_base, log_path, max_len, fmt, model_path=None):
    with lock:
        j = jobs.get(job_id)

    model = pathlib.Path(model_path) if model_path else MODEL

    try:
        with open(log_path, 'a', encoding='utf-8') as logf:
            logf.write(f"Analyzing audio: {in_path}\n")
            logf.write(f"Model: {model.name}\n")
            logf.flush()
            
            # Use ensure_wav to convert to 16k mono wav for better processing
            wav_path = UPLOAD_DIR / f"{job_id}.wav"
            try:
                audio_path = ensure_wav(pathlib.Path(in_path), wav_path)
                if audio_path != pathlib.Path(in_path):
                    logf.write(f"Converted to WAV: {audio_path.name}\n")
                    with lock:
                        j['audio_path'] = str(audio_path)
            except Exception as e:
                logf.write(f"Audio conversion failed: {e}\n")
                audio_path = pathlib.Path(in_path)

            duration = get_audio_duration(str(audio_path))
            if duration > 0:
                logf.write(f"Duration: {duration:.2f}s\n")
                logf.write("Searching for silence points...\n")
                logf.flush()
                silences = find_silences(str(audio_path))
                # Split every 10 mins (600s)
                splits = calculate_splits(duration, silences, 600)
                logf.write(f"Calculated {len(splits)} segments for processing.\n")
            else:
                logf.write("Failed to get duration, proceeding without split.\n")
                splits = [(0, 0)]
            logf.flush()
                
            segments = []
            if len(splits) > 1:
                logf.write("Splitting audio at silent points (this may take a moment)...\n")
                logf.flush()
                segments = split_audio(str(audio_path), splits, UPLOAD_DIR, job_id)
            else:
                segments = [(str(audio_path), 0.0)]
                
            all_txts = []
            all_srts = []
            all_vtts = []
            
            for idx, (seg_path, start_time) in enumerate(segments):
                seg_out_base = f"{out_base}_{idx}" if len(segments) > 1 else str(out_base)
                
                cmd = [str(WHISPER), '-m', str(model), '-f', str(seg_path), '-of', seg_out_base, '-nt']
                cmd.extend(['-l', 'zh', '-ml', str(max_len), '-sow'])
                cmd.extend(['-et', '2.4', '-lpt', '-1.0'])
                cmd.append('-otxt')
                if fmt == 'srt' or fmt == 'vtt':
                    cmd.append('-osrt')
                    cmd.append('-ovtt')
                    
                logf.write(f"\n--- Transcribing Segment {idx+1}/{len(segments)} (Offset: {start_time:.2f}s) ---\n")
                logf.flush()
                
                proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)
                
                with lock:
                    if j['status'] == 'cancelled':
                        proc.kill()
                        return
                    j['proc'] = proc
                    j['pid'] = proc.pid
                
                rc = proc.wait()
                
                with lock:
                    if j['status'] == 'cancelled':
                        return
                        
                if rc != 0:
                    with lock:
                        j['status'] = 'failed'
                        j['returncode'] = rc
                    return
                    
                all_txts.append(f"{seg_out_base}.txt")
                if fmt == 'srt' or fmt == 'vtt':
                    all_srts.append((f"{seg_out_base}.srt", start_time))
                    all_vtts.append((f"{seg_out_base}.vtt", start_time))
                
            if len(segments) > 1:
                logf.write("\nMerging results...\n")
                logf.flush()
                merge_texts(all_txts, f"{out_base}.txt")
                if fmt == 'srt' or fmt == 'vtt':
                    merge_subtitles(all_srts, f"{out_base}.srt", is_srt=True)
                    merge_subtitles(all_vtts, f"{out_base}.vtt", is_srt=False)
                
                for seg_path, _ in segments:
                    if seg_path != str(audio_path): # Don't delete the main wav yet
                        pathlib.Path(seg_path).unlink(missing_ok=True)
                for txt in all_txts: pathlib.Path(txt).unlink(missing_ok=True)
                for srt, _ in all_srts: pathlib.Path(srt).unlink(missing_ok=True)
                for vtt, _ in all_vtts: pathlib.Path(vtt).unlink(missing_ok=True)
                
            # Cleanup main converted wav if it was created
            if audio_path != pathlib.Path(in_path):
                audio_path.unlink(missing_ok=True)

            logf.write("\nAll done.\n")
            logf.flush()
            
            with lock:
                if j['status'] == 'running':
                    j['status'] = 'done'
                    j['returncode'] = 0
                
    except Exception as e:
        with open(log_path, 'a', encoding='utf-8') as logf:
            logf.write(f"\nSystem Error: {str(e)}\n")
        with lock:
            if j['status'] == 'running':
                j['status'] = 'failed'
                j['returncode'] = -1

@app.route('/api/system/capabilities')
def get_system_capabilities():
    default_caps = {
        "capabilities": {
            "whisperx_diarization": False,
            "gemma_e2b_multimodal": False
        }
    }
    if CAPABILITIES_FILE.exists():
        try:
            caps = json.loads(CAPABILITIES_FILE.read_text(encoding='utf-8'))
            return jsonify(caps)
        except Exception:
            return jsonify(default_caps)
    return jsonify(default_caps)

@app.get('/api/models')
def list_models():
    """Which Breeze variants are actually converted and usable right now."""
    return jsonify({
        'ok': True,
        'default': config.MODEL_VARIANT,
        'models': config.available_models(),
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.post('/api/upload_chunk')
def upload_chunk():
    upload_id = request.form.get('upload_id')
    chunk_index = request.form.get('chunk_index')
    chunk_data = request.files.get('file')
    
    print(f"[DEBUG] Received chunk {chunk_index} for upload {upload_id}")
    
    if not upload_id or chunk_index is None or chunk_data is None:
        print(f"[ERROR] Missing params: id={upload_id}, idx={chunk_index}, data={chunk_data is not None}")
        return jsonify({'ok': False, 'error': 'Missing parameters'}), 400
        
    temp_dir = UPLOAD_DIR / upload_id
    temp_dir.mkdir(exist_ok=True)
    
    chunk_path = temp_dir / chunk_index
    chunk_data.save(chunk_path)
    
    print(f"[DEBUG] Saved chunk {chunk_index} to {chunk_path}")
    return jsonify({'ok': True})

@app.post('/api/transcribe')
def transcribe():
    use_whisperx = request.form.get('use_whisperx', 'false') == 'true'
    hf_token = request.form.get('hf_token', '').strip() or None
    
    min_speakers = request.form.get('min_speakers', '').strip()
    max_speakers = request.form.get('max_speakers', '').strip()
    min_speakers = int(min_speakers) if min_speakers and min_speakers.isdigit() else None
    max_speakers = int(max_speakers) if max_speakers and max_speakers.isdigit() else None

    # Which Breeze model to use. Absent means the configured default, so older
    # clients that never send the field keep working.
    variant = request.form.get('model', '').strip()
    try:
        model = config.model_path(variant or None)
    except KeyError:
        return jsonify({'ok': False, 'error': '未知的模型: {} (可用: {})'.format(
            variant, ', '.join(sorted(config.MODEL_VARIANTS)))}), 400

    if not use_whisperx and not WHISPER.exists():
        return jsonify({'ok': False, 'error': '尚未編譯引擎，先執行: scripts/setup_engine.sh'}), 400
    if not use_whisperx and not model.exists():
        return jsonify({'ok': False, 'error': '模型檔不存在: {}，先執行: scripts/fetch_model.sh --convert --variant {}'.format(
            model.name, variant or config.MODEL_VARIANT)}), 400

    upload_id = request.form.get('upload_id')
    filename = request.form.get('filename')
    fmt = request.form.get('format', 'txt')
    max_len = request.form.get('max_len', '20')
    
    if upload_id:
        ext = pathlib.Path(filename).suffix.lower()
    else:
        f = request.files.get('file')
        if not f or not f.filename:
            return jsonify({'ok': False, 'error': '請先選擇音檔'}), 400
        filename = f.filename
        ext = pathlib.Path(filename).suffix.lower()

    if ext not in ALLOWED:
        return jsonify({'ok': False, 'error': f'不支援格式: {ext}'}), 400

    job_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    in_path = UPLOAD_DIR / f"{job_id}-{secure_filename(filename)}"
    out_base = RESULT_DIR / job_id
    log_path = LOG_DIR / f"{job_id}.log"
    
    if upload_id:
        temp_dir = UPLOAD_DIR / upload_id
        if not temp_dir.exists():
            return jsonify({'ok': False, 'error': 'Upload ID not found'}), 400
        chunks = sorted([int(p.name) for p in temp_dir.glob('*') if p.name.isdigit()])
        with open(in_path, 'wb') as outfile:
            for c in chunks:
                chunk_file = temp_dir / str(c)
                with open(chunk_file, 'rb') as infile:
                    outfile.write(infile.read())
                chunk_file.unlink()
        try:
            temp_dir.rmdir()
        except:
            pass
    else:
        f.save(in_path)

    # Initialize log file
    pathlib.Path(log_path).write_text(f"Job {job_id} started.\n", encoding='utf-8')

    with lock:
        jobs[job_id] = {
            'id': job_id, 'status': 'running', 'start_ts': time.time(), 'pid': None,
            'proc': None, 'input_path': str(in_path), 'output_base': str(out_base),
            'log_path': str(log_path), 'returncode': None, 'requested_format': fmt,
            'type': 'whisperx' if use_whisperx else 'whisper-cli',
            'model': None if use_whisperx else (variant or config.MODEL_VARIANT)
        }

    if use_whisperx:
        t = threading.Thread(
            target=run_whisperx_job,
            args=(job_id, in_path, out_base, log_path, 'zh', min_speakers, max_speakers, hf_token)
        )
    else:
        t = threading.Thread(target=process_job_thread,
                             args=(job_id, in_path, out_base, log_path, max_len, fmt, str(model)))
    
    t.start()

    return jsonify({'ok': True, 'job_id': job_id, 'status': 'running'})

@app.get('/api/jobs/<job_id>')
def job_status(job_id):
    with lock:
        j = jobs.get(job_id)
    if not j:
        return jsonify({'ok': False, 'error': 'job not found'}), 404

    # Status is now managed entirely by the background thread.
    txt = ''
    main_output = pathlib.Path(j['output_base'] + '.' + j['requested_format'])
    if j['status'] == 'done' and main_output.exists():
        txt = main_output.read_text(errors='ignore').strip()

    return jsonify({
        'ok': True, 'job_id': job_id, 'status': j['status'],
        'elapsed_sec': round(time.time() - j['start_ts'], 2),
        'returncode': j['returncode'], 'pid': j['pid'],
        'input_path': j['input_path'], 'audio_path': j.get('audio_path'),
        'model': j.get('model'),
        'text': txt, 'log_tail': tail_text(j['log_path'], 50),
    })

@app.get('/api/jobs/<job_id>/download')
def job_download(job_id):
    with lock:
        j = jobs.get(job_id)
    if not j or j['status'] != 'done':
        return 'Not found or not ready', 404

    ext = request.args.get('ext', 'txt')
    target = pathlib.Path(j['output_base'] + '.' + ext)
    if not target.exists():
        return f'Format {ext} not available', 404

    return send_file(target, as_attachment=True)

@app.post('/api/jobs/<job_id>/cancel')
def job_cancel(job_id):
    with lock:
        j = jobs.get(job_id)
        if not j:
            return jsonify({'ok': False, 'error': 'job not found'}), 404
        if j['status'] != 'running':
            return jsonify({'ok': True, 'status': j['status']})
            
        j['status'] = 'cancelled'
        j['returncode'] = -15
        if j['proc']:
            try:
                os.kill(j['proc'].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
                
    return jsonify({'ok': True, 'status': 'cancelled'})

def resolve_llm_url(requested=''):
    """Which LLM server to talk to. The caller's choice wins over .env.

    Which server to use is a user preference, not a property of this machine,
    so the browser can name one per request. LLM_API_URL is only the default
    for a browser that has never been told otherwise.
    """
    requested = (requested or '').strip().rstrip('/')
    if not requested:
        return LLM_API_URL
    # The browser hands this to us and we then fetch it, so refuse anything
    # that is not plain http(s) rather than letting requests try file:// and
    # friends.
    if not requested.startswith(('http://', 'https://')):
        raise ValueError('API 位址必須以 http:// 或 https:// 開頭')
    return requested


def llm_server_models(api_url=None):
    """Ask an LLM server what it serves. [] when it is unreachable.

    The list is whatever that server reports -- llama.cpp, vLLM, Ollama's
    OpenAI shim -- so it cannot be a fixed table the way ASR variants are.
    """
    url = api_url or LLM_API_URL
    try:
        r = requests.get(f"{url}/v1/models", timeout=5)
        if r.status_code == 200:
            return [m.get('id') for m in r.json().get('data', []) if m.get('id')]
    except Exception:
        pass
    return []


def resolve_llm_model(requested='', api_url=None, served=None):
    """Pick the model id to send. Explicit > configured > whatever is served.

    A model the caller remembered from a previous session may no longer exist
    on the server it is now pointed at, so an explicit choice is only honoured
    when the server still lists it.
    """
    requested = (requested or '').strip()
    if served is None:
        served = llm_server_models(api_url)
    if requested and (not served or requested in served):
        return requested
    if config.LLM_MODEL_NAME and (not served or config.LLM_MODEL_NAME in served):
        return config.LLM_MODEL_NAME
    return served[0] if served else ''


@app.get('/api/llm/models')
def llm_models():
    """Models a given LLM server currently offers. ?api_url= overrides .env."""
    try:
        url = resolve_llm_url(request.args.get('api_url', ''))
    except ValueError as exc:
        return jsonify({'ok': False, 'error': str(exc), 'models': []}), 400

    served = llm_server_models(url)
    wanted = (request.args.get('model', '') or '').strip()
    resolved = resolve_llm_model(wanted, url, served)
    return jsonify({
        'ok': bool(served),
        'models': served,
        'default': resolved,
        # Tells the browser its remembered model is gone and it fell back.
        'fallback': bool(wanted and served and wanted not in served),
        'configured': config.LLM_MODEL_NAME,
        'api_url': url,
        'default_api_url': LLM_API_URL,
    })


@app.post('/api/llm')
def llm_process():
    data = request.get_json(force=True)
    text = data.get('text', '').strip()
    action = data.get('action', 'proofread')
    custom_prompt = data.get('custom_prompt', '').strip()

    try:
        api_url = resolve_llm_url(data.get('api_url', ''))
    except ValueError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400
    model = resolve_llm_model(data.get('model', ''), api_url)

    if not text:
        return jsonify({'ok': False, 'error': '文字內容不可為空'}), 400
        
    prompts = {
        'proofread': "你是一位精通繁體中文的專業會議紀錄秘書。請根據輸入的「原始會議逐字稿內容」，在完全不改變原意的前提下，進行同音錯字修正、補上合適的標點符號、拿掉口吃贅字（例如：呃、然後、這個、那個、也就是說），並整理成通順、易讀的精美段落。請直接輸出整理後的逐字稿，不需說明或回答任何額外文字。",
        'diarization': "你是一位精通語境邏輯分析的 AI 秘書。請仔細閱讀下面包含 `【講者 XX】` 聲紋標籤的會議逐字稿內容。請根據對話的上下關係、互相稱呼、對話語境與語意邏輯，推斷出每一位講者的真實姓名或職稱（例如：李總、張經理、小明），並將文中的 `【講者 XX】` 統一替換為對應的真實姓名標籤（例如 `【李總】`、`【小明】`）。如果某個講者無法從語境中推斷出名字，請保留原本的 `【講者 XX】` 或給予合理的臨時角色標籤（如 `【主持人】`、`【女士】`）。請直接輸出替換名字後的完整分段逐字稿，不要輸出任何前言、後記或額外解釋。",
        'minutes': "你是一位高效率的執行秘書。請將以下「會議逐字稿」內容進行整理，產出標準且精美的會議記錄。必須包含以下結構：\n1. 會議核心主題\n2. 重要決議事項 (關鍵點)\n3. 待辦追蹤清單 (Action Items，需指出負責人與任務項目，如果沒有提到負責人，請標記為全體或待定)\n請使用 markdown 格式美化輸出，讓整體結構清晰易讀，不要輸出任何前言或結尾解釋。",
        'summary': "你是一位專業的商業分析師。請精確地為以下「會議逐字稿」做一份摘要，提煉出最重要的核心關鍵與重點摘要，並以 Bullet points (條列式) 清晰呈現，直接給出最精準的精簡重點即可。",
    }
    
    system_prompt = prompts.get(action, prompts['proofread'])
    if action == 'custom' and custom_prompt:
        system_prompt = custom_prompt
        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    
    if not model:
        return jsonify({'ok': False, 'error':
                        f'LLM 伺服器沒有回報任何模型（{api_url}）。'
                        '請確認位址正確且服務已啟動。'}), 503

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "stream": True
    }
    
    def generate():
        try:
            r = requests.post(f"{api_url}/v1/chat/completions", json=payload, stream=True, timeout=600)
            if r.status_code != 200:
                yield f"data: {json.dumps({'error': f'LLM 伺服器回應 {r.status_code}: ' + r.text[:300]}, ensure_ascii=False)}\n\n"
                return
            for line in r.iter_lines():
                if line:
                    decoded = line.decode('utf-8').strip()
                    if decoded.startswith('data:'):
                        data_str = decoded[5:].strip()
                        if data_str == '[DONE]':
                            break
                        try:
                            obj = json.loads(data_str)
                            delta_obj = obj.get('choices', [{}])[0].get('delta', {})
                            
                            thought = delta_obj.get('reasoning_content', '')
                            delta = delta_obj.get('content', '')
                            
                            if thought:
                                yield f"data: {json.dumps({'thought': thought}, ensure_ascii=False)}\n\n"
                            if delta:
                                yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
                        except Exception:
                            continue
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/api/llm/health')
def llm_health():
    try:
        url = resolve_llm_url(request.args.get('api_url', ''))
    except ValueError as exc:
        return jsonify({'ok': False, 'status': 'offline', 'error': str(exc)}), 400
    served = llm_server_models(url)
    if served:
        # Report what is actually being served rather than a name baked in at
        # write time -- this used to always say "gemma-4".
        return jsonify({'ok': True, 'status': 'online', 'api_url': url,
                        'model': resolve_llm_model('', url, served), 'models': served})
    return jsonify({'ok': False, 'status': 'offline', 'api_url': url})

if __name__ == '__main__':
    host, port = config.HOST, config.BATCH_PORT
    print("=" * 60)
    print("breeze-asr-hub :: batch transcription")
    print(config.describe())
    print(f"  webui        : http://{host}:{port}")
    print("=" * 60)
    try:
        from waitress import serve
    except ImportError:
        # Flask's dev server is single-threaded and will choke on chunked
        # uploads, but it beats refusing to start on a fresh machine.
        print("waitress not installed -- falling back to the Flask dev server")
        app.run(host=host, port=port, threaded=True)
    else:
        # 1 GiB ceiling: chunked uploads arrive as multipart bodies.
        serve(app, host=host, port=port, threads=8, max_request_body_size=1073741824)
