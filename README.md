# breeze-asr-hub

在邊緣裝置上跑的中文語音辨識中控台。批次長音檔轉寫與 24/7 即時聽寫兩套服務,
共用同一顆 [whisper.cpp](https://github.com/ggml-org/whisper.cpp) 推理引擎。

參考機器是 NVIDIA Jetson AGX Xavier,但專案本身不綁 Jetson:安裝時探測硬體,
有 CUDA 就編 CUDA,沒有就降級到 CPU,麥克風也是開機時自動挑,不是寫死型號。

---

## 兩個服務

| 服務 | 埠 | 做什麼 |
| --- | --- | --- |
| **批次轉寫 WebUI** | `8013` | 長音檔上傳(分片 + 智慧靜音切割)、字幕輸出 (txt/srt/vtt)、選配 WhisperX 講者分離與 LLM 摘要 |
| **即時聽寫台** | `8015` (HTTP) / `8016` (WS) | 常駐背景 VAD 側錄、即時轉寫推送、10 fps 音壓計、低延遲監聽、選配 webcam 預覽 |

兩者可以同時跑。在 Xavier 上實測雙 `whisper-cli` 併發推理 18.5 秒完成,沒有 OOM。

---

## 安裝

```bash
git clone https://github.com/pondahai/breeze-asr-hub.git
cd breeze-asr-hub
scripts/install.sh
```

`install.sh` 依序做五件事,每一步都可以重跑:系統套件 → 產生 `.env` → 硬體探測 →
Python 套件與編譯 whisper.cpp → 放置模型。

裝完之後:

```bash
# 針對這支麥克風與這個房間校正 VAD 門檻
python3 -m breeze_hub.calibrate --write

# 前景跑起來確認
scripts/run.sh realtime
scripts/run.sh batch

# 確認沒問題再裝成開機服務
scripts/service.sh install
scripts/service.sh status
```

### 模型

MediaTek 發佈了兩顆 Breeze ASR,調校目標不同,而且**哪顆比較好取決於講的是什麼**,
不是取決於機器。兩顆可以並存,呼叫時再指定:

| variant | 上游 | 適用 |
|---|---|---|
| `25` | [Breeze-ASR-25](https://huggingface.co/MediaTek-Research/Breeze-ASR-25) | 台灣華語、中英夾雜 |
| `26` | [Breeze-ASR-26](https://huggingface.co/MediaTek-Research/Breeze-ASR-26) | 台語,輸出中文字 |

兩顆都是 whisper-large-v2 微調,f16 各約 2.9 GB。預設 `26`,用 `.env` 的
`MODEL_VARIANT` 改。實測補充:在一段**華語**會議錄音上,`26` 開頭生成了幻覺字幕
並整段漏掉主席致詞,`25` 則正確轉出 —— 音檔以華語為主的話建議設 `25`。

沒有人發佈現成的 Breeze ASR ggml 檔,所以取得模型有三條路:

```bash
# 1. 已經有一顆
scripts/fetch_model.sh /path/to/ggml-breeze-asr-26.bin

# 2. 自己架的位置:在 .env 設定 MODEL_URL= 後重跑
scripts/fetch_model.sh

# 3. 從 Hugging Face 原始權重轉一顆出來
scripts/fetch_model.sh --convert --variant 26
scripts/fetch_model.sh --convert --variant 25   # 兩顆都要就各跑一次
```

第三條路走 `scripts/convert_model.sh`,用 whisper.cpp 的
`convert-h5-to-ggml.py` 轉成 ggml。`--variant` 會自動帶對上游 repo 與輸出檔名
(`models/ggml-breeze-asr-<variant>.bin`),所以磁碟上的檔案一定說得出自己是誰。

```bash
scripts/convert_model.sh --variant 25                  # 依清單轉一顆
scripts/convert_model.sh --repo openai/whisper-small   # 換一顆來源模型
scripts/convert_model.sh --src ./Breeze-ASR-25         # 用已經下載好的 checkout(離線可行)
scripts/convert_model.sh --quantize q5_0               # 量化,large-v2 從 ~3 GB 降到 ~1 GB
scripts/convert_model.sh --keep-src                    # 保留下載內容供重跑
```

#### 執行時切換

`whisper-cli` 是每個工作才 spawn 一次,模型不常駐記憶體,所以切換只是換一個
`-m` 參數,沒有卸載/重載的成本。

```bash
curl -F file=@meeting.wav -F model=25 localhost:8013/api/transcribe   # 批次:逐次指定
curl localhost:8013/api/models                                        # 有哪幾顆可用

curl localhost:8015/api/models                                        # 即時台:目前用哪顆
curl -X POST -d '{"model":"25"}' localhost:8015/api/model             # 下一段語音起生效
```

`/api/models` 只回報**磁碟上真的存在**的 variant,介面不會列出還沒轉好的模型。
批次 API 不帶 `model` 欄位時沿用 `MODEL_VARIANT`,既有呼叫端不受影響。

轉檔只需要 whisper.cpp 的 **checkout**(轉檔腳本用它的 `convert-h5-to-ggml.py`),
不必先編譯 —— 只是想在桌機轉一顆帶去 Jetson 的話,
`git clone --depth 1 https://github.com/ggml-org/whisper.cpp engine/whisper.cpp` 就夠了,
不需要跑 `setup_engine.sh`。(只有 `--quantize` 會真的去編一個 `whisper-quantize` 出來。)

轉檔需要 torch 與 transformers(`requirements-convert.txt`),推理端完全用不到,
所以刻意不寫進 `requirements.txt`。這些請裝在**獨立的 venv**,並確保跑轉檔時
`python3` 指向它 —— 腳本呼叫的是 PATH 上的 `python3`:

```bash
python3 -m venv .venv-convert
.venv-convert/bin/pip install -r requirements-convert.txt
PATH="$PWD/.venv-convert/bin:$PATH" scripts/fetch_model.sh --convert --variant 26
```

**Jetson 上建議不要在機器上轉**:PyPI 沒有 JetPack 的 torch wheel。
(一般 aarch64 就沒這問題 —— DGX Spark 上 `pip install torch` 直接裝到
CUDA 13 的 aarch64 wheel,轉檔正常。)在桌機轉好之後把 `.bin` 複製過去,再走第一條路即可 ——
轉檔腳本最後會印出 SHA256,填進 `.env` 的 `MODEL_SHA256` 就能讓 `fetch_model.sh` 驗證。

任何 whisper.cpp 相容的 ggml 模型都能用,Breeze ASR 只是對台灣口音的中文特別準。
想先快速驗證整條路徑通不通,拿 `--repo openai/whisper-small` 轉一顆小的最快。

---

## 使用方法

兩個服務都有網頁介面(批次 `http://<IP>:8013`、即時台 `http://<IP>:8015`),
但都是純 HTTP,可以直接當 API 用。

### 批次轉寫 API(`8013`)

| 方法 | 路徑 | 說明 |
| --- | --- | --- |
| `POST` | `/api/transcribe` | 送出轉寫工作,回傳 `job_id` |
| `GET` | `/api/jobs/<job_id>` | 查狀態與結果 |
| `GET` | `/api/jobs/<job_id>/download?ext=txt` | 下載結果檔(`txt` / `srt` / `vtt`) |
| `POST` | `/api/jobs/<job_id>/cancel` | 中止進行中的工作 |
| `POST` | `/api/upload_chunk` | 分片上傳(大檔用,見下) |
| `GET` | `/api/models` | 有哪幾顆模型可用 |
| `GET` | `/api/system/capabilities` | 本機探測到的能力 |
| `POST` | `/api/llm` | 把逐字稿丟給下游 LLM 處理(需設定 `LLM_API_URL`) |
| `GET` | `/api/llm/health` | 下游 LLM 是否活著 |

`POST /api/transcribe` 吃 multipart 表單:

| 欄位 | 預設 | 說明 |
| --- | --- | --- |
| `file` | — | 音檔。支援 `.wav .mp3 .m4a .flac .ogg` |
| `model` | `MODEL_VARIANT` | `25` 或 `26`,見上節 |
| `format` | `txt` | `txt` / `srt` / `vtt` |
| `max_len` | `20` | 每段字幕最長字數 |
| `use_whisperx` | `false` | 改用 WhisperX 做講者分離 |
| `hf_token` | — | WhisperX 取用受管制權重時需要 |
| `min_speakers` / `max_speakers` | — | 提示講者人數,幫助分離 |
| `upload_id` + `filename` | — | 改用分片上傳時,取代 `file` |

最短的一次完整流程:

```bash
# 送出
JOB=$(curl -sS -F file=@meeting.wav -F model=25 -F format=srt \
        localhost:8013/api/transcribe | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')

# 輪詢直到 done(status 會是 running / done / failed / cancelled)
until [ "$(curl -sS localhost:8013/api/jobs/$JOB | python3 -c 'import sys,json;print(json.load(sys.stdin)["status"])')" = done ]; do sleep 5; done

# 取檔
curl -sS -o meeting.srt "localhost:8013/api/jobs/$JOB/download?ext=srt"
```

`GET /api/jobs/<job_id>` 回傳裡除了 `status` 與 `text`,還有 `model`(這份逐字稿是哪顆
模型轉的)、`elapsed_sec` 與 `log_tail`(whisper-cli 的即時輸出,可直接顯示進度)。

**大檔請走分片上傳。** 直接 POST 幾百 MB 容易在反向代理或瀏覽器端斷掉。作法是先
把檔案切塊逐一 `POST /api/upload_chunk`(帶同一個自訂的 `upload_id` 與遞增的
`chunk_index`),全部送完後再 `POST /api/transcribe`,這次不帶 `file`,改帶
`upload_id` 與 `filename`,伺服器會自己組裝。網頁介面就是這樣做的。

### 即時聽寫台 API(`8015` HTTP / `8016` WebSocket)

| 方法 | 路徑 | 說明 |
| --- | --- | --- |
| `GET` | `/api/config` | 埠號、鏡頭、麥克風、門檻等前端需要的資訊 |
| `GET` | `/api/transcriptions` | 目前累積的逐字稿 |
| `GET` | `/api/models` | 可用模型與目前使用中的那顆 |
| `POST` | `/api/model` | 切換模型,`{"model":"25"}`,下一段語音生效 |
| `GET` | `/video_frame` | 單張 webcam JPEG(沒鏡頭時 404) |

即時結果從 WebSocket(`ws://<IP>:8016`)推送,訊息是 JSON,`type` 有三種:

| `type` | 內容 |
| --- | --- |
| `status` | 目前狀態(`LISTENING` / `TRANSCRIBING` 等)與音壓值 |
| `transcription` | 一段語音轉寫完成,含文字、時間與長度 |
| `frame` | webcam 影格(沒鏡頭或 `CAMERA_ENABLED=0` 時不會出現) |

啟動前建議先校正 VAD 門檻,否則會一直誤觸發或完全不觸發:

```bash
python3 -m breeze_hub.calibrate
```

---

## 設定

所有可調參數集中在一個地方,優先序由高到低:

```
環境變數  >  .env  >  hardware.json  >  breeze_hub/config.py 的預設值
```

- **`.env`** — 你手動決定的:埠號、路徑、VAD 參數、下游服務網址。從 `.env.example` 複製。
- **`hardware.json`** — `scripts/probe_hardware.sh` 產生的,machine-specific,不進版控。
  記錄 CUDA 有無與版本、GPU 型號與記憶體、CPU 核數、麥克風裝置、以及一份**能力矩陣**。
- 兩個檔案都不存在時,專案仍然可以用預設值啟動。

換硬體之後重跑一次 `scripts/probe_hardware.sh`,其他部分會自己跟上。

### 能力矩陣與降級

`hardware.json` 裡的 `capabilities` 決定 WebUI 開放哪些按鈕:

| 能力 | 條件 |
| --- | --- |
| `breeze_asr` | 永遠為真 —— whisper.cpp 一定有 CPU 路徑,核心功能在哪都能跑 |
| `realtime_vad` | 找得到可用的錄音裝置 |
| `whisperx_diarization` | CUDA 且 VRAM ≥ 7 GB |
| `gemma_e2b_multimodal` | CUDA 且 VRAM ≥ 7 GB |

不符合條件的功能在介面上直接隱藏或反灰,不會讓使用者按下去才爆炸。

---

## 硬體支援

| 平台 | 加速方式 | 狀態 |
| --- | --- | --- |
| Jetson AGX Xavier (JetPack 5 / L4T R35) | CUDA `sm_72`,關閉 VMM | 參考機,實測 |
| Jetson Orin / NX / Nano | CUDA `sm_87` / `sm_53` | 依 SoC 自動帶入,未實測 |
| 桌機獨顯 (RTX / T4 / A100) | CUDA,架構交給 CMake 自動偵測 | 未實測 |
| 無 GPU 的 x86_64 / ARM | CPU + OpenBLAS / AVX2 / NEON | 未實測 |

Jetson 的偵測順序刻意排在 `nvidia-smi` 之前 —— JetPack 根本沒有 `nvidia-smi`,
先問它就會把每一台 Jetson 都誤判成純 CPU 機器。細節見
[docs/HARDWARE.md](docs/HARDWARE.md)。

---

## 專案結構

```
breeze-asr-hub/
├── breeze_hub/              共用層:設定、硬體、音訊擷取、校正
│   ├── config.py            唯一的設定來源
│   ├── audio.py             麥克風挑選與 PCM 處理
│   └── calibrate.py         python3 -m breeze_hub.calibrate
├── services/
│   ├── batch/               批次轉寫 WebUI (Flask, 8013)
│   └── realtime/            即時聽寫台 (stdlib HTTP + websockets, 8015/8016)
├── scripts/
│   ├── probe_hardware.sh    → hardware.json
│   ├── setup_engine.sh      clone + 依硬體編譯 whisper.cpp
│   ├── fetch_model.sh       放置模型(含 SHA256 驗證)
│   ├── convert_model.sh     Hugging Face 權重 → ggml(選配量化)
│   ├── install.sh           一鍵安裝
│   ├── run.sh               前景執行
│   └── service.sh           systemd 生命週期管理
├── deploy/                  systemd unit 樣板
└── docs/                    架構、硬體、排錯
```

即時聽寫台的前端沒有任何 CDN 依賴,也沒有 build step,離線機器直接可用。

---

## 開發歷程

這個專案是從姊妹專案
[ggml-breeze-asr-26-webui](https://github.com/pondahai/ggml-breeze-asr-26-webui)
長出來的。那邊先有一套跑在 Jetson Xavier 上的網頁介面,這裡把「跑得起來」的部分
抽成不綁單一機器的形式,並補上即時聽寫。兩個專案各自獨立,但模型共用同一套慣例
(`ggml-breeze-asr-<25|26>.bin`),webui 的模型就是用這裡的轉檔腳本產生的。

| 時間 | 里程碑 |
| --- | --- |
| 2026-07 | 初版:硬體探測 → 依探測結果編譯 → 批次與即時兩套服務。參考機 Jetson AGX Xavier |
| 2026-08 | 加入從 Hugging Face 權重轉出 ggml 的能力,模型不再需要「別人給一顆」 |
| 2026-08 | 在 DGX Spark 上完整實測,修掉一批只有真硬體才會現形的問題,並改成兩顆模型並存可切換 |

### 為什麼要有轉檔腳本

原本 `ggml-breeze-asr-26.bin` 是在參考機上手工轉出來的,沒有留下可重現的步驟 ——
任何人拿到這個 repo 都無法自己產生一顆。`scripts/convert_model.sh` 就是補這個洞。

驗證方式是把它拿去重現那顆既有的模型:從 `MediaTek-Research/Breeze-ASR-26` 轉出
的檔案與參考機上那顆 **sha256 完全相同**
(`6d58f81d79155deb5037f995a048856f6deaa9e06f59a89183cc421fa37cb1ad`),不是「看起來
對」而是逐位元組相同。順帶也確認了那顆模型的真實身分是 26 而非 25。

### DGX Spark 實測揭露的問題

「不綁單一機器」這個設計在遇到第一台沒見過的硬體時並沒有直接通過。GB10 不是
Jetson,也不是一般的獨立顯卡,於是:

- `nvidia-smi` 的 `memory.total` 回傳 `[N/A]`(統一記憶體),進到算術運算讓探測腳本
  直接中止 —— 只認 Jetson 的共享記憶體分支救不了它
- CUDA 裝在 `/usr/local/cuda` 但 `nvcc` 不在 `PATH` 上,cmake 找得到 toolkit 卻報
  `No CMAKE_CUDA_COMPILER could be found`,引擎完全編不出來
- `--quantize` 要 cmake 編一個叫 `quantize` 的目標,但 whisper.cpp 早已改名為
  `whisper-quantize` —— 這條路在任何近期版本上都不可能成功過

三個都是阻斷級,也都只在真機器上才會現形。修正後 `cuda.arch` 改成向驅動查詢
compute capability 而非查表,少一層「清單以外就沒轍」的假設。

### 已知邊界

目前實測過的是 Jetson AGX Xavier 與 DGX Spark,也就是 **NVIDIA + Linux**。
`setup_engine.sh` 只處理 CUDA / OpenBLAS / 純 CPU 三條路,**沒有 Metal、ROCm、
Vulkan 或 SYCL 分支**,腳本本身也都是 bash(Windows 需要 WSL)。純 CPU 路徑存在
但尚未實測。

模型面向的是台灣華語與台語;其他語言直接用官方 whisper 模型會更好,不過服務外殼
本身不綁 Breeze —— `MODEL_PATH` 指向任何 whisper.cpp 相容的 ggml 都能跑。

---

## 文件

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — 資料流、執行緒模型、設計取捨
- [docs/HARDWARE.md](docs/HARDWARE.md) — 探測邏輯與各平台注意事項
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — 踩過的坑

## 授權

MIT。whisper.cpp 由安裝腳本另行取得,同樣是 MIT。模型權重的授權依其發布者為準。
