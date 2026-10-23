# breeze-asr-hub

在邊緣裝置上跑的中文語音辨識中控台。批次長音檔轉寫與 24/7 即時聽寫兩套服務,
共用同一顆 [whisper.cpp](https://github.com/ggml-org/whisper.cpp) 推理引擎。

參考機器是 NVIDIA Jetson AGX Xavier,但專案本身不綁 Jetson:安裝時探測硬體,
有 CUDA 就編 CUDA,沒有就降級到 CPU,麥克風也是開機時自動挑,不是寫死型號。

---

## 兩個服務

| 服務 | 埠 | 做什麼 |
| --- | --- | --- |
| **批次轉寫 WebUI** | `8013` | 長音檔上傳 (分片 + 智慧靜音切割 + DSP 聲學濾波)、字幕輸出 (txt/srt/vtt)、選配 WhisperX 講者分離與 LLM 摘要 |
| **即時聽寫台** | `8015` (HTTP) / `8016` (WS) | 常駐背景 VAD 側錄、**DSP 實時語音增強 (帶通濾波 + 人聲 EQ 增益 + AGC + 響度正規化)**、歷史轉錄回溯、10 fps 音壓計、低延遲監聽、選配 webcam 預覽 |

兩者可以同時跑。在 Xavier 上實測雙 `whisper-cli` 併發推理 18.5 秒完成,沒有 OOM。

---

## 🎛️ DSP 語音增強管線 (Speech Enhancement)

本專案內建高效能、超低延遲（<3ms）的純數字訊號處理（DSP）增強模組，於 VAD 語音切片保存與送入 Whisper 推理前自動執行：

* **80Hz ~ 7500Hz 4階帶通濾波**：徹底濾除環境低頻轟鳴、空調風扇震動、桌面敲擊與 50/60Hz 電流哼聲。
* **2.5kHz 人聲共振峰動態增益 (+35%)**：補償麥克風遠距拾音的高頻損失，大幅增強中文聲母與齒音的清晰度。
* **自動增益控制 (AGC) 與動態壓縮**：拉高遠處微弱人聲音量，同時軟拐點壓制近處突發大音量以防爆音。
* **-1.0 dBFS 峰值正規化**：統一所有語音切片的聽感響度。

可在 `.env` 中自由配置開關與頻段參數（`DSP_ENHANCE_ENABLED=1`）。


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

安裝腳本**不會**自動下載 Breeze ASR 模型。參考機上那顆 `ggml-breeze-asr-26.bin`
是本地轉檔產生的,與其寫一個會 404 的 Hugging Face 連結,不如講清楚:

```bash
# 有本地檔案
scripts/fetch_model.sh /path/to/ggml-breeze-asr-26.bin

# 或自己架的位置:在 .env 設定 MODEL_URL= 後重跑
scripts/fetch_model.sh
```

任何 whisper.cpp 相容的 ggml 模型都能用,Breeze ASR 只是對台灣口音的中文特別準。
想先快速驗證整條路徑通不通,可以拿 whisper.cpp 內建的 `download-ggml-model.sh`
抓一顆小模型,把 `MODEL_PATH` 指過去。

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
│   ├── install.sh           一鍵安裝
│   ├── run.sh               前景執行
│   └── service.sh           systemd 生命週期管理
├── deploy/                  systemd unit 樣板
└── docs/                    架構、硬體、排錯
```

即時聽寫台的前端沒有任何 CDN 依賴,也沒有 build step,離線機器直接可用。

---

## 文件

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — 資料流、執行緒模型、設計取捨
- [docs/HARDWARE.md](docs/HARDWARE.md) — 探測邏輯與各平台注意事項
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — 踩過的坑

## 授權

MIT。whisper.cpp 由安裝腳本另行取得,同樣是 MIT。模型權重的授權依其發布者為準。
