# 排錯

## 音壓計不動 / 沒有底噪

多半是選到了不存在輸入的音效卡。

```bash
arecord -L                                    # 列出所有 PCM
arecord -D sysdefault:CARD=C525 -f S16_LE -r 16000 -c 1 -d 3 /tmp/t.wav   # 實測
```

Jetson 上的 `sysdefault:CARD=APE` 是 SoC 內建音訊處理器,開得起來但通常什麼都沒接。
確認可用的裝置名稱後寫進 `.env` 的 `MIC_DEVICE`。

## VAD 抓不到說話,或一直誤觸發

門檻沒有校正。它是原始 int16 RMS 值,換麥克風、換房間、改增益都會變。

```bash
python3 -m breeze_hub.calibrate --write
```

校正時若出現「speech was not louder than the room」警告,表示麥克風增益太低或
距離太遠,先處理硬體再談門檻。

## 網頁開了但沒聲音

瀏覽器規定 Web Audio API 必須先有一次使用者互動才能出聲。點一下
`ENABLE AUDIO MONITOR` 解鎖後 PCM 串流就會播放。

## 探測報告說沒有 CUDA,但機器明明有

Jetson 從別台機器 SSH 過來跑探測時最容易發生:`nvcc` 在 `/usr/local/cuda/bin`,
非互動式 shell 不會載入把它加進 PATH 的 profile。

`probe_hardware.sh` 已經在開頭補上這行 PATH。如果仍然偵測不到:

```bash
ls -l /usr/local/cuda/bin/nvcc     # 確認 CUDA toolkit 真的有裝
cat hardware.json                  # 看 cuda.nvcc 欄位是空的還是有值
```

JetPack 沒有 `nvidia-smi` 是正常的,不代表 CUDA 壞掉。

## whisper.cpp 編譯失敗

先確認 `hardware.json` 的 `accelerator` 與 `cuda.arch` 是不是預期值 —— 探測錯了,
編譯旗標就會錯。要重來:

```bash
rm -rf engine/whisper.cpp/build
bash scripts/probe_hardware.sh
bash scripts/setup_engine.sh
```

指定其他版本:`WHISPER_REF=v1.7.4 bash scripts/setup_engine.sh`。

## 模型轉檔失敗

`scripts/convert_model.sh` 常見的幾種卡點:

**`Conversion needs Python packages that are not installed`** —— 轉檔需要 torch
與 transformers(`requirements-convert.txt`),推理端不需要。Jetson 上 PyPI 沒有
對應 JetPack 的 torch wheel,裝下去不是失敗就是裝到跟 CUDA runtime 對不起來的
CPU 版。**在桌機轉好再把 `.bin` 複製過去**,然後走 `scripts/fetch_model.sh <path>`。

**`is not a readable .npz`** —— 抓 mel filterbank 時被 proxy 或登入頁攔截,存下來
的是 HTML 而不是 npz。刪掉 `var/model-src/openai-whisper/whisper/assets/mel_filters.npz`
重跑;網路受限的話用 `MEL_FILTERS_URL=` 指到自己的鏡像。

**`neither vocab.json nor tokenizer.json is present`** —— 來源不是 Whisper 架構的
repo,或下載被截斷。用 `--keep-src` 重跑後檢查 `var/model-src/` 底下的內容。

**轉到一半被 OOM kill** —— 轉 large-v2 時 torch 會把整份權重讀進記憶體,尖峰大約
需要 8 GB RAM。這一步跟 GPU 無關,加 swap 或換台機器轉都可以。

轉檔中斷後重跑不必重新下載:加 `--keep-src` 會把下載內容留在 `var/model-src/`,
下次用 `--src` 指過去即可。

## 講者分離 API 回 500

長期背景執行的 WhisperX 服務(8088)在 `subprocess` 載入 PyTorch/Pyannote 時,
可能因父進程記憶體碎裂導致 `fork()` 回傳 `ENOMEM`。

重啟該服務即可,不是本專案的 bug:

```bash
kill -9 <8088 的 PID>
# 重新啟動 WhisperX 服務
```

VRAM 低於 7 GB 的機器上,探測會直接把這個功能關掉,介面上不會出現按鈕。

## 服務起不來

```bash
scripts/service.sh status
scripts/service.sh logs realtime
```

啟動時會檢查 `whisper-cli` 與模型是否存在,缺任一個就直接結束並印出路徑 ——
比起跑起來然後每一段都靜默失敗,這樣好除錯。

## 磁碟被吃滿

`var/` 底下是執行時產生的:上傳暫存、轉寫結果、日誌。即時聽寫的音檔切段轉寫完
就刪除,但批次服務的結果會留著。

```bash
du -sh var/*
```

需要放到別的磁碟時,在 `.env` 用絕對路徑改 `UPLOAD_DIR` / `RESULT_DIR` /
`LOG_DIR` / `WORK_DIR`。
