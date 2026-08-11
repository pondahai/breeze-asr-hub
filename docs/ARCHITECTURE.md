# 架構

## 整體

```
                    ┌─────────────────────────────┐
                    │        hardware.json        │
                    │  probe_hardware.sh 產生      │
                    │  CUDA / GPU / mic / 能力矩陣  │
                    └──────────────┬──────────────┘
                                   │  被四方讀取
          ┌────────────────┬───────┴────────┬─────────────────┐
          ▼                ▼                ▼                 ▼
   setup_engine.sh   breeze_hub/       services/batch    services/realtime
   (編譯旗標)         config.py         (功能開關)         (麥克風/門檻)
                          │
                          ▼
                   ┌─────────────┐
                   │    .env     │  使用者手動設定,覆寫探測結果
                   └─────────────┘
```

單一事實來源是刻意的:換一台機器只要重跑探測,編譯旗標、麥克風選擇、UI 按鈕
會一起跟著變,不需要在四個地方各改一次。

## 資料流

### 即時聽寫 (8015 / 8016)

```
麥克風 ──arecord──▶ vad_loop ──┬──▶ WebSocket 廣播 (10 fps: RMS + PCM)
         S16LE 16k             │
         100ms/frame           └──▶ 語音切段 ──▶ asr_queue
                                                    │
                                            asr_worker (單一 worker)
                                                    │
                                              whisper-cli
                                                    │
                                        ┌───────────┼───────────┐
                                        │           │           │
                                   FLAC 錄音   history.jsonl  WebSocket 廣播
                                  audio/日期/      (輪替)       (逐字稿)
```

三條 thread:

| Thread | 職責 | 為什麼獨立 |
| --- | --- | --- |
| `vad_loop` | 讀取音框、算 RMS、切段 | 必須不被阻塞,否則 ALSA buffer 溢位掉音 |
| `asr_worker` | 消化佇列、呼叫 whisper-cli | 推理是秒級的,不能擋住收音 |
| `ws_server` | 擁有 asyncio loop | 另外兩條用 `run_coroutine_threadsafe` 送訊息進來 |

**ASR worker 只有一條**,是刻意的。GPU 一次只有一份記憶體預算,平行送多段進去
只會互相排擠;串列處理讓批次服務仍有餘裕同時工作。

### 批次轉寫 (8014)

```
分片上傳 ──▶ 合併 ──▶ ffmpeg 轉 16k 單聲道
                          │
                          ├─(短)─▶ whisper-cli ──▶ txt/srt/vtt
                          │
                          └─(長)─▶ 靜音偵測切段 ──▶ 逐段轉寫 ──▶ 合併字幕時間軸
                                                                    │
                                          選配:WhisperX 講者分離 ────┤
                                          選配:LLM 摘要 ────────────┘
```

## 留存與回放

逐字稿寫進 `history.jsonl`,語音另存一份 FLAC。兩者都在廣播之前完成,所以送到瀏覽
器的那筆已經帶著 `audio` 旗標,前端不必再問一次就知道能不能播。

錄音檔名直接用該筆的 `id`,而 `id` 本來就是毫秒時戳,於是日期目錄可以從 id 反推:
`/api/audio/<id>` 因此是純函式,不需要任何對照表跟著資料一起維護。

```
entry id 1786439613573
   └─▶ 2026-08-11/1786439613573.flac
```

FLAC 是無損的,回放聽到的就是送進 ASR 的同一份訊號,大小約 wav 的四成。

寫錄音與寫留存檔都是 best effort:磁碟滿了、ffmpeg 不在、SD 卡沒掛上,都只會少一份
備份,不會連帶讓這段逐字稿消失。

`history.jsonl` 依大小輪替;錄音則不自動過期,由 `/api/storage` 的儀表交給人決定刪
除範圍 —— 刪除以整個日期目錄為單位,且只接受長得像日期的目錄名,壞參數碰不到錄音
以外的東西。

## VAD

門檻是**原始 int16 RMS**(0–32767),不是正規化浮點數。理由是這個數字與網頁上
音壓計顯示的完全一致 —— 使用者看到「說話時大約 3000、安靜時大約 400」,就能直接
把門檻設在中間,不需要換算。

```
rms > threshold        ─▶ 進入 RECORDING,累積音框
rms <= threshold 持續   ─▶ 累計靜音;達 VAD_SILENCE_DURATION_SEC 就切段
段落 < VAD_MIN_SPEECH_SEC  ─▶ 丟棄(咳嗽、關門聲)
段落 >= VAD_MAX_SPEECH_SEC ─▶ 強制切段(避免有人講不停造成延遲累積)
```

門檻**不可以跨機器複製**。不同麥克風、不同增益、不同房間的底噪差好幾倍,所以有
`python3 -m breeze_hub.calibrate`:量測環境底噪 (p95) 與說話音量 (p25),取中點。

## 麥克風選擇

`arecord -L` 列出來的裝置不代表讀得到聲音。Jetson 上的 `APE` 是 SoC 內建音訊
處理器,通常什麼都沒接,選到它的結果是音壓計永遠不動 —— 一個不會報錯的失敗。

所以 `breeze_hub.audio.open_capture()` 不是挑一個裝置,而是**建立候選清單依序實測**:

1. `.env` 的 `MIC_DEVICE`(明確指定永遠優先)
2. 名稱含 `MIC_KEYWORD` 的 `sysdefault` PCM
3. 任何非 `APE` 的 `sysdefault:CARD=*`
4. `default`

每個候選實際開起來讀 1 秒,真的吐得出 PCM 才採用。全部失敗才丟例外 —— 一個沒有
麥克風的聽寫服務沒有存在意義,不該假裝正常。

## 前端

即時聽寫台的 HTML 是純靜態檔案,無 CDN、無 build step、無框架。啟動時向
`/api/config` 問三件事:WebSocket 埠、有沒有攝影機、目前的加速器。所以改 `.env`
的埠號、或機器上沒有 OpenCV,前端會自己適應,不必改程式碼。

音訊監聽用 Web Audio API 的 `AudioContext` 解 PCM 播放,而不是 `<audio>`:延遲從
數百毫秒降到 20 毫秒內,也避開瀏覽器的 autoplay 阻擋(仍需一次使用者點擊解鎖)。

回放錄音則反過來,用一般的 `Audio()`:那是點下去才播的離線檔案,延遲不重要,交給
瀏覽器解 FLAC 最省事。整個 console 共用一個播放器,開始一段就停掉上一段。

## 併發

參考機實測:批次與即時兩個 `whisper-cli` 同時載入同一顆 3 GB 模型推理,
18.5 秒內雙雙完成,無 OOM、無死鎖。Xavier 的 14 GB 統一記憶體足以容納兩份。

VRAM 較小的機器上這件事不保證成立。若要保守運作,把批次服務停掉再跑即時,
或把 `ASR_THREADS` 調低。
