# Audio to Summarize

在 Mac 本機把上課錄音轉成逐字稿，再交給 Claude 整理重點，錄音不會離開電腦。

- 模型：[Cohere Transcribe](https://docs.cohere.com/docs/transcribe) 2B（MLX int8 版，透過 [mlx-speech](https://github.com/appautomaton/mlx-speech) 執行）
- 自動判斷每個檔案是英文還是中文授課；中文課的英文術語會保留英文，並轉成台灣繁體
- 實測 M5 / 32GB：約 **50 倍速**（1 小時的課約 72 秒），記憶體峰值約 4GB

## 流程

```
語音備忘錄 ──import_voice_memos.py──▶ recordings/<課程>/*.m4a
                                            │
                                     transcribe.py
                                            ▼
                                  transcripts/<課程>/*.txt / *.srt ──▶ Claude 整理重點
```

## 安裝

需要 Apple Silicon Mac、Python 3.13 以上、[uv](https://docs.astral.sh/uv/)。

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

第一次執行會自動下載模型（約 4GB，存在 `~/.cache/huggingface`）。

## 使用

### 1. 匯入錄音（可選）

`import_voice_memos.py` 會依照錄音的**開始時間**，把語音備忘錄複製到對應課程的資料夾。課表寫在檔案上方的 `SCHEDULE`，可以自行修改：

| 時間 | 課程 |
|---|---|
| 星期一 12:00 前 | 機器學習 |
| 星期二 09:00–10:00 | 實驗室會議 |
| 星期二 10:00–13:00 | 機器學習 |
| 星期二 13:00–16:00 | 影像處理 |
| 星期五 12:00 前 | 資料庫系統 |

```bash
.venv/bin/python import_voice_memos.py --dry   # 先預覽
.venv/bin/python import_voice_memos.py         # 實際複製
```

- 只會**複製**，不會動到語音備忘錄裡的原檔；短於 1 分鐘、不符合任何時段、已經複製過的錄音會略過。
- macOS 會保護語音備忘錄的資料夾，執行時需要暫時開啟終端機的「完整磁碟取用權限」（系統設定 → 隱私權與安全性），用完可以關掉。
- 不想開權限的話，也可以直接從語音備忘錄 app 把錄音拖進 `recordings/<課程>/`。

### 2. 轉錄

```bash
.venv/bin/python transcribe.py                       # 轉 recordings/ 底下全部
.venv/bin/python transcribe.py path/to/lecture.m4a   # 指定檔案或資料夾
.venv/bin/python transcribe.py --overwrite           # 重轉已經轉過的
.venv/bin/python transcribe.py --language en         # 強制語言（預設 auto）
```

已經有逐字稿的檔案會自動跳過，所以可以分批加入新錄音再執行。

## 輸入與輸出

**輸入**：`recordings/<課程>/` 底下的音檔，支援 m4a、mp3、wav、aiff、flac、mp4 等格式。子資料夾結構會照樣保留到輸出。

**輸出**：每個錄音會在 `transcripts/<課程>/` 產生兩個檔案：

| 檔案 | 內容 |
|---|---|
| `<名稱>.txt` | 逐字稿，每 2 分鐘一段，開頭標 `[時:分:秒]`，**拿來給 Claude 整理** |
| `<名稱>.srt` | 字幕檔，每段約 30 秒，可以搭配錄音播放 |

`.txt` 的格式長這樣（示意，實際輸出見下方範例）：

```
[00:00:00] Good morning, everyone. Today, we are going to continue our discussion of gradient descent ...

[00:02:04] So the learning rate controls the step size. If it is too large, the training will diverge ...
```

## 範例

[`examples/`](examples) 裡有兩段**用 macOS 語音合成產生的假課堂錄音**，以及它們實際跑出來的結果，可以先用來確認環境：

```bash
.venv/bin/python transcribe.py examples/audio -o examples/output --overwrite
```

| 範例 | 長度 | 判斷語言 | 錯誤率 | 錯在哪 |
|---|---|---|---|---|
| `english_lecture`（英文，gradient descent） | 1:12 | en | WER 0.9%（2/214 詞） | Adam → "ADOM"、"Atom" |
| `mandarin_lecture`（中文夾英文術語，color model） | 1:20 | zh | CER 1.2%（4/341 字） | 佔→站、慢→滿、B乘→比呈 |

- `examples/audio/`：輸入音檔
- `examples/reference/`：朗讀用的原稿（正確答案）
- `examples/output/`：實際輸出的 `.txt` 和 `.srt`

合成語音很乾淨，真實教室錄音（回音、離麥克風遠、口音）的錯誤率會明顯比較高。

## 參數

| 項目 | 設定 |
|---|---|
| 音訊 | 用 macOS 內建 `afconvert` 轉成 16kHz 單聲道（失敗時改用 ffmpeg） |
| 切段 | 每段最長 35 秒，在最安靜的地方切開 |
| 解碼 | Greedy，每段最多 448 token，含標點 |
| 語言判斷 | 在 20/40/60/80% 位置各取 30 秒用中文模式解碼，中文字比例 > 30% 的樣本過半就判定為中文 |
| 後處理 | 刪除連續重複 3 次以上的片段（模型在靜音或雜音時偶爾會鬼打牆）；中文用 OpenCC `s2twp` 轉台灣繁體 |
| 記憶體 | MLX 快取上限 512MB（不設的話長時間批次會吃到 20GB 以上） |

## 已知限制

- 專有名詞容易錯（例如 training error → "training arrow"），整理重點時把課程講義一起給 Claude 可以修正大部分。
- Cohere Transcribe 沒有內建語言偵測，也無法事先提供詞彙表，所以語言是用上面的取樣方式判斷。

## 為什麼錄音和逐字稿沒有上傳

`recordings/`、`transcripts/` 都在 `.gitignore` 裡。錄音和逐字稿包含老師的授課內容和實驗室會議的討論（可能有未發表的研究），所以只留在本機，這個 repo 只放程式。
