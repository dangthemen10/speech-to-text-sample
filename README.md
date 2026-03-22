# STT + Speaker Diarization

Japanese-optimized speech-to-text with speaker identification.
**Stack:** `faster-whisper large-v3` + `pyannote/speaker-diarization-community-1`

---

## Prerequisites

### 1. System — install ffmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add bin/ to PATH
```

### 2. HuggingFace token

1. Create a **Read** token at <https://hf.co/settings/tokens>
2. Accept model terms at <https://huggingface.co/pyannote/speaker-diarization-community-1>

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
```

### 3. Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip

# PyTorch (macOS / Linux CPU)
pip install "torch>=2.8.0" --prefer-binary

# PyTorch (Linux CUDA 12.1)
# pip install "torch>=2.8.0" --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

---

## Usage

### Basic

```bash
python stt_diarization.py meeting.mp4
```

### With known speaker count (recommended)

```bash
python stt_diarization.py interview.m4a \
  --num-speakers 2 \
  --output transcript.txt
```

### With domain vocabulary (better accuracy for Japanese proper nouns)

```bash
python stt_diarization.py meeting.mp4 \
  --initial-prompt "トヨタ自動車、ソフトバンクグループ、田中部長" \
  --num-speakers 3 \
  --output transcript.txt
```

### Noisy audio (office / café)

```bash
python stt_diarization.py noisy_call.mp3 \
  --denoise \
  --denoise-strength 0.80 \
  --num-speakers 2
```

### Debug alignment issues

```bash
python stt_diarization.py meeting.mp4 \
  --num-speakers 2 \
  --debug-dir ./debug \
  --output transcript.txt
# Produces: debug/01_whisper_raw.txt, 02_diarization_raw.txt,
#           03_final_transcript.txt, 04_alignment_debug.txt
```

### Tune diarization threshold (fast, no Whisper)

```bash
python stt_diarization.py meeting.mp4 --diagnose
# Then apply the best threshold:
python stt_diarization.py meeting.mp4 --clustering-threshold 0.75
```

---

## Output format

```
[00:00 - 00:13] SPEAKER_00: そうですね、これも先ほど…
[00:13 - 00:21] SPEAKER_01: やっぱりその街の良さを…
[00:22 - 00:24] SPEAKER_02: 水をマレーシアから買わなくてはならない
[00:24 - 00:29] SPEAKER_00 [OVERLAP]: えっ、本当ですか？
```

- `SPEAKER_00` = first speaker to appear in the audio
- `[OVERLAP]` = multiple speakers detected simultaneously

---

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--hf-token` | `$HF_TOKEN` | HuggingFace read token |
| `--language` | `ja` | ISO-639-1 language code |
| `--initial-prompt` | — | Domain vocabulary seed for Whisper |
| `--num-speakers` | auto | Exact speaker count (bypasses clustering) |
| `--min/max-speakers` | — | Speaker count bounds |
| `--clustering-threshold` | ~0.715 | Higher = fewer speakers |
| `--segmentation-threshold` | ~0.817 | Lower = catches more short speech |
| `--denoise` | off | Denoise before diarization |
| `--denoise-strength` | 0.85 | 0.0–1.0, sweet spot 0.75–0.85 |
| `--output / -o` | stdout | Save transcript to file |
| `--debug-dir` | — | Save 4 intermediate debug files |
| `--diagnose` | off | Threshold sweep only (no Whisper) |

---

## Troubleshooting

| Error | Fix |
|---|---|
| `HTTP 403` on model load | Accept model terms at the HuggingFace link above |
| `AttributeError: AudioMetaData` | Upgrade to `pyannote.audio>=4.0.1` + `torch>=2.8.0` |
| Wrong speaker count | Run `--diagnose` and adjust `--clustering-threshold` |
| Speaker A/B confused | Add `--denoise` or pass `--num-speakers` |
| Short speech missed | Lower `--segmentation-threshold` to 0.65–0.70 |
