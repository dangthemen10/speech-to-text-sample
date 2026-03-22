# Speech-to-Text + Speaker Diarization System

A fully self-hosted, offline-capable pipeline combining:

- **faster-whisper** (`large-v3`) — word-level transcription via CTranslate2
- **BatchedInferencePipeline** — efficient long-audio processing
- **Silero VAD** — built-in silence removal
- **pyannote.audio 4.0+** — `community-1` neural speaker diarization + overlap detection

---

## Architecture

```
Audio File
    │
    ▼
┌─────────────────────────────────┐
│  faster-whisper (large-v3)      │
│  + BatchedInferencePipeline     │  ← efficient chunked GPU inference
│  + Silero VAD (silence removal) │
│  + word_timestamps=True         │
└──────────────┬──────────────────┘
               │  List[WordToken]
               │  {word, start, end, probability}
               ▼
┌─────────────────────────────────┐
│  pyannote/speaker-diarization-community-1   │
│  - Speaker turn segmentation    │
│  - Overlapping speech detection │
└──────────────┬──────────────────┘
               │  List[SpeakerSegment]
               │  {speaker, start, end, overlapping}
               ▼
┌─────────────────────────────────┐
│  Alignment Engine               │
│  - Intersection-duration method │
│  - Overlap-aware word grouping  │
└──────────────┬──────────────────┘
               │
               ▼
[00:00 - 00:05] Speaker A: Hello.
[00:05 - 00:09] Speaker B: How are you?
[00:09 - 00:11] Speaker A & Speaker B [OVERLAP]: ...
```

---

## Setup

### 1. System dependencies

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

### 2. Python environment (Python 3.9+)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# GPU (CUDA 12.x)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Common dependencies
pip install -r requirements.txt
```

### 3. HuggingFace token (for pyannote models)

pyannote models require accepting terms on HuggingFace:

1. Create account at https://huggingface.co
2. Accept terms at: https://huggingface.co/pyannote/speaker-diarization-community-1
3. Generate a token at: https://huggingface.co/settings/tokens

Pass the token at runtime (see Usage below) or set `HF_TOKEN` in your environment.

---

## Usage

### Basic (auto language detection)

```bash
python stt_diarize.py --audio meeting.wav --hf-token hf_xxxx
```

### Japanese audio with proper noun hints

```bash
python stt_diarize.py \
  --audio interview_jp.wav \
  --language ja \
  --prompt "田中さん、鈴木部長、プロジェクトアルファ" \
  --hf-token hf_xxxx \
  --output transcript.txt
```

### Known number of speakers

```bash
python stt_diarize.py \
  --audio podcast.mp3 \
  --num-speakers 2 \
  --hf-token hf_xxxx
```

### Speaker count range

```bash
python stt_diarize.py \
  --audio roundtable.wav \
  --min-speakers 2 \
  --max-speakers 5 \
  --hf-token hf_xxxx
```

---

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--audio` | *(required)* | Path to audio file |
| `--language` | auto | BCP-47 code (`ja`, `en`, `zh`, …) |
| `--prompt` | — | Initial prompt for Whisper (proper nouns) |
| `--model` | `large-v3` | Whisper model size |
| `--hf-token` | `$HF_TOKEN` | HuggingFace access token |
| `--num-speakers` | — | Exact speaker count |
| `--min-speakers` | — | Min speaker count |
| `--max-speakers` | — | Max speaker count |
| `--output` | — | Save transcript to file |

---

## Output Format

```
[MM:SS - MM:SS] Speaker LABEL: transcribed text
[MM:SS - MM:SS] Speaker A & Speaker B [OVERLAP]: overlapping text
```

Example:

```
[00:00 - 00:04] SPEAKER_00: おはようございます、田中部長。
[00:04 - 00:09] SPEAKER_01: おはよう。今日の会議の準備はできていますか？
[00:09 - 00:11] SPEAKER_00 & SPEAKER_01 [OVERLAP]: はい、もちろん
[00:11 - 00:17] SPEAKER_00: プロジェクトアルファについてご報告があります。
```

---

## Alignment Algorithm

The alignment engine uses **intersection-duration voting**:

```
For each WordToken [ws → we]:
  1. Collect all diarization segments that intersect [ws, we]
  2. Compute intersection duration with each candidate segment
  3. Primary speaker = max intersection duration
  4. If multiple speakers intersect AND any segment is flagged as
     overlapping → mark word as overlapping, collect all speakers
  5. Group consecutive words with same speaker(s) into utterances
```

This handles:
- Words on speaker boundaries (assigned to whoever "owns" most of the word)
- True overlapping speech (multi-speaker labelling + `[OVERLAP]` tag)
- Gaps in diarization (nearest-segment fallback)

---

## Offline / Air-gapped Deployment

```bash
# Pre-download Whisper model
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3')"

# Pre-download pyannote model (requires token once)
python -c "
from pyannote.audio import Pipeline
Pipeline.from_pretrained('pyannote/speaker-diarization-community-1', token='hf_xxxx')
"

# Run fully offline
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python stt_diarize.py --audio meeting.wav
```

---

## Performance Tips

| Scenario | Recommendation |
|----------|---------------|
| GPU available | Runs on CUDA automatically; `float16` compute |
| CPU only | Uses `int8` quantization; slower but functional |
| Very long audio (>1 hr) | Increase `batch_size` in `run_stt()` |
| Many speakers | Set `--max-speakers` to bound search space |
| Low VRAM (<8 GB) | Use `medium` or `small` model via `--model` |