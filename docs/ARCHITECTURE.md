# Architecture — STT + Speaker Diarization Pipeline

## Overview

The pipeline converts any audio/video file into a speaker-labeled transcript
by running two models in parallel and then merging their outputs.

```
Audio file
    │
    ▼
┌─────────────────────────────────────┐
│  ffmpeg: convert to WAV 16kHz mono  │
└─────────────────────────────────────┘
    │                    │
    ▼                    ▼ (if --denoise)
┌──────────┐      ┌─────────────┐
│  original│      │  denoised   │
│   WAV    │      │    WAV      │
└──────────┘      └─────────────┘
    │                    │
    ▼                    ▼
┌──────────────┐  ┌──────────────────┐
│   WHISPER    │  │    PYANNOTE      │
│  large-v3    │  │  community-1     │
│              │  │                  │
│ word-level   │  │ speaker segments │
│ timestamps   │  │ + overlap flags  │
└──────────────┘  └──────────────────┘
    │                    │
    └──────────┬─────────┘
               ▼
    ┌─────────────────────┐
    │  ALIGNMENT          │
    │  max-overlap assign │
    └─────────────────────┘
               │
               ▼
    ┌─────────────────────┐
    │  TRANSCRIPT         │
    │  [MM:SS] SPK: text  │
    └─────────────────────┘
```

---

## Full pipeline flowchart

```mermaid
flowchart TD
    INPUT([🎙 Audio / Video file])

    subgraph PREP ["SECTION 1 — Audio Preparation"]
        CONVERT["convert_audio_to_wav()\nffmpeg → WAV 16kHz mono PCM-16"]
        PROBE["probe_audio_duration()\nffprobe → duration in seconds"]
        DENOISE["denoise_wav()\nnoisereduce spectral gating\n⚠ optional, diarize path only"]
    end

    subgraph ASR ["SECTION 2 — Speech-to-Text  (faster-whisper)"]
        LOAD_W["load_whisper_model()\nlarge-v3, float16/int8"]
        DURATION_CHECK{duration\n> 30 min?}
        BATCHED["BatchedInferencePipeline\nbatch_size=16\nVAD: on by default"]
        STANDARD["WhisperModel.transcribe\nvad_filter=True\nbeam_size=5"]
        WORDS[/"List[WordStamp]\ntext, start_sec, end_sec, confidence"/]
    end

    subgraph DIARIZE ["SECTION 3 — Speaker Diarization  (pyannote community-1)"]
        LOAD_P["DiarizationPipeline.from_pretrained()\npyannote/speaker-diarization-community-1"]
        TUNE["_apply_hyperparameter_overrides()\nclustering_threshold\nsegmentation_threshold"]
        RUN_P["pipeline(wav_path)\nSpeaker Change Detection ✓\nOverlapped Speech Detection ✓"]

        subgraph DIARIZE_OUT ["DiarizeOutput"]
            FULL_ANN["speaker_diarization\nAnnotation — incl. overlap tracks"]
            EXCL_ANN["exclusive_speaker_diarization\nAnnotation — 1 speaker / moment"]
        end

        SWEEP["_find_overlap_intervals()\nevent-sweep algorithm\ndepth ≥ 2 → overlap region"]
        BUILD_SEGS["build SpeakerSegment list\nfrom exclusive annotation\n+ _segment_intersects_overlap()"]
        RELABEL["_relabel_by_first_appearance()\nSPEAKER_00 = first to speak\nSPEAKER_01 = second, etc."]
        SEGMENTS[/"List[SpeakerSegment]\nspeaker_id, start_sec, end_sec, is_overlap"/]
    end

    subgraph ALIGN ["SECTION 4 — Alignment"]
        ALIGN_FN["align_words_to_speakers()\nmax intersection-duration assignment\ngap fallback → inherit last speaker"]
        MERGE["merge consecutive same-speaker words\n→ AlignedUtterance"]
        UTTERANCES[/"List[AlignedUtterance]\nspeaker_id, start_sec, end_sec, text, is_overlap"/]
    end

    subgraph OUTPUT ["SECTION 5 — Output"]
        FORMAT["format_final_transcript()\n[MM:SS - MM:SS] SPEAKER_XX: text\n[OVERLAP] tag for overlap zones"]
        SAVE["write to --output file\n(UTF-8)"]
        TRANSCRIPT([📄 Transcript])
    end

    subgraph DEBUG ["Debug files  (--debug-dir)"]
        D1["01_whisper_raw.txt\nper-word timestamps + confidence"]
        D2["02_diarization_raw.txt\nper-speaker stats + segment list"]
        D3["03_final_transcript.txt"]
        D4["04_alignment_debug.txt\nside-by-side comparison table"]
    end

    INPUT --> CONVERT
    CONVERT --> PROBE
    PROBE --> LOAD_W
    CONVERT --> DENOISE

    LOAD_W --> DURATION_CHECK
    DURATION_CHECK -- yes --> BATCHED
    DURATION_CHECK -- no  --> STANDARD
    BATCHED  --> WORDS
    STANDARD --> WORDS

    CONVERT --> LOAD_P
    DENOISE --> LOAD_P
    LOAD_P --> TUNE --> RUN_P
    RUN_P --> FULL_ANN
    RUN_P --> EXCL_ANN
    FULL_ANN --> SWEEP
    EXCL_ANN --> BUILD_SEGS
    SWEEP    --> BUILD_SEGS
    BUILD_SEGS --> RELABEL --> SEGMENTS

    WORDS    --> ALIGN_FN
    SEGMENTS --> ALIGN_FN
    ALIGN_FN --> MERGE --> UTTERANCES

    UTTERANCES --> FORMAT --> SAVE --> TRANSCRIPT

    WORDS      -.->|debug| D1
    SEGMENTS   -.->|debug| D2
    TRANSCRIPT -.->|debug| D3
    UTTERANCES -.->|debug| D4
```

---

## Diagnose mode flowchart

```mermaid
flowchart LR
    INPUT([Audio file])
    CONVERT["convert_audio_to_wav()"]
    LOAD["load DiarizationPipeline"]

    subgraph SWEEP_LOOP ["run_threshold_sweep()"]
        direction TB
        THRESHOLDS["thresholds:\n0.55 → 0.90"]
        SET["_safe_set_pipeline_param\nclustering.threshold = t"]
        RUN["pipeline(wav_path)"]
        COUNT["count unique speakers\ncount segments"]
        PRINT["print row to table"]
        THRESHOLDS --> SET --> RUN --> COUNT --> PRINT --> THRESHOLDS
    end

    TABLE([📊 Threshold × Speakers table])

    INPUT --> CONVERT --> LOAD --> SWEEP_LOOP --> TABLE
```

---

## Data model relationships

```mermaid
classDiagram
    class WordStamp {
        +str text
        +float start_sec
        +float end_sec
        +float confidence
    }

    class SpeakerSegment {
        +str speaker_id
        +float start_sec
        +float end_sec
        +bool is_overlap
    }

    class AlignedUtterance {
        +str speaker_id
        +float start_sec
        +float end_sec
        +str text
        +bool is_overlap
    }

    WordStamp      "N" --> "1" AlignedUtterance : merged by alignment
    SpeakerSegment "1" --> "N" AlignedUtterance : tagged by speaker
```

---

## Section responsibilities

| Section | Functions | Responsibility |
|---|---|---|
| 1 — Audio | `convert_audio_to_wav` `probe_audio_duration` `denoise_wav` | Normalise input to WAV 16kHz mono; optional denoising for diarize path |
| 2 — STT | `load_whisper_model` `transcribe` | Adaptive ASR (batched vs standard); returns `List[WordStamp]` |
| 3 — Diarize | `diarize` + helpers | Load pyannote, detect overlaps via event-sweep, build + re-label segments |
| 4 — Align | `align_words_to_speakers` | Assign each word to speaker by max temporal overlap; merge into utterances |
| 5 — Format | `format_*` `save_alignment_debug` | Produce human-readable transcript and optional debug files |
| 6 — Orchestrate | `run_pipeline` | Wire steps 1–5; manage temp directory; save outputs |
| 7 — Diagnose | `run_threshold_sweep` | Fast threshold sweep without running Whisper |
| 8 — CLI | `build_arg_parser` `main` | Parse arguments; dispatch to run_pipeline or run_threshold_sweep |

---

## Key design decisions

### Adaptive inference (Section 2)
Audio longer than 30 minutes uses `BatchedInferencePipeline` for throughput.
Shorter audio uses `WhisperModel.transcribe` directly for lower latency.
VAD must be set explicitly (`vad_filter=True`) in the non-batched path.

### Split-input denoising (Section 1 + 6)
When `--denoise` is enabled, only the diarization pass receives the denoised
audio. Whisper always receives the original WAV because spectral gating can
distort phonemes (especially Japanese short vowels and geminate consonants),
which would increase word error rate.

### Overlap detection via event-sweep (Section 3)
Rather than comparing timestamps between two annotations directly (fragile due
to floating-point differences), the pipeline builds an event list of
segment-open (+1) and segment-close (-1) events from `speaker_diarization`,
then sweeps through time. Any interval where depth ≥ 2 is an overlap region.
`exclusive_speaker_diarization` segments are then tagged via interval intersection.

### Chronological speaker re-labeling (Section 3)
pyannote assigns speaker IDs from internal clustering order, not appearance order.
After sorting segments by `start_sec`, the pipeline walks the list once and
assigns `SPEAKER_00` to the first new speaker encountered, `SPEAKER_01` to the
second, and so on. This makes transcripts easier to read.

### Word assignment by max overlap (Section 4)
Each `WordStamp` is assigned to the `SpeakerSegment` whose time interval
overlaps it the most. This is more robust than simple containment checks,
especially at segment boundaries where a word's timestamp may straddle
two adjacent segments.
