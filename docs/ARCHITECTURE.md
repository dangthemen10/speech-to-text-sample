# Kiến trúc — Pipeline STT + Phân tách người nói

## Tổng quan

Pipeline nhận bất kỳ file audio/video nào và tạo ra bản transcript có gán nhãn người nói,
bằng cách chạy song song hai model rồi gộp kết quả lại.

```
File audio
    │
    ▼
┌─────────────────────────────────────┐
│  ffmpeg: chuyển sang WAV 16kHz mono │
└─────────────────────────────────────┘
    │                    │
    ▼                    ▼ (nếu --denoise)
┌──────────┐      ┌─────────────┐
│  WAV gốc │      │  WAV đã khử │
│          │      │   tiếng ồn  │
└──────────┘      └─────────────┘
    │                    │
    ▼                    ▼
┌──────────────┐  ┌──────────────────┐
│   WHISPER    │  │    PYANNOTE      │
│  large-v3    │  │  community-1     │
│              │  │                  │
│ timestamps   │  │ speaker segments │
│ từng từ      │  │ + overlap flags  │
└──────────────┘  └──────────────────┘
    │                    │
    └──────────┬─────────┘
               ▼
    ┌─────────────────────┐
    │  ALIGNMENT          │
    │  gán từ → người nói │
    └─────────────────────┘
               │
               ▼
    ┌─────────────────────┐
    │  TRANSCRIPT         │
    │  [MM:SS] SPK: text  │
    └─────────────────────┘
```

---

## Sơ đồ luồng toàn bộ pipeline

```mermaid
flowchart TD
    INPUT([🎙 File Audio / Video đầu vào])

    subgraph PREP ["SECTION 1 — Chuẩn bị Audio"]
        CONVERT["convert_audio_to_wav()\nffmpeg → WAV 16kHz mono PCM-16"]
        PROBE["probe_audio_duration()\nffprobe → độ dài tính bằng giây"]
        DENOISE["denoise_wav()\nnoisereduce spectral gating\n⚠ tuỳ chọn, chỉ cho nhánh diarize"]
    end

    subgraph ASR ["SECTION 2 — Nhận dạng giọng nói  (faster-whisper)"]
        LOAD_W["load_whisper_model()\nlarge-v3, float16/int8"]
        DURATION_CHECK{độ dài\n> 30 phút?}
        BATCHED["BatchedInferencePipeline\nbatch_size=16\nVAD: bật mặc định"]
        STANDARD["WhisperModel.transcribe\nvad_filter=True\nbeam_size=5"]
        WORDS[/"List[WordStamp]\ntext, start_sec, end_sec, confidence"/]
    end

    subgraph DIARIZE ["SECTION 3 — Phân tách người nói  (pyannote community-1)"]
        LOAD_P["DiarizationPipeline.from_pretrained()\npyannote/speaker-diarization-community-1"]
        TUNE["_apply_hyperparameter_overrides()\nclustering_threshold\nsegmentation_threshold"]
        RUN_P["pipeline(wav_path)\nSpeaker Change Detection ✓\nOverlapped Speech Detection ✓"]

        subgraph DIARIZE_OUT ["DiarizeOutput"]
            FULL_ANN["speaker_diarization\nAnnotation — bao gồm các track overlap"]
            EXCL_ANN["exclusive_speaker_diarization\nAnnotation — 1 người nói / thời điểm"]
        end

        SWEEP["_find_overlap_intervals()\nthuật toán event-sweep\ndepth ≥ 2 → vùng overlap"]
        BUILD_SEGS["xây dựng danh sách SpeakerSegment\ntừ exclusive annotation\n+ _segment_intersects_overlap()"]
        RELABEL["_relabel_by_first_appearance()\nSPEAKER_00 = người nói đầu tiên\nSPEAKER_01 = người thứ hai, v.v."]
        SEGMENTS[/"List[SpeakerSegment]\nspeaker_id, start_sec, end_sec, is_overlap"/]
    end

    subgraph ALIGN ["SECTION 4 — Alignment"]
        ALIGN_FN["align_words_to_speakers()\ngán theo max intersection-duration\ngap fallback → kế thừa speaker trước"]
        MERGE["gộp các từ liên tiếp cùng speaker\n→ AlignedUtterance"]
        UTTERANCES[/"List[AlignedUtterance]\nspeaker_id, start_sec, end_sec, text, is_overlap"/]
    end

    subgraph OUTPUT ["SECTION 5 — Xuất kết quả"]
        FORMAT["format_final_transcript()\n[MM:SS - MM:SS] SPEAKER_XX: text\nthêm tag [OVERLAP] cho vùng chen lấn"]
        SAVE["ghi ra file --output\n(UTF-8)"]
        TRANSCRIPT([📄 Transcript])
    end

    subgraph DEBUG ["File debug  (--debug-dir)"]
        D1["01_whisper_raw.txt\ntimestamp từng từ + confidence"]
        D2["02_diarization_raw.txt\nthống kê theo speaker + danh sách segment"]
        D3["03_final_transcript.txt"]
        D4["04_alignment_debug.txt\nbảng so sánh song song"]
    end

    INPUT --> CONVERT
    CONVERT --> PROBE
    PROBE --> LOAD_W
    CONVERT --> DENOISE

    LOAD_W --> DURATION_CHECK
    DURATION_CHECK -- có --> BATCHED
    DURATION_CHECK -- không --> STANDARD
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

## Sơ đồ chế độ Diagnose

```mermaid
flowchart LR
    INPUT([File audio])
    CONVERT["convert_audio_to_wav()"]
    LOAD["tải DiarizationPipeline"]

    subgraph SWEEP_LOOP ["run_threshold_sweep()"]
        direction TB
        THRESHOLDS["các threshold:\n0.55 → 0.90"]
        SET["_safe_set_pipeline_param\nclustering.threshold = t"]
        RUN["pipeline(wav_path)"]
        COUNT["đếm số speaker\nđếm số segment"]
        PRINT["in dòng ra bảng"]
        THRESHOLDS --> SET --> RUN --> COUNT --> PRINT --> THRESHOLDS
    end

    TABLE([📊 Bảng Threshold × Số người nói])

    INPUT --> CONVERT --> LOAD --> SWEEP_LOOP --> TABLE
```

---

## Quan hệ giữa các data model

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

    WordStamp      "N" --> "1" AlignedUtterance : gộp bởi alignment
    SpeakerSegment "1" --> "N" AlignedUtterance : gán nhãn speaker
```

---

## Trách nhiệm từng section

| Section | Hàm | Trách nhiệm |
|---|---|---|
| 1 — Audio | `convert_audio_to_wav` `probe_audio_duration` `denoise_wav` | Chuẩn hoá đầu vào thành WAV 16kHz mono; khử tiếng ồn tuỳ chọn cho nhánh diarize |
| 2 — STT | `load_whisper_model` `transcribe` | ASR thích ứng (batched vs standard); trả về `List[WordStamp]` |
| 3 — Diarize | `diarize` + các hàm hỗ trợ | Tải pyannote, phát hiện overlap bằng event-sweep, xây dựng và đặt lại nhãn segment |
| 4 — Align | `align_words_to_speakers` | Gán từng từ vào speaker theo max temporal overlap; gộp thành utterance |
| 5 — Format | `format_*` `save_alignment_debug` | Tạo transcript dạng người đọc được và các file debug tuỳ chọn |
| 6 — Điều phối | `run_pipeline` | Kết nối các bước 1–5; quản lý thư mục tạm; lưu kết quả |
| 7 — Diagnose | `run_threshold_sweep` | Quét threshold nhanh, không chạy Whisper |
| 8 — CLI | `build_arg_parser` `main` | Phân tích tham số; điều hướng sang `run_pipeline` hoặc `run_threshold_sweep` |

---

## Các quyết định thiết kế quan trọng

### Inference thích ứng theo độ dài (Section 2)

Audio dài hơn 30 phút sử dụng `BatchedInferencePipeline` để tăng throughput.
Audio ngắn hơn dùng trực tiếp `WhisperModel.transcribe` để giảm latency.
Ở nhánh non-batched, VAD phải được bật tường minh (`vad_filter=True`) vì nó không tự động bật như ở nhánh batched.

### Khử tiếng ồn tách biệt cho từng nhánh (Section 1 + 6)

Khi bật `--denoise`, chỉ nhánh diarization nhận file WAV đã khử nhiễu.
Whisper luôn nhận WAV gốc vì spectral gating có thể làm biến dạng âm vị —
đặc biệt là các nguyên âm ngắn và phụ âm geminata trong tiếng Nhật —
dẫn đến tăng word error rate.

### Phát hiện overlap bằng event-sweep (Section 3)

Thay vì so sánh trực tiếp timestamp giữa hai annotation (dễ sai do sai số dấu phẩy động),
pipeline xây dựng danh sách sự kiện mở (+1) và đóng (-1) của từng segment từ `speaker_diarization`,
rồi quét theo thời gian. Bất kỳ khoảng nào có depth ≥ 2 là vùng overlap.
Các segment trong `exclusive_speaker_diarization` sau đó được đánh dấu qua phép kiểm tra giao khoảng thời gian.

### Đặt lại nhãn speaker theo thứ tự xuất hiện (Section 3)

pyannote gán speaker ID theo thứ tự clustering nội bộ, không theo thứ tự xuất hiện trong audio.
Sau khi sắp xếp các segment theo `start_sec`, pipeline duyệt danh sách một lần
và gán `SPEAKER_00` cho speaker mới đầu tiên gặp được, `SPEAKER_01` cho speaker thứ hai, v.v.
Cách này giúp transcript dễ đọc hơn vì số thứ tự khớp với thứ tự lên tiếng thực tế.

### Gán từ theo max overlap (Section 4)

Mỗi `WordStamp` được gán cho `SpeakerSegment` có khoảng thời gian giao nhau nhiều nhất với từ đó.
Cách này bền vững hơn kiểm tra containment đơn thuần, đặc biệt ở ranh giới segment
nơi timestamp của một từ có thể nằm vắt ngang giữa hai segment liền kề.