"""
=============================================================================
  Speech-to-Text + Speaker Diarization Demo (Japanese-optimized)
  Stack : faster-whisper (large-v3) + pyannote/speaker-diarization-community-1
  Author: AI Engineer Demo
  Python: 3.9+
=============================================================================

SETUP CHECKLIST (đọc trước khi chạy)
--------------------------------------
1. Cài ffmpeg ở cấp hệ điều hành:
     Ubuntu/Debian : sudo apt install ffmpeg
     macOS (brew)  : brew install ffmpeg
     Windows       : https://ffmpeg.org/download.html  (thêm vào PATH)

2. Tạo Hugging Face Access Token:
     → https://hf.co/settings/tokens
     Chọn loại "Read", đặt tên, rồi copy token.

3. Chấp nhận điều khoản sử dụng model diarization:
     → https://huggingface.co/pyannote/speaker-diarization-community-1
     Nhấn "Agree and access repository" trên trang model.
     (Nếu bỏ qua bước này, pipeline sẽ báo lỗi 403.)

4. Đặt token vào biến môi trường (khuyến nghị – không hard-code):
     export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
   Hoặc truyền trực tiếp qua tham số CLI: --hf-token hf_xxx...

5. Cài dependencies:
     pip install -r requirements.txt
=============================================================================
"""

import os
import sys
import subprocess
import argparse
import logging
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ── Tắt telemetry của pyannote.audio TRƯỚC KHI import bất kỳ thứ gì ──────────
os.environ["PYANNOTE_METRICS_ENABLED"] = "false"
# Tắt thêm analytics của Lightning / PyTorch Lightning nếu có
os.environ["LIGHTNING_DISABLE_ANALYTICS"] = "1"
os.environ["PL_DISABLE_TELEMETRY"] = "1"

# ── Third-party imports (lazy để bắt ImportError rõ ràng) ────────────────────
try:
    from faster_whisper import WhisperModel, BatchedInferencePipeline
except ImportError:
    sys.exit("❌  faster-whisper chưa được cài.  Chạy: pip install faster-whisper")

try:
    from pyannote.audio import Pipeline as PyannotePipeline
except ImportError:
    sys.exit("❌  pyannote.audio chưa được cài.  Chạy: pip install pyannote.audio")

try:
    import torch
except ImportError:
    sys.exit("❌  torch chưa được cài.  Chạy: pip install torch")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
LONG_AUDIO_THRESHOLD_SECONDS = 30 * 60   # 30 phút
WHISPER_MODEL_SIZE            = "large-v3"
DIARIZATION_MODEL_ID          = "pyannote/speaker-diarization-community-1"
TARGET_SAMPLE_RATE            = 16_000
TARGET_CHANNELS               = 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class WordToken:
    """Một từ đơn với thông tin thời gian từ faster-whisper."""
    word:        str
    start:       float
    end:         float
    probability: float = 1.0


@dataclass
class DiarizationSegment:
    """Một khoảng thời gian được gán cho một người nói cụ thể."""
    speaker: str
    start:   float
    end:     float
    is_overlap: bool = False   # True nếu đây là vùng overlapped speech


@dataclass
class AlignedSegment:
    """Kết quả sau khi gán từng từ vào người nói."""
    speaker:  str
    start:    float
    end:      float
    text:     str
    is_overlap: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Audio utilities
# ─────────────────────────────────────────────────────────────────────────────
def convert_to_wav(input_path: str, output_path: str) -> None:
    """Dùng ffmpeg chuyển bất kỳ file audio → WAV 16 kHz mono PCM-16."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", str(TARGET_CHANNELS),
        "-sample_fmt", "s16",
        output_path,
    ]
    log.info("🔄  Đang chuyển đổi audio → WAV 16 kHz mono …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg thất bại:\n{result.stderr}"
        )
    log.info("✅  Chuyển đổi xong: %s", output_path)


def denoise_wav(input_wav: str, output_wav: str, strength: float = 0.85) -> None:
    """
    Khử tiếng ồn nền bằng noisereduce (spectral gating).

    Đặc biệt hữu ích khi:
      - Audio ghi ở văn phòng / quán cà phê (tiếng ồn stationary)
      - Giọng 2 người gần nhau → embedding bị nhiễu → diarization nhầm speaker
      - File ngắn < 5 phút → ít data để pipeline tự phân biệt

    Cơ chế: ước tính noise profile từ toàn bộ signal (stationary noise),
    sau đó dùng spectral gating để lọc các tần số dưới ngưỡng noise.

    strength: 0.0 = không lọc, 1.0 = lọc mạnh nhất
              0.75–0.85 là sweet spot cho audio hội thoại văn phòng
              Quá cao (>0.95) có thể làm mất âm vị → Whisper WER tăng

    Requires: pip install noisereduce soundfile
    """
    try:
        import noisereduce as nr
        import soundfile as sf
    except ImportError:
        log.warning(
            "⚠️  noisereduce chưa được cài → bỏ qua bước denoising.\n"
            "   Để bật: pip install noisereduce soundfile"
        )
        # Copy file gốc sang output mà không làm gì
        import shutil
        shutil.copy2(input_wav, output_wav)
        return

    log.info("🔇  Đang khử tiếng ồn (strength=%.2f) …", strength)
    data, sr = sf.read(input_wav, dtype="float32")

    # noisereduce ước tính noise từ toàn bộ clip (stationary mode)
    # prop_decrease = strength: tỉ lệ giảm noise (0→1)
    reduced = nr.reduce_noise(
        y             = data,
        sr            = sr,
        stationary    = True,
        prop_decrease = strength,
        n_fft         = 1024,
        hop_length    = 256,
    )

    sf.write(output_wav, reduced, sr, subtype="PCM_16")
    log.info("✅  Denoising xong: %s", output_wav)


def get_audio_duration_seconds(wav_path: str) -> float:
    """Lấy độ dài (giây) của file WAV bằng ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe thất bại:\n{result.stderr}")
    return float(result.stdout.strip())


# ─────────────────────────────────────────────────────────────────────────────
# 2. Whisper transcription (Adaptive Inference)
# ─────────────────────────────────────────────────────────────────────────────
def build_whisper_model(device: str, compute_type: str) -> WhisperModel:
    """Khởi tạo WhisperModel large-v3."""
    log.info(
        "🧠  Đang tải Whisper %s (device=%s, compute=%s) …",
        WHISPER_MODEL_SIZE, device, compute_type,
    )
    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=device,
        compute_type=compute_type,
    )
    log.info("✅  Whisper model đã sẵn sàng.")
    return model


def transcribe_audio(
    wav_path: str,
    model: WhisperModel,
    duration_seconds: float,
    language: str = "ja",
    initial_prompt: Optional[str] = None,
) -> List[WordToken]:
    """
    Nhận dạng giọng nói và trả về danh sách WordToken (word-level timestamps).

    Adaptive Inference:
      - duration > 30 phút  → BatchedInferencePipeline  (throughput cao hơn)
      - duration ≤ 30 phút  → WhisperModel.transcribe   (latency thấp hơn)

    Tham số initial_prompt giúp mồi mô hình nhận diện đúng danh từ riêng /
    từ vựng chuyên ngành tiếng Nhật.
    Ví dụ:
        initial_prompt = "トヨタ自動車、本田技研工業、ソフトバンクグループ"
    """
    common_kwargs = dict(
        language=language,
        word_timestamps=True,       # ← bắt buộc để lấy word-level timestamps
        initial_prompt=initial_prompt,
        beam_size=5,
        best_of=5,
        temperature=0.0,            # greedy, ổn định hơn cho tác vụ business
        condition_on_previous_text=True,
    )

    if duration_seconds > LONG_AUDIO_THRESHOLD_SECONDS:
        log.info(
            "⏱  Độ dài %.1f phút > 30 phút → dùng BatchedInferencePipeline.",
            duration_seconds / 60,
        )
        pipeline = BatchedInferencePipeline(model=model)
        # VAD được bật mặc định trong BatchedInferencePipeline
        segments_iter, info = pipeline.transcribe(
            wav_path,
            batch_size=16,
            **common_kwargs,
        )
    else:
        log.info(
            "⏱  Độ dài %.1f phút ≤ 30 phút → dùng WhisperModel.transcribe.",
            duration_seconds / 60,
        )
        segments_iter, info = model.transcribe(
            wav_path,
            vad_filter=True,            # ← phải truyền thủ công ở đây
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=400,
            ),
            **common_kwargs,
        )

    log.info(
        "🌐  Ngôn ngữ phát hiện: %s (xác suất %.2f)",
        info.language, info.language_probability,
    )

    words: List[WordToken] = []
    for seg in segments_iter:
        if seg.words is None:
            continue
        for w in seg.words:
            words.append(WordToken(
                word=w.word,
                start=w.start,
                end=w.end,
                probability=w.probability,
            ))

    log.info("✅  Transcription xong: %d từ.", len(words))
    return words


# ─────────────────────────────────────────────────────────────────────────────
# 3. Speaker Diarization
# ─────────────────────────────────────────────────────────────────────────────
def run_diarization(
    wav_path: str,
    hf_token: str,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    clustering_threshold:   Optional[float] = None,
    segmentation_threshold: Optional[float] = None,
) -> List[DiarizationSegment]:
    """
    Chạy pyannote/speaker-diarization-community-1 và trả về
    List[DiarizationSegment] đã đánh dấu is_overlap.

    ── Cấu trúc DiarizeOutput của community-1 ──────────────────────────────
    Pipeline trả về dataclass DiarizeOutput với 3 fields:

      .exclusive_speaker_diarization  →  pyannote Annotation
          Kết quả diarization "sạch": mỗi thời điểm chỉ có DUY NHẤT 1 speaker.
          Đây là output chính để dùng cho alignment với Whisper transcript.
          Speaker change detection đã được tích hợp tự động bên trong pipeline.

      .speaker_diarization            →  pyannote Annotation
          Kết quả đầy đủ CÓ overlap: các vùng nhiều người nói cùng lúc
          được biểu diễn bằng NHIỀU TRACK song song (track A + track B
          cùng tồn tại trong khoảng [t1, t2]).
          Overlapped speech detection đã được tích hợp tự động.

      .speaker_embeddings             →  numpy array
          Vector embedding của từng speaker (không dùng trong pipeline này).

    ── Chiến lược implementation đúng ──────────────────────────────────────
    1. Parse TOÀN BỘ tracks từ .speaker_diarization → danh sách thô
    2. Xây dựng "overlap timeline" bằng interval sweep:
       - Với mỗi điểm thời gian, đếm số tracks đang active
       - Nếu > 1 track active → đó là vùng overlapped speech
    3. Với mỗi segment trong .exclusive_speaker_diarization,
       kiểm tra xem nó có giao với vùng overlap không → set is_overlap
    4. Trả về List[DiarizationSegment] từ exclusive (để alignment ổn định)
       kèm is_overlap flag chính xác.

    ── Tại sao KHÔNG dùng key-matching float ───────────────────────────────
    .speaker_diarization và .exclusive_speaker_diarization được tính toán
    độc lập, timestamps của chúng KHÔNG khớp chính xác nhau dù round().
    Phải dùng interval intersection thay vì key lookup.

    ── HF Token ────────────────────────────────────────────────────────────
    pyannote.audio >= 3.x dùng tham số `token` (không phải use_auth_token).
    Token cần có quyền Read và user phải accept điều khoản tại:
    https://huggingface.co/pyannote/speaker-diarization-community-1
    """
    log.info("🎙  Đang tải diarization pipeline: %s …", DIARIZATION_MODEL_ID)
    pipe = PyannotePipeline.from_pretrained(
        DIARIZATION_MODEL_ID,
        token=hf_token,
    )

    device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device_obj)

    # ── Apply threshold overrides nếu có (sau khi pipe đã được instantiate) ─
    _apply_pipeline_overrides(pipe, clustering_threshold, segmentation_threshold)

    log.info("🎙  Diarization đang chạy trên %s …", device_obj)

    diarize_kwargs: dict = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers

    output = pipe(wav_path, **diarize_kwargs)
    # output là DiarizeOutput dataclass
    # output.exclusive_speaker_diarization → Annotation (1 speaker/thời điểm)
    # output.speaker_diarization           → Annotation (có overlap tracks)

    # ── Helper: iterate Annotation tương thích pyannote 3.x và 4.x ──────────
    # pyannote 3.x: itertracks(yield_label=True) → (Segment, track_id, label)
    # pyannote 4.x: for turn, speaker in annotation → (Segment, label)  [official example]
    #               itertracks vẫn còn nhưng signature có thể thay đổi
    def _iter_annotation(annotation):
        """Yield (turn, speaker) từ pyannote Annotation, tương thích 3.x và 4.x."""
        try:
            # Thử 3-tuple trước (pyannote 3.x standard)
            for item in annotation.itertracks(yield_label=True):
                if len(item) == 3:
                    turn, _track, speaker = item
                else:
                    turn, speaker = item
                yield turn, speaker
        except Exception:
            # Fallback: iterate trực tiếp theo official 4.x example
            for turn, speaker in annotation:
                yield turn, speaker

    # ── Bước 1: Xây dựng overlap timeline bằng interval sweep ───────────────
    events: List[Tuple[float, int]] = []
    for turn, _ in _iter_annotation(output.speaker_diarization):
        events.append((turn.start, +1))
        events.append((turn.end,   -1))

    # Sort: cùng timestamp thì event mở (+1) đến trước event đóng (-1)
    events.sort(key=lambda e: (e[0], -e[1]))

    # Quét để tìm các khoảng thời gian có depth >= 2 (overlap)
    overlap_intervals: List[Tuple[float, float]] = []
    depth      = 0
    overlap_start: Optional[float] = None

    for t, delta in events:
        if depth >= 2 and overlap_start is not None:
            if t > overlap_start:
                overlap_intervals.append((overlap_start, t))
        depth += delta
        if depth >= 2:
            overlap_start = t
        else:
            overlap_start = None

    log.info("📊  Overlap intervals phát hiện: %d vùng", len(overlap_intervals))

    # ── Bước 2: Helper kiểm tra segment có giao với overlap interval không ──
    def _is_in_overlap(seg_start: float, seg_end: float) -> bool:
        for ov_start, ov_end in overlap_intervals:
            if ov_start < seg_end and ov_end > seg_start:
                return True
        return False

    # ── Bước 3: Build DiarizationSegment từ exclusive annotation ────────────
    segments: List[DiarizationSegment] = []

    for turn, speaker in _iter_annotation(output.exclusive_speaker_diarization):
        segments.append(DiarizationSegment(
            speaker    = speaker,
            start      = turn.start,
            end        = turn.end,
            is_overlap = _is_in_overlap(turn.start, turn.end),
        ))

    segments.sort(key=lambda s: s.start)

    # ── Re-label speaker theo thứ tự xuất hiện trong timeline ───────────────
    # Pyannote gán label (SPEAKER_00, SPEAKER_01...) theo clustering index,
    # không theo thứ tự thời gian → đôi khi SPEAKER_00 xuất hiện sau SPEAKER_01.
    # Re-map lại để SPEAKER_00 = người nói đầu tiên, SPEAKER_01 = người thứ hai...
    label_map: dict = {}
    for seg in segments:
        if seg.speaker not in label_map:
            idx = len(label_map)
            label_map[seg.speaker] = f"SPEAKER_{idx:02d}"

    if label_map:
        original_labels = sorted(label_map.keys())
        new_labels      = [label_map[k] for k in original_labels]
        if original_labels != new_labels:
            log.info("🔀  Re-label speakers theo thứ tự xuất hiện:")
            for orig, new in label_map.items():
                if orig != new:
                    log.info("     %s → %s", orig, new)
            for seg in segments:
                seg.speaker = label_map[seg.speaker]

    overlap_count   = sum(1 for s in segments if s.is_overlap)
    unique_speakers = {s.speaker for s in segments}

    log.info(
        "✅  Diarization xong: %d segments | %d người nói | %d segments có overlap",
        len(segments), len(unique_speakers), overlap_count,
    )
    log.info("👥  Speakers (theo thứ tự xuất hiện): %s", sorted(unique_speakers))

    return segments


# ─────────────────────────────────────────────────────────────────────────────
# 4. Alignment – gán từng từ vào người nói
# ─────────────────────────────────────────────────────────────────────────────
def _overlap_duration(
    w_start: float, w_end: float,
    s_start: float, s_end: float,
) -> float:
    """Tính độ dài overlap (giây) giữa hai khoảng thời gian."""
    return max(0.0, min(w_end, s_end) - max(w_start, s_start))


def align_words_to_speakers(
    words: List[WordToken],
    diarization: List[DiarizationSegment],
) -> List[AlignedSegment]:
    """
    Thuật toán alignment word-level timestamps ↔ diarization segments.

    Chiến lược:
    1. Mỗi từ được gán cho speaker có THỜI GIAN OVERLAP LỚN NHẤT với từ đó.
    2. Nếu từ nằm trong vùng overlapped speech, đánh dấu is_overlap=True.
    3. Gộp các từ liên tiếp của cùng một speaker thành một AlignedSegment.
    4. Nếu một từ không trùng với bất kỳ speaker nào (khoảng lặng / edge),
       gán cho speaker của từ liền trước (fallback).
    """
    if not words:
        return []

    assigned: List[Tuple[str, bool]] = []   # (speaker, is_overlap) per word

    last_speaker = diarization[0].speaker if diarization else "UNKNOWN"

    for w in words:
        best_speaker = None
        best_overlap = 0.0
        word_is_overlap = False

        for seg in diarization:
            # Bỏ qua các segment hoàn toàn nằm ngoài cửa sổ từ
            if seg.end < w.start or seg.start > w.end:
                continue
            ov = _overlap_duration(w.start, w.end, seg.start, seg.end)
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = seg.speaker
                word_is_overlap = seg.is_overlap

        if best_speaker is None:
            # Fallback: dùng speaker của từ trước
            best_speaker = last_speaker
            word_is_overlap = False

        assigned.append((best_speaker, word_is_overlap))
        last_speaker = best_speaker

    # ── Gộp từ liên tiếp của cùng speaker ───────────────────────────────────
    aligned: List[AlignedSegment] = []
    current_speaker, current_overlap = assigned[0]
    current_words: List[WordToken] = [words[0]]

    for i in range(1, len(words)):
        spk, ov = assigned[i]
        if spk == current_speaker:
            current_words.append(words[i])
            current_overlap = current_overlap or ov
        else:
            aligned.append(AlignedSegment(
                speaker=current_speaker,
                start=current_words[0].start,
                end=current_words[-1].end,
                text="".join(w.word for w in current_words).strip(),
                is_overlap=current_overlap,
            ))
            current_speaker = spk
            current_overlap = ov
            current_words = [words[i]]

    # Flush đoạn cuối
    aligned.append(AlignedSegment(
        speaker=current_speaker,
        start=current_words[0].start,
        end=current_words[-1].end,
        text="".join(w.word for w in current_words).strip(),
        is_overlap=current_overlap,
    ))

    return aligned


# ─────────────────────────────────────────────────────────────────────────────
# 5. Formatting output
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_time(seconds: float) -> str:
    """Chuyển giây → MM:SS (hoặc HH:MM:SS nếu >= 1 giờ)."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_transcript(aligned: List[AlignedSegment]) -> str:
    """
    Định dạng kết quả cuối cùng:
      [MM:SS - MM:SS] Speaker X: <text>
    Đoạn overlapped speech được đánh dấu thêm "[OVERLAP]".
    """
    lines = []
    for seg in aligned:
        if not seg.text:
            continue
        overlap_tag = " [OVERLAP]" if seg.is_overlap else ""
        lines.append(
            f"[{_fmt_time(seg.start)} - {_fmt_time(seg.end)}] "
            f"{seg.speaker}{overlap_tag}: {seg.text}"
        )
    return "\n".join(lines)


def format_whisper_raw(words: List[WordToken]) -> str:
    """
    Format kết quả RAW của Whisper (word-level timestamps).
    Dùng để debug: so sánh với output diarization để kiểm tra alignment.

    Output mẫu:
      [00:00.120 - 00:00.480]  そう        (p=0.99)
      [00:00.480 - 00:00.840]  です        (p=0.98)
      ...
    """
    if not words:
        return "(Không có từ nào được transcribe)"

    lines = [
        "=" * 65,
        "WHISPER RAW OUTPUT — word-level timestamps",
        f"Tổng số từ: {len(words)}",
        "=" * 65,
    ]
    for w in words:
        # Format timestamp đến millisecond
        def ms(sec: float) -> str:
            m, s = divmod(sec, 60)
            return f"{int(m):02d}:{s:06.3f}"
        prob_bar = "█" * int(w.probability * 10) + "░" * (10 - int(w.probability * 10))
        lines.append(
            f"[{ms(w.start)} → {ms(w.end)}]  "
            f"{w.word:<20s}  p={w.probability:.2f} {prob_bar}"
        )
    return "\n".join(lines)


def format_diarization_raw(segments: List[DiarizationSegment]) -> str:
    """
    Format kết quả RAW của pyannote diarization.
    Dùng để debug: xem pipeline phân tách ai nói ở khoảng thời gian nào
    trước khi matching với Whisper output.

    Output mẫu:
      [00:00 - 00:05]  SPEAKER_01   (5.00s)
      [00:05 - 00:08]  SPEAKER_02   (3.00s)  [OVERLAP]
      ...
    """
    if not segments:
        return "(Không có segment nào)"

    # Thống kê tổng thời lượng theo speaker
    speaker_duration: dict = {}
    for seg in segments:
        d = seg.end - seg.start
        speaker_duration[seg.speaker] = speaker_duration.get(seg.speaker, 0.0) + d

    total_duration = max((s.end for s in segments), default=0.0)
    overlap_count  = sum(1 for s in segments if s.is_overlap)

    lines = [
        "=" * 65,
        "PYANNOTE DIARIZATION RAW OUTPUT",
        f"Tổng segments  : {len(segments)}",
        f"Overlap segs   : {overlap_count}",
        f"Audio duration : {_fmt_time(total_duration)}",
        f"Speakers       : {sorted(speaker_duration.keys())}",
        "-" * 65,
        "Thống kê thời lượng mỗi speaker:",
    ]
    for spk, dur in sorted(speaker_duration.items()):
        pct = dur / total_duration * 100 if total_duration > 0 else 0
        bar = "█" * int(pct / 5)
        lines.append(f"  {spk:<14s}  {dur:6.1f}s  ({pct:5.1f}%)  {bar}")

    lines += [
        "-" * 65,
        "Chi tiết từng segment:",
        f"  {'[START - END]':<16}  {'SPEAKER':<14}  {'DUR':>6}  {'OVERLAP':>8}",
        "  " + "-" * 55,
    ]
    for seg in segments:
        dur = seg.end - seg.start
        ov  = "⚠ OVERLAP" if seg.is_overlap else ""
        lines.append(
            f"  [{_fmt_time(seg.start)} - {_fmt_time(seg.end)}]  "
            f"{seg.speaker:<14s}  {dur:5.2f}s  {ov}"
        )
    lines.append("=" * 65)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    input_audio:             str,
    hf_token:                str,
    language:                str = "ja",
    initial_prompt:          Optional[str] = None,
    num_speakers:            Optional[int] = None,
    min_speakers:            Optional[int] = None,
    max_speakers:            Optional[int] = None,
    output_file:             Optional[str] = None,
    clustering_threshold:    Optional[float] = None,
    segmentation_threshold:  Optional[float] = None,
    denoise:                 bool = False,
    denoise_strength:        float = 0.85,
    debug_dir:               Optional[str] = None,
) -> str:
    """
    Pipeline đầy đủ end-to-end:
      audio file → WAV → [Denoise] → Whisper & Diarization → Alignment → transcript

    debug_dir: nếu được truyền vào, pipeline sẽ lưu 2 file intermediate:
      • {debug_dir}/01_whisper_raw.txt    — word-level timestamps từ Whisper
      • {debug_dir}/02_diarization_raw.txt — speaker segments từ pyannote
    Dùng để so sánh 2 kết quả và debug lỗi matching.
    """
    if not Path(input_audio).exists():
        raise FileNotFoundError(f"File không tồn tại: {input_audio}")

    # Tạo debug_dir nếu cần
    debug_path: Optional[Path] = None
    if debug_dir:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        log.info("🐛  Debug mode: intermediate files sẽ lưu vào %s", debug_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path     = str(Path(tmpdir) / "audio_16k.wav")
        denoise_path = str(Path(tmpdir) / "audio_denoised.wav")

        convert_to_wav(input_audio, wav_path)

        if denoise:
            denoise_wav(wav_path, denoise_path, strength=denoise_strength)
            diarize_input = denoise_path
            log.info("🎙  Diarization ← denoised WAV | Whisper ← WAV gốc")
        else:
            diarize_input = wav_path

        duration = get_audio_duration_seconds(wav_path)
        log.info("📏  Độ dài audio: %.1f giây (%.1f phút)", duration, duration / 60)

        device       = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        whisper_model = build_whisper_model(device, compute_type)

        words = transcribe_audio(
            wav_path         = wav_path,
            model            = whisper_model,
            duration_seconds = duration,
            language         = language,
            initial_prompt   = initial_prompt,
        )

        # ── Lưu intermediate file 1: Whisper raw output ──────────────────────
        if debug_path:
            whisper_raw_file = debug_path / "01_whisper_raw.txt"
            whisper_raw_file.write_text(format_whisper_raw(words), encoding="utf-8")
            log.info("💾  [DEBUG] Whisper raw → %s", whisper_raw_file)

        diarization = run_diarization(
            wav_path               = diarize_input,
            hf_token               = hf_token,
            num_speakers           = num_speakers,
            min_speakers           = min_speakers,
            max_speakers           = max_speakers,
            clustering_threshold   = clustering_threshold,
            segmentation_threshold = segmentation_threshold,
        )

        # ── Lưu intermediate file 2: Diarization raw output ─────────────────
        if debug_path:
            diarize_raw_file = debug_path / "02_diarization_raw.txt"
            diarize_raw_file.write_text(format_diarization_raw(diarization), encoding="utf-8")
            log.info("💾  [DEBUG] Diarization raw → %s", diarize_raw_file)

    log.info("🔗  Đang alignment word-timestamps ↔ speaker segments …")
    aligned = align_words_to_speakers(words, diarization)
    log.info("✅  Alignment xong: %d đoạn.", len(aligned))

    transcript = format_transcript(aligned)

    if output_file:
        Path(output_file).write_text(transcript, encoding="utf-8")
        log.info("💾  Transcript đã lưu: %s", output_file)

    # ── Lưu intermediate file 3: Final aligned transcript (luôn lưu nếu debug) ─
    if debug_path:
        final_file = debug_path / "03_final_transcript.txt"
        final_file.write_text(transcript, encoding="utf-8")
        log.info("💾  [DEBUG] Final transcript → %s", final_file)
        _print_debug_summary(words, diarization, aligned, debug_path)

    return transcript


def _print_debug_summary(
    words: List[WordToken],
    diarization: List[DiarizationSegment],
    aligned: List[AlignedSegment],
    debug_path: Path,
) -> None:
    """
    In và lưu bảng tóm tắt alignment để dễ phát hiện lỗi matching.
    So sánh trực tiếp: diarization timeline vs whisper word timeline.
    """
    lines = [
        "=" * 75,
        "DEBUG ALIGNMENT SUMMARY",
        "=" * 75,
        f"{'DIARIZATION SEGMENTS':<38}  {'WHISPER WORDS (first 3 per segment)'}",
        "-" * 75,
    ]

    for seg in diarization:
        # Tìm các từ Whisper nằm trong segment này
        seg_words = [
            w for w in words
            if not (w.end <= seg.start or w.start >= seg.end)
        ]

        seg_label = (
            f"[{_fmt_time(seg.start)}-{_fmt_time(seg.end)}] "
            f"{seg.speaker}"
            f"{' ⚠OVL' if seg.is_overlap else ''}"
        )

        if seg_words:
            preview = "".join(w.word for w in seg_words[:8]).strip()
            if len(seg_words) > 8:
                preview += f"… (+{len(seg_words)-8} từ)"
            word_count = f"({len(seg_words)} từ)"
        else:
            preview   = "⚠️  KHÔNG CÓ TỪ NÀO MATCH — có thể gap trong timeline"
            word_count = "(0 từ)"

        lines.append(f"{seg_label:<38}  {word_count} {preview}")

    # Kiểm tra từ không được map vào segment nào
    unmapped = []
    for w in words:
        matched = any(
            not (w.end <= seg.start or w.start >= seg.end)
            for seg in diarization
        )
        if not matched:
            unmapped.append(w)

    if unmapped:
        lines += [
            "",
            f"⚠️  {len(unmapped)} TỪ KHÔNG MAP ĐƯỢC vào segment diarization nào:",
            "   (Nguyên nhân: gap giữa các diarization segment, hoặc Whisper detect âm thanh ngoài vùng nói)",
        ]
        for w in unmapped[:10]:
            def ms(sec: float) -> str:
                m, s = divmod(sec, 60)
                return f"{int(m):02d}:{s:05.2f}"
            lines.append(f"   [{ms(w.start)}-{ms(w.end)}] '{w.word}'")
        if len(unmapped) > 10:
            lines.append(f"   ... và {len(unmapped)-10} từ khác")

    lines.append("=" * 75)
    summary = "\n".join(lines)

    # In ra console
    print("\n" + summary)

    # Lưu ra file
    summary_file = debug_path / "04_alignment_debug.txt"
    summary_file.write_text(summary, encoding="utf-8")
    log.info("💾  [DEBUG] Alignment summary → %s", summary_file)


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Speech-to-Text + Speaker Diarization (Japanese)\n"
            "faster-whisper large-v3  +  pyannote community-1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "audio",
        help="Đường dẫn file audio đầu vào (bất kỳ định dạng ffmpeg hỗ trợ).",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help=(
            "Hugging Face access token. "
            "Nếu không truyền, script đọc biến môi trường HF_TOKEN. "
            "Tạo token tại: https://hf.co/settings/tokens"
        ),
    )
    parser.add_argument(
        "--language", default="ja",
        help="Mã ngôn ngữ ISO-639-1 (mặc định: ja).",
    )
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help=(
            "Prompt ngữ cảnh để mồi Whisper nhận diện danh từ riêng / "
            "từ vựng chuyên ngành. "
            "Ví dụ: 'トヨタ自動車、ソフトバンク、田中部長'"
        ),
    )
    parser.add_argument(
        "--num-speakers", type=int, default=None,
        help="Số người nói chính xác (nếu đã biết trước).",
    )
    parser.add_argument(
        "--min-speakers", type=int, default=None,
        help="Số người nói tối thiểu.",
    )
    parser.add_argument(
        "--max-speakers", type=int, default=None,
        help="Số người nói tối đa.",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Lưu transcript ra file văn bản (UTF-8).",
    )
    parser.add_argument(
        "--denoise", action="store_true",
        help=(
            "Bật khử tiếng ồn nền trước khi diarization (dùng noisereduce). "
            "Khuyến nghị khi audio có nhiễu stationary (văn phòng, quán cà phê) "
            "hoặc 2 giọng nói gần nhau hay bị nhầm speaker. "
            "Whisper vẫn dùng audio gốc để giữ chất lượng ASR. "
            "Cần cài thêm: pip install noisereduce soundfile"
        ),
    )
    parser.add_argument(
        "--denoise-strength", type=float, default=0.85,
        help=(
            "Mức độ khử nhiễu (0.0–1.0, mặc định 0.85). "
            "0.75–0.85 phù hợp cho hội thoại văn phòng. "
            "Quá cao (>0.95) có thể làm mất âm vị."
        ),
    )
    parser.add_argument(
        "--clustering-threshold", type=float, default=None,
        help=(
            "Override clustering threshold của pyannote (mặc định ~0.715). "
            "Tăng lên (0.80-0.90) nếu bị over-segmentation (1 người thành nhiều). "
            "Giảm xuống (0.55-0.65) nếu bị under-segmentation (nhiều người thành 1)."
        ),
    )
    parser.add_argument(
        "--segmentation-threshold", type=float, default=None,
        help=(
            "Override segmentation threshold (mặc định ~0.817). "
            "Giảm xuống (0.60-0.75) nếu pipeline bỏ sót nhiều đoạn nói ngắn."
        ),
    )
    parser.add_argument(
        "--diagnose", action="store_true",
        help=(
            "Chế độ chẩn đoán: in hyperparameters hiện tại của pipeline, "
            "thử nhiều clustering threshold và báo cáo số speaker detected "
            "mà KHÔNG chạy Whisper. Dùng để tìm threshold phù hợp trước."
        ),
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help=(
            "Lưu các file intermediate để debug alignment. "
            "Tạo thư mục chứa 4 file:\n"
            "  01_whisper_raw.txt      — word timestamps từ Whisper\n"
            "  02_diarization_raw.txt  — speaker segments từ pyannote\n"
            "  03_final_transcript.txt — transcript đã alignment\n"
            "  04_alignment_debug.txt  — bảng so sánh để tìm lỗi matching\n"
            "Ví dụ: --debug-dir ./debug_output"
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic helper
# ─────────────────────────────────────────────────────────────────────────────
def _apply_pipeline_overrides(
    pipe,
    clustering_threshold: Optional[float],
    segmentation_threshold: Optional[float],
) -> None:
    """
    Override hyperparameters của pyannote pipeline sau khi đã load.

    pyannote lưu params dưới dạng lồng nhau, truy cập qua instantiated_parameters.
    Nếu key không tồn tại (pipeline version khác nhau), log warning và bỏ qua.
    """
    try:
        params = pipe.parameters(instantiated=True)
        log.info("📐  Hyperparameters hiện tại:\n%s",
                 "\n".join(f"   {k}: {v}" for k, v in _flatten(params).items()))
    except Exception:
        pass

    if clustering_threshold is not None:
        _set_nested(pipe, ["clustering", "threshold"], clustering_threshold)
        log.info("🔧  clustering.threshold  → %.3f", clustering_threshold)

    if segmentation_threshold is not None:
        _set_nested(pipe, ["segmentation", "threshold"], segmentation_threshold)
        log.info("🔧  segmentation.threshold → %.3f", segmentation_threshold)


def _flatten(d: dict, parent: str = "") -> dict:
    """Flatten nested dict thành {a.b.c: value}."""
    out = {}
    for k, v in d.items():
        key = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _set_nested(pipe, path: List[str], value: float) -> None:
    """
    Cố gắng set nested param trong pipeline.
    pyannote lưu params qua instantiated_parameters hoặc trực tiếp trên sub-models.
    """
    # Thử qua instantiated_parameters dict
    try:
        params = pipe.parameters(instantiated=True)
        section = path[0]
        key     = path[1]
        if section in params and key in params[section]:
            params[section][key] = value
            pipe.instantiate(params)
            return
    except Exception:
        pass

    # Thử set trực tiếp qua attribute path trên pipe object
    try:
        obj = pipe
        for attr in path[:-1]:
            obj = getattr(obj, attr)
        setattr(obj, path[-1], value)
        return
    except Exception:
        pass

    log.warning("⚠️  Không thể set %s — pipeline version này có thể dùng key khác.", ".".join(path))


def run_diagnose(
    wav_path: str,
    pipe,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> None:
    """
    Chế độ chẩn đoán: thử một dải clustering threshold,
    in số speaker detected ở mỗi mức để người dùng chọn giá trị phù hợp.
    """
    print("\n" + "=" * 70)
    print("🔬  DIAGNOSE MODE — Clustering Threshold Sweep")
    print("=" * 70)
    print("Mục tiêu: tìm threshold cho số speaker detected khớp thực tế.")
    print(f"{'Threshold':>12}  {'Speakers detected':>18}  {'Segments':>10}")
    print("-" * 46)

    diarize_kwargs: dict = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers
        print(f"  (num_speakers={num_speakers} được truyền vào → threshold ít ảnh hưởng)")
    else:
        if min_speakers: diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers: diarize_kwargs["max_speakers"] = max_speakers

    for threshold in [0.55, 0.60, 0.65, 0.70, 0.715, 0.75, 0.80, 0.85, 0.90]:
        _set_nested(pipe, ["clustering", "threshold"], threshold)
        try:
            out = pipe(wav_path, **diarize_kwargs)
            speakers = set()
            seg_count = 0
            for turn, _, spk in out.exclusive_speaker_diarization.itertracks(yield_label=True):
                speakers.add(spk)
                seg_count += 1
            print(f"{threshold:>12.3f}  {len(speakers):>18}  {seg_count:>10}")
        except Exception as e:
            print(f"{threshold:>12.3f}  ERROR: {e}")

    print("=" * 70)
    print("💡  Gợi ý:")
    print("   • Nếu số speaker quá nhiều → tăng threshold (0.80+)")
    print("   • Nếu số speaker quá ít    → giảm threshold (0.60-)")
    print("   • Sau khi chọn được threshold, chạy lại với --clustering-threshold <value>")
    print("=" * 70)


def main() -> None:
    args = parse_args()

    if not args.hf_token:
        sys.exit(
            "❌  Hugging Face token chưa được cung cấp.\n"
            "   Cách 1: export HF_TOKEN='hf_xxx...'\n"
            "   Cách 2: truyền --hf-token hf_xxx...\n"
            "   Tạo token tại: https://hf.co/settings/tokens\n"
            "   Nhớ accept điều khoản model tại:\n"
            "   https://huggingface.co/pyannote/speaker-diarization-community-1"
        )

    # ── Chế độ chẩn đoán (--diagnose): chỉ chạy diarization sweep, không Whisper ──
    if args.diagnose:
        import tempfile
        log.info("🔬  Chế độ chẩn đoán được kích hoạt.")
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = str(Path(tmpdir) / "audio_16k.wav")
            convert_to_wav(args.audio, wav_path)
            pipe = PyannotePipeline.from_pretrained(
                DIARIZATION_MODEL_ID, token=args.hf_token
            )
            device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pipe = pipe.to(device_obj)
            run_diagnose(
                wav_path     = wav_path,
                pipe         = pipe,
                num_speakers = args.num_speakers,
                min_speakers = args.min_speakers,
                max_speakers = args.max_speakers,
            )
        return

    # ── Chạy pipeline đầy đủ ─────────────────────────────────────────────────
    transcript = run_pipeline(
        input_audio            = args.audio,
        hf_token               = args.hf_token,
        language               = args.language,
        initial_prompt         = args.initial_prompt,
        num_speakers           = args.num_speakers,
        min_speakers           = args.min_speakers,
        max_speakers           = args.max_speakers,
        output_file            = args.output,
        clustering_threshold   = args.clustering_threshold,
        segmentation_threshold = args.segmentation_threshold,
        denoise                = args.denoise,
        denoise_strength       = args.denoise_strength,
        debug_dir              = args.debug_dir,
    )

    print("\n" + "=" * 70)
    print("📝  TRANSCRIPT")
    print("=" * 70)
    print(transcript)
    print("=" * 70)


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# USAGE EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────
# Cơ bản:
#   python stt_diarization.py meeting.mp4 --hf-token hf_xxx
#
# Với initial_prompt (danh từ riêng tiếng Nhật):
#   python stt_diarization.py interview.m4a \
#       --hf-token hf_xxx \
#       --initial-prompt "トヨタ自動車、田中一郎部長、ソフトバンクグループ" \
#       --num-speakers 2 \
#       --output transcript.txt
#
# Dùng biến môi trường:
#   export HF_TOKEN="hf_xxx"
#   python stt_diarization.py long_meeting.mp3 --max-speakers 5 -o out.txt