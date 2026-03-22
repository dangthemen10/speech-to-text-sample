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

    # ── Bước 1: Xây dựng overlap timeline bằng interval sweep ───────────────
    # Mỗi track trong speaker_diarization là một interval [start, end].
    # Nếu tại bất kỳ điểm nào có >= 2 tracks cùng active → overlap region.
    # Dùng "event sweep": tạo list (time, +1/-1) rồi quét từ trái sang phải.

    events: List[Tuple[float, int]] = []   # (timestamp, +1=open / -1=close)
    for turn, _track, _spk in output.speaker_diarization.itertracks(yield_label=True):
        events.append((turn.start, +1))
        events.append((turn.end,   -1))

    # Sort: cùng timestamp thì event mở (+1) đến trước event đóng (-1)
    events.sort(key=lambda e: (e[0], -e[1]))

    # Quét để tìm các khoảng thời gian có depth >= 2 (overlap)
    overlap_intervals: List[Tuple[float, float]] = []
    depth      = 0
    prev_time  = 0.0
    overlap_start: Optional[float] = None

    for t, delta in events:
        if depth >= 2 and overlap_start is not None:
            # Đang trong vùng overlap, ghi nhận đến t
            if t > overlap_start:
                overlap_intervals.append((overlap_start, t))
        depth += delta
        if depth >= 2:
            overlap_start = t
        else:
            overlap_start = None

    log.info(
        "📊  Overlap intervals phát hiện: %d vùng",
        len(overlap_intervals),
    )

    # ── Bước 2: Helper kiểm tra segment có giao với overlap interval không ──
    def _is_in_overlap(seg_start: float, seg_end: float) -> bool:
        """True nếu segment [seg_start, seg_end] giao với bất kỳ overlap interval nào."""
        for ov_start, ov_end in overlap_intervals:
            if ov_start < seg_end and ov_end > seg_start:   # interval intersection
                return True
        return False

    # ── Bước 3: Build DiarizationSegment từ exclusive annotation ────────────
    # Dùng exclusive vì: 1 speaker/thời điểm → alignment ổn định, không bị
    # duplicate word assignment. is_overlap flag vẫn chính xác nhờ bước trên.
    segments: List[DiarizationSegment] = []
    excl_count = 0

    for turn, _track, speaker in output.exclusive_speaker_diarization.itertracks(yield_label=True):
        excl_count += 1
        segments.append(DiarizationSegment(
            speaker    = speaker,
            start      = turn.start,
            end        = turn.end,
            is_overlap = _is_in_overlap(turn.start, turn.end),
        ))

    segments.sort(key=lambda s: s.start)

    overlap_count = sum(1 for s in segments if s.is_overlap)
    unique_speakers = {s.speaker for s in segments}

    log.info(
        "✅  Diarization xong: %d segments | %d người nói | %d segments có overlap",
        len(segments), len(unique_speakers), overlap_count,
    )
    log.info("👥  Speakers: %s", sorted(unique_speakers))

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


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    input_audio:    str,
    hf_token:       str,
    language:       str = "ja",
    initial_prompt: Optional[str] = None,
    num_speakers:   Optional[int] = None,
    min_speakers:   Optional[int] = None,
    max_speakers:   Optional[int] = None,
    output_file:    Optional[str] = None,
) -> str:
    """
    Pipeline đầy đủ end-to-end:
      audio file → WAV → Whisper → Diarization → Alignment → transcript
    """
    # ── Bước 0: kiểm tra đầu vào ────────────────────────────────────────────
    if not Path(input_audio).exists():
        raise FileNotFoundError(f"File không tồn tại: {input_audio}")

    # ── Bước 1: chuyển đổi sang WAV 16 kHz mono (ffmpeg) ────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = str(Path(tmpdir) / "audio_16k.wav")
        convert_to_wav(input_audio, wav_path)

        # ── Bước 2: lấy độ dài để quyết định inference mode ─────────────────
        duration = get_audio_duration_seconds(wav_path)
        log.info("📏  Độ dài audio: %.1f giây (%.1f phút)", duration, duration / 60)

        # ── Bước 3: khởi tạo Whisper ─────────────────────────────────────────
        device       = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        whisper_model = build_whisper_model(device, compute_type)

        # ── Bước 4: transcription (adaptive) ─────────────────────────────────
        words = transcribe_audio(
            wav_path        = wav_path,
            model           = whisper_model,
            duration_seconds= duration,
            language        = language,
            initial_prompt  = initial_prompt,
        )

        # ── Bước 5: diarization ───────────────────────────────────────────────
        diarization = run_diarization(
            wav_path     = wav_path,
            hf_token     = hf_token,
            num_speakers = num_speakers,
            min_speakers = min_speakers,
            max_speakers = max_speakers,
        )

    # ── Bước 6: alignment ────────────────────────────────────────────────────
    log.info("🔗  Đang alignment word-timestamps ↔ speaker segments …")
    aligned = align_words_to_speakers(words, diarization)
    log.info("✅  Alignment xong: %d đoạn.", len(aligned))

    # ── Bước 7: format transcript ────────────────────────────────────────────
    transcript = format_transcript(aligned)

    # ── Bước 8: lưu file (tuỳ chọn) ─────────────────────────────────────────
    if output_file:
        Path(output_file).write_text(transcript, encoding="utf-8")
        log.info("💾  Transcript đã lưu: %s", output_file)

    return transcript


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
    return parser.parse_args()


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

    transcript = run_pipeline(
        input_audio    = args.audio,
        hf_token       = args.hf_token,
        language       = args.language,
        initial_prompt = args.initial_prompt,
        num_speakers   = args.num_speakers,
        min_speakers   = args.min_speakers,
        max_speakers   = args.max_speakers,
        output_file    = args.output,
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