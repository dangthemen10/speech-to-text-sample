"""
Speech-to-Text + Speaker Diarization — Japanese-optimized pipeline
===================================================================
Stack  : faster-whisper (large-v3)  +  pyannote/speaker-diarization-community-1
Python : 3.9+

BEFORE RUNNING — see README.md for full setup instructions.
"""

import os
import sys
import logging
import argparse
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Disable pyannote / Lightning telemetry BEFORE any third-party imports
# ---------------------------------------------------------------------------
os.environ["PYANNOTE_METRICS_ENABLED"]    = "false"
os.environ["LIGHTNING_DISABLE_ANALYTICS"] = "1"
os.environ["PL_DISABLE_TELEMETRY"]        = "1"

# ---------------------------------------------------------------------------
# Third-party imports — fail fast with actionable messages
# ---------------------------------------------------------------------------
try:
    from faster_whisper import WhisperModel, BatchedInferencePipeline
except ImportError:
    sys.exit("❌  faster-whisper not installed.  Run: pip install faster-whisper")

try:
    from pyannote.audio import Pipeline as DiarizationPipeline
except ImportError:
    sys.exit("❌  pyannote.audio not installed.  Run: pip install pyannote.audio>=4.0.1")

try:
    import torch
except ImportError:
    sys.exit("❌  torch not installed.  Run: pip install torch>=2.8.0")


# ===========================================================================
# CONSTANTS
# ===========================================================================

WHISPER_MODEL_SIZE                  = "large-v3"
DIARIZATION_MODEL_ID                = "pyannote/speaker-diarization-community-1"
AUDIO_SAMPLE_RATE                   = 16_000   # Hz — required by both models
AUDIO_CHANNELS                      = 1        # mono
BATCHED_PIPELINE_THRESHOLD_MINUTES  = 30       # use BatchedInferencePipeline above this

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# DATA MODELS
# ===========================================================================

@dataclass
class WordStamp:
    """A single word with millisecond-precise timestamps from Whisper."""
    text:        str
    start_sec:   float
    end_sec:     float
    confidence:  float = 1.0


@dataclass
class SpeakerSegment:
    """A time interval attributed to one speaker by pyannote."""
    speaker_id:  str
    start_sec:   float
    end_sec:     float
    is_overlap:  bool = False   # True when multiple speakers talked simultaneously


@dataclass
class AlignedUtterance:
    """Final output unit: speaker + text + timestamps, produced after alignment."""
    speaker_id:  str
    start_sec:   float
    end_sec:     float
    text:        str
    is_overlap:  bool = False


# ===========================================================================
# SECTION 1 — AUDIO UTILITIES
# ===========================================================================

def convert_audio_to_wav(src_path: str, dst_path: str) -> None:
    """
    Convert any ffmpeg-supported audio/video file to WAV 16 kHz mono PCM-16.
    Both Whisper and pyannote require this exact format.
    """
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", str(AUDIO_CHANNELS),
        "-sample_fmt", "s16",
        dst_path,
    ]
    log.info("🔄  Converting audio → WAV 16 kHz mono …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    log.info("✅  Converted: %s", dst_path)


def probe_audio_duration(wav_path: str) -> float:
    """Return duration in seconds using ffprobe (no extra library needed)."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr}")
    return float(result.stdout.strip())


def denoise_wav(src_path: str, dst_path: str, strength: float = 0.85) -> None:
    """
    Apply spectral-gating noise reduction via noisereduce.

    Recommended when:
      - Background noise is stationary (office fan, AC, café ambience)
      - Two speakers have similar voice profiles — diarization confuses them
      - Audio is short (< 5 min) and noise corrupts speaker embeddings

    strength: 0.0 = no reduction  |  1.0 = maximum  |  sweet spot = 0.75–0.85
    Values above 0.95 may distort phonemes and hurt Whisper accuracy.

    NOTE: Only used for the diarization pass. Whisper always receives the
    original WAV to preserve ASR quality.

    Requires: pip install noisereduce soundfile
    """
    try:
        import noisereduce as nr
        import soundfile as sf
    except ImportError:
        log.warning("⚠️  noisereduce not installed — skipping denoising. "
                    "Run: pip install noisereduce soundfile")
        import shutil
        shutil.copy2(src_path, dst_path)
        return

    log.info("🔇  Denoising (strength=%.2f) …", strength)
    audio_data, sample_rate = sf.read(src_path, dtype="float32")
    denoised = nr.reduce_noise(
        y=audio_data, sr=sample_rate,
        stationary=True,
        prop_decrease=strength,
        n_fft=1024,
        hop_length=256,
    )
    sf.write(dst_path, denoised, sample_rate, subtype="PCM_16")
    log.info("✅  Denoised: %s", dst_path)


# ===========================================================================
# SECTION 2 — SPEECH-TO-TEXT  (faster-whisper, adaptive inference)
# ===========================================================================

def load_whisper_model(device: str, compute_type: str) -> WhisperModel:
    """
    Load Whisper large-v3.
    compute_type = float16 on GPU for speed, int8 on CPU to save RAM.
    """
    log.info("🧠  Loading Whisper %s (device=%s, compute=%s) …",
             WHISPER_MODEL_SIZE, device, compute_type)
    model = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
    log.info("✅  Whisper ready.")
    return model


def transcribe(
    wav_path:       str,
    model:          WhisperModel,
    duration_sec:   float,
    language:       str = "ja",
    initial_prompt: Optional[str] = None,
) -> List[WordStamp]:
    """
    Run ASR and return per-word timestamps.

    Adaptive inference:
      > 30 min  → BatchedInferencePipeline  (higher throughput, VAD on by default)
      ≤ 30 min  → WhisperModel.transcribe   (lower latency, VAD set explicitly)

    initial_prompt:
      Seed Whisper with domain vocabulary or proper nouns to improve accuracy.
      Example: "トヨタ自動車、ソフトバンクグループ、田中部長"
    """
    shared_params = dict(
        language=language,
        word_timestamps=True,           # required for word-level alignment
        initial_prompt=initial_prompt,
        beam_size=5,
        best_of=5,
        temperature=0.0,                # greedy decoding — stable for business use
        condition_on_previous_text=True,
    )

    duration_min = duration_sec / 60
    if duration_sec > BATCHED_PIPELINE_THRESHOLD_MINUTES * 60:
        log.info("⏱  %.1f min > %d min threshold → BatchedInferencePipeline",
                 duration_min, BATCHED_PIPELINE_THRESHOLD_MINUTES)
        batched = BatchedInferencePipeline(model=model)
        segments_iter, info = batched.transcribe(wav_path, batch_size=16, **shared_params)
    else:
        log.info("⏱  %.1f min ≤ %d min threshold → WhisperModel.transcribe",
                 duration_min, BATCHED_PIPELINE_THRESHOLD_MINUTES)
        segments_iter, info = model.transcribe(
            wav_path,
            vad_filter=True,    # must be set manually in non-batched path
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=400),
            **shared_params,
        )

    log.info("🌐  Detected language: %s (confidence=%.2f)",
             info.language, info.language_probability)

    words: List[WordStamp] = []
    for segment in segments_iter:
        if segment.words is None:
            continue
        for w in segment.words:
            words.append(WordStamp(
                text=w.word,
                start_sec=w.start,
                end_sec=w.end,
                confidence=w.probability,
            ))

    log.info("✅  Transcription done: %d words.", len(words))
    return words


# ===========================================================================
# SECTION 3 — SPEAKER DIARIZATION  (pyannote community-1)
# ===========================================================================

def _iter_annotation(annotation):
    """
    Yield (Segment, speaker_label) from a pyannote Annotation.
    Handles both pyannote 3.x (3-tuple) and 4.x (2-tuple) iteration APIs.
    """
    try:
        for item in annotation.itertracks(yield_label=True):
            # pyannote 3.x returns (Segment, track_id, label)
            # pyannote 4.x returns (Segment, label)
            turn, speaker = (item[0], item[2]) if len(item) == 3 else item
            yield turn, speaker
    except Exception:
        for turn, speaker in annotation:   # 4.x direct iteration fallback
            yield turn, speaker


def _find_overlap_intervals(full_annotation) -> List[Tuple[float, float]]:
    """
    Return time intervals where 2+ speakers spoke simultaneously.

    Uses an event-sweep algorithm:
      - Each segment contributes a +1 (open) and -1 (close) event.
      - Whenever active depth >= 2, we are in an overlap region.
    """
    events: List[Tuple[float, int]] = []
    for turn, _ in _iter_annotation(full_annotation):
        events.append((turn.start, +1))
        events.append((turn.end,   -1))

    events.sort(key=lambda e: (e[0], -e[1]))  # ties: open before close

    overlap_intervals: List[Tuple[float, float]] = []
    active_depth = 0
    overlap_start: Optional[float] = None

    for timestamp, delta in events:
        if active_depth >= 2 and overlap_start is not None and timestamp > overlap_start:
            overlap_intervals.append((overlap_start, timestamp))
        active_depth += delta
        overlap_start = timestamp if active_depth >= 2 else None

    return overlap_intervals


def _segment_intersects_overlap(
    seg_start: float,
    seg_end:   float,
    overlap_intervals: List[Tuple[float, float]],
) -> bool:
    """Return True if [seg_start, seg_end] overlaps any overlap interval."""
    return any(
        ov_start < seg_end and ov_end > seg_start
        for ov_start, ov_end in overlap_intervals
    )


def _relabel_by_first_appearance(segments: List[SpeakerSegment]) -> None:
    """
    Reassign speaker IDs so SPEAKER_00 = first to speak, SPEAKER_01 = second, etc.

    pyannote assigns IDs from internal clustering order, which does NOT
    guarantee chronological order. This function fixes that discrepancy.
    Mutates segments in-place.
    """
    id_map: dict = {}
    for seg in segments:
        if seg.speaker_id not in id_map:
            id_map[seg.speaker_id] = f"SPEAKER_{len(id_map):02d}"

    if any(orig != new for orig, new in id_map.items()):
        log.info("🔀  Re-labeling speakers by first appearance:")
        for orig, new in id_map.items():
            if orig != new:
                log.info("     %s → %s", orig, new)
        for seg in segments:
            seg.speaker_id = id_map[seg.speaker_id]


def _apply_hyperparameter_overrides(
    pipeline,
    clustering_threshold:   Optional[float],
    segmentation_threshold: Optional[float],
) -> None:
    """
    Override pyannote pipeline hyperparameters post-load.

    clustering_threshold:
      Higher (0.80+) → fewer speakers, fixes over-segmentation
      Lower  (0.55–) → more speakers, fixes under-segmentation

    segmentation_threshold:
      Lower (0.60–0.75) → catches more short speech bursts

    Tries two paths: pipeline.instantiate() then direct attribute set.
    Warns if neither works (API varies across pyannote versions).
    """
    try:
        current = pipeline.parameters(instantiated=True)
        flat    = _flatten_dict(current)
        log.info("📐  Pipeline params: %s", ", ".join(f"{k}={v}" for k, v in flat.items()))
    except Exception:
        pass

    for param_path, value in [
        (["clustering",   "threshold"], clustering_threshold),
        (["segmentation", "threshold"], segmentation_threshold),
    ]:
        if value is None:
            continue
        _safe_set_pipeline_param(pipeline, param_path, value)
        log.info("🔧  %s → %.3f", ".".join(param_path), value)


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict: {a: {b: v}} → {'a.b': v}."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        out.update(_flatten_dict(v, key) if isinstance(v, dict) else {key: v})
    return out


def _safe_set_pipeline_param(pipeline, path: List[str], value: float) -> None:
    """
    Set a nested pipeline parameter via two strategies:
      1. pipeline.instantiate() (preferred, pyannote official API)
      2. Direct attribute traversal (fallback)
    """
    try:
        params = pipeline.parameters(instantiated=True)
        section, key = path[0], path[1]
        if section in params and key in params[section]:
            params[section][key] = value
            pipeline.instantiate(params)
            return
    except Exception:
        pass

    try:
        obj = pipeline
        for attr in path[:-1]:
            obj = getattr(obj, attr)
        setattr(obj, path[-1], value)
    except Exception:
        log.warning("⚠️  Could not set %s — may differ across pyannote versions.",
                    ".".join(path))


def diarize(
    wav_path:               str,
    hf_token:               str,
    num_speakers:           Optional[int]   = None,
    min_speakers:           Optional[int]   = None,
    max_speakers:           Optional[int]   = None,
    clustering_threshold:   Optional[float] = None,
    segmentation_threshold: Optional[float] = None,
) -> List[SpeakerSegment]:
    """
    Run pyannote/speaker-diarization-community-1 and return speaker segments.

    community-1 DiarizeOutput fields:
      .speaker_diarization           — full annotation incl. simultaneous tracks
      .exclusive_speaker_diarization — 1 speaker per moment (used for alignment)
      .speaker_embeddings            — per-speaker embedding vectors

    Pipeline steps:
      1. Detect overlap regions from .speaker_diarization via event-sweep
      2. Build SpeakerSegment list from .exclusive_speaker_diarization
      3. Tag segments that intersect detected overlap regions
      4. Re-label speaker IDs in chronological order of first appearance

    HF token: must have "read" scope and the model's terms must be accepted at
      https://huggingface.co/pyannote/speaker-diarization-community-1
    """
    log.info("🎙  Loading diarization pipeline: %s …", DIARIZATION_MODEL_ID)
    pipeline = DiarizationPipeline.from_pretrained(DIARIZATION_MODEL_ID, token=hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)
    log.info("🎙  Running diarization on %s …", device)

    _apply_hyperparameter_overrides(pipeline, clustering_threshold, segmentation_threshold)

    run_kwargs: dict = {}
    if num_speakers is not None:
        run_kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            run_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            run_kwargs["max_speakers"] = max_speakers

    output = pipeline(wav_path, **run_kwargs)

    # Step 1: find overlap regions
    overlap_intervals = _find_overlap_intervals(output.speaker_diarization)
    log.info("📊  Overlap intervals detected: %d", len(overlap_intervals))

    # Steps 2–3: build segment list from exclusive annotation
    segments: List[SpeakerSegment] = [
        SpeakerSegment(
            speaker_id=speaker,
            start_sec=turn.start,
            end_sec=turn.end,
            is_overlap=_segment_intersects_overlap(turn.start, turn.end, overlap_intervals),
        )
        for turn, speaker in _iter_annotation(output.exclusive_speaker_diarization)
    ]
    segments.sort(key=lambda s: s.start_sec)

    # Step 4: re-label chronologically
    _relabel_by_first_appearance(segments)

    unique_speakers = sorted({s.speaker_id for s in segments})
    overlap_count   = sum(1 for s in segments if s.is_overlap)
    log.info("✅  Diarization done: %d segments | %d speakers | %d overlap",
             len(segments), len(unique_speakers), overlap_count)
    log.info("👥  Speakers: %s", unique_speakers)

    return segments


# ===========================================================================
# SECTION 4 — ALIGNMENT  (word timestamps ↔ speaker segments)
# ===========================================================================

def _intersection_duration(
    a_start: float, a_end: float,
    b_start: float, b_end: float,
) -> float:
    """Return the length (seconds) of the intersection of two intervals."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def align_words_to_speakers(
    words:    List[WordStamp],
    segments: List[SpeakerSegment],
) -> List[AlignedUtterance]:
    """
    Assign each WordStamp to the SpeakerSegment with the greatest temporal overlap,
    then merge consecutive same-speaker words into AlignedUtterance objects.

    Edge cases:
      - Word falls in a gap between segments → assigned to previous speaker (fallback).
      - is_overlap flag propagates if ANY word in the merged chunk was in an overlap zone.
    """
    if not words:
        return []

    fallback_speaker = segments[0].speaker_id if segments else "UNKNOWN"
    word_assignments: List[Tuple[str, bool]] = []

    for word in words:
        best_speaker    = None
        best_overlap    = 0.0
        word_is_overlap = False

        for seg in segments:
            if seg.end_sec <= word.start_sec or seg.start_sec >= word.end_sec:
                continue
            overlap = _intersection_duration(
                word.start_sec, word.end_sec,
                seg.start_sec,  seg.end_sec,
            )
            if overlap > best_overlap:
                best_overlap    = overlap
                best_speaker    = seg.speaker_id
                word_is_overlap = seg.is_overlap

        if best_speaker is None:
            best_speaker = fallback_speaker   # gap fallback: inherit last speaker

        word_assignments.append((best_speaker, word_is_overlap))
        fallback_speaker = best_speaker

    # Merge consecutive same-speaker words into utterances
    utterances: List[AlignedUtterance] = []
    current_speaker, current_is_overlap = word_assignments[0]
    current_chunk: List[WordStamp] = [words[0]]

    for i in range(1, len(words)):
        speaker, is_overlap = word_assignments[i]
        if speaker == current_speaker:
            current_chunk.append(words[i])
            current_is_overlap = current_is_overlap or is_overlap
        else:
            utterances.append(AlignedUtterance(
                speaker_id=current_speaker,
                start_sec=current_chunk[0].start_sec,
                end_sec=current_chunk[-1].end_sec,
                text="".join(w.text for w in current_chunk).strip(),
                is_overlap=current_is_overlap,
            ))
            current_speaker    = speaker
            current_is_overlap = is_overlap
            current_chunk      = [words[i]]

    # Flush the final chunk
    utterances.append(AlignedUtterance(
        speaker_id=current_speaker,
        start_sec=current_chunk[0].start_sec,
        end_sec=current_chunk[-1].end_sec,
        text="".join(w.text for w in current_chunk).strip(),
        is_overlap=current_is_overlap,
    ))

    return utterances


# ===========================================================================
# SECTION 5 — OUTPUT FORMATTERS
# ===========================================================================

def _fmt_ts(seconds: float) -> str:
    """MM:SS or HH:MM:SS for display in transcript lines."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _fmt_ts_ms(seconds: float) -> str:
    """MM:SS.mmm for millisecond-precision debug output."""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:06.3f}"


def format_final_transcript(utterances: List[AlignedUtterance]) -> str:
    """
    Produce the human-readable transcript:
      [MM:SS - MM:SS] SPEAKER_XX: <text>
    Overlap segments are tagged with [OVERLAP].
    """
    lines = []
    for utt in utterances:
        if not utt.text:
            continue
        tag = " [OVERLAP]" if utt.is_overlap else ""
        lines.append(
            f"[{_fmt_ts(utt.start_sec)} - {_fmt_ts(utt.end_sec)}] "
            f"{utt.speaker_id}{tag}: {utt.text}"
        )
    return "\n".join(lines)


def format_whisper_debug(words: List[WordStamp]) -> str:
    """
    Debug view of Whisper output: one word per line with ms timestamps
    and a confidence bar. Saved to 01_whisper_raw.txt in --debug-dir.
    """
    if not words:
        return "(no words transcribed)"
    lines = [
        "=" * 65,
        "WHISPER RAW — word-level timestamps",
        f"Total words: {len(words)}",
        "=" * 65,
    ]
    for w in words:
        bar = "█" * int(w.confidence * 10) + "░" * (10 - int(w.confidence * 10))
        lines.append(
            f"[{_fmt_ts_ms(w.start_sec)} → {_fmt_ts_ms(w.end_sec)}]  "
            f"{w.text:<20s}  p={w.confidence:.2f}  {bar}"
        )
    return "\n".join(lines)


def format_diarization_debug(segments: List[SpeakerSegment]) -> str:
    """
    Debug view of diarization output: per-speaker stats + segment list.
    Saved to 02_diarization_raw.txt in --debug-dir.
    """
    if not segments:
        return "(no segments)"

    speaker_totals: dict = {}
    for seg in segments:
        dur = seg.end_sec - seg.start_sec
        speaker_totals[seg.speaker_id] = speaker_totals.get(seg.speaker_id, 0.0) + dur

    total_dur     = max((s.end_sec for s in segments), default=0.0)
    overlap_count = sum(1 for s in segments if s.is_overlap)

    lines = [
        "=" * 65,
        "PYANNOTE DIARIZATION RAW",
        f"Segments     : {len(segments)}",
        f"Overlap segs : {overlap_count}",
        f"Duration     : {_fmt_ts(total_dur)}",
        f"Speakers     : {sorted(speaker_totals.keys())}",
        "-" * 65,
        "Speaker share:",
    ]
    for spk, dur in sorted(speaker_totals.items()):
        pct = dur / total_dur * 100 if total_dur else 0
        bar = "█" * int(pct / 5)
        lines.append(f"  {spk:<14s}  {dur:6.1f}s  ({pct:5.1f}%)  {bar}")

    lines += [
        "-" * 65,
        f"  {'[START - END]':<18}  {'SPEAKER':<14}  {'DUR':>6}  NOTE",
        "  " + "-" * 55,
    ]
    for seg in segments:
        dur  = seg.end_sec - seg.start_sec
        note = "⚠ OVERLAP" if seg.is_overlap else ""
        lines.append(
            f"  [{_fmt_ts(seg.start_sec)} - {_fmt_ts(seg.end_sec)}]  "
            f"{seg.speaker_id:<14s}  {dur:5.2f}s  {note}"
        )
    lines.append("=" * 65)
    return "\n".join(lines)


def save_alignment_debug(
    words:      List[WordStamp],
    segments:   List[SpeakerSegment],
    utterances: List[AlignedUtterance],
    save_path:  Path,
) -> None:
    """
    Build and save a side-by-side alignment comparison table to save_path.

    Highlights:
      - Segments with NO matched words (timeline gap issues)
      - Words that fell outside ALL segments (VAD / boundary mismatch)

    Saved to 04_alignment_debug.txt in --debug-dir.
    """
    lines = [
        "=" * 75,
        "ALIGNMENT DEBUG",
        "=" * 75,
        f"{'DIARIZATION SEGMENT':<38}  WHISPER WORDS",
        "-" * 75,
    ]

    for seg in segments:
        matched = [
            w for w in words
            if not (w.end_sec <= seg.start_sec or w.start_sec >= seg.end_sec)
        ]
        label = (
            f"[{_fmt_ts(seg.start_sec)}-{_fmt_ts(seg.end_sec)}] "
            f"{seg.speaker_id}"
            + (" ⚠OVL" if seg.is_overlap else "")
        )
        if matched:
            preview = "".join(w.text for w in matched[:8]).strip()
            if len(matched) > 8:
                preview += f"… (+{len(matched)-8})"
            detail = f"({len(matched)} words) {preview}"
        else:
            detail = "⚠️  NO WORDS MATCHED — possible timeline gap"
        lines.append(f"{label:<38}  {detail}")

    unmapped = [
        w for w in words
        if not any(
            not (w.end_sec <= seg.start_sec or w.start_sec >= seg.end_sec)
            for seg in segments
        )
    ]
    if unmapped:
        lines += ["", f"⚠️  {len(unmapped)} words outside all segments:"]
        for w in unmapped[:10]:
            lines.append(f"   [{_fmt_ts_ms(w.start_sec)}-{_fmt_ts_ms(w.end_sec)}] '{w.text}'")
        if len(unmapped) > 10:
            lines.append(f"   ... and {len(unmapped)-10} more")

    lines.append("=" * 75)
    summary = "\n".join(lines)
    print("\n" + summary)
    save_path.write_text(summary, encoding="utf-8")
    log.info("💾  [DEBUG] Alignment summary → %s", save_path)


# ===========================================================================
# SECTION 6 — PIPELINE ORCHESTRATOR
# ===========================================================================

def run_pipeline(
    audio_path:             str,
    hf_token:               str,
    language:               str            = "ja",
    initial_prompt:         Optional[str]  = None,
    num_speakers:           Optional[int]  = None,
    min_speakers:           Optional[int]  = None,
    max_speakers:           Optional[int]  = None,
    output_file:            Optional[str]  = None,
    clustering_threshold:   Optional[float] = None,
    segmentation_threshold: Optional[float] = None,
    enable_denoise:         bool           = False,
    denoise_strength:       float          = 0.85,
    debug_dir:              Optional[str]  = None,
) -> str:
    """
    End-to-end orchestrator:
      audio → WAV 16kHz → [denoise] → Whisper (STT) + pyannote (diarize) → align → transcript

    Split-input design:
      Whisper  always uses the original WAV for maximum ASR accuracy.
      pyannote uses the denoised WAV when enable_denoise=True,
      which improves speaker embedding quality in noisy environments.

    debug_dir: when provided, saves 4 intermediate inspection files:
      01_whisper_raw.txt      — per-word timestamps from Whisper
      02_diarization_raw.txt  — speaker segments from pyannote
      03_final_transcript.txt — aligned transcript
      04_alignment_debug.txt  — side-by-side comparison table
    """
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    debug_path: Optional[Path] = None
    if debug_dir:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        log.info("🐛  Debug mode ON → %s", debug_path)

    with tempfile.TemporaryDirectory() as tmp:
        wav_path      = str(Path(tmp) / "audio.wav")
        denoised_path = str(Path(tmp) / "audio_denoised.wav")

        # Step 1: normalise to WAV 16 kHz mono
        convert_audio_to_wav(audio_path, wav_path)

        # Step 2 (optional): denoise — diarization path only
        diarize_input = wav_path
        if enable_denoise:
            denoise_wav(wav_path, denoised_path, strength=denoise_strength)
            diarize_input = denoised_path
            log.info("🎙  Diarize ← denoised  |  Whisper ← original")

        duration = probe_audio_duration(wav_path)
        log.info("📏  Duration: %.1fs (%.1f min)", duration, duration / 60)

        # Step 3: ASR
        device        = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type  = "float16" if device == "cuda" else "int8"
        whisper_model = load_whisper_model(device, compute_type)

        words = transcribe(
            wav_path=wav_path,
            model=whisper_model,
            duration_sec=duration,
            language=language,
            initial_prompt=initial_prompt,
        )

        if debug_path:
            p = debug_path / "01_whisper_raw.txt"
            p.write_text(format_whisper_debug(words), encoding="utf-8")
            log.info("💾  [DEBUG] Whisper raw → %s", p)

        # Step 4: Diarization
        segments = diarize(
            wav_path=diarize_input,
            hf_token=hf_token,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            clustering_threshold=clustering_threshold,
            segmentation_threshold=segmentation_threshold,
        )

        if debug_path:
            p = debug_path / "02_diarization_raw.txt"
            p.write_text(format_diarization_debug(segments), encoding="utf-8")
            log.info("💾  [DEBUG] Diarization raw → %s", p)

    # Step 5: Alignment
    log.info("🔗  Aligning words to speakers …")
    utterances = align_words_to_speakers(words, segments)
    log.info("✅  Alignment done: %d utterances.", len(utterances))

    # Step 6: Format & save
    transcript = format_final_transcript(utterances)

    if output_file:
        Path(output_file).write_text(transcript, encoding="utf-8")
        log.info("💾  Transcript saved → %s", output_file)

    if debug_path:
        (debug_path / "03_final_transcript.txt").write_text(transcript, encoding="utf-8")
        save_alignment_debug(words, segments, utterances, debug_path / "04_alignment_debug.txt")

    return transcript


# ===========================================================================
# SECTION 7 — DIAGNOSE MODE  (threshold sweep, Whisper-free)
# ===========================================================================

def run_threshold_sweep(
    wav_path:     str,
    pipeline,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> None:
    """
    Sweep clustering thresholds and report detected speaker counts.
    Does NOT run Whisper — provides fast feedback for threshold tuning.
    Use before a full pipeline run to find the optimal threshold value.
    """
    print("\n" + "=" * 70)
    print("🔬  DIAGNOSE — Clustering Threshold Sweep")
    print("=" * 70)
    print(f"{'Threshold':>12}  {'Speakers':>10}  {'Segments':>10}")
    print("-" * 38)

    run_kwargs: dict = {}
    if num_speakers is not None:
        run_kwargs["num_speakers"] = num_speakers
        print(f"  (num_speakers={num_speakers} fixed — threshold has limited effect)")
    else:
        if min_speakers:
            run_kwargs["min_speakers"] = min_speakers
        if max_speakers:
            run_kwargs["max_speakers"] = max_speakers

    for threshold in [0.55, 0.60, 0.65, 0.70, 0.715, 0.75, 0.80, 0.85, 0.90]:
        _safe_set_pipeline_param(pipeline, ["clustering", "threshold"], threshold)
        try:
            out       = pipeline(wav_path, **run_kwargs)
            speakers  = set()
            seg_count = 0
            for item in out.exclusive_speaker_diarization.itertracks(yield_label=True):
                speakers.add(item[-1])
                seg_count += 1
            print(f"{threshold:>12.3f}  {len(speakers):>10}  {seg_count:>10}")
        except Exception as exc:
            print(f"{threshold:>12.3f}  ERROR: {exc}")

    print("=" * 70)
    print("💡  Too many speakers → raise threshold  |  Too few → lower threshold")
    print("💡  Then run with: --clustering-threshold <value>")
    print("=" * 70)


# ===========================================================================
# SECTION 8 — CLI
# ===========================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="STT + Speaker Diarization (Japanese)\n"
                    "faster-whisper large-v3  +  pyannote community-1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("audio",
                        help="Input audio/video file (any format ffmpeg supports).")

    parser.add_argument("--hf-token",
                        default=os.environ.get("HF_TOKEN", ""),
                        help="HuggingFace read token. Defaults to $HF_TOKEN. "
                             "Create at: https://hf.co/settings/tokens")

    parser.add_argument("--language", default="ja",
                        help="ISO-639-1 language code (default: ja).")
    parser.add_argument("--initial-prompt",
                        help="Seed Whisper with domain vocabulary. "
                             "Example: 'トヨタ自動車、ソフトバンク、田中部長'")

    parser.add_argument("--num-speakers",  type=int,
                        help="Exact speaker count (if known — bypasses clustering).")
    parser.add_argument("--min-speakers",  type=int, help="Minimum speaker count.")
    parser.add_argument("--max-speakers",  type=int, help="Maximum speaker count.")
    parser.add_argument("--clustering-threshold", type=float,
                        help="Clustering threshold (~0.715 default). "
                             "Raise → merge over-split; Lower → split under-merged.")
    parser.add_argument("--segmentation-threshold", type=float,
                        help="Segmentation threshold (~0.817 default). "
                             "Lower → capture more short bursts.")

    parser.add_argument("--denoise", action="store_true",
                        help="Denoise before diarization. "
                             "Recommended for noisy/office recordings. "
                             "Requires: pip install noisereduce soundfile")
    parser.add_argument("--denoise-strength", type=float, default=0.85,
                        help="Noise reduction strength 0.0–1.0 (default 0.85).")

    parser.add_argument("--output", "-o", help="Save transcript to file (UTF-8).")
    parser.add_argument("--debug-dir",
                        help="Save 4 intermediate debug files to this directory.")
    parser.add_argument("--diagnose", action="store_true",
                        help="Sweep clustering thresholds only — no Whisper. "
                             "Use to tune diarization before a full run.")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.hf_token:
        sys.exit(
            "❌  HuggingFace token missing.\n"
            "    Set env var : export HF_TOKEN='hf_xxx'\n"
            "    Or pass flag: --hf-token hf_xxx\n"
            "    Create token: https://hf.co/settings/tokens\n"
            "    Accept model: https://huggingface.co/pyannote/speaker-diarization-community-1"
        )

    if args.diagnose:
        log.info("🔬  Diagnose mode.")
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = str(Path(tmp) / "audio.wav")
            convert_audio_to_wav(args.audio, wav_path)
            pipeline = DiarizationPipeline.from_pretrained(
                DIARIZATION_MODEL_ID, token=args.hf_token
            )
            pipeline = pipeline.to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            run_threshold_sweep(
                wav_path=wav_path, pipeline=pipeline,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
            )
        return

    transcript = run_pipeline(
        audio_path=args.audio,
        hf_token=args.hf_token,
        language=args.language,
        initial_prompt=args.initial_prompt,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        output_file=args.output,
        clustering_threshold=args.clustering_threshold,
        segmentation_threshold=args.segmentation_threshold,
        enable_denoise=args.denoise,
        denoise_strength=args.denoise_strength,
        debug_dir=args.debug_dir,
    )

    print("\n" + "=" * 70)
    print("📝  TRANSCRIPT")
    print("=" * 70)
    print(transcript)
    print("=" * 70)


if __name__ == "__main__":
    main()
