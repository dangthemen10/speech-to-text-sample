"""
Speech-to-Text + Speaker Diarization Pipeline
==============================================
Models
  STT        : faster-whisper large-v3  (CTranslate2)
  Diarization: pyannote/speaker-diarization-community-1  (pyannote.audio 4.0)

Tunable parameters (all exposed as CLI flags)
  Whisper     : --beam-size, --temperature, --no-speech-threshold, --compression-ratio
  VAD (Silero): --vad-threshold, --vad-min-speech-ms, --vad-min-silence-ms, --vad-speech-pad-ms
  Diarization : --clustering-threshold (pyannote internal)
  Post-proc   : --merge-gap, --min-segment, --embedding-threshold
  Pipeline    : --batched-threshold

Usage
  python stt_diarize.py --audio meeting.wav --language ja --hf-token hf_xxxx
  python stt_diarize.py --audio meeting.wav --language ja \\
      --beam-size 10 --vad-min-silence-ms 300 --clustering-threshold 0.7
"""

import os
import shutil
import subprocess
import tempfile
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# ── Disable pyannote telemetry before ANY pyannote import ─────────────────────
os.environ["PYANNOTE_METRICS_ENABLED"] = "false"
os.environ["PYANNOTE_DATABASE_CONFIG"] = ""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
# Module-level tuning constants
# All of these are overridable via CLI flags.
# Defined here (before any function) so they can be used as default= values.
# =============================================================================

# ── Pipeline selection ────────────────────────────────────────────────────────
BATCHED_THRESHOLD_SECONDS = 600     # Use BatchedInferencePipeline above this (GPU only)

# ── Whisper / STT ─────────────────────────────────────────────────────────────
BEAM_SIZE              = 5          # Higher → more accurate but slower (5–10)
TEMPERATURE            = 0.0        # 0 = greedy; increase if hallucinations occur
NO_SPEECH_THRESHOLD    = 0.6        # Probability above which a segment is silence
COMPRESSION_RATIO      = 2.4        # Segments with higher ratio are likely garbage

# ── Silero VAD ────────────────────────────────────────────────────────────────
VAD_THRESHOLD          = 0.5        # Speech probability threshold (0–1)
VAD_MIN_SPEECH_MS      = 250        # Minimum speech chunk duration to keep (ms)
VAD_MIN_SILENCE_MS     = 500        # Minimum silence to split on (ms)
                                    # Lower → keeps more short pauses (good for Japanese)
VAD_SPEECH_PAD_MS      = 200        # Padding added to each side of speech chunk (ms)

# ── Diarization post-processing ───────────────────────────────────────────────
MAX_MERGE_GAP          = 0.5        # Merge same-speaker gaps shorter than this (s)
MIN_SEGMENT_DURATION   = 0.3        # Drop isolated segments shorter than this (s)
EMBEDDING_THRESHOLD    = 0.85       # Cosine similarity to merge confused speakers
                                    # 0.90=conservative  0.85=recommended  0.75=aggressive
                                    # 1.0 = disable Pass 3 entirely

# ── pyannote clustering ───────────────────────────────────────────────────────
CLUSTERING_THRESHOLD   = None       # None = use pyannote default (~0.7)
                                    # Lower → more speakers detected
                                    # Higher → fewer speakers (aggressive merge)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class WordToken:
    """Single word with time boundaries from Whisper word_timestamps."""
    word: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class SpeakerSegment:
    """Speaker turn from pyannote diarization."""
    speaker: str
    start: float
    end: float
    overlapping: bool = False


@dataclass
class AlignedUtterance:
    """Final aligned output: speaker(s) + text for a contiguous time range."""
    speakers: List[str]
    start: float
    end: float
    text: str
    overlapping: bool = False


# =============================================================================
# Helpers
# =============================================================================

def fmt(seconds: float) -> str:
    """Format float seconds → MM:SS string."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def get_audio_duration(audio_path: str) -> float:
    """Return audio duration in seconds via torchaudio metadata (no decode)."""
    try:
        import torchaudio
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate
    except Exception:
        return 0.0


def _stem(audio_path: str) -> str:
    return Path(audio_path).stem


# =============================================================================
# Model download with progress bar
# =============================================================================

def ensure_hf_model(repo_id: str, token: Optional[str]) -> None:
    """
    Pre-fetch a HuggingFace model repo so users see a tqdm progress bar on
    first run.  On subsequent runs the local HF cache is used instantly.
    Cache location: $HF_HOME  or  ~/.cache/huggingface/hub
    """
    try:
        from huggingface_hub import snapshot_download
        log.info("Checking / downloading '%s' …", repo_id)
        snapshot_download(repo_id=repo_id, token=token)
        log.info("'%s' ready in local cache.", repo_id)
    except Exception as e:
        log.warning("Pre-fetch of '%s' failed (%s) — will retry during load.", repo_id, e)


# =============================================================================
# Audio pre-processing  (pyannote requires mono 16 kHz WAV)
# =============================================================================

def prepare_audio(audio_path: str) -> Tuple[str, bool]:
    """
    Convert audio to mono 16 kHz 16-bit PCM WAV for pyannote compatibility.
    Returns (path, is_tmp).  Caller must os.unlink(path) when is_tmp=True.

    Strategy:
      1. Already mono 16 kHz WAV  → return as-is
      2. torchaudio  (pure Python, fast)
      3. ffmpeg subprocess  (handles any codec)
    """
    src = Path(audio_path).resolve()

    if src.suffix.lower() == ".wav":
        try:
            import torchaudio
            info = torchaudio.info(str(src))
            if info.num_channels == 1 and info.sample_rate == 16_000:
                log.info("Audio is already mono 16 kHz WAV — skipping conversion.")
                return str(src), False
            log.info("WAV needs conversion: %d ch, %d Hz → mono 16 kHz",
                     info.num_channels, info.sample_rate)
        except Exception:
            pass

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    tmp_path = tmp.name

    try:
        import torchaudio
        waveform, sr = torchaudio.load(str(src))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16_000:
            waveform = torchaudio.transforms.Resample(sr, 16_000)(waveform)
        torchaudio.save(tmp_path, waveform, 16_000, encoding="PCM_S", bits_per_sample=16)
        log.info("Audio converted via torchaudio → %s", tmp_path)
        return tmp_path, True
    except Exception as e:
        log.warning("torchaudio conversion failed (%s) — trying ffmpeg …", e)

    if shutil.which("ffmpeg") is None:
        os.unlink(tmp_path)
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (apt/brew/choco) "
            "or ensure torchaudio can read the input file."
        )
    cmd = ["ffmpeg", "-y", "-i", str(src),
           "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", tmp_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        os.unlink(tmp_path)
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}")
    log.info("Audio converted via ffmpeg → %s", tmp_path)
    return tmp_path, True


# =============================================================================
# Step 1 — Speech-to-Text
# =============================================================================

def run_stt(
    audio_path: str,
    # ── Model ────────────────────────────────────────────────────────────────
    model_size: str = "large-v3",
    device: str = "auto",
    compute_type: str = "auto",
    # ── Language ─────────────────────────────────────────────────────────────
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    # ── Whisper decoding ─────────────────────────────────────────────────────
    beam_size: int = BEAM_SIZE,
    temperature: float = TEMPERATURE,
    no_speech_threshold: float = NO_SPEECH_THRESHOLD,
    compression_ratio_threshold: float = COMPRESSION_RATIO,
    # ── VAD ──────────────────────────────────────────────────────────────────
    vad_threshold: float = VAD_THRESHOLD,
    vad_min_speech_ms: int = VAD_MIN_SPEECH_MS,
    vad_min_silence_ms: int = VAD_MIN_SILENCE_MS,
    vad_speech_pad_ms: int = VAD_SPEECH_PAD_MS,
    # ── Output ───────────────────────────────────────────────────────────────
    stt_output_path: Optional[str] = None,
) -> List[WordToken]:
    """
    Transcribe audio with faster-whisper.

    Pipeline selection
    ------------------
    • GPU + duration > BATCHED_THRESHOLD_SECONDS → BatchedInferencePipeline
      (parallel chunk processing, much faster for long recordings)
    • CPU or short audio → WhisperModel.transcribe() sequential

    Whisper parameters
    ------------------
    beam_size                 : Search width. 5 is default; 10 is better but ~2× slower.
    temperature               : 0 = deterministic greedy decoding.
                                Increase to 0.2–0.4 if Whisper hallucinates on silence.
    no_speech_threshold       : Segments with silence probability above this are skipped.
                                Lower (0.4) = keep more; Higher (0.8) = skip more.
    compression_ratio_threshold: Segments with gzip compression ratio above this are
                                likely repetition/garbage and are discarded.

    VAD (Silero) parameters
    -----------------------
    vad_threshold      : Speech probability threshold. Lower = keep more borderline audio.
    vad_min_speech_ms  : Minimum speech chunk to keep. Raise for noisy audio.
    vad_min_silence_ms : Silence shorter than this is NOT used as a split point.
                         Lower this (e.g. 300) for Japanese where inter-word pauses are short.
    vad_speech_pad_ms  : Padding added to both ends of each detected speech segment.
    """
    import torch
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    ensure_hf_model(f"Systran/faster-whisper-{model_size}", token=None)

    log.info("Loading Whisper '%s' on %s (%s) …", model_size, device, compute_type)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    duration = get_audio_duration(audio_path)
    use_batched = (device == "cuda") and (duration > BATCHED_THRESHOLD_SECONDS)

    # Common transcription kwargs
    transcribe_kwargs = dict(
        language=language,
        initial_prompt=initial_prompt,
        beam_size=beam_size,
        temperature=temperature,
        no_speech_threshold=no_speech_threshold,
        compression_ratio_threshold=compression_ratio_threshold,
        word_timestamps=True,
        vad_filter=False,
        vad_parameters={
            "threshold":             vad_threshold,
            "min_speech_duration_ms": vad_min_speech_ms,
            "min_silence_duration_ms": vad_min_silence_ms,
            "speech_pad_ms":          vad_speech_pad_ms,
        },
    )

    if use_batched:
        log.info(
            "Duration %.0f s > %d s + GPU → BatchedInferencePipeline (batch_size=16)",
            duration, BATCHED_THRESHOLD_SECONDS,
        )
        batched = BatchedInferencePipeline(model=model)
        segments_iter, info = batched.transcribe(audio_path, batch_size=16, **transcribe_kwargs)
    else:
        reason = "CPU" if device != "cuda" else f"short audio ({duration:.0f} s)"
        log.info("Sequential transcription (%s).", reason)
        segments_iter, info = model.transcribe(audio_path, **transcribe_kwargs)

    log.info("Detected language: %s (prob %.2f)", info.language, info.language_probability)
    log.info(
        "Whisper params — beam_size=%d  temperature=%.2f  "
        "no_speech_thr=%.2f  compression_ratio=%.1f",
        beam_size, temperature, no_speech_threshold, compression_ratio_threshold,
    )
    log.info(
        "VAD params — threshold=%.2f  min_speech=%d ms  "
        "min_silence=%d ms  speech_pad=%d ms",
        vad_threshold, vad_min_speech_ms, vad_min_silence_ms, vad_speech_pad_ms,
    )

    words: List[WordToken] = []
    stt_lines: List[str] = []

    for seg in segments_iter:
        stt_lines.append(f"[{fmt(seg.start)} → {fmt(seg.end)}]  {seg.text.strip()}")
        if seg.words is None:
            continue
        for w in seg.words:
            words.append(WordToken(
                word=w.word, start=w.start, end=w.end, probability=w.probability,
            ))

    log.info("STT complete — %d words, %d segments.", len(words), len(stt_lines))

    if stt_output_path:
        Path(stt_output_path).write_text("\n".join(stt_lines), encoding="utf-8")
        log.info("Raw STT saved → %s", stt_output_path)

    return words


# =============================================================================
# Step 2 — Speaker Diarization
# =============================================================================

def run_diarization(
    audio_path: str,
    hf_token: Optional[str] = None,
    # ── Speaker count ─────────────────────────────────────────────────────────
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    # ── pyannote clustering ───────────────────────────────────────────────────
    clustering_threshold: Optional[float] = CLUSTERING_THRESHOLD,
    # ── Post-processing ───────────────────────────────────────────────────────
    embedding_threshold: float = EMBEDDING_THRESHOLD,
    # ── Debug output ─────────────────────────────────────────────────────────
    diar_output_path: Optional[str] = None,
) -> List[SpeakerSegment]:
    """
    Diarize with pyannote/speaker-diarization-community-1.

    community-1 tracks
    ------------------
    • exclusive_speaker_diarization  — one speaker per frame (used as base)
    • speaker_diarization (raw)      — may overlap (used for overlap detection only)

    clustering_threshold
    --------------------
    Controls pyannote's internal AgglomerativeClustering step.
    This is the most impactful single parameter for speaker accuracy.

      Lower value (e.g. 0.5)  → cluster less aggressively → MORE speakers detected
                                 Use when pyannote merges distinct speakers into one.
      Higher value (e.g. 0.8) → cluster more aggressively → FEWER speakers detected
                                 Use when pyannote splits one person into multiple IDs.
      None (default)          → use pyannote's built-in default (~0.7)

    Post-processing passes
    ----------------------
    Pass 1 — Gap merge   : same-speaker turns with gap ≤ MAX_MERGE_GAP are joined
    Pass 2 — Micro-drop  : isolated segments < MIN_SEGMENT_DURATION are removed
    Pass 3 — Embedding   : speaker IDs with cosine similarity ≥ embedding_threshold
                           are merged (catches split-speaker errors from clustering)
    """
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    import torch

    token = hf_token or os.environ.get("HF_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_hf_model("pyannote/speaker-diarization-community-1", token=token)

    log.info("Loading pyannote/speaker-diarization-community-1 …")
    diarize_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token,
    ).to(device)

    # ── Apply clustering threshold ─────────────────────────────────────────────
    # community-1 exposes the clustering threshold via instantiated_pipeline.klustering
    # We set it on the segmentation sub-pipeline's clustering module.
    if clustering_threshold is not None:
        try:
            diarize_pipeline.klustering.threshold = clustering_threshold
            log.info("Clustering threshold set to %.3f", clustering_threshold)
        except AttributeError:
            try:
                # Fallback: older pyannote attribute path
                diarize_pipeline.clustering.threshold = clustering_threshold
                log.info("Clustering threshold (fallback attr) set to %.3f", clustering_threshold)
            except AttributeError:
                log.warning(
                    "Could not set clustering threshold — pyannote attribute path changed. "
                    "Proceeding with pipeline default."
                )

    # ── Build diarize call kwargs ──────────────────────────────────────────────
    diarize_kwargs: dict = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers

    # ── Prepare audio & run ────────────────────────────────────────────────────
    wav_path, is_tmp = prepare_audio(audio_path)
    log.info("Running diarization …")
    try:
        with ProgressHook() as hook:
            output = diarize_pipeline(wav_path, hook=hook, **diarize_kwargs)
    finally:
        if is_tmp:
            os.unlink(wav_path)
            log.debug("Removed temp WAV %s", wav_path)

    # ── Extract exclusive track ────────────────────────────────────────────────
    raw_exclusive: List[SpeakerSegment] = []
    for turn, speaker in output.exclusive_speaker_diarization:
        raw_exclusive.append(SpeakerSegment(
            speaker=speaker,
            start=round(turn.start, 3),
            end=round(turn.end, 3),
        ))
    log.info("Exclusive track — %d raw segments.", len(raw_exclusive))

    # ── Post-processing ────────────────────────────────────────────────────────
    exclusive = _merge_short_segments(raw_exclusive)
    log.info("After Pass 1+2 — %d segments.", len(exclusive))

    exclusive = _correct_speaker_confusion(
        exclusive, audio_path=audio_path,
        hf_token=hf_token, threshold=embedding_threshold,
    )
    log.info("After Pass 3 — %d segments.", len(exclusive))

    # ── Overlap detection from raw track ──────────────────────────────────────
    raw_segments: List[SpeakerSegment] = [
        SpeakerSegment(speaker=spk, start=round(t.start, 3), end=round(t.end, 3))
        for t, _, spk in output.speaker_diarization.itertracks(yield_label=True)
    ]
    overlap_intervals = _compute_overlap_intervals(raw_segments)
    log.info("Overlap intervals: %d", len(overlap_intervals))

    for seg in exclusive:
        for ov_s, ov_e in overlap_intervals:
            if min(seg.end, ov_e) - max(seg.start, ov_s) > 0.05:
                seg.overlapping = True
                break

    flagged = sum(1 for s in exclusive if s.overlapping)
    log.info("Overlap-flagged: %d / %d segments.", flagged, len(exclusive))

    # ── Debug output ───────────────────────────────────────────────────────────
    if diar_output_path:
        lines = [
            f"[{fmt(s.start)} → {fmt(s.end)}]  {s.speaker}"
            + ("  [OVERLAP]" if s.overlapping else "")
            for s in exclusive
        ]
        Path(diar_output_path).write_text("\n".join(lines), encoding="utf-8")
        log.info("Raw diarization saved → %s", diar_output_path)

    return exclusive


# =============================================================================
# Diarization post-processing helpers
# =============================================================================

def _merge_short_segments(segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
    """
    Pass 1: Merge same-speaker turns separated by ≤ MAX_MERGE_GAP seconds.
    Pass 2: Drop micro-segments < MIN_SEGMENT_DURATION flanked by a different speaker.
    """
    if not segments:
        return segments

    # Pass 1 — gap merge
    merged: List[SpeakerSegment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.speaker == prev.speaker and (seg.start - prev.end) <= MAX_MERGE_GAP:
            merged[-1] = SpeakerSegment(speaker=prev.speaker, start=prev.start, end=seg.end)
        else:
            merged.append(seg)

    # Pass 2 — micro-segment drop
    if len(merged) < 3:
        return merged

    cleaned: List[SpeakerSegment] = [merged[0]]
    for i in range(1, len(merged) - 1):
        seg = merged[i]
        prev_spk = merged[i - 1].speaker
        next_spk = merged[i + 1].speaker
        dur = seg.end - seg.start
        if dur < MIN_SEGMENT_DURATION and prev_spk == next_spk and prev_spk != seg.speaker:
            log.debug(
                "Dropping micro-segment [%s→%s] %s (%.2f s) between %s",
                fmt(seg.start), fmt(seg.end), seg.speaker, dur, prev_spk,
            )
            cleaned[-1] = SpeakerSegment(
                speaker=cleaned[-1].speaker,
                start=cleaned[-1].start,
                end=merged[i + 1].start,
            )
        else:
            cleaned.append(seg)
    cleaned.append(merged[-1])
    return cleaned


def _correct_speaker_confusion(
    segments: List[SpeakerSegment],
    audio_path: str,
    hf_token: Optional[str],
    threshold: float = EMBEDDING_THRESHOLD,
) -> List[SpeakerSegment]:
    """
    Pass 3: Re-verify speaker identity via embedding cosine similarity.

    Extracts a mean embedding (centroid) per speaker ID over ALL their segments,
    computes pairwise cosine similarity, and merges IDs above `threshold`.
    Gracefully skips if the embedding model cannot be loaded.
    """
    try:
        import torch
        import torchaudio
        from pyannote.audio import Model, Inference
    except ImportError:
        log.warning("pyannote.audio not available — skipping speaker confusion correction.")
        return segments

    token = hf_token or os.environ.get("HF_TOKEN")
    unique_speakers = list(dict.fromkeys(s.speaker for s in segments))
    if len(unique_speakers) < 2:
        return segments

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        ensure_hf_model("pyannote/wespeaker-voxceleb-resnet34-LM", token=token)
        embedding_model = Model.from_pretrained(
            "pyannote/wespeaker-voxceleb-resnet34-LM", token=token,
        ).to(device)
        inference = Inference(embedding_model, window="whole")
        log.info("Speaker embedding model loaded.")
    except Exception as e:
        log.warning("Could not load embedding model (%s) — skipping Pass 3.", e)
        return segments

    try:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16_000:
            waveform = torchaudio.transforms.Resample(sr, 16_000)(waveform)
    except Exception as e:
        log.warning("Could not load audio for embedding extraction (%s) — skipping.", e)
        return segments

    total_frames = waveform.shape[1]
    speaker_embeddings: dict = {}

    for spk in unique_speakers:
        emb_list = []
        for seg in (s for s in segments if s.speaker == spk):
            if (seg.end - seg.start) < 0.5:
                continue
            start_f = int(seg.start * 16_000)
            end_f   = min(int(seg.end * 16_000), total_frames)
            chunk   = waveform[:, start_f:end_f]
            try:
                emb = inference({"waveform": chunk.unsqueeze(0), "sample_rate": 16_000})
                emb_list.append(torch.tensor(emb))
            except Exception:
                continue
        if emb_list:
            centroid = torch.stack(emb_list).mean(dim=0)
            centroid = centroid / (centroid.norm() + 1e-8)
            speaker_embeddings[spk] = centroid

    embeddable = list(speaker_embeddings.keys())
    if len(embeddable) < 2:
        return segments

    merge_map = {spk: spk for spk in unique_speakers}

    for i in range(len(embeddable)):
        for j in range(i + 1, len(embeddable)):
            a, b = embeddable[i], embeddable[j]
            sim = float(torch.dot(speaker_embeddings[a], speaker_embeddings[b]))
            log.info("Embedding similarity  %s ↔ %s : %.4f  (threshold %.2f)", a, b, sim, threshold)
            if sim >= threshold:
                canonical = merge_map[a]
                if merge_map[b] != canonical:
                    merge_map[b] = canonical
                    log.info("  → Merging %s into %s", b, canonical)

    def resolve(spk: str) -> str:
        visited: set = set()
        while merge_map[spk] != spk and spk not in visited:
            visited.add(spk)
            spk = merge_map[spk]
        return spk

    corrected = [
        SpeakerSegment(speaker=resolve(s.speaker), start=s.start,
                       end=s.end, overlapping=s.overlapping)
        for s in segments
    ]
    n_changed = sum(1 for o, c in zip(segments, corrected) if o.speaker != c.speaker)
    if n_changed:
        log.info("Speaker confusion correction: %d segment(s) re-labelled.", n_changed)
        corrected = _merge_short_segments(corrected)   # re-merge after relabelling
    else:
        log.info("Speaker confusion correction: no merges needed.")
    return corrected


def _compute_overlap_intervals(
    raw_segments: List[SpeakerSegment],
) -> List[Tuple[float, float]]:
    """Return merged list of (start, end) where ≥2 speakers are simultaneously active."""
    intervals: List[Tuple[float, float]] = []
    n = len(raw_segments)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = raw_segments[i], raw_segments[j]
            if a.speaker == b.speaker:
                continue
            ov_s, ov_e = max(a.start, b.start), min(a.end, b.end)
            if ov_s < ov_e:
                intervals.append((ov_s, ov_e))
    if not intervals:
        return []
    intervals.sort()
    merged: List[Tuple[float, float]] = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


# =============================================================================
# Step 3 — Alignment  (words → speaker segments)
# =============================================================================

def align(
    words: List[WordToken],
    diar_segments: List[SpeakerSegment],
) -> List[AlignedUtterance]:
    """
    Assign each WordToken to the speaker with the largest time intersection.
    Uses two-pointer O(N+M) traversal for efficiency.
    """
    if not words or not diar_segments:
        return []

    words = sorted(words, key=lambda w: w.start)
    diar_segments = sorted(diar_segments, key=lambda s: s.start)
    n_segs = len(diar_segments)

    assigned: List[Tuple[WordToken, List[str], bool]] = []
    seg_idx = 0

    for word in words:
        ws, we = word.start, word.end
        while seg_idx < n_segs - 1 and diar_segments[seg_idx].end <= ws:
            seg_idx += 1

        candidates: List[Tuple[str, float, bool]] = []
        for seg in diar_segments[seg_idx:]:
            if seg.start >= we:
                break
            intersection = min(we, seg.end) - max(ws, seg.start)
            if intersection > 0:
                candidates.append((seg.speaker, intersection, seg.overlapping))

        if not candidates:
            nearest = min(diar_segments,
                          key=lambda s: min(abs(s.start - ws), abs(s.end - we)))
            assigned.append((word, [nearest.speaker], nearest.overlapping))
            continue

        candidates.sort(key=lambda x: -x[1])
        is_overlap = len(candidates) > 1 and any(c[2] for c in candidates)
        speakers = (list(dict.fromkeys(c[0] for c in candidates))
                    if is_overlap else [candidates[0][0]])
        assigned.append((word, speakers, is_overlap))

    utterances: List[AlignedUtterance] = []
    if not assigned:
        return utterances

    cur_spk, cur_ovlp, cur_words = assigned[0][1], assigned[0][2], [assigned[0][0]]

    def flush(spk, buf, ovlp):
        text = "".join(w.word for w in buf).strip()
        if text:
            utterances.append(AlignedUtterance(
                speakers=spk, start=buf[0].start, end=buf[-1].end,
                text=text, overlapping=ovlp,
            ))

    for word, spk, ovlp in assigned[1:]:
        if spk == cur_spk and ovlp == cur_ovlp:
            cur_words.append(word)
        else:
            flush(cur_spk, cur_words, cur_ovlp)
            cur_spk, cur_ovlp, cur_words = spk, ovlp, [word]

    flush(cur_spk, cur_words, cur_ovlp)
    log.info("Alignment complete — %d utterances.", len(utterances))
    return utterances


# =============================================================================
# Step 4 — Format output
# =============================================================================

def format_output(utterances: List[AlignedUtterance]) -> str:
    lines = []
    for utt in utterances:
        time_range = f"[{fmt(utt.start)} - {fmt(utt.end)}]"
        label = (" & ".join(utt.speakers) + " [OVERLAP]") if utt.overlapping else utt.speakers[0]
        lines.append(f"{time_range} {label}: {utt.text}")
    return "\n".join(lines)


# =============================================================================
# Main orchestrator
# =============================================================================

def process(
    audio_path: str,
    # ── Language ─────────────────────────────────────────────────────────────
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    # ── Model ────────────────────────────────────────────────────────────────
    model_size: str = "large-v3",
    hf_token: Optional[str] = None,
    # ── Whisper ──────────────────────────────────────────────────────────────
    beam_size: int = BEAM_SIZE,
    temperature: float = TEMPERATURE,
    no_speech_threshold: float = NO_SPEECH_THRESHOLD,
    compression_ratio_threshold: float = COMPRESSION_RATIO,
    # ── VAD ──────────────────────────────────────────────────────────────────
    vad_threshold: float = VAD_THRESHOLD,
    vad_min_speech_ms: int = VAD_MIN_SPEECH_MS,
    vad_min_silence_ms: int = VAD_MIN_SILENCE_MS,
    vad_speech_pad_ms: int = VAD_SPEECH_PAD_MS,
    # ── Diarization ──────────────────────────────────────────────────────────
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    clustering_threshold: Optional[float] = CLUSTERING_THRESHOLD,
    embedding_threshold: float = EMBEDDING_THRESHOLD,
    # ── Output ───────────────────────────────────────────────────────────────
    output_path: Optional[str] = None,
) -> str:
    """
    End-to-end pipeline. Always writes two debug files alongside the audio:
      <stem>.stt_raw.txt   — raw Whisper segments (no speaker labels)
      <stem>.diar_raw.txt  — diarization segments after post-processing
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    stem    = _stem(audio_path)
    out_dir = Path(audio_path).parent
    stt_debug_path  = str(out_dir / f"{stem}.stt_raw.txt")
    diar_debug_path = str(out_dir / f"{stem}.diar_raw.txt")

    # 1. STT
    words = run_stt(
        audio_path,
        model_size=model_size,
        language=language,
        initial_prompt=initial_prompt,
        beam_size=beam_size,
        temperature=temperature,
        no_speech_threshold=no_speech_threshold,
        compression_ratio_threshold=compression_ratio_threshold,
        vad_threshold=vad_threshold,
        vad_min_speech_ms=vad_min_speech_ms,
        vad_min_silence_ms=vad_min_silence_ms,
        vad_speech_pad_ms=vad_speech_pad_ms,
        stt_output_path=stt_debug_path,
    )

    # 2. Diarization
    diar_segments = run_diarization(
        audio_path,
        hf_token=hf_token,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        clustering_threshold=clustering_threshold,
        embedding_threshold=embedding_threshold,
        diar_output_path=diar_debug_path,
    )

    # 3. Align
    utterances = align(words, diar_segments)

    # 4. Format & print
    transcript = format_output(utterances)
    print("\n" + "=" * 64)
    print("FINAL TRANSCRIPT")
    print("=" * 64)
    print(transcript)
    print("=" * 64)
    print(f"\nDebug files:")
    print(f"  STT (no speaker) → {stt_debug_path}")
    print(f"  Diarization only → {diar_debug_path}")

    # 5. Save
    if output_path:
        Path(output_path).write_text(transcript, encoding="utf-8")
        log.info("Transcript saved → %s", output_path)
        print(f"  Final transcript → {output_path}")

    return transcript


# =============================================================================
# CLI
# =============================================================================

def main():
    # global declared first — required by Python before any reference to these names
    global BATCHED_THRESHOLD_SECONDS
    global BEAM_SIZE, TEMPERATURE, NO_SPEECH_THRESHOLD, COMPRESSION_RATIO
    global VAD_THRESHOLD, VAD_MIN_SPEECH_MS, VAD_MIN_SILENCE_MS, VAD_SPEECH_PAD_MS
    global MAX_MERGE_GAP, MIN_SEGMENT_DURATION, EMBEDDING_THRESHOLD, CLUSTERING_THRESHOLD

    parser = argparse.ArgumentParser(
        description="Speech-to-Text + Speaker Diarization pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input / output ────────────────────────────────────────────────────────
    g = parser.add_argument_group("Input / Output")
    g.add_argument("--audio",     required=True, help="Path to audio file (any format)")
    g.add_argument("--output",    default=None,  help="Path to save final transcript")
    g.add_argument("--language",  default=None,  help="Language code e.g. 'ja', 'en'. Auto-detect if omitted.")
    g.add_argument("--prompt",    default=None,  help="Initial Whisper prompt (proper nouns, domain terms)")
    g.add_argument("--model",     default="large-v3", help="Whisper model size")
    g.add_argument("--hf-token",  default=None,  help="HuggingFace token (or set HF_TOKEN env var)")

    # ── Whisper decoding ──────────────────────────────────────────────────────
    g = parser.add_argument_group("Whisper decoding")
    g.add_argument("--beam-size",            type=int,   default=BEAM_SIZE,
                   help="Beam search width. Higher = more accurate but slower. (5–10)")
    g.add_argument("--temperature",          type=float, default=TEMPERATURE,
                   help="Decoding temperature. 0=greedy. Raise to 0.2–0.4 if hallucinations occur.")
    g.add_argument("--no-speech-threshold",  type=float, default=NO_SPEECH_THRESHOLD,
                   help="Silence probability threshold. Segments above this are skipped.")
    g.add_argument("--compression-ratio",    type=float, default=COMPRESSION_RATIO,
                   help="Gzip compression ratio threshold. Segments above this are discarded as garbage.")

    # ── VAD (Silero) ──────────────────────────────────────────────────────────
    g = parser.add_argument_group("VAD (Silero)")
    g.add_argument("--vad-threshold",      type=float, default=VAD_THRESHOLD,
                   help="Speech probability threshold (0–1). Lower = keep more borderline audio.")
    g.add_argument("--vad-min-speech-ms",  type=int,   default=VAD_MIN_SPEECH_MS,
                   help="Minimum speech chunk duration to keep (ms).")
    g.add_argument("--vad-min-silence-ms", type=int,   default=VAD_MIN_SILENCE_MS,
                   help="Minimum silence to split on (ms). Lower for Japanese short pauses (e.g. 300).")
    g.add_argument("--vad-speech-pad-ms",  type=int,   default=VAD_SPEECH_PAD_MS,
                   help="Padding added to both ends of each speech chunk (ms).")

    # ── Speaker diarization ───────────────────────────────────────────────────
    g = parser.add_argument_group("Speaker diarization")
    g.add_argument("--num-speakers",         type=int,   default=None,
                   help="Exact number of speakers. Best accuracy when known.")
    g.add_argument("--min-speakers",         type=int,   default=None,
                   help="Minimum expected speakers.")
    g.add_argument("--max-speakers",         type=int,   default=None,
                   help="Maximum expected speakers.")
    g.add_argument("--clustering-threshold", type=float, default=CLUSTERING_THRESHOLD,
                   help=(
                       "pyannote clustering threshold. "
                       "Lower (~0.5) = more speakers detected. "
                       "Higher (~0.8) = fewer speakers. "
                       "Default: pyannote built-in default."
                   ))

    # ── Post-processing ───────────────────────────────────────────────────────
    g = parser.add_argument_group("Diarization post-processing")
    g.add_argument("--merge-gap",            type=float, default=MAX_MERGE_GAP,
                   help="Max silence gap (s) between same-speaker turns to merge.")
    g.add_argument("--min-segment",          type=float, default=MIN_SEGMENT_DURATION,
                   help="Min segment duration (s). Shorter isolated segments are dropped.")
    g.add_argument("--embedding-threshold",  type=float, default=EMBEDDING_THRESHOLD,
                   help=(
                       "Cosine similarity to merge confused speaker IDs. "
                       "0.90=conservative  0.85=recommended  0.75=aggressive  1.0=disable."
                   ))

    # ── Pipeline ──────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Pipeline")
    g.add_argument("--batched-threshold", type=int, default=BATCHED_THRESHOLD_SECONDS,
                   help="Audio duration (s) above which BatchedInferencePipeline is used (GPU only).")

    args = parser.parse_args()

    # Apply CLI overrides to module-level constants
    BATCHED_THRESHOLD_SECONDS = args.batched_threshold
    BEAM_SIZE                 = args.beam_size
    TEMPERATURE               = args.temperature
    NO_SPEECH_THRESHOLD       = args.no_speech_threshold
    COMPRESSION_RATIO         = args.compression_ratio
    VAD_THRESHOLD             = args.vad_threshold
    VAD_MIN_SPEECH_MS         = args.vad_min_speech_ms
    VAD_MIN_SILENCE_MS        = args.vad_min_silence_ms
    VAD_SPEECH_PAD_MS         = args.vad_speech_pad_ms
    MAX_MERGE_GAP             = args.merge_gap
    MIN_SEGMENT_DURATION      = args.min_segment
    EMBEDDING_THRESHOLD       = args.embedding_threshold
    CLUSTERING_THRESHOLD      = args.clustering_threshold

    process(
        audio_path=args.audio,
        language=args.language,
        initial_prompt=args.prompt,
        model_size=args.model,
        hf_token=args.hf_token,
        beam_size=args.beam_size,
        temperature=args.temperature,
        no_speech_threshold=args.no_speech_threshold,
        compression_ratio_threshold=args.compression_ratio,
        vad_threshold=args.vad_threshold,
        vad_min_speech_ms=args.vad_min_speech_ms,
        vad_min_silence_ms=args.vad_min_silence_ms,
        vad_speech_pad_ms=args.vad_speech_pad_ms,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        clustering_threshold=args.clustering_threshold,
        embedding_threshold=args.embedding_threshold,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
