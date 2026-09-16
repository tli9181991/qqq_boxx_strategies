"""Transcribe downloaded media with faster-whisper.

Where this sits
---------------
Between the download and the upload, and it is *additive*: the media still goes
to Drive unless you turn that off. You said you might hand Gemini either the
video or the transcript, so the pipeline produces both and lets you decide per
post. A transcript is ~50 KB against a gigabyte of video, so keeping both costs
essentially nothing.

Failure here is not fatal
-------------------------
`worker.py` catches everything this module raises and uploads the media anyway.
That is deliberate. Transcription is the slowest, most memory-hungry and most
optional stage in the pipeline; a box that OOMs on a three-hour post should
still end up with the video in Drive. Losing a download because the *nice to
have* failed would be the wrong trade.

Why ffmpeg first
----------------
faster-whisper can decode an mp4 itself through PyAV, but pulling a 2 GB
container through it to get at the audio is slow and occasionally fails on odd
codecs. ffmpeg is already a hard requirement of yt-dlp's format merging, so it
is guaranteed present, and 16 kHz mono PCM is exactly what the model wants --
no resampling inside the inference loop. The intermediate wav is deleted
afterwards.

Model loading is cached
-----------------------
`large-v3-turbo` takes tens of seconds to load and roughly a gigabyte of RAM.
Loading it per job would dominate the runtime of a short post, so the worker
process keeps one instance alive across jobs.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import PipelineConfig

log = logging.getLogger(__name__)

# Containers we can hand to ffmpeg directly. Anything else still works -- this
# only decides whether we bother checking.
_AUDIO_EXT = {".mp3", ".m4a", ".aac", ".opus", ".ogg", ".wav", ".flac"}

_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}


class TranscribeError(RuntimeError):
    """Transcription failed. Never fatal to a job -- see worker.py."""


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    txt_path: str
    srt_path: Optional[str] = None
    language: Optional[str] = None
    duration: float = 0.0
    segments: int = 0
    paths: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# Formatting (pure -- this is the part worth testing)
# ----------------------------------------------------------------------

def format_timestamp(seconds: float, *, srt: bool = False) -> str:
    """Seconds -> HH:MM:SS (or HH:MM:SS,mmm for SRT)."""
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    if srt:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def to_srt(segments: List[Segment]) -> str:
    """Standard SRT. Useful as subtitles, and Gemini reads it as timed text."""
    out: List[str] = []
    for i, seg in enumerate(segments, start=1):
        out.append(str(i))
        out.append(f"{format_timestamp(seg.start, srt=True)} --> "
                   f"{format_timestamp(seg.end, srt=True)}")
        out.append(seg.text.strip())
        out.append("")
    return "\n".join(out)


def to_paragraphs(segments: List[Segment],
                  *,
                  timestamps: bool = True,
                  gap_seconds: float = 2.0,
                  min_chars: int = 400) -> str:
    """Readable prose, broken into paragraphs at natural pauses.

    Whisper emits a segment every few seconds, and a file of 3,000 one-line
    segments is painful to read and wasteful to feed to a model. Breaking on a
    real pause -- but only once the paragraph has some bulk -- gives something
    closer to a transcript a person would write.

    The `[HH:MM:SS]` prefix is what lets you (or Gemini) point back at the
    moment in the video a claim came from. That is most of the value of having
    the transcript next to the media rather than instead of it.
    """
    if not segments:
        return ""
    paras: List[str] = []
    buf: List[str] = []
    buf_start = segments[0].start
    prev_end = segments[0].start

    def flush() -> None:
        if not buf:
            return
        body = " ".join(t.strip() for t in buf if t.strip())
        body = re.sub(r"\s+", " ", body).strip()
        if not body:
            return
        paras.append(f"[{format_timestamp(buf_start)}] {body}" if timestamps
                     else body)

    for seg in segments:
        gap = seg.start - prev_end
        if buf and gap >= gap_seconds and len(" ".join(buf)) >= min_chars:
            flush()
            buf = []
            buf_start = seg.start
        buf.append(seg.text)
        prev_end = seg.end
    flush()
    return "\n\n".join(paras) + "\n"


def build_initial_prompt(vocabulary: List[str]) -> Optional[str]:
    """Turn a vocabulary list into Whisper's `initial_prompt`.

    Whisper conditions on this text as though it were the transcript of the
    audio immediately before the file. Feeding it the jargon you expect is the
    single highest-leverage knob for domain audio: without it, tickers and
    options terms come back mangled -- "QQQ" as "Q3", "0DTE" as "zero DT" --
    and no amount of a bigger model fixes it the way this does.
    """
    terms = [t.strip() for t in vocabulary if t and t.strip()]
    if not terms:
        return None
    return ", ".join(terms) + "."


# ----------------------------------------------------------------------
# Audio extraction
# ----------------------------------------------------------------------

def extract_audio(media_path: str, dest_wav: str,
                  *, ffmpeg: str = "ffmpeg", timeout: int = 3600) -> str:
    """Decode any media file to 16 kHz mono PCM, which is what Whisper wants."""
    cmd = [ffmpeg, "-nostdin", "-y", "-i", media_path,
           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
           "-loglevel", "error", dest_wav]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise TranscribeError(
            f"{ffmpeg} not found on PATH (yt-dlp needs it too)") from exc
    except subprocess.TimeoutExpired as exc:
        raise TranscribeError("ffmpeg timed out extracting audio") from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-5:])
        raise TranscribeError(f"ffmpeg failed: {tail}")
    if not os.path.exists(dest_wav) or os.path.getsize(dest_wav) == 0:
        raise TranscribeError("ffmpeg produced no audio (is the post silent?)")
    return dest_wav


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

def load_model(cfg: PipelineConfig):
    """Load (and cache) the faster-whisper model.

    `download_root` is pinned into the state directory rather than the default
    `~/.cache/huggingface`, because the systemd units run with ProtectHome and
    a first run would otherwise fail trying to create a cache in a home
    directory it cannot write.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscribeError(
            "faster-whisper is not installed: pip install -r "
            "requirements-patreon.txt") from exc

    key = (cfg.whisper_model, cfg.whisper_device, cfg.whisper_compute_type)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    root = cfg.resolved_whisper_root()
    os.makedirs(root, exist_ok=True)
    log.info("loading whisper model %s (%s/%s); first run downloads it to %s",
             cfg.whisper_model, cfg.whisper_device, cfg.whisper_compute_type, root)
    try:
        model = WhisperModel(cfg.whisper_model,
                             device=cfg.whisper_device,
                             compute_type=cfg.whisper_compute_type,
                             download_root=root,
                             cpu_threads=cfg.whisper_cpu_threads or 0)
    except Exception as exc:
        raise TranscribeError(
            f"could not load whisper model {cfg.whisper_model!r}: {exc}") from exc
    _MODEL_CACHE[key] = model
    return model


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def transcribe(cfg: PipelineConfig, media_path: str,
               out_dir: Optional[str] = None,
               *, base_name: Optional[str] = None) -> TranscriptResult:
    """Transcribe one media file. Writes .txt (and .srt) next to it.

    Raises TranscribeError on any failure. Callers in the pipeline treat that
    as "upload the media without a transcript", never as a lost job.
    """
    if not os.path.isfile(media_path):
        raise TranscribeError(f"not a file: {media_path}")
    out_dir = out_dir or os.path.dirname(os.path.abspath(media_path))
    stem = base_name or os.path.splitext(os.path.basename(media_path))[0]

    model = load_model(cfg)

    tmp_wav: Optional[str] = None
    ext = os.path.splitext(media_path)[1].lower()
    try:
        if ext in _AUDIO_EXT and cfg.whisper_skip_extract_for_audio:
            audio = media_path
        else:
            fd, tmp_wav = tempfile.mkstemp(suffix=".wav", dir=out_dir)
            os.close(fd)
            log.info("extracting audio from %s", os.path.basename(media_path))
            audio = extract_audio(media_path, tmp_wav,
                                  timeout=cfg.transcribe_timeout)

        prompt = build_initial_prompt(cfg.whisper_vocabulary)
        log.info("transcribing with %s (vad=%s, lang=%s)", cfg.whisper_model,
                 cfg.whisper_vad, cfg.whisper_language or "auto")
        try:
            raw_segments, info = model.transcribe(
                audio,
                language=cfg.whisper_language or None,
                beam_size=cfg.whisper_beam_size,
                vad_filter=cfg.whisper_vad,
                initial_prompt=prompt,
                condition_on_previous_text=False,
            )
        except Exception as exc:
            raise TranscribeError(f"whisper failed: {exc}") from exc

        # faster-whisper yields lazily; inference happens during this loop.
        segments = [Segment(float(s.start), float(s.end), s.text)
                    for s in raw_segments]
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)

    if not segments:
        raise TranscribeError("no speech detected")

    duration = float(getattr(info, "duration", 0.0) or segments[-1].end)
    language = getattr(info, "language", None)

    written: List[str] = []
    txt_path = os.path.join(out_dir, f"{stem}.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(to_paragraphs(segments, timestamps=cfg.whisper_txt_timestamps))
    written.append(txt_path)

    srt_path: Optional[str] = None
    if "srt" in cfg.whisper_formats:
        srt_path = os.path.join(out_dir, f"{stem}.srt")
        with open(srt_path, "w", encoding="utf-8") as fh:
            fh.write(to_srt(segments))
        written.append(srt_path)

    log.info("transcribed %s: %d segments, %.1f min of audio, language=%s",
             os.path.basename(media_path), len(segments), duration / 60.0,
             language)
    return TranscriptResult(txt_path=txt_path, srt_path=srt_path,
                            language=language, duration=duration,
                            segments=len(segments), paths=written)
