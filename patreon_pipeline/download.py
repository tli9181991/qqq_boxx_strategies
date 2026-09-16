"""The yt-dlp wrapper.

yt-dlp is invoked as a subprocess rather than imported. Two reasons: the
command that gets run is the same one you already tested by hand, so a failure
can be reproduced by copy-pasting the line out of the log; and yt-dlp's Python
API is explicitly not stable between releases, while its CLI is.

Output discovery
----------------
Each job downloads into its own empty staging directory and we then look at
what appeared. That is version-proof, where parsing `--print filepath` is not:
the printing options have moved around across releases, and merged formats make
the "final" filename differ from anything announced mid-run.

Failure classification
----------------------
`classify_error` exists because the two failure modes need opposite responses.
A post that is genuinely broken should burn its retries and land in `failed`.
An expired cookie fails *every* post identically -- retrying just empties the
whole queue into `failed` while the actual fix (log in again) goes undone. So
auth failures halt the worker instead, and the queue waits.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import PipelineConfig

log = logging.getLogger(__name__)

# Extensions yt-dlp leaves behind that are not the media itself.
SIDECAR_EXT = {".json", ".jpg", ".jpeg", ".png", ".webp", ".vtt", ".srt",
               ".ass", ".lrc", ".part", ".ytdl", ".temp"}


class DownloadError(RuntimeError):
    """A post failed to download. Retryable unless it is an AuthError."""


class AuthError(DownloadError):
    """The Patreon session is not valid: log in again.

    Never counted against a job's attempts -- it is a property of the box, not
    of the post.
    """


class NoMediaError(DownloadError):
    """The post exists but has nothing downloadable (text-only, or images)."""


# Matched case-insensitively against yt-dlp's stderr. Ordered most specific
# first; the first hit wins.
_AUTH_PATTERNS = (
    r"this post is for patrons only",
    r"patrons only",
    r"http error 401",
    r"http error 403",
    r"unable to (?:download|extract).*(?:requires|log[ -]?in)",
    r"sign in to confirm",
    r"login required",
    r"cookies are no longer valid",
    r"failed to (?:open|read|decrypt).*cookie",
    r"could not (?:copy|find).*cookie",
)

_NO_MEDIA_PATTERNS = (
    r"no video formats found",
    r"there's no video",
    r"unsupported url",
    r"no media found",
    r"does not have a video",
)


def classify_error(stderr: str) -> type:
    """Map yt-dlp stderr onto an exception class."""
    low = (stderr or "").lower()
    for pat in _AUTH_PATTERNS:
        if re.search(pat, low):
            return AuthError
    for pat in _NO_MEDIA_PATTERNS:
        if re.search(pat, low):
            return NoMediaError
    return DownloadError


@dataclass
class DownloadResult:
    media_path: str
    info_path: Optional[str] = None
    size_bytes: int = 0
    title: Optional[str] = None
    creator: Optional[str] = None
    upload_date: Optional[str] = None
    post_id: Optional[str] = None
    extra_paths: List[str] = field(default_factory=list)


def _cookie_args(cfg: PipelineConfig) -> List[str]:
    if cfg.cookies_file:
        return ["--cookies", cfg.cookies_file]
    if cfg.cookies_from_browser:
        return ["--cookies-from-browser", cfg.cookies_from_browser]
    return []


def build_command(cfg: PipelineConfig, url: str, dest_dir: str) -> List[str]:
    """The exact argv used for a download. Pure, so it can be asserted on."""
    cmd = [cfg.ytdlp_binary]
    cmd += _cookie_args(cfg)
    cmd += [
        "--no-playlist",
        "--no-progress",
        "--newline",
        # Retries inside yt-dlp handle a blip mid-transfer; the queue's retries
        # handle everything bigger. Both are wanted.
        "--retries", "5",
        "--fragment-retries", "10",
        # yt-dlp resumes a .part file, so a killed job costs minutes, not the
        # whole download.
        "--continue",
        "--write-info-json",
        "--sleep-requests", str(cfg.sleep_requests),
        "--sleep-interval", str(cfg.sleep_interval),
        "--max-sleep-interval", str(cfg.max_sleep_interval),
        "--paths", dest_dir,
        "-o", "%(title).180B [%(id)s].%(ext)s",
    ]
    if cfg.audio_only:
        cmd += ["-f", "bestaudio/best", "-x",
                "--audio-format", cfg.audio_format,
                "--audio-quality", cfg.audio_quality]
    elif cfg.ytdlp_format:
        cmd += ["-f", cfg.ytdlp_format]
    cmd.append(url)
    return cmd


def _pick_media(dest_dir: str) -> Optional[str]:
    """The largest non-sidecar file in the staging directory.

    Size is the discriminator because a merged download can leave the original
    audio and video streams behind on some failures; the merged result is
    always the biggest thing there.
    """
    best: Optional[str] = None
    best_size = -1
    for path in glob.glob(os.path.join(dest_dir, "*")):
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext in SIDECAR_EXT or path.endswith(".info.json"):
            continue
        size = os.path.getsize(path)
        if size > best_size:
            best, best_size = path, size
    return best


def _read_info(dest_dir: str) -> Dict:
    matches = sorted(glob.glob(os.path.join(dest_dir, "*.info.json")))
    if not matches:
        return {}
    try:
        with open(matches[0], "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        log.warning("could not read %s: %s", matches[0], exc)
        return {}


def download(cfg: PipelineConfig, url: str, dest_dir: str) -> DownloadResult:
    """Fetch one post into `dest_dir`. Raises a DownloadError subclass on failure.

    `dest_dir` is emptied first. A leftover from a previous attempt would
    confuse the "largest file wins" rule, and a half-written file is worse than
    no file.
    """
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir, ignore_errors=True)
    os.makedirs(dest_dir, exist_ok=True)

    cmd = build_command(cfg, url, dest_dir)
    log.info("yt-dlp: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=cfg.download_timeout)
    except FileNotFoundError as exc:
        raise DownloadError(f"{cfg.ytdlp_binary} not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise DownloadError(
            f"timed out after {cfg.download_timeout}s") from exc

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        tail = "\n".join(stderr.splitlines()[-8:])
        raise classify_error(stderr)(
            f"yt-dlp exited {proc.returncode}: {tail or 'no stderr'}")

    media = _pick_media(dest_dir)
    if media is None:
        raise NoMediaError("yt-dlp reported success but produced no media file")

    info = _read_info(dest_dir)
    info_matches = sorted(glob.glob(os.path.join(dest_dir, "*.info.json")))
    return DownloadResult(
        media_path=media,
        info_path=info_matches[0] if info_matches else None,
        size_bytes=os.path.getsize(media),
        title=info.get("title"),
        creator=info.get("uploader") or info.get("channel") or info.get("creator"),
        upload_date=info.get("upload_date"),
        post_id=str(info.get("id")) if info.get("id") is not None else None,
    )


def probe_session(cfg: PipelineConfig, url: str) -> Optional[str]:
    """Check the Patreon session without downloading anything.

    Returns None when the session works, or a human-readable reason when it
    does not. This is the cheap version of the Selenium session-keeper: it asks
    yt-dlp to resolve one patron-only post's metadata and reports whether the
    cookies still carry the entitlement.

    Run it from a timer and you learn about an expired login the day it
    happens, instead of the day you go looking for a post that never arrived.
    """
    cmd = [cfg.ytdlp_binary] + _cookie_args(cfg) + [
        "--simulate", "--no-warnings", "--print", "%(id)s", url]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        return f"{cfg.ytdlp_binary} not found on PATH"
    except subprocess.TimeoutExpired:
        return "yt-dlp timed out while probing the session"
    if proc.returncode == 0:
        return None
    stderr = (proc.stderr or "").strip()
    kind = classify_error(stderr)
    tail = "\n".join(stderr.splitlines()[-4:])
    if kind is AuthError:
        return f"Patreon session is not valid -- log in again. {tail}"
    return f"probe failed ({kind.__name__}): {tail}"


def list_campaign_posts(cfg: PipelineConfig, campaign_url: str,
                        *, limit: int = 0) -> List[str]:
    """List post URLs for a campaign without downloading anything.

    This is the backstop sweep. Email triggering is the fast path, but it has
    failure modes the sweep does not -- a filter edited by accident, a change
    to Patreon's notification format, a week of downtime. Running this from a
    timer means the worst case for a missed notification is a few hours of
    latency rather than a post lost forever.
    """
    cmd = [cfg.ytdlp_binary] + _cookie_args(cfg) + [
        "--flat-playlist", "--no-warnings", "--ignore-errors",
        "--print", "%(webpage_url)s",
    ]
    if limit > 0:
        cmd += ["--playlist-items", f"1-{int(limit)}"]
    cmd.append(campaign_url)
    log.info("sweeping campaign: %s", campaign_url)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except FileNotFoundError as exc:
        raise DownloadError(f"{cfg.ytdlp_binary} not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise DownloadError("campaign sweep timed out") from exc

    # --ignore-errors means a partial listing still exits non-zero when one
    # entry is unavailable. Trust whatever came out of stdout, and only raise
    # when nothing did.
    urls = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if not urls and proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        tail = "\n".join(stderr.splitlines()[-6:])
        raise classify_error(stderr)(f"campaign sweep failed: {tail}")
    return urls
