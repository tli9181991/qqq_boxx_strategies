"""Upload through rclone.

Why this exists alongside `drive.py`
------------------------------------
`drive.py` talks to the Drive API with an OAuth client you register yourself.
That is the tidy answer right up until you try to publish the consent screen:
Google requires a homepage and a privacy policy URL on a domain you control
before an External app leaves "Testing", and an app left in Testing gets
refresh tokens that expire after seven days. For a personal pipeline on a home
box, that is a recurring chore in exchange for nothing.

rclone ships with its own OAuth client, already registered and verified. You
authorise *rclone* -- an app Google already knows -- rather than standing up a
Cloud project of your own. No consent screen, no verification, no seven-day
expiry, and none of the `drive.file` scope's blind spots: rclone can see
folders you made by hand in the Drive web UI, which the API client could not.

The cost is one more binary on the box, and that rclone's shared client ID is
rate-limited across everyone using it. At a few uploads a day that is
irrelevant; if you ever push hundreds, `rclone config` can be given a client
ID of your own.

What we get for free
--------------------
`rclone copy` is already idempotent -- it compares size and modification time
and skips a file it has seen -- so the "did this upload already succeed before
the crash" dance in `drive.py` is simply not needed here. It also creates the
destination path on demand, so there is no find-or-create-folder step.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from typing import Any, Dict, List, Optional

from .config import PipelineConfig

log = logging.getLogger(__name__)


class RcloneError(RuntimeError):
    pass


def _run(cmd: List[str], *, timeout: int) -> subprocess.CompletedProcess:
    log.debug("rclone: %s", " ".join(cmd))
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise RcloneError(
            f"{cmd[0]} not found on PATH. Install it with "
            "`winget install Rclone.Rclone` on Windows, or your package "
            "manager on Linux.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RcloneError(f"rclone timed out after {timeout}s") from exc


def check(cfg: PipelineConfig) -> None:
    """Verify the binary is present and the remote is configured.

    Called once at worker startup rather than per job: a missing remote is a
    setup mistake, and failing every job individually would bury that in
    retries instead of stating it once.
    """
    proc = _run([cfg.rclone_binary, "listremotes"], timeout=60)
    if proc.returncode != 0:
        raise RcloneError(
            f"`rclone listremotes` failed: {(proc.stderr or '').strip()}")
    remotes = [r.strip() for r in (proc.stdout or "").splitlines() if r.strip()]
    wanted = f"{cfg.rclone_remote}:"
    if wanted not in remotes:
        raise RcloneError(
            f"rclone has no remote named {cfg.rclone_remote!r}. "
            f"Configured remotes: {remotes or 'none'}. "
            f"Run `rclone config`, choose 'n' for a new remote, name it "
            f"{cfg.rclone_remote!r}, and pick the Google Drive backend.")
    log.info("rclone remote %s is configured", wanted)


def destination(cfg: PipelineConfig, creator: Optional[str]) -> str:
    """Build the remote path a job's files belong in.

    rclone creates whatever part of this does not exist, so unlike the API
    client there is nothing to look up or create first.
    """
    parts = [cfg.rclone_base_path.strip("/")] if cfg.rclone_base_path else []
    if cfg.drive_subfolder_per_creator and creator:
        # Forward slashes separate remote path segments, so a creator name
        # containing one would silently nest an extra folder.
        safe = re.sub(r"[/\\]", "-", creator).strip() or "unknown"
        parts.append(safe)
    return f"{cfg.rclone_remote}:" + "/".join(parts)


def _file_id(cfg: PipelineConfig, dest: str, name: str) -> Optional[str]:
    """Ask rclone for the Drive file id, so `status` can link to the file.

    Best-effort: a missing id costs a convenience link and nothing else, so
    every failure here is swallowed. Note this is `lsjson`, not `rclone link`
    -- the latter would create a public sharing link, which is not something
    to do to someone's files as a side effect of recording a URL.
    """
    try:
        proc = _run([cfg.rclone_binary, "lsjson", dest, "--files-only"],
                    timeout=120)
        if proc.returncode != 0:
            return None
        for entry in json.loads(proc.stdout or "[]"):
            if entry.get("Name") == name:
                return entry.get("ID")
    except (RcloneError, ValueError, TypeError) as exc:
        log.debug("could not read file id for %s: %s", name, exc)
    return None


def upload(cfg: PipelineConfig, path: str, dest: str,
           description: Optional[str] = None) -> Dict[str, Any]:
    """Copy one file to `dest`. Returns metadata shaped like the Drive client's.

    `description` is accepted and ignored: rclone has no way to set Drive's
    description field. The same information is in the uploaded `.info.json`,
    which is why that upload is on by default.
    """
    if not os.path.isfile(path):
        raise RcloneError(f"not a file: {path}")
    name = os.path.basename(path)
    size = os.path.getsize(path)
    log.info("uploading %s (%.1f MB) to %s", name, size / 1e6, dest)

    cmd = [
        cfg.rclone_binary, "copy", path, dest,
        # One transfer at a time: the worker is already serial, and letting
        # rclone parallelise here would undo that.
        "--transfers", "1",
        "--retries", "3",
        "--low-level-retries", "10",
        "--drive-chunk-size", f"{max(1, cfg.drive_chunk_mb)}M",
        "--stats-one-line", "--stats", "30s",
    ]
    proc = _run(cmd, timeout=cfg.rclone_timeout)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-8:])
        raise RcloneError(f"rclone copy failed: {tail or 'no stderr'}")

    file_id = _file_id(cfg, dest, name)
    meta: Dict[str, Any] = {"name": name, "path": f"{dest}/{name}"}
    if file_id:
        meta["id"] = file_id
        meta["webViewLink"] = f"https://drive.google.com/file/d/{file_id}/view"
    log.info("uploaded %s", name)
    return meta
