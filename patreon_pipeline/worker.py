"""Drain the queue: download each post, upload it, record where it went.

One job at a time, on purpose. Patreon and Vimeo both notice concurrency, and
the whole point of the queue is that a burst of notifications becomes a
sequence of polite requests rather than a thundering herd.

The auth stop
-------------
An expired Patreon session fails every job in exactly the same way. Treating
that as a per-job failure would march the entire queue into `failed` in a few
minutes, and the queue would then be empty of work *and* wrong -- the posts
were fine, the login was not.

So `AuthError` does something different: the job goes back on the queue without
spending an attempt, a flag is set, and the worker exits. systemd restarts it
on a long timer, so once you log in again it picks up by itself with nothing
lost. That is why `EXIT_AUTH` is its own code.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import time
from typing import Any, Dict, Optional

from . import download, store, transcribe as transcribe_mod, upload
from .config import PipelineConfig

log = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_AUTH = 3

AUTH_FLAG = "auth_alert"


def _describe(job: sqlite3.Row, res: download.DownloadResult,
              tr: Optional["transcribe_mod.TranscriptResult"] = None) -> str:
    """The Drive file description.

    Worth populating: it is what tells you which post a file came from months
    later, and it gives Gemini the surrounding context -- creator, date, source
    URL -- without needing the .info.json opened alongside.
    """
    bits = [f"Patreon post: {job['post_url']}"]
    if res.title:
        bits.append(f"Title: {res.title}")
    if res.creator:
        bits.append(f"Creator: {res.creator}")
    if res.upload_date:
        bits.append(f"Posted: {res.upload_date}")
    if job["title"]:
        bits.append(f"Email subject: {job['title']}")
    if tr:
        bits.append(f"Duration: {tr.duration / 60.0:.1f} min")
        if tr.language:
            bits.append(f"Language: {tr.language}")
        bits.append(f"Transcript: {os.path.basename(tr.txt_path)}")
    return "\n".join(bits)


def process_job(cfg: PipelineConfig, conn: sqlite3.Connection,
                job: sqlite3.Row, uploader: "upload._Backend") -> str:
    """Run one job end to end. Returns its resulting state.

    Raises AuthError upward -- that is the one failure the caller must handle
    differently from every other.
    """
    job_id = int(job["id"])
    url = job["post_url"]
    staging = os.path.join(cfg.resolved_staging_dir(), str(job_id))
    log.info("job %s: %s", job_id, url)

    try:
        res = download.download(cfg, url, staging)
    except download.AuthError:
        raise
    except download.NoMediaError as exc:
        log.warning("job %s: nothing to download (%s)", job_id, exc)
        shutil.rmtree(staging, ignore_errors=True)
        store.mark_skipped(conn, job_id, str(exc))
        return "skipped"
    except download.DownloadError as exc:
        log.error("job %s: download failed: %s", job_id, exc)
        return store.mark_retry(conn, job_id, str(exc),
                                max_attempts=cfg.max_attempts,
                                backoff_base=cfg.backoff_base,
                                backoff_cap=cfg.backoff_cap)

    store._update(conn, job_id, title=res.title or job["title"],
                  creator=res.creator, upload_date=res.upload_date,
                  local_path=res.media_path, size_bytes=res.size_bytes,
                  post_id=res.post_id or job["post_id"])

    # Transcription is best-effort by design. It is the slowest and most
    # memory-hungry stage here, and it is the optional one: a box that runs out
    # of RAM on a three-hour post should still end up with the video in Drive.
    # So every failure is logged and swallowed, and the job carries on.
    tr = None
    if cfg.transcribe:
        try:
            tr = transcribe_mod.transcribe(cfg, res.media_path)
            store._update(conn, job_id, transcript_path=tr.txt_path,
                          language=tr.language, duration_sec=tr.duration)
        except transcribe_mod.TranscribeError as exc:
            log.warning("job %s: transcription failed, continuing without it: %s",
                        job_id, exc)
        except Exception as exc:
            log.warning("job %s: transcription failed (%s), continuing: %s",
                        job_id, type(exc).__name__, exc)

    try:
        dest = uploader.destination(res.creator)
        meta: Dict[str, Any] = {}
        transcript_meta: Dict[str, Any] = {}

        if cfg.upload_media:
            meta = uploader.send(res.media_path, dest,
                                 description=_describe(job, res, tr))
        if tr and cfg.upload_transcript:
            for path in tr.paths:
                got = uploader.send(path, dest,
                                    description=_describe(job, res, tr))
                if path == tr.txt_path:
                    transcript_meta = got
        if cfg.upload_info_json and res.info_path:
            uploader.send(res.info_path, dest)

        # With upload_media off, the transcript is the artefact, so it is what
        # `status` should link to.
        primary = meta or transcript_meta
    except Exception as exc:
        # Drive failures are overwhelmingly transient -- quota, a 5xx, the
        # connection dropping. Retry rather than fail, and keep the staged
        # file so the retry does not re-download it.
        log.error("job %s: upload failed: %s: %s", job_id, type(exc).__name__, exc)
        return store.mark_retry(conn, job_id, f"upload: {exc}",
                                max_attempts=cfg.max_attempts,
                                backoff_base=cfg.backoff_base,
                                backoff_cap=cfg.backoff_cap)

    store.mark_done(conn, job_id,
                    drive_file_id=primary.get("id"),
                    drive_link=primary.get("webViewLink"),
                    transcript_link=transcript_meta.get("webViewLink"))
    if cfg.delete_after_upload:
        shutil.rmtree(staging, ignore_errors=True)
        store._update(conn, job_id, local_path=None, transcript_path=None)
    log.info("job %s: done -> %s", job_id,
             primary.get("webViewLink") or primary.get("id"))
    return "done"


def run(cfg: PipelineConfig, *, once: bool = False,
        max_jobs: int = 0) -> int:
    """Drain the queue until it is empty (`once`) or forever.

    Returns a process exit code.
    """
    db_path = cfg.resolved_db_path()
    os.makedirs(cfg.resolved_staging_dir(), exist_ok=True)

    # Checked once here rather than per job: a missing rclone remote or an
    # unusable Drive token is a setup mistake, and failing every job in turn
    # would bury that under retries instead of stating it once.
    try:
        uploader = upload.backend(cfg)
        uploader.check()
    except upload.UploadError as exc:
        log.error("uploads are not usable (%s backend): %s", cfg.uploader, exc)
        return EXIT_CONFIG

    processed = 0
    with store.connect(db_path) as conn:
        orphaned = store.reset_running(conn)
        if orphaned:
            log.warning("returned %d interrupted job(s) to the queue", orphaned)
        store.set_flag(conn, AUTH_FLAG, None)

        while True:
            job = store.claim_next(conn)
            if job is None:
                if once:
                    log.info("queue empty")
                    return EXIT_OK
                time.sleep(cfg.poll_seconds)
                continue

            try:
                process_job(cfg, conn, job, uploader)
            except download.AuthError as exc:
                store.requeue(conn, int(job["id"]))
                store.set_flag(conn, AUTH_FLAG, str(exc))
                log.error(
                    "PATREON SESSION EXPIRED -- %s\n"
                    "  Log in again in the browser profile this box reads "
                    "cookies from, then the worker will resume by itself.\n"
                    "  Job %s was put back on the queue; no attempt was spent.",
                    exc, job["id"])
                return EXIT_AUTH
            except KeyboardInterrupt:
                store.requeue(conn, int(job["id"]))
                log.info("interrupted; job %s returned to the queue", job["id"])
                return EXIT_OK
            except Exception as exc:
                # Anything unclassified is still the job's problem, not a
                # reason to take the worker down.
                log.exception("job %s: unexpected failure", job["id"])
                store.mark_retry(conn, int(job["id"]),
                                 f"{type(exc).__name__}: {exc}",
                                 max_attempts=cfg.max_attempts,
                                 backoff_base=cfg.backoff_base,
                                 backoff_cap=cfg.backoff_cap)

            processed += 1
            if max_jobs and processed >= max_jobs:
                log.info("stopping after %d job(s)", processed)
                return EXIT_OK


def sweep(cfg: PipelineConfig, *, limit: int = 20) -> int:
    """Enqueue anything on the configured campaigns that is not already known."""
    if not cfg.campaign_urls:
        # A hard error, not a warning. A sweep that swept nothing did not do
        # what it was asked, and reporting success for it hides the real
        # problem -- which is almost always that the config file is not being
        # read at all, so every other setting is silently a default too.
        source = os.environ.get("PATREON_CONFIG") or "(no PATREON_CONFIG set)"
        log.error(
            "no campaign_urls configured -- nothing to sweep.\n"
            "  config file read: %s\n"
            "  Set campaign_urls there, or PATREON_CAMPAIGN_URLS in the "
            "environment (which overrides the file).\n"
            "  `runner config` prints what actually got loaded.",
            source)
        return EXIT_CONFIG
    from . import mail

    created = 0
    with store.connect(cfg.resolved_db_path()) as conn:
        for campaign in cfg.campaign_urls:
            try:
                urls = download.list_campaign_posts(cfg, campaign, limit=limit)
            except download.AuthError as exc:
                store.set_flag(conn, AUTH_FLAG, str(exc))
                log.error("sweep: Patreon session expired: %s", exc)
                return EXIT_AUTH
            except download.DownloadError as exc:
                log.error("sweep of %s failed: %s", campaign, exc)
                continue
            skipped = []
            for url in urls:
                canon = mail.canonical_post_url(url)
                if not canon:
                    # Never silent. A listing whose entries do not look like
                    # posts means the wrong field was printed, and that failure
                    # is otherwise indistinguishable from "nothing new".
                    skipped.append(url)
                    continue
                _, is_new = store.enqueue(conn, canon, source="sweep",
                                          post_id=mail.post_id_of(canon))
                if is_new:
                    created += 1
                    log.info("sweep queued %s", canon)
            if skipped:
                log.warning(
                    "%d/%d listed url(s) were not Patreon posts and were "
                    "skipped, e.g. %s", len(skipped), len(urls), skipped[0])
            if urls and not any(mail.canonical_post_url(u) for u in urls):
                log.error(
                    "the campaign listing returned no post URLs at all. Every "
                    "entry looked like %s -- that is the campaign page, not a "
                    "post, so yt-dlp printed the wrong field.", urls[0])
    log.info("sweep finished: %d new job(s)", created)
    return EXIT_OK
