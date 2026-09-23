"""Command-line entry point for the Patreon pipeline.

    python -m patreon_pipeline.runner auth          # once, with a browser
    python -m patreon_pipeline.runner watch         # long-running: Gmail IDLE
    python -m patreon_pipeline.runner work          # long-running: the queue
    python -m patreon_pipeline.runner sweep         # from a timer: backstop
    python -m patreon_pipeline.runner folders       # which Gmail labels exist
    python -m patreon_pipeline.runner status        # what is in the queue
    python -m patreon_pipeline.runner add URL ...   # queue a post by hand
    python -m patreon_pipeline.runner probe URL     # is the session still good?
    python -m patreon_pipeline.runner transcribe FILE  # one-off, no queue

`watch` and `work` are two units rather than one process with two threads. They
fail for unrelated reasons -- one is a mail socket, the other is a two-hour
download -- and keeping them separate means a crash in either leaves the other
running, with the queue holding the state between them.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

from . import drive, mail, store, transcribe as transcribe_mod, upload, worker
from .config import PipelineConfig
from .worker import EXIT_AUTH, EXIT_CONFIG, EXIT_ERROR, EXIT_OK

log = logging.getLogger("patreon_pipeline")


def setup_logging(verbose: bool, logfile: Optional[str] = None) -> None:
    """Log to stderr, and optionally to a rotating file.

    The file matters on Windows, where Task Scheduler keeps only a task's exit
    code and nothing at all of its output -- there is no journald to fall back
    on. Rotation is not optional either: these are long-running processes, and
    an unbounded log on a mini-PC eventually becomes the disk-full incident.
    """
    from logging.handlers import RotatingFileHandler

    level = logging.DEBUG if verbose else logging.INFO
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if logfile:
        os.makedirs(os.path.dirname(os.path.abspath(logfile)) or ".",
                    exist_ok=True)
        handlers.append(RotatingFileHandler(
            logfile, maxBytes=5 * 1024 * 1024, backupCount=3,
            encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    # The Google client library logs every HTTP request at INFO, which buries
    # everything this pipeline has to say.
    logging.getLogger("googleapiclient").setLevel(logging.WARNING)
    logging.getLogger("google_auth_httplib2").setLevel(logging.WARNING)


def cmd_status(cfg: PipelineConfig, args) -> int:
    with store.connect(cfg.resolved_db_path()) as conn:
        counts = store.counts(conn)
        alert = store.get_flag(conn, worker.AUTH_FLAG)
        rows = store.recent(conn, limit=args.limit, state=args.state)

    print("queue: " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    if alert:
        print(f"\n!! PATREON SESSION PROBLEM: {alert}\n"
              f"   Log in again in the browser profile this box reads, then "
              f"the worker resumes on its own.")
    if not rows:
        return EXIT_OK
    print()
    for r in rows:
        line = f"[{r['id']:>5}] {r['state']:<8} {r['post_url']}"
        if r["title"]:
            line += f"\n        {r['title'][:90]}"
        if r["drive_link"]:
            line += f"\n        {r['drive_link']}"
        if r["transcript_link"] and r["transcript_link"] != r["drive_link"]:
            line += f"\n        transcript: {r['transcript_link']}"
        if r["duration_sec"]:
            line += f"\n        {r['duration_sec'] / 60.0:.0f} min"
            if r["language"]:
                line += f" ({r['language']})"
        if r["last_error"] and r["state"] in ("failed", "queued", "skipped"):
            line += f"\n        ! {r['last_error'][:160]}"
        print(line)
    return EXIT_OK


def cmd_add(cfg: PipelineConfig, args) -> int:
    added = 0
    with store.connect(cfg.resolved_db_path()) as conn:
        for raw in args.urls:
            canon = mail.canonical_post_url(raw)
            if not canon:
                log.error("not a Patreon post URL: %s", raw)
                continue
            job_id, is_new = store.enqueue(conn, canon, source="manual",
                                           post_id=mail.post_id_of(canon))
            print(f"{'queued' if is_new else 'already known'}: "
                  f"job {job_id} {canon}")
            added += int(is_new)
    return EXIT_OK if added or not args.urls else EXIT_OK


def cmd_probe(cfg: PipelineConfig, args) -> int:
    problem = download_probe(cfg, args.url)
    if problem is None:
        print("session OK")
        with store.connect(cfg.resolved_db_path()) as conn:
            store.set_flag(conn, worker.AUTH_FLAG, None)
        return EXIT_OK
    print(f"session PROBLEM: {problem}", file=sys.stderr)
    with store.connect(cfg.resolved_db_path()) as conn:
        store.set_flag(conn, worker.AUTH_FLAG, problem)
    return EXIT_AUTH


def download_probe(cfg: PipelineConfig, url: str):
    from .download import probe_session
    return probe_session(cfg, url)


def cmd_folders(cfg: PipelineConfig, args) -> int:
    """List the IMAP folders the account exposes, and say whether ours is one.

    Gmail labels are IMAP folders, but not always under the name you see in the
    web UI: a nested label arrives as "Parent/Child", and the name is
    case-sensitive. Guessing at that produces a select-folder error that says
    nothing useful, so this prints the real list instead.
    """
    try:
        from imapclient import IMAPClient
    except ImportError:
        log.error("imapclient is not installed: pip install -r "
                  "requirements-patreon.txt")
        return EXIT_CONFIG

    with IMAPClient(cfg.imap_host, port=cfg.imap_port, ssl=True,
                    timeout=60) as client:
        client.login(cfg.imap_user, cfg.imap_password)
        rows = client.list_folders()

    names = []
    for _flags, _delimiter, name in rows:
        names.append(name if isinstance(name, str) else name.decode("utf-8"))

    print(f"folders visible to {cfg.imap_user}:\n")
    for name in sorted(names):
        marker = "   <-- PATREON_IMAP_FOLDER" if name == cfg.imap_folder else ""
        print(f"  {name}{marker}")

    if cfg.imap_folder not in names:
        print(f"\n!! {cfg.imap_folder!r} is not in that list, so the watcher "
              f"cannot select it.\n"
              f"   Set PATREON_IMAP_FOLDER to one of the names above, exactly "
              f"as printed.")
        return EXIT_CONFIG
    print(f"\n{cfg.imap_folder!r} exists -- the watcher can select it.")
    return EXIT_OK


def cmd_transcribe(cfg: PipelineConfig, args) -> int:
    """Transcribe a local file without touching the queue or Drive.

    This is how you benchmark the model on your own hardware before trusting
    it to an unattended worker -- run it on one real post and see whether an
    hour of audio takes ten minutes or ninety.
    """
    import time
    started = time.monotonic()
    try:
        res = transcribe_mod.transcribe(cfg, args.path, args.out_dir)
    except transcribe_mod.TranscribeError as exc:
        log.error("transcription failed: %s", exc)
        return EXIT_ERROR
    elapsed = time.monotonic() - started
    speed = (res.duration / elapsed) if elapsed > 0 else 0.0
    for path in res.paths:
        print(path)
    print(f"\n{res.segments} segments, {res.duration / 60.0:.1f} min of audio "
          f"in {elapsed / 60.0:.1f} min ({speed:.2f}x real time), "
          f"language={res.language}")
    return EXIT_OK


def cmd_retry(cfg: PipelineConfig, args) -> int:
    with store.connect(cfg.resolved_db_path()) as conn:
        n = store.retry_failed(conn)
    print(f"requeued {n} failed job(s)")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="patreon_pipeline",
        description="Gmail-triggered Patreon downloads, uploaded to Drive.")
    p.add_argument("--config", default=None,
                   help="JSON config file (or set PATREON_CONFIG)")
    p.add_argument("--logfile", default=None, help="also write the log here")
    p.add_argument("-v", "--verbose", action="store_true")

    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("auth", help="one-time Google Drive authorisation")
    a.add_argument("--console", action="store_true",
                   help="print the URL instead of opening a browser (for SSH)")

    sub.add_parser("folders",
                   help="list the IMAP folders (Gmail labels) this account has")

    w = sub.add_parser("watch", help="hold a Gmail IDLE connection open")
    w.add_argument("--once", action="store_true",
                   help="check for new mail once and exit")

    k = sub.add_parser("work", help="drain the download queue")
    k.add_argument("--once", action="store_true",
                   help="stop when the queue is empty instead of waiting")
    k.add_argument("--max-jobs", type=int, default=0,
                   help="stop after N jobs (0 = no limit)")

    s = sub.add_parser("sweep", help="enqueue recent campaign posts (backstop)")
    s.add_argument("--limit", type=int, default=20,
                   help="how many recent posts to look at per campaign")

    st = sub.add_parser("status", help="show the queue")
    st.add_argument("--limit", type=int, default=20)
    st.add_argument("--state", default=None, choices=list(store.STATES))

    ad = sub.add_parser("add", help="queue one or more post URLs by hand")
    ad.add_argument("urls", nargs="+")

    pr = sub.add_parser("probe", help="check the Patreon session is still valid")
    pr.add_argument("url", help="any patron-only post URL")

    tr = sub.add_parser("transcribe",
                        help="transcribe a local media file (no queue, no Drive)")
    tr.add_argument("path")
    tr.add_argument("--out-dir", default=None,
                    help="where to write .txt/.srt (default: next to the file)")

    sub.add_parser("retry", help="move failed jobs back to the queue")
    sub.add_parser("config", help="print the resolved configuration")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose, args.logfile)

    try:
        cfg = PipelineConfig.from_env(args.config)
    except (ValueError, OSError) as exc:
        log.error("bad config: %s", exc)
        return EXIT_CONFIG

    os.makedirs(cfg.state_dir, exist_ok=True)

    if args.command == "config":
        print(cfg.describe())
        try:
            upload.backend(cfg).check()
            print(f"\nuploader: {cfg.uploader} -- OK")
        except upload.UploadError as exc:
            print(f"\nuploader: {cfg.uploader} -- NOT READY: {exc}")
        return EXIT_OK

    problems = cfg.validate(need_mail=args.command in ("watch", "folders"))
    if problems:
        for pb in problems:
            log.error("config: %s", pb)
        return EXIT_CONFIG

    try:
        if args.command == "auth":
            if cfg.uploader == "rclone":
                # Nothing for this program to do: rclone owns its own
                # credentials, and duplicating its setup wizard would only
                # create a second place for it to go wrong.
                print(
                    "This install uploads through rclone, which keeps its own\n"
                    "credentials -- there is no token for the pipeline to fetch.\n"
                    "\n"
                    "Run rclone's own setup once:\n"
                    "\n"
                    "    rclone config\n"
                    "\n"
                    f"  n) New remote\n"
                    f"  name> {cfg.rclone_remote}\n"
                    "  Storage> drive\n"
                    "  client_id / client_secret> (leave both blank)\n"
                    "  scope> 1   (full access)\n"
                    "  Edit advanced config? n\n"
                    "  Use web browser to automatically authenticate? y\n"
                    "\n"
                    "Then check it with:\n"
                    "\n"
                    f"    rclone lsd {cfg.rclone_remote}:\n")
                return EXIT_OK
            path = drive.authorize(cfg, console=args.console)
            print(f"authorised; token saved to {path}")
            return EXIT_OK
        if args.command == "folders":
            return cmd_folders(cfg, args)
        if args.command == "watch":
            return mail.watch(cfg, once=args.once)
        if args.command == "work":
            return worker.run(cfg, once=args.once, max_jobs=args.max_jobs)
        if args.command == "sweep":
            return worker.sweep(cfg, limit=args.limit)
        if args.command == "status":
            return cmd_status(cfg, args)
        if args.command == "add":
            return cmd_add(cfg, args)
        if args.command == "probe":
            return cmd_probe(cfg, args)
        if args.command == "transcribe":
            return cmd_transcribe(cfg, args)
        if args.command == "retry":
            return cmd_retry(cfg, args)
    except drive.DriveError as exc:
        log.error("drive: %s", exc)
        return EXIT_CONFIG
    except KeyboardInterrupt:
        return EXIT_OK
    except Exception as exc:
        log.exception("%s failed: %s", args.command, exc)
        return EXIT_ERROR
    return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
