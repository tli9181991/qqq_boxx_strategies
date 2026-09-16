"""Patreon -> Google Drive ingestion pipeline.

Unrelated to the trading strategy in `qbs`. It lives in this repo because it
runs on the same box, under the same systemd conventions, and nothing about it
is worth a second deployment story.

Shape: a Gmail watcher turns "new post" notifications into rows in a SQLite
queue; a single worker drains that queue, pulling each post with yt-dlp and
pushing the result to Drive. The queue is the seam -- see `store.py` for why
that matters more than it looks.
"""

__all__ = ["config", "store", "mail", "download", "drive", "worker"]
