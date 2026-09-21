"""Tests for the Patreon pipeline. Run with `python -m pytest tests -q`
or `python tests/test_patreon_pipeline.py`.

Scope: the pure logic where the bugs actually live -- URL canonicalisation
(which is the dedupe key), queue semantics under retry, and the error
classification that decides whether a failure costs a job its attempts or
stops the worker. Nothing here touches the network, Gmail, Drive or yt-dlp.
"""

from __future__ import annotations

import email
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patreon_pipeline import download, mail, rclone, store, transcribe, upload
from patreon_pipeline.config import PipelineConfig


# ----------------------------------------------------------------------
# URL handling
# ----------------------------------------------------------------------

def test_canonical_post_url_collapses_slug_variants():
    """The same post linked three ways must produce one key, or dedupe fails."""
    want = "https://www.patreon.com/posts/12345678"
    for url in (
        "https://www.patreon.com/posts/12345678",
        "https://www.patreon.com/posts/friday-recap-12345678",
        "https://patreon.com/posts/friday-recap-12345678/",
        "https://www.patreon.com/posts/friday-recap-12345678?utm_source=email",
        "https://www.patreon.com/posts/spx-0dte-recap-12345678#comments",
    ):
        assert mail.canonical_post_url(url) == want, url


def test_canonical_post_url_rejects_non_posts():
    for url in (
        "https://www.patreon.com/c/somecreator",
        "https://www.patreon.com/home",
        "https://example.com/posts/12345678",
        "not a url",
        "",
    ):
        assert mail.canonical_post_url(url) is None, url


def test_lookalike_domains_are_not_patreon():
    """A plain endswith() would accept the first two. That is the bug this
    guards: the URLs come out of email, which anyone can send."""
    assert mail.host_matches("www.patreon.com", ["patreon.com"])
    assert mail.host_matches("click.patreon.com", ["patreon.com"])
    assert mail.host_matches("patreon.com", ["patreon.com"])
    assert not mail.host_matches("evilpatreon.com", ["patreon.com"])
    assert not mail.host_matches("patreon.com.evil.net", ["patreon.com"])
    assert mail.canonical_post_url("https://evilpatreon.com/posts/1") is None


def test_extract_urls_strips_trailing_punctuation():
    text = 'See <a href="https://www.patreon.com/posts/111">this</a> (and more).'
    urls = mail.extract_urls(text)
    assert "https://www.patreon.com/posts/111" in urls


def _message(from_addr: str, body: str, subject: str = "New post") -> email.message.Message:
    raw = (f"From: {from_addr}\r\n"
           f"Subject: {subject}\r\n"
           f"Content-Type: text/html; charset=utf-8\r\n"
           f"\r\n{body}\r\n")
    return email.message_from_string(raw)


def test_direct_post_link_is_found_without_any_network():
    msg = _message(
        "Patreon <noreply@patreon.com>",
        '<a href="https://www.patreon.com/posts/friday-9988776?utm_medium=e">go</a>')

    def explode(url):
        raise AssertionError("resolver must not be called for a direct link")

    assert mail.post_urls_from_message(msg, resolver=explode) == [
        "https://www.patreon.com/posts/9988776"]


def test_tracking_link_is_resolved():
    msg = _message("bingo@mg.patreon.com",
                   '<a href="https://click.patreon.com/ss/c/abc123">read</a>')
    calls = []

    def resolver(url):
        calls.append(url)
        return ["https://www.patreon.com/posts/weekly-review-5551212"]

    assert mail.post_urls_from_message(msg, resolver=resolver) == [
        "https://www.patreon.com/posts/5551212"]
    assert calls == ["https://click.patreon.com/ss/c/abc123"]


def test_mail_from_other_senders_is_ignored():
    msg = _message("attacker@example.com",
                   '<a href="https://www.patreon.com/posts/13579">click</a>')
    assert mail.post_urls_from_message(msg, resolver=lambda u: []) == []


# ----------------------------------------------------------------------
# Queue
# ----------------------------------------------------------------------

def _db():
    tmp = tempfile.mkdtemp(prefix="patreon-test-")
    return os.path.join(tmp, "pipeline.db")


def test_enqueue_is_idempotent():
    """The email trigger and the cron sweep both find the same post. The
    second one must be a no-op, or every post downloads twice."""
    url = "https://www.patreon.com/posts/12345678"
    with store.connect(_db()) as conn:
        first_id, created = store.enqueue(conn, url, source="email")
        assert created
        second_id, created_again = store.enqueue(conn, url, source="sweep")
        assert not created_again
        assert second_id == first_id
        assert store.counts(conn)["queued"] == 1


def test_claim_next_takes_one_job_then_none():
    with store.connect(_db()) as conn:
        store.enqueue(conn, "https://www.patreon.com/posts/1")
        job = store.claim_next(conn)
        assert job is not None and job["state"] == "running"
        assert store.claim_next(conn) is None


def test_retry_backs_off_then_fails_at_the_ceiling():
    with store.connect(_db()) as conn:
        job_id, _ = store.enqueue(conn, "https://www.patreon.com/posts/2")
        for _ in range(3):
            state = store.mark_retry(conn, job_id, "boom", max_attempts=4,
                                     backoff_base=60, backoff_cap=1800)
            assert state == "queued"
        assert store.mark_retry(conn, job_id, "boom", max_attempts=4,
                                backoff_base=60, backoff_cap=1800) == "failed"
        assert store.counts(conn)["failed"] == 1


def test_backed_off_job_is_not_claimable_yet():
    with store.connect(_db()) as conn:
        job_id, _ = store.enqueue(conn, "https://www.patreon.com/posts/3")
        store.mark_retry(conn, job_id, "boom", max_attempts=4,
                         backoff_base=600, backoff_cap=1800)
        assert store.claim_next(conn) is None


def test_reset_running_recovers_an_interrupted_job():
    """A worker killed mid-download leaves a row marked running. Nothing else
    can free it, so startup must."""
    with store.connect(_db()) as conn:
        store.enqueue(conn, "https://www.patreon.com/posts/4")
        store.claim_next(conn)
        assert store.reset_running(conn) == 1
        assert store.claim_next(conn) is not None


def test_requeue_does_not_spend_an_attempt():
    """An expired cookie is not the job's fault."""
    with store.connect(_db()) as conn:
        job_id, _ = store.enqueue(conn, "https://www.patreon.com/posts/5")
        store.claim_next(conn)
        store.requeue(conn, job_id)
        row = store.recent(conn, limit=1)[0]
        assert row["state"] == "queued"
        assert row["attempts"] == 0


def test_mail_cursor_round_trips():
    with store.connect(_db()) as conn:
        assert store.get_mail_state(conn, "INBOX")["last_uid"] == 0
        store.set_mail_state(conn, "INBOX", uidvalidity=99, last_uid=1200)
        got = store.get_mail_state(conn, "INBOX")
        assert got == {"uidvalidity": 99, "last_uid": 1200}


def test_flags_set_and_clear():
    with store.connect(_db()) as conn:
        store.set_flag(conn, "auth_alert", "expired")
        assert store.get_flag(conn, "auth_alert") == "expired"
        store.set_flag(conn, "auth_alert", None)
        assert store.get_flag(conn, "auth_alert") is None


# ----------------------------------------------------------------------
# yt-dlp wrapper
# ----------------------------------------------------------------------

def test_auth_failures_are_classified_apart_from_real_failures():
    """This split is what stops an expired login marching the whole queue
    into `failed`."""
    assert download.classify_error(
        "ERROR: This post is for patrons only") is download.AuthError
    assert download.classify_error(
        "ERROR: unable to download: HTTP Error 403: Forbidden") is download.AuthError
    assert download.classify_error(
        "ERROR: No video formats found!") is download.NoMediaError
    assert download.classify_error(
        "ERROR: unable to connect to host") is download.DownloadError
    assert download.classify_error("") is download.DownloadError


def test_build_command_carries_the_session_and_the_destination():
    cfg = PipelineConfig(cookies_from_browser="firefox")
    cmd = download.build_command(cfg, "https://www.patreon.com/posts/1", "/tmp/j")
    assert "--cookies-from-browser" in cmd and "firefox" in cmd
    assert cmd[cmd.index("--paths") + 1] == "/tmp/j"
    assert cmd[-1] == "https://www.patreon.com/posts/1"
    assert "--write-info-json" in cmd


def test_cookies_file_takes_precedence_over_browser():
    cfg = PipelineConfig(cookies_from_browser="firefox",
                         cookies_file="/etc/patreon/cookies.txt")
    cmd = download.build_command(cfg, "https://www.patreon.com/posts/1", "/tmp/j")
    assert "--cookies" in cmd and "--cookies-from-browser" not in cmd


def test_audio_only_switches_the_format():
    cfg = PipelineConfig(audio_only=True)
    cmd = download.build_command(cfg, "https://www.patreon.com/posts/1", "/tmp/j")
    assert "-x" in cmd and "bestaudio/best" in cmd


# ----------------------------------------------------------------------
# Transcription
# ----------------------------------------------------------------------

def test_timestamp_formatting():
    assert transcribe.format_timestamp(0) == "00:00:00"
    assert transcribe.format_timestamp(3725.5) == "01:02:05"
    assert transcribe.format_timestamp(3725.5, srt=True) == "01:02:05,500"
    assert transcribe.format_timestamp(-1) == "00:00:00"


def test_srt_is_well_formed():
    segs = [transcribe.Segment(0.0, 2.5, "First line."),
            transcribe.Segment(2.5, 5.0, "Second line.")]
    out = transcribe.to_srt(segs)
    lines = out.splitlines()
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:02,500"
    assert lines[2] == "First line."
    assert lines[4] == "2"


def test_paragraphs_break_on_a_real_pause():
    """Whisper emits a segment every few seconds. 3,000 one-line segments is
    painful to read and wasteful to feed to a model, so they get grouped."""
    segs = [transcribe.Segment(0, 3, "A" * 200),
            transcribe.Segment(3, 6, "B" * 250),
            transcribe.Segment(30, 33, "After a long gap.")]
    out = transcribe.to_paragraphs(segs, gap_seconds=2.0, min_chars=400)
    paras = [p for p in out.split("\n\n") if p.strip()]
    assert len(paras) == 2
    assert paras[0].startswith("[00:00:00]")
    assert paras[1].startswith("[00:00:30]")


def test_paragraphs_do_not_split_on_a_short_pause():
    segs = [transcribe.Segment(0, 3, "A" * 500),
            transcribe.Segment(3.5, 6, "Still the same thought.")]
    out = transcribe.to_paragraphs(segs, gap_seconds=2.0, min_chars=400)
    assert len([p for p in out.split("\n\n") if p.strip()]) == 1


def test_paragraph_timestamps_can_be_turned_off():
    segs = [transcribe.Segment(10, 12, "Hello.")]
    assert transcribe.to_paragraphs(segs, timestamps=False).strip() == "Hello."
    assert transcribe.to_paragraphs(segs, timestamps=True).startswith("[00:00:10]")


def test_empty_transcript_formats_cleanly():
    assert transcribe.to_paragraphs([]) == ""
    assert transcribe.to_srt([]) == ""


def test_initial_prompt_from_vocabulary():
    """The single highest-leverage knob for jargon-heavy audio."""
    assert transcribe.build_initial_prompt(["QQQ", "0DTE"]) == "QQQ, 0DTE."
    assert transcribe.build_initial_prompt([]) is None
    assert transcribe.build_initial_prompt(["  ", ""]) is None


def test_default_vocabulary_is_populated():
    cfg = PipelineConfig()
    prompt = transcribe.build_initial_prompt(cfg.whisper_vocabulary)
    assert prompt and "QQQ" in prompt and "0DTE" in prompt


# ----------------------------------------------------------------------
# Uploads
# ----------------------------------------------------------------------

def test_rclone_is_the_default_backend():
    """Chosen because it needs no Cloud project, no consent screen and has no
    seven-day token expiry."""
    assert PipelineConfig().uploader == "rclone"
    assert upload.backend(PipelineConfig()).name == "rclone"


def test_drive_backend_is_still_selectable():
    assert upload.backend(PipelineConfig(uploader="drive")).name == "drive"


def test_unknown_uploader_is_refused():
    try:
        upload.backend(PipelineConfig(uploader="dropbox"))
    except upload.UploadError as exc:
        assert "dropbox" in str(exc)
    else:
        raise AssertionError("an unknown uploader should not be accepted")


def test_rclone_destination_nests_by_creator():
    cfg = PipelineConfig(rclone_remote="gdrive", rclone_base_path="Patreon")
    assert rclone.destination(cfg, "Some Creator") == "gdrive:Patreon/Some Creator"


def test_rclone_destination_without_creator_subfolder():
    cfg = PipelineConfig(rclone_remote="gdrive", rclone_base_path="Patreon",
                         drive_subfolder_per_creator=False)
    assert rclone.destination(cfg, "Some Creator") == "gdrive:Patreon"


def test_rclone_destination_sanitises_separators_in_a_creator_name():
    """A slash in a creator name would silently nest an extra remote folder."""
    cfg = PipelineConfig(rclone_remote="gdrive", rclone_base_path="Patreon")
    assert rclone.destination(cfg, "A/B") == "gdrive:Patreon/A-B"


def test_validate_rejects_a_bad_uploader_and_an_empty_remote():
    assert any("uploader" in p for p in PipelineConfig(uploader="nope").validate())
    assert any("rclone_remote" in p
               for p in PipelineConfig(rclone_remote="").validate())


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

def test_validate_reports_every_problem_at_once():
    cfg = PipelineConfig(cookies_from_browser="", cookies_file="")
    problems = cfg.validate(need_mail=True)
    assert len(problems) >= 3


def test_the_shipped_example_configs_actually_load():
    """The installer copies these verbatim. Shipping an example the loader
    rejects means a fresh install fails on its first command -- which is
    exactly what happened, because the examples carry `_comment` keys."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example = os.path.join(here, "deploy", "patreon", "patreon.example.json")
    cfg = PipelineConfig.from_env(example)
    assert cfg.campaign_urls, "example config should carry a campaign URL"


def test_underscore_keys_are_comments_not_settings():
    """JSON has no comment syntax, so `_note` keys document the file in
    place and are dropped on load."""
    import json
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump({"_comment": "explain something", "max_attempts": 7}, fh)
    cfg = PipelineConfig.from_env(path)
    assert cfg.max_attempts == 7
    assert not hasattr(cfg, "_comment")


def test_a_misspelled_setting_is_still_rejected():
    """The underscore escape hatch must not turn into blanket permissiveness:
    a typo that silently does nothing is worse than a startup error."""
    import json
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump({"campaign_url": "https://www.patreon.com/c/x"}, fh)
    try:
        PipelineConfig.from_env(path)
    except ValueError as exc:
        assert "campaign_url" in str(exc)
    else:
        raise AssertionError("a misspelled key should not be accepted")


def test_secrets_are_not_read_from_the_config_file():
    """A Gmail app password in a tracked JSON file is how it ends up in a
    commit. It must come from the environment or not at all."""
    import json
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump({"imap_user": "me@gmail.com", "imap_password": "hunter2"}, fh)
    os.environ.pop("PATREON_IMAP_PASSWORD", None)
    cfg = PipelineConfig.from_env(path)
    assert cfg.imap_user == "me@gmail.com"
    assert cfg.imap_password == ""
    assert "hunter2" not in cfg.describe()


def test_title_length_is_configurable_for_windows_path_limits():
    """Windows caps a path at 260 chars unless long paths are on, and yt-dlp
    fails *after* the download when the name crosses it."""
    cfg = PipelineConfig(title_bytes=100)
    cmd = download.build_command(cfg, "https://www.patreon.com/posts/1", "/tmp/j")
    assert "%(title).100B [%(id)s].%(ext)s" in cmd

    cfg = PipelineConfig(title_bytes=180)
    cmd = download.build_command(cfg, "https://www.patreon.com/posts/1", "/tmp/j")
    assert "%(title).180B [%(id)s].%(ext)s" in cmd


def test_validate_rejects_unreadable_title_length():
    assert any("title_bytes" in p for p in PipelineConfig(title_bytes=5).validate())


def test_validate_rejects_uploading_nothing():
    cfg = PipelineConfig(upload_media=False, upload_transcript=False)
    assert any("upload nothing" in p for p in cfg.validate())


def test_validate_rejects_float16_on_cpu():
    cfg = PipelineConfig(whisper_device="cpu", whisper_compute_type="float16")
    assert any("float16" in p for p in cfg.validate())


def test_transcript_only_mode_is_valid():
    """Video off, transcript on: the archive that costs megabytes."""
    cfg = PipelineConfig(upload_media=False, transcribe=True,
                         upload_transcript=True)
    assert cfg.validate() == []


def test_new_columns_survive_a_migration_from_v1():
    """A database created before transcription existed must gain the columns
    without losing rows."""
    import sqlite3
    path = _db()
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT,
          post_url TEXT NOT NULL UNIQUE, post_id TEXT, source TEXT NOT NULL,
          state TEXT NOT NULL DEFAULT 'queued', attempts INTEGER NOT NULL DEFAULT 0,
          next_attempt_at TEXT, creator TEXT, title TEXT, upload_date TEXT,
          local_path TEXT, size_bytes INTEGER, drive_file_id TEXT, drive_link TEXT,
          last_error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO jobs (post_url, source, state, created_at, updated_at)
          VALUES ('https://www.patreon.com/posts/9', 'email', 'done', 'x', 'x');
    """)
    old.commit()
    old.close()

    with store.connect(path) as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
        assert {"transcript_path", "transcript_link", "language",
                "duration_sec"} <= cols
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
        job_id = conn.execute("SELECT id FROM jobs").fetchone()["id"]
        store._update(conn, job_id, language="en", duration_sec=1800.0)
        row = conn.execute("SELECT * FROM jobs").fetchone()
        assert row["language"] == "en" and row["duration_sec"] == 1800.0


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as exc:
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)
