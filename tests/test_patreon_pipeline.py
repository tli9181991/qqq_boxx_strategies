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

from patreon_pipeline import download, mail, store
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
# Config
# ----------------------------------------------------------------------

def test_validate_reports_every_problem_at_once():
    cfg = PipelineConfig(cookies_from_browser="", cookies_file="")
    problems = cfg.validate(need_mail=True)
    assert len(problems) >= 3


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
