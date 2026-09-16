"""Deployment settings for the Patreon pipeline.

Everything here can be overridden by an environment variable, so the mini-PC is
configured without editing tracked files. Two settings are secrets and are
read *only* from the environment, never from the JSON config file:
`PATREON_IMAP_PASSWORD` and the Drive OAuth token path. Putting a Gmail app
password in a file that sits next to tracked code is how it ends up in a
commit.

On the two Google credentials
-----------------------------
This pipeline touches Google twice and deliberately uses a different mechanism
for each. That looks inconsistent; it is the only combination that survives
unattended running.

* **Gmail: IMAP + an app password.** The Gmail API would be tidier, but its
  read scopes are *restricted*, which means an unverified personal project is
  stuck in "Testing" publishing status -- and there a refresh token expires
  after seven days. A pipeline that silently stops every Tuesday is worse than
  one that uses IMAP. An app password does not expire.

* **Drive: OAuth with the `drive.file` scope.** A service account is the usual
  unattended answer (it is what `qbs.live.sheets` uses) but it cannot work
  here: a service account has no Drive storage quota of its own, so uploading
  to a personal My Drive fails outright. User OAuth is required. `drive.file`
  is a non-sensitive scope -- it grants access only to files this app itself
  creates -- so the project can be pushed to "In production" without Google's
  verification review, and *that* is what stops the refresh token expiring.

  Set the OAuth consent screen to "In production". Left in "Testing", the
  token dies after seven days and the uploads stop.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STATE_DIR = os.environ.get(
    "PATREON_STATE_DIR", os.path.join(REPO_ROOT, "var", "patreon"))


def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v not in (None, "") else default


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    return float(v) if v not in (None, "") else default


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v in (None, ""):
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_list(key: str, default: List[str]) -> List[str]:
    """Comma-separated env var -> list. Blank entries dropped."""
    v = os.environ.get(key)
    if v in (None, ""):
        return list(default)
    return [p.strip() for p in v.split(",") if p.strip()]


@dataclass
class PipelineConfig:
    """Settings for the watcher, the downloader and the uploader.

    Defaults are chosen to be safe on a home connection: one download at a
    time, polite sleeps between requests, and a retry ceiling low enough that a
    genuinely broken post stops being retried within an hour rather than being
    hammered all week.
    """

    # ---- state -----------------------------------------------------------
    state_dir: str = DEFAULT_STATE_DIR
    db_path: str = ""            # blank -> <state_dir>/pipeline.db
    staging_dir: str = ""        # blank -> <state_dir>/staging

    # ---- gmail (IMAP) ----------------------------------------------------
    imap_host: str = "imap.gmail.com"
    imap_port: int = 993
    imap_user: str = ""
    imap_password: str = ""      # env only: PATREON_IMAP_PASSWORD
    imap_folder: str = "INBOX"   # a Gmail label works here too
    # Gmail drops an idle connection at ~29 minutes. Re-issuing every 5 keeps
    # it alive and bounds how long a dead socket goes unnoticed.
    imap_idle_seconds: int = 300
    # Which senders count. Matched against the envelope From address as a
    # domain suffix, so "patreon.com" also accepts "bingo@mg.patreon.com".
    sender_domains: List[str] = field(default_factory=lambda: ["patreon.com"])

    # ---- patreon / yt-dlp ------------------------------------------------
    campaign_urls: List[str] = field(default_factory=list)
    ytdlp_binary: str = "yt-dlp"
    # One of these two supplies the patron session. `cookies_from_browser` is
    # the low-maintenance option on a box with a logged-in Firefox profile;
    # `cookies_file` is the exported-cookies fallback.
    cookies_from_browser: str = "firefox"
    cookies_file: str = ""
    ytdlp_format: str = ""       # blank -> let yt-dlp choose
    audio_only: bool = False     # True -> bestaudio, extracted to mp3
    audio_format: str = "mp3"
    audio_quality: str = "64K"
    sleep_requests: float = 2.0
    sleep_interval: float = 5.0
    max_sleep_interval: float = 15.0
    download_timeout: int = 7200  # seconds; a 2h ceiling on one post

    # ---- google drive ----------------------------------------------------
    # The folder uploads land in. An ID is exact; a name is found-or-created
    # under My Drive root. ID wins when both are set.
    drive_folder_id: str = ""
    drive_folder_name: str = "Patreon"
    # One subfolder per creator keeps a year of posts navigable, and gives
    # Gemini a natural unit to be pointed at.
    drive_subfolder_per_creator: bool = True
    drive_client_secret: str = ""  # blank -> <state_dir>/client_secret.json
    drive_token: str = ""          # blank -> <state_dir>/drive_token.json
    drive_chunk_mb: int = 8
    upload_info_json: bool = True  # the .info.json alongside the media
    delete_after_upload: bool = True

    # ---- worker ----------------------------------------------------------
    max_attempts: int = 4
    backoff_base: float = 60.0   # seconds; doubled per attempt
    backoff_cap: float = 1800.0
    poll_seconds: float = 30.0   # how often an idle worker re-checks the queue

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------
    def resolved_db_path(self) -> str:
        return self.db_path or os.path.join(self.state_dir, "pipeline.db")

    def resolved_staging_dir(self) -> str:
        return self.staging_dir or os.path.join(self.state_dir, "staging")

    def resolved_client_secret(self) -> str:
        return self.drive_client_secret or os.path.join(
            self.state_dir, "client_secret.json")

    def resolved_token(self) -> str:
        return self.drive_token or os.path.join(self.state_dir, "drive_token.json")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, path: Optional[str] = None) -> "PipelineConfig":
        """Build from defaults, then a JSON file, then the environment.

        Later sources win. The JSON file is for the settings you want in
        version control adjacent to the box (campaign URLs, folder names); the
        environment is for the secrets and anything host-specific.
        """
        data: Dict[str, Any] = {}
        path = path or os.environ.get("PATREON_CONFIG")
        if path:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            unknown = set(data) - {f for f in cls.__dataclass_fields__}
            if unknown:
                raise ValueError(f"unknown config keys: {sorted(unknown)}")

        cfg = cls(**data)

        cfg.state_dir = _env_str("PATREON_STATE_DIR", cfg.state_dir)
        cfg.db_path = _env_str("PATREON_DB", cfg.db_path)
        cfg.staging_dir = _env_str("PATREON_STAGING_DIR", cfg.staging_dir)

        cfg.imap_host = _env_str("PATREON_IMAP_HOST", cfg.imap_host)
        cfg.imap_port = _env_int("PATREON_IMAP_PORT", cfg.imap_port)
        cfg.imap_user = _env_str("PATREON_IMAP_USER", cfg.imap_user)
        # Secret: environment only. Never read from the JSON file.
        cfg.imap_password = os.environ.get("PATREON_IMAP_PASSWORD", "")
        cfg.imap_folder = _env_str("PATREON_IMAP_FOLDER", cfg.imap_folder)
        cfg.imap_idle_seconds = _env_int(
            "PATREON_IMAP_IDLE_SECONDS", cfg.imap_idle_seconds)
        cfg.sender_domains = _env_list("PATREON_SENDER_DOMAINS", cfg.sender_domains)

        cfg.campaign_urls = _env_list("PATREON_CAMPAIGN_URLS", cfg.campaign_urls)
        cfg.ytdlp_binary = _env_str("PATREON_YTDLP", cfg.ytdlp_binary)
        cfg.cookies_from_browser = _env_str(
            "PATREON_COOKIES_FROM_BROWSER", cfg.cookies_from_browser)
        cfg.cookies_file = _env_str("PATREON_COOKIES_FILE", cfg.cookies_file)
        cfg.ytdlp_format = _env_str("PATREON_FORMAT", cfg.ytdlp_format)
        cfg.audio_only = _env_bool("PATREON_AUDIO_ONLY", cfg.audio_only)
        cfg.download_timeout = _env_int(
            "PATREON_DOWNLOAD_TIMEOUT", cfg.download_timeout)

        cfg.drive_folder_id = _env_str("PATREON_DRIVE_FOLDER_ID", cfg.drive_folder_id)
        cfg.drive_folder_name = _env_str(
            "PATREON_DRIVE_FOLDER_NAME", cfg.drive_folder_name)
        cfg.drive_client_secret = _env_str(
            "PATREON_DRIVE_CLIENT_SECRET", cfg.drive_client_secret)
        cfg.drive_token = _env_str("PATREON_DRIVE_TOKEN", cfg.drive_token)
        cfg.delete_after_upload = _env_bool(
            "PATREON_DELETE_AFTER_UPLOAD", cfg.delete_after_upload)

        cfg.max_attempts = _env_int("PATREON_MAX_ATTEMPTS", cfg.max_attempts)
        cfg.poll_seconds = _env_float("PATREON_POLL_SECONDS", cfg.poll_seconds)
        return cfg

    def validate(self, *, need_mail: bool = False) -> List[str]:
        """Return a list of problems, empty if the config is usable.

        Returned rather than raised so the CLI can print every problem at once.
        A first run typically has three.
        """
        problems: List[str] = []
        if need_mail:
            if not self.imap_user:
                problems.append("imap_user is unset (PATREON_IMAP_USER)")
            if not self.imap_password:
                problems.append(
                    "imap_password is unset (PATREON_IMAP_PASSWORD); use a "
                    "Gmail app password, not the account password")
        if not self.cookies_from_browser and not self.cookies_file:
            problems.append(
                "no Patreon session: set cookies_from_browser or cookies_file")
        if self.cookies_file and not os.path.exists(self.cookies_file):
            problems.append(f"cookies_file does not exist: {self.cookies_file}")
        if self.max_attempts < 1:
            problems.append("max_attempts must be >= 1")
        return problems

    def describe(self) -> str:
        """Human-readable dump with the secret redacted."""
        d = asdict(self)
        d["imap_password"] = "***" if self.imap_password else ""
        return json.dumps(d, indent=2, sort_keys=True)
