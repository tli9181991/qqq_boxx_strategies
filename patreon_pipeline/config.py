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
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

IS_WINDOWS = sys.platform == "win32"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_state_dir() -> str:
    """Where state lives when nothing overrides it.

    On Windows the repo may sit anywhere (often a synced folder), and a
    Scheduled Task runs as a specific user, so LOCALAPPDATA is the right home
    for a database, staging files and a 1.5 GB model cache. On Linux the
    systemd units set PATREON_STATE_DIR to /var/lib/patreon and this default
    only ever applies to a developer running from a clone.
    """
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("PROGRAMDATA")
        if base:
            return os.path.join(base, "PatreonPipeline")
    return os.path.join(REPO_ROOT, "var", "patreon")


DEFAULT_STATE_DIR = os.environ.get("PATREON_STATE_DIR", _default_state_dir())

# Bytes of post title kept in the output filename. Windows caps a path at 260
# characters unless long-path support is switched on, and a staging directory
# plus a 180-byte title plus " [id].ext" runs right up against that -- yt-dlp
# then fails late, after the download. Linux has no such limit worth worrying
# about.
DEFAULT_TITLE_BYTES = 100 if IS_WINDOWS else 180


# Values the shipped example files carry. Left in place they produce failures
# that look like credential problems -- Gmail rejects you@gmail.com with the
# same AUTHENTICATIONFAILED it gives a wrong password -- so they are caught as
# config errors instead.
PLACEHOLDERS = {
    "you@gmail.com", "your@gmail.com", "user@gmail.com", "youremail@gmail.com",
    "xxxxxxxxxxxxxxxx",
    "https://www.patreon.com/posts/12345678",
    "https://www.patreon.com/c/somecreator",
}


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
    ytdlp_binary: str = "yt-dlp"     # "yt-dlp.exe" is found by this name too
    ffmpeg_binary: str = "ffmpeg"
    # See DEFAULT_TITLE_BYTES: this is a Windows path-length guard.
    title_bytes: int = DEFAULT_TITLE_BYTES
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

    # ---- transcription ---------------------------------------------------
    # Whisper runs between the download and the upload and is *additive*: the
    # media is still uploaded unless `upload_media` is turned off. A transcript
    # is ~50 KB against a gigabyte of video, so keeping both is free.
    transcribe: bool = True
    # large-v3-turbo is the sweet spot on a fanless mini-PC: near large-v3
    # quality at a fraction of the compute. Drop to "small" if an hour of audio
    # takes materially longer than an hour to process.
    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "cpu"          # "cuda" if the box has a usable GPU
    whisper_compute_type: str = "int8"   # "float16" on CUDA
    whisper_cpu_threads: int = 0         # 0 -> let ctranslate2 decide
    whisper_language: str = "en"         # "" -> autodetect (slower, less exact)
    whisper_beam_size: int = 5
    whisper_vad: bool = True             # skip silence; a real speedup
    whisper_txt_timestamps: bool = True  # [HH:MM:SS] prefix on each paragraph
    whisper_formats: List[str] = field(default_factory=lambda: ["txt", "srt"])
    whisper_root: str = ""               # blank -> <state_dir>/whisper-models
    whisper_skip_extract_for_audio: bool = True
    transcribe_timeout: int = 14400      # 4h ceiling on one file

    # Domain vocabulary fed to Whisper as `initial_prompt`. This is the single
    # highest-leverage setting for jargon-heavy audio: without it tickers and
    # options terms come back mangled, and a bigger model does not fix it the
    # way this does. Edit it to match what you actually archive.
    whisper_vocabulary: List[str] = field(default_factory=lambda: [
        "QQQ", "SPX", "SPY", "ES", "NDX", "IWM", "VIX", "BOXX",
        "0DTE", "delta", "gamma", "theta", "vega", "implied volatility",
        "IV crush", "box spread", "credit spread", "debit spread",
        "iron condor", "strike", "expiry", "assignment", "the bid", "the ask",
    ])

    # ---- uploads ---------------------------------------------------------
    # "rclone" authorises against rclone's own registered OAuth client, so
    # there is no Cloud project, no consent screen to publish and no seven-day
    # token expiry -- it just needs the rclone binary. "drive" uses the Drive
    # API with an OAuth client you register yourself. See upload.py.
    uploader: str = "rclone"
    rclone_binary: str = "rclone"
    rclone_remote: str = "gdrive"    # the name given in `rclone config`
    rclone_base_path: str = "Patreon"
    rclone_timeout: int = 7200

    # ---- google drive (uploader = "drive" only) --------------------------
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
    # Both default on: you said you might hand Gemini either artefact, so the
    # pipeline ships both and lets you choose per post. Set upload_media to
    # False to keep only transcripts -- an archive that then costs megabytes
    # rather than terabytes.
    upload_media: bool = True
    upload_transcript: bool = True
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

    def resolved_whisper_root(self) -> str:
        return self.whisper_root or os.path.join(self.state_dir, "whisper-models")

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
            # JSON has no comment syntax, so a key beginning with an
            # underscore is treated as an annotation and dropped. The shipped
            # example config uses them to explain settings in place, and
            # without this the file the installer copies is one this loader
            # refuses to load.
            data = {k: v for k, v in data.items() if not k.startswith("_")}
            unknown = set(data) - {f for f in cls.__dataclass_fields__}
            if unknown:
                # Still strict about everything else: a typo like
                # "campaign_url" for "campaign_urls" would otherwise be
                # ignored silently and the setting would never take effect.
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
        cfg.ffmpeg_binary = _env_str("PATREON_FFMPEG", cfg.ffmpeg_binary)
        cfg.title_bytes = _env_int("PATREON_TITLE_BYTES", cfg.title_bytes)
        cfg.cookies_from_browser = _env_str(
            "PATREON_COOKIES_FROM_BROWSER", cfg.cookies_from_browser)
        cfg.cookies_file = _env_str("PATREON_COOKIES_FILE", cfg.cookies_file)
        cfg.ytdlp_format = _env_str("PATREON_FORMAT", cfg.ytdlp_format)
        cfg.audio_only = _env_bool("PATREON_AUDIO_ONLY", cfg.audio_only)
        cfg.download_timeout = _env_int(
            "PATREON_DOWNLOAD_TIMEOUT", cfg.download_timeout)

        cfg.transcribe = _env_bool("PATREON_TRANSCRIBE", cfg.transcribe)
        cfg.whisper_model = _env_str("PATREON_WHISPER_MODEL", cfg.whisper_model)
        cfg.whisper_device = _env_str("PATREON_WHISPER_DEVICE", cfg.whisper_device)
        cfg.whisper_compute_type = _env_str(
            "PATREON_WHISPER_COMPUTE_TYPE", cfg.whisper_compute_type)
        cfg.whisper_language = _env_str(
            "PATREON_WHISPER_LANGUAGE", cfg.whisper_language)
        cfg.whisper_root = _env_str("PATREON_WHISPER_ROOT", cfg.whisper_root)
        cfg.whisper_vocabulary = _env_list(
            "PATREON_WHISPER_VOCABULARY", cfg.whisper_vocabulary)
        cfg.upload_media = _env_bool("PATREON_UPLOAD_MEDIA", cfg.upload_media)
        cfg.upload_transcript = _env_bool(
            "PATREON_UPLOAD_TRANSCRIPT", cfg.upload_transcript)

        cfg.uploader = _env_str("PATREON_UPLOADER", cfg.uploader)
        cfg.rclone_binary = _env_str("PATREON_RCLONE", cfg.rclone_binary)
        cfg.rclone_remote = _env_str("PATREON_RCLONE_REMOTE", cfg.rclone_remote)
        cfg.rclone_base_path = _env_str(
            "PATREON_RCLONE_BASE_PATH", cfg.rclone_base_path)

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
            elif self.imap_user.strip().lower() in PLACEHOLDERS:
                problems.append(
                    f"imap_user is still the example value "
                    f"{self.imap_user!r} -- set PATREON_IMAP_USER to your own "
                    f"address. Gmail rejects it exactly like a wrong password.")
            elif "@" not in self.imap_user:
                problems.append(
                    f"imap_user {self.imap_user!r} is not an email address; "
                    f"Gmail needs the full address including the domain")
            if self.imap_password.strip().lower() in PLACEHOLDERS:
                problems.append(
                    "imap_password is still the example value -- paste the "
                    "16-character app password from "
                    "https://myaccount.google.com/apppasswords")
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
        if not self.upload_media and not self.upload_transcript:
            problems.append(
                "upload_media and upload_transcript are both off: the pipeline "
                "would download and then upload nothing")
        if not self.upload_media and not self.transcribe:
            problems.append(
                "upload_media is off but transcribe is off too: nothing would "
                "be produced")
        if self.campaign_urls and any(
                u.strip().lower() in PLACEHOLDERS for u in self.campaign_urls):
            problems.append(
                "campaign_urls still contains the example URL -- replace it "
                "with the creator you actually follow")
        if self.uploader not in ("rclone", "drive"):
            problems.append(
                f"uploader must be 'rclone' or 'drive', not {self.uploader!r}")
        if self.uploader == "rclone" and not self.rclone_remote:
            problems.append("rclone_remote is empty; name the remote you made "
                            "with `rclone config`")
        if self.title_bytes < 20:
            problems.append("title_bytes below 20 makes filenames unreadable")
        if self.whisper_device == "cpu" and self.whisper_compute_type == "float16":
            problems.append(
                "whisper_compute_type float16 is not supported on CPU; use int8")
        return problems

    def password_shape(self) -> str:
        """Describe the app password without revealing it.

        "***" tells you nothing when Gmail says AUTHENTICATIONFAILED. Almost
        every cause of that is visible in the *shape* of the string -- pasted
        with the spaces Google displays it with, wrapped in quotes, or the
        OAuth client secret pasted by mistake -- so report those and keep the
        value itself out of the output.
        """
        pw = self.imap_password
        if not pw:
            return "(unset)"
        notes = [f"{len(pw)} chars"]
        if pw != pw.strip():
            notes.append("SURROUNDING WHITESPACE -- trim it")
        core = pw.strip()
        if core[:1] in ("\"", "'") or core[-1:] in ("\"", "'"):
            notes.append("QUOTED -- remove the quotes")
        if any(c.isspace() for c in core):
            notes.append("CONTAINS SPACES -- enter the 16 characters unspaced")
        if core.startswith("GOCSPX-"):
            notes.append("this is an OAuth client secret, NOT an app password")
        elif len(core) != 16:
            notes.append("expected 16 characters for a Gmail app password")
        return "<" + "; ".join(notes) + ">"

    def describe(self) -> str:
        """Human-readable dump. The password is described, never printed."""
        d = asdict(self)
        d["imap_password"] = self.password_shape()
        return json.dumps(d, indent=2, sort_keys=True)
