"""Upload finished downloads to Google Drive.

Scope choice, and the trap in it
--------------------------------
This uses `drive.file`, which grants access only to files the app itself
creates. That is what keeps the OAuth project out of Google's verification
review, which in turn is what lets the consent screen go to "In production",
which is what stops the refresh token expiring every seven days. The chain
matters: take the wider `drive` scope and you are back to a token that dies
weekly unless you submit the project for review.

The trap: under `drive.file` the app **cannot see a folder you made by hand in
the Drive web UI**. Pasting that folder's ID into `drive_folder_id` gives a 404
that reads like a permissions bug. Let the pipeline create its own folder by
name (the default) and everything works. If you genuinely need to target a
pre-existing folder, that is the one case for widening the scope.

Gemini reads what lands here either way -- a folder this app created is an
ordinary folder in your Drive, visible and shareable like any other.

Uploads are resumable
---------------------
A two-gigabyte upload over a home connection will be interrupted eventually.
Resumable chunks mean an interruption costs one chunk, not the file.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import re
from typing import Any, Dict, List, Optional

from .config import PipelineConfig

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
FOLDER_MIME = "application/vnd.google-apps.folder"


class DriveError(RuntimeError):
    pass


def _require_libs():
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError as exc:
        raise DriveError(
            "Google client libraries are missing: pip install -r "
            "requirements-patreon.txt") from exc
    return Request, Credentials, InstalledAppFlow, build, MediaFileUpload


def authorize(cfg: PipelineConfig, *, console: bool = False) -> str:
    """Run the one-time OAuth consent flow and save the token.

    Run this once, on the mini-PC, with a browser available. `console=True`
    prints a URL to paste into a browser elsewhere instead -- useful over SSH.
    Returns the token path.
    """
    Request, Credentials, InstalledAppFlow, build, _ = _require_libs()
    secret = cfg.resolved_client_secret()
    if not os.path.exists(secret):
        raise DriveError(
            f"OAuth client secret not found at {secret}. Create a Desktop-app "
            "OAuth client in the Google Cloud console and download its JSON "
            "here.")
    token_path = cfg.resolved_token()
    os.makedirs(os.path.dirname(os.path.abspath(token_path)), exist_ok=True)

    flow = InstalledAppFlow.from_client_secrets_file(secret, SCOPES)
    # access_type=offline plus prompt=consent is what actually returns a
    # refresh token. Without them a re-authorisation can come back with only an
    # access token, and the pipeline stops an hour later.
    kwargs = {"access_type": "offline", "prompt": "consent"}
    if console:
        creds = flow.run_local_server(port=0, open_browser=False, **kwargs)
    else:
        creds = flow.run_local_server(port=0, **kwargs)

    with open(token_path, "w", encoding="utf-8") as fh:
        fh.write(creds.to_json())
    try:
        os.chmod(token_path, 0o600)
    except OSError as exc:
        log.debug("could not restrict permissions on %s: %s", token_path, exc)
    if os.name == "nt":
        # chmod on Windows only toggles the read-only bit -- it does not
        # restrict other users. The token is a live credential, so say so
        # rather than letting the call above imply a protection it did not
        # provide. NTFS ACLs under LOCALAPPDATA are already per-user, which is
        # why the default state directory lives there.
        log.info("token saved under your user profile; on Windows its "
                 "protection is the folder's ACL, not file permissions")
    log.info("saved Drive token to %s", token_path)
    return token_path


def service(cfg: PipelineConfig):
    """Build an authenticated Drive client, refreshing the token if needed."""
    Request, Credentials, _, build, _ = _require_libs()
    token_path = cfg.resolved_token()
    if not os.path.exists(token_path):
        raise DriveError(
            f"no Drive token at {token_path}; run `python -m "
            "patreon_pipeline.runner auth` once on this machine")
    creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(token_path, "w", encoding="utf-8") as fh:
                fh.write(creds.to_json())
        else:
            raise DriveError(
                "Drive credentials are invalid and cannot be refreshed. If the "
                "OAuth consent screen is still in 'Testing', tokens expire "
                "after 7 days -- set it to 'In production' and re-run `auth`.")
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _escape(name: str) -> str:
    """Escape a name for a Drive query string literal."""
    return name.replace("\\", "\\\\").replace("'", "\\'")


def find_folder(svc, name: str, parent: Optional[str] = None) -> Optional[str]:
    q = (f"name = '{_escape(name)}' and mimeType = '{FOLDER_MIME}' "
         f"and trashed = false")
    q += f" and '{parent}' in parents" if parent else " and 'root' in parents"
    resp = svc.files().list(q=q, spaces="drive", fields="files(id, name)",
                            pageSize=10).execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def ensure_folder(svc, name: str, parent: Optional[str] = None) -> str:
    """Find a folder by name, or create it. Returns its id."""
    existing = find_folder(svc, name, parent)
    if existing:
        return existing
    body: Dict[str, Any] = {"name": name, "mimeType": FOLDER_MIME}
    body["parents"] = [parent] if parent else ["root"]
    created = svc.files().create(body=body, fields="id").execute()
    log.info("created Drive folder %r (%s)", name, created["id"])
    return created["id"]


def find_file(svc, name: str, folder_id: str) -> Optional[Dict[str, str]]:
    """Look for a file already uploaded under this name.

    Guards the narrow window where an upload completed but the job was not
    marked done -- a crash between the two would otherwise upload a second copy
    on the retry.
    """
    q = (f"name = '{_escape(name)}' and '{folder_id}' in parents "
         f"and trashed = false")
    resp = svc.files().list(q=q, spaces="drive",
                            fields="files(id, name, webViewLink, size)",
                            pageSize=10).execute()
    files = resp.get("files", [])
    return files[0] if files else None


def upload(svc, path: str, folder_id: str, *,
           chunk_mb: int = 8,
           description: Optional[str] = None,
           skip_if_present: bool = True) -> Dict[str, Any]:
    """Resumable-upload one file into `folder_id`. Returns the Drive metadata."""
    _, _, _, _, MediaFileUpload = _require_libs()
    if not os.path.isfile(path):
        raise DriveError(f"not a file: {path}")
    name = os.path.basename(path)

    if skip_if_present:
        existing = find_file(svc, name, folder_id)
        if existing:
            log.info("already on Drive, skipping upload: %s (%s)",
                     name, existing["id"])
            return existing

    mime, _ = mimetypes.guess_type(path)
    body: Dict[str, Any] = {"name": name, "parents": [folder_id]}
    if description:
        # Drive caps descriptions; keep well under it.
        body["description"] = description[:4000]

    size = os.path.getsize(path)
    media = MediaFileUpload(path, mimetype=mime or "application/octet-stream",
                            chunksize=max(1, chunk_mb) * 1024 * 1024,
                            resumable=True)
    request = svc.files().create(body=body, media_body=media,
                                 fields="id, name, webViewLink, size")
    log.info("uploading %s (%.1f MB)", name, size / 1e6)
    response = None
    last_pct = -10
    while response is None:
        # num_retries gives the client library its own backoff for 5xx and
        # rate-limit responses, which is the overwhelming majority of what goes
        # wrong mid-upload.
        status, response = request.next_chunk(num_retries=5)
        if status:
            pct = int(status.progress() * 100)
            if pct - last_pct >= 10:
                log.info("  %s: %d%%", name, pct)
                last_pct = pct
    log.info("uploaded %s -> %s", name, response.get("id"))
    return response


def target_folder(svc, cfg: PipelineConfig, creator: Optional[str]) -> str:
    """Resolve the folder a job's files belong in, creating it if needed."""
    if cfg.drive_folder_id:
        root = cfg.drive_folder_id
    else:
        root = ensure_folder(svc, cfg.drive_folder_name)
    if cfg.drive_subfolder_per_creator and creator:
        safe = re.sub(r"[/\\]", "-", creator).strip() or "unknown"
        return ensure_folder(svc, safe, root)
    return root
