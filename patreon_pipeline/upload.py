"""One seam over the two upload backends.

The worker should not know or care which one is configured, so both are
reduced to the same two operations: work out where a job's files belong, and
send a file there.

Two backends because they fail in different places:

* **rclone** (the default) authorises against rclone's own registered OAuth
  client. No Cloud project, no consent screen, no verification, no seven-day
  token expiry. It needs the `rclone` binary on the box.
* **drive** talks to the Drive API with an OAuth client you register yourself.
  No extra binary, but publishing the consent screen means supplying a
  homepage and privacy policy on a domain you control, and leaving it in
  "Testing" means re-authorising every seven days.

`drive` is kept rather than deleted because it is already working for anyone
who got through that setup, and because a pure-Python path is worth having if
rclone is ever unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .config import PipelineConfig

log = logging.getLogger(__name__)

BACKENDS = ("rclone", "drive")


class UploadError(RuntimeError):
    pass


class _Backend:
    """What the worker is allowed to assume about an uploader."""

    name = "?"

    def check(self) -> None:
        """Raise UploadError if this backend cannot work. Called once."""
        raise NotImplementedError

    def destination(self, creator: Optional[str]) -> Any:
        """An opaque handle for where this job's files go."""
        raise NotImplementedError

    def send(self, path: str, dest: Any,
             description: Optional[str] = None) -> Dict[str, Any]:
        """Upload one file. Returns metadata carrying at least a name, and an
        `id` and `webViewLink` where the backend can supply them."""
        raise NotImplementedError


class _Rclone(_Backend):
    name = "rclone"

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

    def check(self) -> None:
        from . import rclone
        try:
            rclone.check(self.cfg)
        except rclone.RcloneError as exc:
            raise UploadError(str(exc)) from exc

    def destination(self, creator: Optional[str]) -> str:
        from . import rclone
        return rclone.destination(self.cfg, creator)

    def send(self, path, dest, description=None):
        from . import rclone
        try:
            return rclone.upload(self.cfg, path, dest, description=description)
        except rclone.RcloneError as exc:
            raise UploadError(str(exc)) from exc


class _Drive(_Backend):
    name = "drive"

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._svc = None

    def check(self) -> None:
        from . import drive
        try:
            self._svc = drive.service(self.cfg)
        except drive.DriveError as exc:
            raise UploadError(str(exc)) from exc

    def destination(self, creator: Optional[str]) -> str:
        from . import drive
        if self._svc is None:
            self.check()
        return drive.target_folder(self._svc, self.cfg, creator)

    def send(self, path, dest, description=None):
        from . import drive
        if self._svc is None:
            self.check()
        try:
            return drive.upload(self._svc, path, dest,
                                chunk_mb=self.cfg.drive_chunk_mb,
                                description=description)
        except drive.DriveError as exc:
            raise UploadError(str(exc)) from exc


def backend(cfg: PipelineConfig) -> _Backend:
    """Build the configured upload backend."""
    if cfg.uploader == "rclone":
        return _Rclone(cfg)
    if cfg.uploader == "drive":
        return _Drive(cfg)
    raise UploadError(
        f"unknown uploader {cfg.uploader!r}; expected one of {list(BACKENDS)}")
