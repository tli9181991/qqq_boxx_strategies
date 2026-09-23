"""Turn Patreon notification emails into queued jobs.

The parsing half of this file is deliberately pure -- no sockets -- because it
is where the bugs live and it is the only part worth unit-testing. The IMAP
half is a reconnect loop around it.

Why IDLE and not polling
------------------------
IMAP IDLE is a push: the server tells us a message landed, typically within a
second or two. Polling every minute would work, but it is sixty times the
connections for a worse latency, and Gmail starts throttling logins long before
it throttles a held-open idle socket.

The cursor is a UID, not a flag
-------------------------------
Marking messages read or moving them would fight with the user's own inbox --
you would open Gmail to find the pipeline had been marking things for you. So
we store the highest UID processed and only ever look past it. The one hazard
is UIDVALIDITY: if Gmail ever changes it, every stored UID is meaningless, so
we detect that and restart the cursor from the folder's current end rather than
re-downloading a year of posts.

On trusting the email
---------------------
A notification is an untrusted input -- anyone who can get a message into the
inbox can put links in it. So links are only followed when the host is already
on patreon.com, and the URL that comes back out of the redirect chain must
still be on patreon.com before it is queued. Without that check, a spoofed
"new post" email would have the box fetching arbitrary URLs on demand.
"""

from __future__ import annotations

import email
import html
import logging
import re
import time
import urllib.error
import urllib.request
from email.message import Message
from email.utils import parseaddr
from typing import Callable, Iterable, List, Optional, Sequence, Set

from . import store
from .config import PipelineConfig

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) patreon-pipeline/1.0"

# Any http(s) URL. Applied to both the HTML and plain-text parts.
_URL_RE = re.compile(r"""https?://[^\s"'<>)\]]+""", re.IGNORECASE)

# The post id is the final run of digits in the path. Patreon serves the same
# post under several shapes -- /posts/12345678, /posts/friday-recap-12345678,
# and /CreatorName/posts/friday-recap-12345678 (which is what a campaign
# listing returns) -- and all of them must collapse to the same key or the
# dedupe does nothing and the sweep re-downloads the archive every run.
#
# The leading segments are optional and bounded rather than open-ended: real
# URLs carry at most a vanity name, or a /c/ or /cw/ prefix plus the vanity.
_POST_PATH_RE = re.compile(r"^(?:/[\w-]+){0,2}/posts/(?:.*?-)?(\d+)/?$")

_HOST_RE = re.compile(r"^https?://([^/:?#]+)", re.IGNORECASE)


def url_host(url: str) -> str:
    m = _HOST_RE.match(url.strip())
    return m.group(1).lower() if m else ""


def host_matches(host: str, domains: Sequence[str]) -> bool:
    """True when `host` is one of `domains` or a subdomain of one.

    Suffix matching is done on a label boundary on purpose: a plain
    `endswith("patreon.com")` would also accept `evilpatreon.com`.
    """
    host = host.lower().strip(".")
    for d in domains:
        d = d.lower().strip(".")
        if host == d or host.endswith("." + d):
            return True
    return False


def canonical_post_url(url: str) -> Optional[str]:
    """Normalise a Patreon post URL, or return None if it is not one.

    Strips query strings (Patreon appends utm_* to every link in an email) and
    reduces slug variants to the numeric id.
    """
    url = html.unescape(url.strip())
    host = url_host(url)
    if not host_matches(host, ["patreon.com"]):
        return None
    # Path without query or fragment.
    rest = url[len(_HOST_RE.match(url).group(0)):] if _HOST_RE.match(url) else ""
    path = rest.split("?", 1)[0].split("#", 1)[0]
    if not path.startswith("/"):
        path = "/" + path
    m = _POST_PATH_RE.match(path)
    if m:
        return f"https://www.patreon.com/posts/{m.group(1)}"
    return None


def post_id_of(canonical_url: str) -> Optional[str]:
    m = re.search(r"/posts/(\d+)$", canonical_url)
    return m.group(1) if m else None


def extract_urls(text: str) -> List[str]:
    """All http(s) URLs in a chunk of text or HTML, de-duplicated, in order."""
    if not text:
        return []
    text = html.unescape(text)
    seen: Set[str] = set()
    out: List[str] = []
    for raw in _URL_RE.findall(text):
        u = raw.rstrip(".,;:!)\"'")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def message_text(msg: Message) -> str:
    """Flatten a message to one string: every text/plain and text/html part.

    Both parts are kept rather than preferring one. Patreon's HTML and plain
    alternatives do not always carry the same links, and concatenating is
    cheaper than deciding which to trust.
    """
    chunks: List[str] = []
    for part in msg.walk():
        if part.get_content_maintype() != "text":
            continue
        if part.get_content_subtype() not in ("plain", "html"):
            continue
        try:
            payload = part.get_payload(decode=True)
        except Exception:
            continue
        if payload is None:
            continue
        charset = part.get_content_charset() or "utf-8"
        try:
            chunks.append(payload.decode(charset, errors="replace"))
        except LookupError:
            chunks.append(payload.decode("utf-8", errors="replace"))
    return "\n".join(chunks)


def sender_allowed(msg: Message, domains: Sequence[str]) -> bool:
    _, addr = parseaddr(msg.get("From", ""))
    if "@" not in addr:
        return False
    return host_matches(addr.rsplit("@", 1)[1], domains)


def resolve_tracking_url(url: str, *, timeout: float = 20.0) -> List[str]:
    """Follow a Patreon click-tracker to whatever it points at.

    Returns every candidate URL learned: the final URL after redirects, plus
    any URLs found in the response body (some trackers land on an interstitial
    that redirects with JavaScript, where the real link is only in the HTML).

    Refuses to fetch anything not already on patreon.com. The caller has taken
    this URL out of an email, and an email is not a trustworthy source of
    things to go and fetch.
    """
    if not host_matches(url_host(url), ["patreon.com"]):
        return []
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            final = resp.geturl()
            body = resp.read(65536).decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log.debug("could not resolve %s: %s", url, exc)
        return []
    out = [final]
    out.extend(extract_urls(body))
    return out


def post_urls_from_message(
        msg: Message,
        *,
        domains: Sequence[str] = ("patreon.com",),
        resolver: Optional[Callable[[str], List[str]]] = None) -> List[str]:
    """Canonical post URLs found in one notification email.

    Direct links are taken as-is. Anything else on a patreon.com host is
    treated as a possible tracker and resolved, which costs one request per
    unique link but is the only way to see through `click.patreon.com`.
    """
    if not sender_allowed(msg, domains):
        return []
    resolver = resolver if resolver is not None else resolve_tracking_url

    found: List[str] = []
    seen: Set[str] = set()

    def keep(u: str) -> None:
        c = canonical_post_url(u)
        if c and c not in seen:
            seen.add(c)
            found.append(c)

    candidates = extract_urls(message_text(msg))
    for url in candidates:
        keep(url)

    if found:
        return found

    # No direct post link: the email used trackers. Resolve the patreon.com
    # ones, newest-looking first, and stop as soon as something resolves --
    # a notification is about one post, and every link in it leads there.
    for url in candidates:
        if canonical_post_url(url):
            continue
        if not host_matches(url_host(url), ["patreon.com"]):
            continue
        for resolved in resolver(url):
            keep(resolved)
        if found:
            break
    return found


# ----------------------------------------------------------------------
# IMAP
# ----------------------------------------------------------------------

def _process_uids(client, conn, cfg: PipelineConfig, uids: Sequence[int]) -> int:
    """Fetch, parse and enqueue a batch of UIDs. Returns jobs created."""
    if not uids:
        return 0
    created = 0
    # BODY.PEEK[] rather than BODY[]: fetching must not mark the user's mail
    # read. The pipeline is a bystander in this mailbox.
    response = client.fetch(list(uids), ["BODY.PEEK[]"])
    for uid in sorted(response):
        raw = response[uid].get(b"BODY[]")
        if not raw:
            continue
        try:
            msg = email.message_from_bytes(raw)
        except Exception as exc:
            log.warning("uid %s: unparseable message: %s", uid, exc)
            continue
        subject = (msg.get("Subject") or "").strip()
        urls = post_urls_from_message(msg, domains=cfg.sender_domains)
        if not urls:
            log.debug("uid %s: no post link (%s)", uid, subject[:80])
            continue
        for url in urls:
            job_id, is_new = store.enqueue(
                conn, url, source="email", post_id=post_id_of(url),
                title=subject or None)
            if is_new:
                created += 1
                log.info("queued job %s from uid %s: %s", job_id, uid, url)
            else:
                log.debug("already known: %s", url)
    return created


def check_now(client, conn, cfg: PipelineConfig) -> int:
    """Look for messages past the stored cursor and enqueue them.

    Returns the number of new jobs. Safe to call as often as you like: the
    cursor only moves forward and `enqueue` is idempotent.
    """
    folder = cfg.imap_folder
    info = client.select_folder(folder)
    uidvalidity = int(info.get(b"UIDVALIDITY", 0))
    state = store.get_mail_state(conn, folder)

    if state["uidvalidity"] and state["uidvalidity"] != uidvalidity:
        # Every stored UID now refers to a different message, or to none.
        # Restarting at the current end is the safe choice: the alternative is
        # re-queuing the entire mailbox. The cron sweep covers anything missed
        # in the gap.
        log.warning("UIDVALIDITY changed for %s (%s -> %s); restarting cursor "
                    "at the current end of the folder",
                    folder, state["uidvalidity"], uidvalidity)
        state["last_uid"] = 0

    last_uid = state["last_uid"]
    if last_uid == 0 and state["uidvalidity"] != uidvalidity:
        uids = client.search(["ALL"])
        highest = max(uids) if uids else 0
        store.set_mail_state(conn, folder, uidvalidity, highest)
        log.info("watching %s from uid %s onward", folder, highest)
        return 0

    # "N:*" always returns at least one message even when none are new, so the
    # explicit filter below is load-bearing, not belt-and-braces.
    uids = [u for u in client.search([f"UID {last_uid + 1}:*"]) if u > last_uid]
    created = _process_uids(client, conn, cfg, uids)
    if uids:
        store.set_mail_state(conn, folder, uidvalidity, max(uids))
    return created


def watch(cfg: PipelineConfig, *, once: bool = False) -> int:
    """Hold an IDLE connection open and enqueue posts as notifications land.

    Reconnects with backoff on any network-shaped error. A home connection
    drops; this loop treating that as routine is the difference between a
    pipeline that runs for months and one you restart by hand.
    """
    try:
        from imapclient import IMAPClient
    except ImportError as exc:
        raise RuntimeError(
            "imapclient is not installed: pip install -r "
            "requirements-patreon.txt") from exc

    backoff = 5.0
    db_path = cfg.resolved_db_path()

    while True:
        try:
            with IMAPClient(cfg.imap_host, port=cfg.imap_port, ssl=True,
                            timeout=60) as client:
                client.login(cfg.imap_user, cfg.imap_password)
                log.info("connected to %s as %s", cfg.imap_host, cfg.imap_user)
                backoff = 5.0
                with store.connect(db_path) as conn:
                    check_now(client, conn, cfg)
                    if once:
                        return 0
                    while True:
                        client.idle()
                        try:
                            responses = client.idle_check(
                                timeout=cfg.imap_idle_seconds)
                        finally:
                            client.idle_done()
                        if responses:
                            log.debug("idle woke: %s", responses)
                            check_now(client, conn, cfg)
        except KeyboardInterrupt:
            log.info("watcher stopped")
            return 0
        except Exception as exc:
            log.error("imap error (%s: %s); reconnecting in %.0fs",
                      type(exc).__name__, exc, backoff)
            if once:
                return 1
            time.sleep(backoff)
            backoff = min(backoff * 2, 300.0)
