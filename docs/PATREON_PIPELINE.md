# Patreon → Google Drive pipeline

Gmail notification arrives → the post is queued → a worker downloads it with
yt-dlp → the file lands in Google Drive, where Gemini can read it.

Unrelated to the trading strategy. It lives here because it runs on the same
mini-PC under the same systemd conventions, and a second repo for ~1,500 lines
would cost more than it saves.

## How it fits together

```
Gmail (IMAP IDLE)          patreon-watch.service ─┐
                                                  ├─→  jobs table (SQLite)
campaign sweep (6-hourly)  patreon-sweep.timer  ──┘           │
                                                              ▼
                                            patreon-work.service
                                          yt-dlp ──→ staging ──→ Drive
```

The queue in the middle is the whole design. The watcher only writes rows; a
single worker drains them. That is what gives you serialised downloads (rather
than three at once when a creator posts a batch), retry with backoff, recovery
from a mid-download reboot, and something you can inspect with `sqlite3` when a
post does not show up.

Two paths feed the queue because the fast one is not the reliable one. Email
triggering gets a post within seconds, but it breaks quietly — an edited
filter, a change to Patreon's notification format, a week of downtime. The
6-hourly campaign sweep catches whatever email missed. `enqueue()` is an upsert
on the canonical post URL, so the two paths finding the same post costs
nothing.

## First-time setup

### 1. Install

```bash
sudo useradd -r -m -d /home/patreon patreon
sudo mkdir -p /opt/qbs /var/lib/patreon /etc/patreon
sudo chown -R patreon:patreon /var/lib/patreon
git clone <this repo> /opt/qbs && cd /opt/qbs
python3 -m venv .venv && .venv/bin/pip install -r requirements-patreon.txt
```

### 2. Log into Patreon, once, by hand

As the `patreon` user, open Firefox and log in. That is the entire auth story
for Patreon — the session cookies are long-lived and yt-dlp reads them live
from the profile with `--cookies-from-browser firefox`.

**Do not automate this with Selenium.** Patreon's login sits behind Cloudflare
bot detection plus email OTP device verification; driving it programmatically
means fighting a system built to spot exactly that, and repeated automated
login attempts are a good way to get an account flagged. Logging in by hand
every few months is less work than maintaining the alternative. The
`patreon-probe` timer below tells you which month.

Firefox specifically: Chrome's cookie store is encrypted against the desktop
keyring on Linux and locked while the browser runs.

### 3. Gmail app password

Google Account → Security → 2-Step Verification → App passwords. Put it in
`/etc/patreon/patreon.env` as `PATREON_IMAP_PASSWORD`, `chmod 600`.

Not the Gmail API — its read scopes are *restricted*, so an unverified personal
project is stuck in "Testing" publishing status, where refresh tokens expire
after seven days. IMAP with an app password does not expire.

Then add a Gmail filter: from `patreon.com`, subject contains "posted", apply
label `Patreon`. Point `imap_folder` at that label so the watcher is not
reading your whole inbox.

### 4. Google Drive OAuth

In the Google Cloud console: new project → enable the Drive API → OAuth
consent screen → **set publishing status to "In production"** → Credentials →
create an OAuth client ID of type **Desktop app** → download the JSON to
`/var/lib/patreon/client_secret.json`.

Then, on the mini-PC, with a browser:

```bash
sudo -u patreon /opt/qbs/.venv/bin/python -m patreon_pipeline.runner auth
```

Over SSH, add `--console`.

Two things here bite people:

- **Leave the consent screen in "Testing" and your refresh token dies after
  seven days.** The uploads stop and nothing says why. "In production" is the
  fix, and the `drive.file` scope this uses is non-sensitive, so it needs no
  verification review.
- **`drive.file` cannot see folders you created by hand in the Drive web UI.**
  Pasting such a folder's ID into `drive_folder_id` gives a 404 that reads like
  a permissions bug. Let the pipeline create its own folder by name (the
  default, `drive_folder_name`). It is an ordinary Drive folder once created —
  visible, shareable, and readable by Gemini.

### 5. Configure and start

```bash
sudo cp deploy/patreon/patreon.example.json /etc/patreon/config.json
sudo cp deploy/patreon/patreon.env.example  /etc/patreon/patreon.env
sudo chmod 600 /etc/patreon/patreon.env
# edit both: campaign URLs, Gmail address, app password, probe URL

sudo cp deploy/systemd/patreon-*.service deploy/systemd/patreon-*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now patreon-watch patreon-work
sudo systemctl enable --now patreon-sweep.timer patreon-probe.timer
```

Verify before trusting it:

```bash
sudo -u patreon /opt/qbs/.venv/bin/python -m patreon_pipeline.runner \
  add https://www.patreon.com/posts/12345678
sudo -u patreon /opt/qbs/.venv/bin/python -m patreon_pipeline.runner status
journalctl -u patreon-work -f
```

## Day to day

```bash
runner status                 # queue counts, recent jobs, session warnings
runner status --state failed  # what needs a human
runner add URL ...            # queue a post by hand
runner sweep --limit 50       # catch up after downtime
runner retry                  # move failed jobs back to the queue
runner probe URL              # is the Patreon session still good?
```

## When the Patreon session expires

This is the failure mode you will actually hit, and the pipeline is built
around it. An expired cookie fails *every* post identically, so treating it as
a normal per-job failure would march the whole queue into `failed` in minutes
while the real problem — a login — went unnoticed.

Instead: `AuthError` puts the job back on the queue **without spending an
attempt**, sets a flag that `runner status` prints in red, and exits with code
3. systemd restarts the worker every five minutes; each restart costs one
failed request until you log back in, and then it resumes on its own with
nothing lost.

The `patreon-probe` timer checks this daily against one patron-only post, so
you find out the day it happens rather than the day you go looking for a post
that never arrived.

## Notes

- **Keep yt-dlp updated.** It is the one dependency that should move —
  extractor fixes land within days of Patreon or Vimeo changing something. A
  stale yt-dlp is the most common cause of "this post is not available".
  `pip install -U yt-dlp` on a schedule.
- **Transcripts instead of video.** Set `audio_only: true` and an hour of
  content goes from ~1 GB to ~30 MB. Whisper on the audio is the natural next
  step; it is not built here yet.
- **Untrusted input.** Notification emails are parsed as untrusted: sender
  domain is checked on a label boundary (so `evilpatreon.com` is not
  `patreon.com`), links are only followed when already on a patreon.com host,
  and the URL that comes back out of the redirect chain has to still be on
  patreon.com before it is queued.
- **Scope.** Downloading what you pay for, as a personal archive, is the use
  case here. Patreon's ToS is not enthusiastic about bulk downloading, and
  redistributing anything you pull is a separate matter entirely.

## Tests

```bash
python -m pytest tests/test_patreon_pipeline.py -q
# or, with no pytest installed:
python tests/test_patreon_pipeline.py
```

No network, no Gmail, no Drive, no yt-dlp — they cover URL canonicalisation
(the dedupe key), queue semantics under retry, and the error classification
that decides whether a failure costs a job its attempts or stops the worker.
