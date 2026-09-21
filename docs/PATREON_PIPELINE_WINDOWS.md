# Running the Patreon pipeline on Windows

The pipeline itself is portable Python. Only the deployment layer was
Linux-specific, so this is the Windows half of
[`PATREON_PIPELINE.md`](PATREON_PIPELINE.md) — read that first for what the
thing actually does and why. This file covers installing it, and the handful of
places Windows behaves differently enough to cost you an evening.

## Native Windows, not WSL2

WSL2 would let the systemd units run unchanged, which is tempting. It is still
the worse option here:

- **Cookies.** yt-dlp on native Windows reads the Firefox profile directly.
  From inside WSL2 you have to point it at the Windows profile across
  `/mnt/c/...`, which works but is one more thing to re-find when Firefox
  updates its profile directory.
- **Boot.** WSL2 does not start at boot on its own. You end up registering a
  Windows Scheduled Task to launch it — so you need Task Scheduler anyway, and
  now you have Task Scheduler *and* systemd.
- **Disk.** Large media files either live on the WSL ext4 disk, where Windows
  cannot easily see them, or on `/mnt/c`, where I/O is markedly slower.

The only thing WSL2 genuinely buys is reusing the systemd units, and the four
Scheduled Tasks below cover the same ground. Go native.

## Prerequisites

```powershell
winget install Python.Python.3.12
winget install Gyan.FFmpeg          # both stages need it
winget install Mozilla.Firefox
```

Open a **new** terminal afterwards so PATH is picked up.

## Install

```powershell
cd <repo>\deploy\windows
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

That creates the virtualenv, installs dependencies, makes
`C:\ProgramData\PatreonPipeline\`, copies starter config, and registers four
Scheduled Tasks that mirror the systemd units one-for-one:

| Task | Schedule | Linux equivalent |
| --- | --- | --- |
| `PatreonPipeline-Watch` | at logon, restart every 1 min on failure | `patreon-watch.service` |
| `PatreonPipeline-Work` | at logon, restart every 5 min on failure | `patreon-work.service` |
| `PatreonPipeline-Sweep` | every 6 hours | `patreon-sweep.timer` |
| `PatreonPipeline-Probe` | daily at 09:00 | `patreon-probe.timer` |

Task Scheduler's restart-on-failure stands in for `Restart=always`. The
intervals differ on purpose: the worker exits 3 when the Patreon session has
expired, and retrying that every minute would just fill the log.

Then the five things the installer cannot do for you:

1. **Log into Patreon in Firefox**, on this machine, as this user. Once.
2. Edit `deploy\windows\patreon.env` — Gmail app password, probe URL.
3. Edit `C:\ProgramData\PatreonPipeline\config.json` — campaign URLs, Whisper
   vocabulary.
4. `.\run.ps1 auth` — the one-time Google Drive consent flow.
5. Stop the machine sleeping (below).

## The Windows-specific traps

### Sleep will silently stop everything

This is the one that gets people. A mini-PC left on default power settings
sleeps after 30 minutes and neither the IMAP connection nor the scheduled
sweeps survive it. Nothing errors — the pipeline simply stops existing for
hours at a time.

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

"Allow wake timers" also has to stay enabled if you want the 6-hourly sweep to
pull the box out of a low-power state.

### Firefox, not Chrome or Edge

Chrome and Edge encrypt their cookie store with Windows DPAPI, tied to the
logged-in user, and hold a lock on it while running. Extracting from them is
fragile at best. Firefox keeps `cookies.sqlite` unencrypted, so yt-dlp reads it
whether or not Firefox is open. Use Firefox for the Patreon session and leave
Chrome for everything else.

### The 260-character path limit

Windows truncates paths at 260 characters unless long-path support is on, and
yt-dlp fails *after* the download when the final filename crosses that line —
the most annoying possible moment. The pipeline trims titles to 100 bytes on
Windows (180 on Linux) to stay clear of it.

If you would rather have the longer names, enable long paths and raise the
limit:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
# then in patreon.env:  PATREON_TITLE_BYTES=180
```

### Logs, because Task Scheduler keeps none

Task Scheduler records a task's exit code and throws away everything it
printed. There is no journald. So each task writes its own rotating log (5 MB,
three generations):

```powershell
Get-Content -Wait C:\ProgramData\PatreonPipeline\logs\work.log
```

### Antivirus

Real-time scanning inspects every file as it is written, which on a
multi-gigabyte download is a measurable slowdown, and it occasionally holds a
handle open just long enough that cleanup fails. The pipeline tolerates the
second problem. For the first, exclude
`C:\ProgramData\PatreonPipeline\staging` in Defender.

### Credentials are protected by folder ACLs, not file permissions

`chmod` on Windows only toggles the read-only bit — it does not restrict other
users. The Drive token and the Gmail app password are protected by living under
a per-user profile path, which is why the default state directory is what it
is. Do not move them into a shared or synced folder.

## Day to day

Every command goes through the wrapper, which loads `patreon.env` and hands off
to the venv's Python:

```powershell
.\run.ps1 status
.\run.ps1 add https://www.patreon.com/posts/12345678
.\run.ps1 sweep --limit 50
.\run.ps1 probe https://www.patreon.com/posts/12345678
.\run.ps1 transcribe "D:\media\one-post.mp4"
```

Task control:

```powershell
Start-ScheduledTask   -TaskName PatreonPipeline-Work
Stop-ScheduledTask    -TaskName PatreonPipeline-Work
Get-ScheduledTaskInfo -TaskName PatreonPipeline-Work   # last result, last run
```

`LastTaskResult` of `3` means the Patreon session expired: log into Patreon in
Firefox again and the worker picks up by itself on its next restart, with
nothing lost from the queue.

## Benchmark Whisper before trusting it

```powershell
.\run.ps1 transcribe "D:\media\one-real-post.mp4"
```

It prints a real-time multiple. Windows makes no difference to inference speed
— the same advice as on Linux applies: if `large-v3-turbo` is much slower than
1x real time on your CPU, drop `whisper_model` to `small`, and if the box has a
usable GPU, `cuda` with `float16` makes the question go away. On a GPU you will
also need the CUDA runtime libraries that CTranslate2 expects; on CPU there is
nothing extra to install.
