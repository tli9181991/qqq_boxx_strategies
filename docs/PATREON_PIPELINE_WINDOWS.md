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
winget install Rclone.Rclone        # how files reach Drive
winget install Mozilla.Firefox
```

Open a **new** terminal afterwards so PATH is picked up.

## Install

Run this from an **elevated** PowerShell — registering a Scheduled Task in the
Task Scheduler root requires administrator rights. Right-click PowerShell →
*Run as administrator*, then:

```powershell
cd <repo>\deploy\windows
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

Elevation changes the token you hold, not which user you are, so the tasks
still register against your account and still read your Firefox profile. The
pipeline itself does not run elevated.

`-SkipTasks` does everything except registering the tasks and needs no
elevation.

That creates the virtualenv, installs dependencies, makes
`C:\ProgramData\PatreonPipeline\`, copies starter config, and registers four
Scheduled Tasks that mirror the systemd units one-for-one:

| Task | Schedule | Linux equivalent |
| --- | --- | --- |
| `PatreonPipeline-Watch` | at logon, re-checked every 5 min | `patreon-watch.service` |
| `PatreonPipeline-Work` | at logon, re-checked every 5 min | `patreon-work.service` |
| `PatreonPipeline-Sweep` | every 6 hours | `patreon-sweep.timer` |
| `PatreonPipeline-Probe` | daily at 09:00 | `patreon-probe.timer` |

The stand-in for `Restart=always` is a **repetition on the trigger**, not Task
Scheduler's restart-on-failure. Restart-on-failure only fires on what Task
Scheduler classes as a failure, which misses a process that simply went away;
a repetition fires regardless. `MultipleInstances=IgnoreNew` discards a tick
that lands while the task is still running, so a five-minute repetition reads
as "start it if it is not running".

That also sidesteps a real trap: `RestartInterval` can only be assigned onto
the settings object after it is built, and that path serialises the TimeSpan
into a form the task XML schema rejects — `Register-ScheduledTask` fails with
*"The task XML contains a value which is incorrectly formatted or out of
range"*.

Then the five things the installer cannot do for you:

1. **Log into Patreon in Firefox**, on this machine, as this user. Once.
2. Edit `deploy\windows\patreon.env` — Gmail app password, probe URL.
3. Edit `C:\ProgramData\PatreonPipeline\config.json` — campaign URLs, Whisper
   vocabulary.
4. `rclone config` — the one-time Google Drive authorisation. rclone carries
   its own registered OAuth client, so there is no Google Cloud project to
   create and no consent screen to publish. `.\run.ps1 auth` prints the exact
   answers to give its prompts.
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

## Uninstalling

```powershell
.\uninstall.ps1
```

By default that does the safe half only: stops and unregisters the four
Scheduled Tasks, and touches nothing else. "Uninstall" usually means "make it
stop", and the data left behind is the part that is expensive to get back.

Everything destructive is opt-in, prompts first, and honours `-WhatIf`:

| Flag | Removes |
| --- | --- |
| `-RemoveState` | Queue database, staging, logs, `config.json` |
| `-RemoveModels` | Cached Whisper models (~1.5 GB) |
| `-RemoveVenv` | The virtualenv |
| `-RemoveCredentials` | `patreon.env`, plus any leftover Drive API secret or token |
| `-RemoveRcloneRemote` | Only the pipeline's rclone remote, leaving your others alone |
| `-All` | All of the above |

See exactly what a full teardown would delete, without deleting anything:

```powershell
.\uninstall.ps1 -All -WhatIf
```

Two things worth knowing:

- **`pipeline.db` is the record of every post already downloaded.** Delete it
  and the next sweep re-downloads everything it finds. That is why it is not
  part of the default.
- **Deleting a local credential does not revoke it.** The Gmail app password
  and the Drive grant stay live in your Google account until you revoke them at
  [apppasswords](https://myaccount.google.com/apppasswords) and
  [permissions](https://myaccount.google.com/permissions). The script prints
  both links when it removes credentials.

The repository, and anything already uploaded to Drive, are never touched.

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
