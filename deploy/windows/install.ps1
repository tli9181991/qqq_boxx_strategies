<#
.SYNOPSIS
    Install the Patreon pipeline on Windows: virtualenv, folders, scheduled tasks.

.DESCRIPTION
    The Windows counterpart to the systemd units in deploy\systemd. Four tasks,
    mapping one-to-one onto the four Linux units:

      PatreonPipeline-Watch   at logon, restarts on failure  (patreon-watch.service)
      PatreonPipeline-Work    at logon, restarts on failure  (patreon-work.service)
      PatreonPipeline-Sweep   every 6 hours                  (patreon-sweep.timer)
      PatreonPipeline-Probe   daily at 09:00                 (patreon-probe.timer)

    Task Scheduler's restart-on-failure is the stand-in for systemd's
    Restart=always. The intervals match: one minute for the watcher, five for
    the worker, because the worker's exit code 3 means "the Patreon session
    expired" and retrying that every minute just fills the log.

.PARAMETER LogonType
    Interactive (default) runs the tasks only while you are logged in, and
    needs no stored password. That suits a home mini-PC with automatic logon.
    S4U runs them whether or not you are logged on, without storing a password
    either, but a task running that way has no access to network locations.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\install.ps1
    powershell -ExecutionPolicy Bypass -File .\install.ps1 -LogonType S4U
#>
[CmdletBinding()]
param(
    [ValidateSet('Interactive', 'S4U')]
    [string]$LogonType = 'Interactive',

    [string]$StateDir = 'C:\ProgramData\PatreonPipeline',

    [switch]$SkipTasks
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python   = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$RunPs1   = Join-Path $PSScriptRoot 'run.ps1'

Write-Host "Repo:  $RepoRoot"
Write-Host "State: $StateDir"

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------
$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command python -ErrorAction SilentlyContinue }
if (-not $py) {
    throw "No Python found. Install Python 3.10+ from python.org (tick 'Add to PATH')."
}

if (-not (Get-Command rclone -ErrorAction SilentlyContinue)) {
    Write-Warning @"
rclone is not on PATH. It is how files reach Google Drive, and it carries its
own registered OAuth client -- which is why this setup needs no Google Cloud
project at all. Install it with:  winget install Rclone.Rclone
Then run `rclone config` once to authorise Google Drive.
"@
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning @"
ffmpeg is not on PATH. Both stages need it: yt-dlp merges separate audio and
video streams with it, and Whisper is fed audio extracted with it. Install it
with:  winget install Gyan.FFmpeg
Then open a new terminal so PATH is picked up.
"@
}

# ---------------------------------------------------------------------------
# Virtualenv
# ---------------------------------------------------------------------------
if (-not (Test-Path $Python)) {
    Write-Host "Creating virtualenv..."
    & $py.Source -m venv (Join-Path $RepoRoot '.venv')
}
Write-Host "Installing dependencies (this pulls ~2 GB for faster-whisper)..."
& $Python -m pip install --upgrade pip --quiet
& $Python -m pip install -r (Join-Path $RepoRoot 'requirements-patreon.txt')

# ---------------------------------------------------------------------------
# Folders and config
# ---------------------------------------------------------------------------
foreach ($dir in @($StateDir, (Join-Path $StateDir 'staging'),
                   (Join-Path $StateDir 'whisper-models'),
                   (Join-Path $StateDir 'logs'))) {
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
}

$cfgTarget = Join-Path $StateDir 'config.json'
if (-not (Test-Path $cfgTarget)) {
    Copy-Item (Join-Path $RepoRoot 'deploy\patreon\patreon.example.json') $cfgTarget
    Write-Host "Wrote starter config to $cfgTarget -- edit it."
}
$envTarget = Join-Path $PSScriptRoot 'patreon.env'
if (-not (Test-Path $envTarget)) {
    Copy-Item (Join-Path $PSScriptRoot 'patreon.env.example') $envTarget
    Write-Host "Wrote starter env file to $envTarget -- edit it (Gmail app password)."
}

if ($SkipTasks) { Write-Host "`nSkipping task registration."; return }

# ---------------------------------------------------------------------------
# Scheduled tasks
# ---------------------------------------------------------------------------
function New-PipelineTask {
    param(
        [string]$Name,
        [string]$Description,
        [string]$CommandArgs,
        [object[]]$Triggers,
        [int]$RestartMinutes = 0,
        [switch]$Unlimited
    )

    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument (
        "-NoProfile -NonInteractive -WindowStyle Hidden " +
        "-ExecutionPolicy Bypass -File `"$RunPs1`" $CommandArgs")

    $settingsArgs = @{
        AllowStartIfOnBatteries    = $true
        DontStopIfGoingOnBatteries = $true
        StartWhenAvailable         = $true
        MultipleInstances          = 'IgnoreNew'
    }
    if ($Unlimited) {
        # A download plus a transcription can legitimately run for hours, and
        # Task Scheduler's default is to kill a task after three days. The
        # pipeline enforces its own per-stage ceilings instead.
        $settingsArgs['ExecutionTimeLimit'] = (New-TimeSpan -Seconds 0)
    }
    $settings = New-ScheduledTaskSettingsSet @settingsArgs
    if ($RestartMinutes -gt 0) {
        # The stand-in for systemd Restart=always.
        $settings.RestartInterval = (New-TimeSpan -Minutes $RestartMinutes)
        $settings.RestartCount = 999
    }

    $principal = if ($LogonType -eq 'S4U') {
        New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
            -LogonType S4U -RunLevel Limited
    } else {
        New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
            -LogonType Interactive -RunLevel Limited
    }

    Unregister-ScheduledTask -TaskName $Name -Confirm:$false -ErrorAction SilentlyContinue
    Register-ScheduledTask -TaskName $Name -Description $Description `
        -Action $action -Trigger $Triggers -Settings $settings `
        -Principal $principal | Out-Null
    Write-Host "  registered $Name"
}

Write-Host "`nRegistering scheduled tasks ($LogonType)..."

New-PipelineTask -Name 'PatreonPipeline-Watch' `
    -Description 'Patreon pipeline: Gmail IDLE watcher' `
    -CommandArgs "--logfile `"$StateDir\logs\watch.log`" watch" `
    -Triggers @((New-ScheduledTaskTrigger -AtLogOn)) `
    -RestartMinutes 1 -Unlimited

New-PipelineTask -Name 'PatreonPipeline-Work' `
    -Description 'Patreon pipeline: download, transcribe and upload queue worker' `
    -CommandArgs "--logfile `"$StateDir\logs\work.log`" work" `
    -Triggers @((New-ScheduledTaskTrigger -AtLogOn)) `
    -RestartMinutes 5 -Unlimited

New-PipelineTask -Name 'PatreonPipeline-Sweep' `
    -Description 'Patreon pipeline: campaign sweep (backstop for missed emails)' `
    -CommandArgs "--logfile `"$StateDir\logs\sweep.log`" sweep --limit 20" `
    -Triggers @((New-ScheduledTaskTrigger -Once -At (Get-Date).Date.AddMinutes(20) `
                 -RepetitionInterval (New-TimeSpan -Hours 6)))

New-PipelineTask -Name 'PatreonPipeline-Probe' `
    -Description 'Patreon pipeline: daily Patreon session health check' `
    -CommandArgs "--logfile `"$StateDir\logs\probe.log`" probe `"`$env:PATREON_PROBE_URL`"" `
    -Triggers @((New-ScheduledTaskTrigger -Daily -At 9am))

Write-Host @"

Done. Before this works end to end:

  1. Log into Patreon in Firefox, on this machine, as this user. Once.
  2. Edit $envTarget       (Gmail app password, probe URL)
  3. Edit $cfgTarget       (campaign URLs, Whisper vocabulary)
  4. Authorise Google Drive, once, with a browser:
         .\run.ps1 auth
  5. Stop the box from sleeping, or none of this runs:
         powercfg /change standby-timeout-ac 0
         powercfg /change hibernate-timeout-ac 0

Logs (Task Scheduler keeps only exit codes):

     Get-Content -Wait $StateDir\logs\work.log

Then start the two long-running tasks without logging out and back in:

     Start-ScheduledTask -TaskName PatreonPipeline-Watch
     Start-ScheduledTask -TaskName PatreonPipeline-Work

Check it:

     .\run.ps1 status
     .\run.ps1 add https://www.patreon.com/posts/12345678
"@
