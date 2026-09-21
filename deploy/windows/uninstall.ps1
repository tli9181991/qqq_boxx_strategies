<#
.SYNOPSIS
    Remove the Patreon pipeline's Scheduled Tasks, and optionally its data.

.DESCRIPTION
    The counterpart to install.ps1. By default it does the safe half only:
    stops and unregisters the four Scheduled Tasks, so the pipeline stops
    running, and touches nothing else.

    That default is deliberate. "Uninstall" almost always means "make it stop",
    and the data it leaves behind is the part you cannot get back cheaply:
    pipeline.db is the record of every post already downloaded, and deleting it
    means the next sweep re-downloads the lot. Everything destructive is
    opt-in, prompts before acting, and honours -WhatIf.

    Nothing here touches the repository, and nothing touches files already
    uploaded to Drive.

.PARAMETER StateDir
    Where the pipeline keeps its data. Must match what install.ps1 used.

.PARAMETER RemoveState
    Delete the queue database, staging files, logs and config.json. This is
    the one that loses the download history.

.PARAMETER RemoveModels
    Delete the cached Whisper models (~1.5 GB). Safe -- they re-download on
    demand -- but slow to undo on a home connection.

.PARAMETER RemoveVenv
    Delete the virtualenv in the repo.

.PARAMETER RemoveCredentials
    Delete patreon.env (which holds the Gmail app password) and any leftover
    Google Drive API client secret or token.

.PARAMETER RemoveRcloneRemote
    Delete the pipeline's rclone remote from rclone's own config. Leaves every
    other remote alone -- rclone is often used for unrelated things.

.PARAMETER All
    Everything above.

.PARAMETER Force
    Skip confirmation prompts. For scripted teardown.

.EXAMPLE
    .\uninstall.ps1
    Stop the pipeline. Keeps all data.

.EXAMPLE
    .\uninstall.ps1 -All -WhatIf
    Show exactly what a full teardown would delete, without deleting anything.

.EXAMPLE
    .\uninstall.ps1 -All -Force
    Full teardown, no prompts.
#>
[CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'High')]
param(
    [string]$StateDir = 'C:\ProgramData\PatreonPipeline',
    [switch]$RemoveState,
    [switch]$RemoveModels,
    [switch]$RemoveVenv,
    [switch]$RemoveCredentials,
    [switch]$RemoveRcloneRemote,
    [switch]$All,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

if ($All) {
    $RemoveState = $true
    $RemoveModels = $true
    $RemoveVenv = $true
    $RemoveCredentials = $true
    $RemoveRcloneRemote = $true
}

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$TaskNames = @(
    'PatreonPipeline-Watch',
    'PatreonPipeline-Work',
    'PatreonPipeline-Sweep',
    'PatreonPipeline-Probe'
)

$removed = [System.Collections.Generic.List[string]]::new()
$kept    = [System.Collections.Generic.List[string]]::new()

function Confirm-Step {
    param([string]$Message)
    if ($Force) { return $true }
    $answer = Read-Host "$Message [y/N]"
    return $answer -match '^(y|yes)$'
}

function Remove-Tree {
    param([string]$Path, [string]$What)
    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Verbose "not present: $Path"
        return $false
    }
    if ($PSCmdlet.ShouldProcess($Path, "Delete $What")) {
        Remove-Item -LiteralPath $Path -Recurse -Force
        $script:removed.Add("$What  ($Path)")
        return $true
    }
    return $false
}

Write-Host "Repo:  $RepoRoot"
Write-Host "State: $StateDir"
Write-Host ""

# ---------------------------------------------------------------------------
# 1. Scheduled Tasks -- always, and first
# ---------------------------------------------------------------------------
# Before anything is deleted: a running worker holds the SQLite file open, and
# on Windows an open handle makes the delete fail rather than succeed quietly.
Write-Host "Scheduled Tasks"
$foundAny = $false
foreach ($name in $TaskNames) {
    $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if (-not $task) {
        Write-Host "  $name  - not registered"
        continue
    }
    $foundAny = $true
    if ($PSCmdlet.ShouldProcess($name, 'Stop and unregister scheduled task')) {
        Stop-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
        Write-Host "  $name  - removed"
        $removed.Add("scheduled task $name")
    }
}
if (-not $foundAny) {
    Write-Host "  (none were registered)"
}

# Give a worker mid-upload a moment to notice it has been stopped.
if ($foundAny -and ($RemoveState -or $RemoveModels)) {
    Start-Sleep -Seconds 2
}

# ---------------------------------------------------------------------------
# 2. Lingering processes
# ---------------------------------------------------------------------------
# Matched on the command line rather than the image name, so this can never
# catch an unrelated Python you are running.
$lingering = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -like '*patreon_pipeline*' })

if ($lingering.Count -gt 0) {
    Write-Host ""
    Write-Host "Still running:"
    foreach ($proc in $lingering) {
        Write-Host "  PID $($proc.ProcessId)  $($proc.Name)"
    }
    if ($RemoveState -or $RemoveModels -or $RemoveVenv) {
        if (Confirm-Step "Stop these $($lingering.Count) pipeline process(es)?") {
            foreach ($proc in $lingering) {
                if ($PSCmdlet.ShouldProcess("PID $($proc.ProcessId)", 'Stop process')) {
                    Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
                }
            }
            Start-Sleep -Seconds 1
        } else {
            Write-Warning "Leaving them running. Deletes below may fail on open files."
        }
    }
}

# ---------------------------------------------------------------------------
# 3. State
# ---------------------------------------------------------------------------
# A recursive delete driven by a parameter deserves a sanity check: refuse a
# path that does not look like ours, so a mistyped -StateDir cannot take out a
# drive root or a Documents folder.
$stateLooksRight = $StateDir -and
                   (Split-Path -Leaf $StateDir) -match 'Patreon' -and
                   $StateDir.TrimEnd('\').Length -gt 3

if (($RemoveState -or $RemoveModels -or $RemoveCredentials) -and -not $stateLooksRight) {
    throw ("Refusing to delete anything under '$StateDir': it does not look " +
           "like a PatreonPipeline state directory. Pass the same -StateDir " +
           "you gave install.ps1.")
}

if ($RemoveModels) {
    Write-Host ""
    Write-Host "Whisper models"
    $models = Join-Path $StateDir 'whisper-models'
    if (Test-Path -LiteralPath $models) {
        $size = (Get-ChildItem -LiteralPath $models -Recurse -File -ErrorAction SilentlyContinue |
                 Measure-Object -Property Length -Sum).Sum
        $gb = if ($size) { [math]::Round($size / 1GB, 2) } else { 0 }
        Write-Host "  $models  ($gb GB)"
        if (Confirm-Step "  Delete the cached models? They re-download on demand") {
            [void](Remove-Tree -Path $models -What 'Whisper model cache')
        } else {
            $kept.Add("Whisper model cache  ($models)")
        }
    } else {
        Write-Host "  (not present)"
    }
}

if ($RemoveState) {
    Write-Host ""
    Write-Host "Queue, logs and config"
    $db = Join-Path $StateDir 'pipeline.db'
    if (Test-Path -LiteralPath $db) {
        Write-Warning ("pipeline.db is the record of every post already " +
                       "downloaded. Delete it and the next sweep will " +
                       "re-download everything it finds.")
    }
    if (Confirm-Step "  Delete the queue database, staging, logs and config?") {
        foreach ($leaf in @('pipeline.db', 'pipeline.db-wal', 'pipeline.db-shm',
                            'staging', 'logs', 'config.json')) {
            [void](Remove-Tree -Path (Join-Path $StateDir $leaf) -What $leaf)
        }
        # Only remove the directory itself once it is empty -- models or
        # credentials the user chose to keep still live here.
        $left = @(Get-ChildItem -LiteralPath $StateDir -Force -ErrorAction SilentlyContinue)
        if ($left.Count -eq 0) {
            [void](Remove-Tree -Path $StateDir -What 'state directory')
        } else {
            Write-Host "  keeping $StateDir ($($left.Count) item(s) still in it)"
        }
    } else {
        $kept.Add("queue database and logs  ($StateDir)")
    }
}

# ---------------------------------------------------------------------------
# 4. Credentials
# ---------------------------------------------------------------------------
if ($RemoveCredentials) {
    Write-Host ""
    Write-Host "Credentials"
    $envFile = Join-Path $PSScriptRoot 'patreon.env'
    foreach ($item in @(
        @{ Path = $envFile; What = 'patreon.env (Gmail app password)' },
        @{ Path = (Join-Path $StateDir 'client_secret.json'); What = 'Drive API client secret' },
        @{ Path = (Join-Path $StateDir 'drive_token.json');   What = 'Drive API token' }
    )) {
        if (Test-Path -LiteralPath $item.Path) {
            Write-Host "  $($item.What)"
            [void](Remove-Tree -Path $item.Path -What $item.What)
        }
    }
}

if ($RemoveRcloneRemote) {
    Write-Host ""
    Write-Host "rclone remote"
    $rclone = Get-Command rclone -ErrorAction SilentlyContinue
    if (-not $rclone) {
        Write-Host "  rclone is not on PATH - nothing to do"
    } else {
        # Read the remote name from patreon.env when it is still there, so a
        # non-default name is honoured rather than silently missed.
        $remote = 'gdrive'
        $envFile = Join-Path $PSScriptRoot 'patreon.env'
        if (Test-Path -LiteralPath $envFile) {
            $line = Select-String -LiteralPath $envFile -Pattern '^\s*PATREON_RCLONE_REMOTE\s*=' -ErrorAction SilentlyContinue
            if ($line) { $remote = ($line.Line -split '=', 2)[1].Trim() }
        }
        $existing = & $rclone.Source listremotes 2>$null
        if ($existing -contains "${remote}:") {
            if ($PSCmdlet.ShouldProcess($remote, 'Delete rclone remote')) {
                & $rclone.Source config delete $remote
                Write-Host "  removed remote '$remote'"
                $removed.Add("rclone remote '$remote'")
            }
        } else {
            Write-Host "  no remote named '$remote' - nothing to do"
        }
    }
}

# ---------------------------------------------------------------------------
# 5. Virtualenv
# ---------------------------------------------------------------------------
if ($RemoveVenv) {
    Write-Host ""
    Write-Host "Virtualenv"
    $venv = Join-Path $RepoRoot '.venv'
    if (Test-Path -LiteralPath $venv) {
        Write-Host "  $venv"
        if (Confirm-Step "  Delete it? Other things in this repo may use it") {
            [void](Remove-Tree -Path $venv -What 'virtualenv')
        } else {
            $kept.Add("virtualenv  ($venv)")
        }
    } else {
        Write-Host "  (not present)"
    }
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "-------------------------------------------------------------"
if ($removed.Count -gt 0) {
    Write-Host "Removed:"
    foreach ($item in $removed) { Write-Host "  - $item" }
} else {
    Write-Host "Nothing was removed."
}

if ($kept.Count -gt 0) {
    Write-Host ""
    Write-Host "Kept (you declined):"
    foreach ($item in $kept) { Write-Host "  - $item" }
}

if (-not $All) {
    Write-Host ""
    Write-Host "Left alone by default. Add the flag to remove each:"
    if (-not $RemoveState)       { Write-Host "  -RemoveState        queue database, staging, logs, config" }
    if (-not $RemoveModels)      { Write-Host "  -RemoveModels       cached Whisper models (~1.5 GB)" }
    if (-not $RemoveVenv)        { Write-Host "  -RemoveVenv         the virtualenv" }
    if (-not $RemoveCredentials) { Write-Host "  -RemoveCredentials  patreon.env and any Drive API token" }
    if (-not $RemoveRcloneRemote){ Write-Host "  -RemoveRcloneRemote the pipeline's rclone remote" }
    Write-Host "  -All                all of the above"
}

# Deleting a local copy of a credential does not revoke it. Both of these stay
# live in the Google account until they are revoked there, so say so plainly
# rather than leaving the impression that a full teardown has happened.
if ($RemoveCredentials -or $RemoveRcloneRemote -or $All) {
    Write-Host ""
    Write-Host "Still to do by hand - deleting a local file does not revoke access:"
    Write-Host "  Gmail app password:  https://myaccount.google.com/apppasswords"
    Write-Host "  Drive access:        https://myaccount.google.com/permissions"
    Write-Host "                       (revoke 'rclone', or your own OAuth app)"
}

Write-Host ""
Write-Host "The repository, and anything already uploaded to Drive, were not touched."
