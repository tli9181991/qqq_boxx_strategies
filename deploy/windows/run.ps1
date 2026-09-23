<#
.SYNOPSIS
    Wrapper that loads the environment file and invokes the pipeline.

.DESCRIPTION
    Windows Task Scheduler has no equivalent of systemd's EnvironmentFile, so
    this stands in for it: read patreon.env, set the variables, then hand the
    arguments to the venv's Python.

    Every scheduled task goes through here, which means there is exactly one
    place that knows where the venv and the config live.

.EXAMPLE
    .\run.ps1 status
    .\run.ps1 add https://www.patreon.com/posts/12345678
    .\run.ps1 transcribe "D:\media\one-post.mp4"
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python   = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$EnvFile  = Join-Path $PSScriptRoot 'patreon.env'

if (-not (Test-Path $Python)) {
    throw "No virtualenv at $Python. Run deploy\windows\install.ps1 first."
}

# KEY=VALUE, one per line. '#' starts a comment; blank lines are skipped.
#
# Values are trimmed, and one matching pair of surrounding quotes is removed.
# An earlier version took them literally, on the theory that guessing at
# quoting could corrupt a credential. That was backwards: a Gmail app password
# is sixteen lowercase letters and can contain neither a quote nor a space, so
# there is nothing to corrupt -- while pasting one with the spaces Google
# displays it with, or in quotes, produces AUTHENTICATIONFAILED and no clue
# why. `runner config` reports the shape of what was read.
# Say which file was read, every time. There is a patreon.env in deploy\windows
# AND one in deploy\patreon (the systemd copy), and editing the wrong one
# produces a login failure that blames the credential rather than the path.
# One dim line here answers "which file am I editing?" at a glance.
if (Test-Path $EnvFile) {
    Write-Host "env: $EnvFile" -ForegroundColor DarkGray
    foreach ($line in Get-Content -LiteralPath $EnvFile) {
        $trimmed = $line.Trim()
        if ($trimmed -eq '' -or $trimmed.StartsWith('#')) { continue }
        $idx = $trimmed.IndexOf('=')
        if ($idx -lt 1) { continue }
        $key = $trimmed.Substring(0, $idx).Trim()
        $val = $trimmed.Substring($idx + 1).Trim()
        if ($val.Length -ge 2 -and
            (($val.StartsWith('"') -and $val.EndsWith('"')) -or
             ($val.StartsWith("'") -and $val.EndsWith("'")))) {
            $val = $val.Substring(1, $val.Length - 2)
        }
        Set-Item -Path "env:$key" -Value $val
    }
} else {
    Write-Warning @"
No $EnvFile -- running on defaults and whatever is already in the environment.

  This is the file run.ps1 reads. Note there is also a patreon.env under
  deploy\patreon, which is the systemd copy and is NOT read on Windows.
  Create this one from its example:

      Copy-Item .\patreon.env.example .\patreon.env
"@
}

# Push/Pop rather than Set-Location. PowerShell's current directory belongs to
# the session, not to the script, so a bare Set-Location here leaves the
# caller's prompt somewhere it never asked to be -- you run `.\run.ps1 status`
# from deploy\windows and land in the repo root. The finally restores it
# however this exits: clean run, failure, thrown error or Ctrl-C.
#
# The directory has to change at all because patreon_pipeline is imported from
# the working tree rather than installed, so `python -m` needs the repo root
# as the working directory.
$exitCode = 1
Push-Location -LiteralPath $RepoRoot
try {
    & $Python -m patreon_pipeline.runner @Arguments
    $exitCode = if ($null -ne $LASTEXITCODE) { $LASTEXITCODE } else { 0 }
} finally {
    Pop-Location
}
exit $exitCode
