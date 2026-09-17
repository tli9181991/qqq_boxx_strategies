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
# Values are taken literally -- no quote stripping, no expansion -- because a
# Gmail app password can contain almost anything and guessing at quoting is how
# a credential silently becomes the wrong string.
if (Test-Path $EnvFile) {
    foreach ($line in Get-Content -LiteralPath $EnvFile) {
        $trimmed = $line.Trim()
        if ($trimmed -eq '' -or $trimmed.StartsWith('#')) { continue }
        $idx = $trimmed.IndexOf('=')
        if ($idx -lt 1) { continue }
        $key = $trimmed.Substring(0, $idx).Trim()
        $val = $trimmed.Substring($idx + 1)
        Set-Item -Path "env:$key" -Value $val
    }
} else {
    Write-Warning "No $EnvFile -- running on defaults and whatever is already in the environment."
}

Set-Location $RepoRoot
& $Python -m patreon_pipeline.runner @Arguments
exit $LASTEXITCODE
