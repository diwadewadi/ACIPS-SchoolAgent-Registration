Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
$ProjectRoot = Join-Path $RepoRoot "school-agent-prototype"
$SdkPath = Join-Path $RepoRoot "ACPs-SDK"

Set-Location $ProjectRoot

if (-not (Test-Path $SdkPath)) {
    Write-Error "ACPs-SDK not found at $SdkPath"
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    $pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pythonLauncher) {
        py -3.11 -m venv .venv
    }
    else {
        python -m venv .venv
    }
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $Python -m pip install -U pip
& $Python -m pip install -r requirements.txt
& $Python -m pip install -e $SdkPath

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created school-agent-prototype\.env. Fill OPENAI_API_KEY before running the demo."
}

Write-Host "Install complete."
