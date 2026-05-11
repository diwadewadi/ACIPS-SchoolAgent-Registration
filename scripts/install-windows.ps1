Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SdkPath = Resolve-Path -Path (Join-Path $ProjectRoot "..\ACPs-SDK") -ErrorAction SilentlyContinue

Set-Location $ProjectRoot

if (-not $SdkPath) {
    Write-Error "ACPs-SDK not found. Put it next to this project, for example: ..\ACPs-SDK"
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
& $Python -m pip install -e $SdkPath.Path

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env. Fill OPENAI_API_KEY before running the demo."
}

Write-Host "Install complete."
