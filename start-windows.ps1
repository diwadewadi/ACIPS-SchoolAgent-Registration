Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
$ProjectRoot = Join-Path $RepoRoot "school-agent-prototype"
$SdkPath = Join-Path $RepoRoot "ACPs-SDK"

if (-not (Test-Path (Join-Path $ProjectRoot ".venv\Scripts\python.exe"))) {
    Write-Error "Virtual environment not found. Run .\install-windows.ps1 first."
}

if (-not (Test-Path $SdkPath)) {
    Write-Error "ACPs-SDK not found at $SdkPath"
}

if (-not (Test-Path (Join-Path $ProjectRoot ".env"))) {
    Write-Warning "school-agent-prototype\.env not found. Copy .env.example to .env and fill OPENAI_API_KEY."
}

$PythonPath = "$ProjectRoot;$ProjectRoot\leader;$SdkPath"

function Escape-SingleQuoted {
    param([string]$Value)
    return $Value.Replace("'", "''")
}

function New-ServiceCommand {
    param([string]$Command)

    $RootLiteral = Escape-SingleQuoted $ProjectRoot
    $PythonPathLiteral = Escape-SingleQuoted $PythonPath
    return "Set-Location '$RootLiteral'; . .\.venv\Scripts\Activate.ps1; `$env:PYTHONPATH = '$PythonPathLiteral'; $Command"
}

$PartnerCommand = New-ServiceCommand "python -m partners.main"
$LeaderCommand = New-ServiceCommand "python -m uvicorn leader.main:app --host 0.0.0.0 --port 59210"
$WebCommand = New-ServiceCommand "python web_app\webserver.py"

Start-Process powershell -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $PartnerCommand)
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $LeaderCommand)
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $WebCommand)

Write-Host "Started Partner, Leader, and Web windows."
Write-Host "Open http://localhost:59200"
