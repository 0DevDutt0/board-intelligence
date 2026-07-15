# scripts/start_backend.ps1
# Starts the FastAPI backend natively on Windows using the repo venv.
# Reads configuration from .env in the repo root.

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (-not (Test-Path "$repo\venv\Scripts\python.exe")) {
    Write-Host 'ERROR: venv not found. Create it first:'
    Write-Host '  python -m venv venv; venv\Scripts\pip install -r requirements.txt'
    exit 1
}

# run_backend.py preloads pyarrow before uvicorn/asyncio -- required on this
# machine to avoid a DLL access violation during model loading. Do not replace
# with a direct 'python -m uvicorn' call.
& "$repo\venv\Scripts\python.exe" scripts\run_backend.py
