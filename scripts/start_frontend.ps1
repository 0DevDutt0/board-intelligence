# scripts/start_frontend.ps1
# Starts the React/Vite dev server natively on Windows.

$repo = Split-Path -Parent $PSScriptRoot
Set-Location "$repo\frontend"

if (-not (Test-Path "$repo\frontend\node_modules")) {
    Write-Host 'Installing frontend dependencies (first run only)...'
    npm install
}

npm run dev
