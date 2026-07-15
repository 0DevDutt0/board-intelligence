# scripts/start_all.ps1
# Launches the full stack natively (no Docker) in three windows:
#   1. vLLM in WSL Ubuntu     -> 127.0.0.1:11436
#   2. FastAPI backend        -> 127.0.0.1:8000
#   3. React/Vite frontend    -> 127.0.0.1:5173
# Each service gets its own window so logs stay separate.

$repo = Split-Path -Parent $PSScriptRoot

Write-Host '[1/3] Starting vLLM (WSL)...'
Start-Process powershell -ArgumentList '-NoExit', '-File', "$repo\scripts\start_vllm.ps1"

Write-Host 'Waiting for vLLM to become healthy (model load can take a few minutes)...'
$deadline = (Get-Date).AddMinutes(15)
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest -Uri 'http://127.0.0.1:11436/health' -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) { break }
    } catch {}
    Start-Sleep -Seconds 5
}

Write-Host '[2/3] Starting backend...'
Start-Process powershell -ArgumentList '-NoExit', '-File', "$repo\scripts\start_backend.ps1"

Write-Host 'Waiting for backend to load models...'
$deadline = (Get-Date).AddMinutes(5)
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) { break }
    } catch {}
    Start-Sleep -Seconds 3
}

Write-Host '[3/3] Starting frontend...'
Start-Process powershell -ArgumentList '-NoExit', '-File', "$repo\scripts\start_frontend.ps1"

Write-Host ''
Write-Host 'All services launching. Open http://localhost:5173 when ready.'
