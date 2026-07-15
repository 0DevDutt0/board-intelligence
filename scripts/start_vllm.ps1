# scripts/start_vllm.ps1
# Starts the vLLM server inside WSL Ubuntu (no Docker).
# First run installs vLLM into ~/.venvs/vllm inside WSL (several GB).

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

wsl -d Ubuntu -- bash scripts/start_vllm_wsl.sh
