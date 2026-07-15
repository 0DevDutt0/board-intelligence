# scripts/start_vllm_wsl.sh
# Runs the vLLM OpenAI-compatible server directly in WSL Ubuntu -- no Docker.
# Same wheels, flags, and environment as the old Dockerfile.vllm container.
#
# First run: creates ~/.venvs/vllm, installs vllm + flashinfer (cu128 wheels),
# and applies the Blackwell sm_120 flashinfer patch. Later runs start in seconds.
#
# Usage (from Windows): scripts/start_vllm.ps1
# Usage (inside WSL):   bash scripts/start_vllm_wsl.sh

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$HOME/.venvs/vllm"
MODEL_DIR="${VLLM_MODEL_DIR:-$REPO_DIR/models/Qwen3-14B-AWQ}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: model directory not found: $MODEL_DIR"
    echo "Download models first: python scripts/download_models.py"
    exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
    echo "Creating vLLM virtualenv at $VENV (first run only)..."
    python3 -m venv "$VENV"
fi

# vllm is pinned to 0.23.0: newer releases regress on WSL2 Blackwell
# (0.25 V2 runner needs UVA which dxgkrnl lacks; flashinfer 0.6 warmup
# segfaults during kernel warmup on sm_120). 0.23.0 is the version the
# proven Docker image was built against.
if ! "$VENV/bin/python" -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("vllm") else 1)' 2>/dev/null; then
    echo "Installing vLLM + flashinfer (first run only, several GB)..."
    "$VENV/bin/pip" install --no-cache-dir --timeout 600 \
        'vllm==0.23.0' \
        --extra-index-url https://download.pytorch.org/whl/cu128
    "$VENV/bin/python" "$REPO_DIR/scripts/patch_flashinfer.py"
fi

# flashinfer JIT-compiles attention kernels at startup and needs nvcc 12.8
# (CUDA 13.x miscompiles for Blackwell sm_120 -- project rule). Install a
# user-local CUDA 12.8 toolkit on first run.
if [ ! -x "$HOME/cuda-12.8/bin/nvcc" ]; then
    bash "$REPO_DIR/scripts/setup_cuda128_wsl.sh"
fi
export CUDA_HOME="$HOME/cuda-12.8"
# venv bin must be on PATH too: flashinfer's JIT invokes 'ninja', which is
# installed as a pip package inside the venv.
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
# JIT link step needs -lcuda; on WSL the driver library lives in
# /usr/lib/wsl/lib (the Docker image used the CUDA base image's stub instead).
export LIBRARY_PATH="/usr/lib/wsl/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

# vLLM 0.23+ ships CUDA 13 runtime libs (libcudart.so.13) as pip wheels under
# site-packages/nvidia/*/lib. The Docker image registered them via ldconfig
# (root); in a venv we use LD_LIBRARY_PATH instead.
SITE_PACKAGES="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
NVIDIA_LIB_DIRS="$(find "$SITE_PACKAGES/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -)"
if [ -n "$NVIDIA_LIB_DIRS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Same runtime environment as docker-compose.yml:
# - flashinfer JIT sampler rejects sm_120, so it stays disabled
# - JIT (if it ever runs) must target Blackwell
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_CUDA_ARCHS=12.0
# vLLM 0.25's V2 GPU model runner requires UVA (pinned mapped host memory),
# which the WSL2 dxgkrnl GPU layer does not support. Force the V1 runner --
# same engine path the proven vLLM 0.23 Docker setup used. CUDA graphs,
# flashinfer attention, and AWQ-Marlin are unaffected.
export VLLM_USE_V2_MODEL_RUNNER=0
# Note: expandable_segments is NOT usable here -- CUDA VMM allocations are
# broken under WSL2 dxgkrnl and fail far earlier than plain cudaMalloc.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Starting vLLM on 127.0.0.1:11436 (model: $MODEL_DIR)"
echo "Tip: loading from /mnt/e is slow; copy the model into WSL and set"
echo "     VLLM_MODEL_DIR=~/models/Qwen3-14B-AWQ for faster startup."

exec "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --quantization awq_marlin \
    --dtype float16 \
    --gpu-memory-utilization 0.60 \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --port 11436 \
    --host 127.0.0.1 \
    --served-model-name qwen3-14b \
    --attention-backend flashinfer \
    --no-enable-log-requests
