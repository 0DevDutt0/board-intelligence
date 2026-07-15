# scripts/setup_cuda128_wsl.sh
# Installs a user-local CUDA 12.8 toolkit into ~/cuda-12.8 (no root needed).
# Downloads the official NVIDIA WSL-Ubuntu debs and extracts them with dpkg -x.
#
# Why: flashinfer JIT-compiles attention kernels at vLLM startup and needs a
# real nvcc. The pip cu12 wheels do not ship the nvcc compiler driver, and the
# project mandates CUDA 12.8 (CUDA 13.x miscompiles for Blackwell sm_120).
# This mirrors the nvcc the old Dockerfile.vllm got from its CUDA 12.8 base image.
#
# Called automatically by start_vllm_wsl.sh on first run; safe to re-run.

set -e

CUDA_DIR="$HOME/cuda-12.8"
STAGE="$HOME/.cache/cuda-12.8-debs"
REPO=https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64

if [ -x "$CUDA_DIR/bin/nvcc" ]; then
    echo "CUDA 12.8 toolkit already present at $CUDA_DIR"
    exit 0
fi

DEBS="cuda-nvcc-12-8_12.8.93-1_amd64.deb \
cuda-crt-12-8_12.8.93-1_amd64.deb \
cuda-nvvm-12-8_12.8.93-1_amd64.deb \
cuda-cudart-12-8_12.8.90-1_amd64.deb \
cuda-cudart-dev-12-8_12.8.90-1_amd64.deb \
cuda-cccl-12-8_12.8.90-1_amd64.deb \
cuda-nvrtc-12-8_12.8.93-1_amd64.deb \
cuda-nvrtc-dev-12-8_12.8.93-1_amd64.deb"

mkdir -p "$STAGE"
cd "$STAGE"
for d in $DEBS; do
    if [ ! -f "$d" ]; then
        echo "Downloading $d"
        curl -sfL "$REPO/$d" -o "$d"
    fi
done

echo 'Extracting CUDA 12.8 toolkit (no root required)...'
EXTRACT="$STAGE/extract"
rm -rf "$EXTRACT"
mkdir -p "$EXTRACT"
for d in $DEBS; do
    dpkg -x "$d" "$EXTRACT"
done
rm -rf "$CUDA_DIR"
mv "$EXTRACT/usr/local/cuda-12.8" "$CUDA_DIR"
rm -rf "$EXTRACT"

echo 'Verifying nvcc...'
"$CUDA_DIR/bin/nvcc" --version | tail -2
echo "CUDA 12.8 toolkit installed at $CUDA_DIR"
