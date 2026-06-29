#!/usr/bin/env bash
# Build a universal Linux release of keryx-miner (multi-arch fat binary) in a CUDA 12.2 devel container.
#
# Builds in nvidia/cuda:12.2.2-devel-ubuntu22.04 and creates a single binary that auto-selects
# the optimal PTX kernel for any NVIDIA GPU (from Pascal P40 to Hopper H100).
# All 7 architectures (SM 61, 70, 75, 80, 86, 89, 90) are compiled into one binary.
#
# Runs in WSL Ubuntu. Mounts the Windows source tree, builds inside docker, and stages
# the binary + .so plugins + README into dist/.
#
# Usage (from WSL):
#   bash /mnt/e/keryx/keryx-miner-src/tests/build_linux_universal.sh
#
# Requires: docker daemon running (sudo service docker start, or Docker Desktop WSL integration).
set -euo pipefail

SRC="/mnt/e/keryx/keryx-miner-src"
OUT="$SRC/dist/keryx-miner-v0.3.3-linux-amd64-universal"
IMG="docker.io/nvidia/cuda:12.2.2-devel-ubuntu22.04"

echo "==> Building keryx-miner Linux UNIVERSAL (all GPU architectures) in $IMG"
echo "    source: $SRC"
echo "    output: $OUT"
echo "    architectures: Pascal (SM61) to Hopper (SM90)"

mkdir -p "$OUT"

# Build universal binary with all 7 GPU architectures embedded.
# The binary auto-selects the optimal PTX kernel at runtime.
docker run --rm \
  -v "$SRC:/src:rw" -w /src \
  -e CARGO_TARGET_DIR=/src/target-linux-universal \
  "$IMG" bash -c '
    set -e
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq curl build-essential pkg-config libssl-dev ca-certificates protobuf-compiler >/dev/null
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
    . "$HOME/.cargo/env"
    export CUDA_PATH=/usr/local/cuda
    echo "Building universal binary with PTX for all GPU architectures (SM 61, 70, 75, 80, 86, 89, 90)..."
    cargo build --release --bin keryx-miner
    cargo build --release -p keryxcuda -p keryxopencl
  '

echo "==> staging output"
cp "$SRC/target-linux-universal/release/keryx-miner" "$OUT/"
cp "$SRC/target-linux-universal/release/libkeryxcuda.so" "$OUT/"
cp "$SRC/target-linux-universal/release/libkeryxopencl.so" "$OUT/"
cp "$SRC/tests/install_cuda_deps.sh" "$OUT/"

cat > "$OUT/README.txt" <<EOF
Keryx Miner (Pascal Fork) v0.3.3 — Linux x86-64 UNIVERSAL
===========================================================

Universal binary with auto-selected GPU kernels for ANY NVIDIA GPU.
Includes optimized PTX for: Pascal (P40), Volta (V100), Turing (RTX 2080),
Ampere (RTX 3090/A100), Ada (RTX 6000 Ada), and Hopper (H100).

The binary detects your GPU's compute capability at runtime and automatically
uses the optimal kernel — no need for separate binaries for different GPUs.

Built in nvidia/cuda:12.2.2-devel-ubuntu22.04 (glibc 2.35). Runs on any Linux
distro with glibc >= 2.35 (Ubuntu 22.04+, HiveOS, etc.) and an NVIDIA driver
supporting the CUDA 12.x runtime.

Files
-----
  keryx-miner            the miner (universal — works on any NVIDIA GPU)
  libkeryxcuda.so        CUDA PoW + inference plugin (REQUIRED for GPU mining)
  libkeryxopencl.so      OpenCL plugin (optional alternative backend)
  install_cuda_deps.sh   one-time dependency installer (sudo required)

Quick start (Ubuntu/Debian)
---------------------------
  sudo bash install_cuda_deps.sh          # one-time: installs CUDA 12.x runtime
  LD_LIBRARY_PATH=. ./keryx-miner --mining-address keryx:YOUR_ADDRESS --light

The second command works on any Linux with glibc >= 2.35 and an NVIDIA driver
supporting CUDA 12.x (driver >= 535). Set KERYX_CUDA_DEVICE=N to select a
specific GPU when multiple are present.

This fork charges a 2% maintainer fee by default (to the fork author).
Disable with --devfund-percent=0.

Source: https://github.com/Muskwak/keryx-pascal-miner
EOF

echo "==> creating tarball"
tar -czf "$OUT.tar.gz" -C "$SRC/dist" "keryx-miner-v0.3.3-linux-amd64-universal"
echo "==> done:"
ls -la "$OUT" "$OUT.tar.gz"
