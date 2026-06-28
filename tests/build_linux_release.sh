#!/usr/bin/env bash
# Build a portable Linux release of keryx-miner (Pascal fork) in a CUDA 12.2 devel container.
#
# Runs in WSL Ubuntu (docker available). Mounts the Windows source tree, builds inside
# nvidia/cuda:12.2.2-devel-ubuntu22.04 (glibc 2.35 — runs on HiveOS / Ubuntu 22.04+ rigs),
# and stages the binary + .so plugins + README into dist/.
#
# Usage (from WSL):
#   bash /mnt/e/keryx/keryx-miner-src/tests/build_linux_release.sh [SM_ARCH]
#     SM_ARCH default: 61 (Pascal). Pass 89 for Ada/Blackwell.
#
# Requires: docker daemon running (sudo service docker start, or Docker Desktop WSL integration).
set -euo pipefail

SM_ARCH="${1:-61}"
SRC="/mnt/e/keryx/keryx-miner-src"
OUT="$SRC/dist/keryx-miner-v0.3.3-linux-amd64-sm${SM_ARCH}"
IMG="docker.io/nvidia/cuda:12.2.2-devel-ubuntu22.04"

echo "==> Building keryx-miner Linux sm_${SM_ARCH} in $IMG"
echo "    source: $SRC"
echo "    output: $OUT"

mkdir -p "$OUT"

# Pull (no-op if cached) then build. The container installs build deps, Rust, protoc,
# sets SM_ARCH + CUDA_COMPUTE_CAP, and builds release. We set CUDA_COMPUTE_CAP (read by
# candle-kernels' bindgen_cuda) AND SM_ARCH (read by this fork's build.rs for pom_mine.ptx).
docker run --rm \
  -v "$SRC:/src:rw" -w /src \
  -e SM_ARCH="$SM_ARCH" \
  -e CUDA_COMPUTE_CAP="$SM_ARCH" \
  -e CARGO_TARGET_DIR=/src/target-linux-sm"${SM_ARCH}" \
  "$IMG" bash -c '
    set -e
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq curl build-essential pkg-config libssl-dev ca-certificates protobuf-compiler >/dev/null
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
    . "$HOME/.cargo/env"
    export CUDA_PATH=/usr/local/cuda
    cargo build --release --bin keryx-miner
    cargo build --release -p keryxcuda -p keryxopencl
  '

echo "==> staging output"
cp "$SRC/target-linux-sm${SM_ARCH}/release/keryx-miner" "$OUT/"
cp "$SRC/target-linux-sm${SM_ARCH}/release/libkeryxcuda.so" "$OUT/"
cp "$SRC/target-linux-sm${SM_ARCH}/release/libkeryxopencl.so" "$OUT/"

cat > "$OUT/README.txt" <<EOF
Keryx Miner (Pascal Fork) v0.3.3 — Linux x86-64, sm_${SM_ARCH}
============================================================

Built in nvidia/cuda:12.2.2-devel-ubuntu22.04 (glibc 2.35). Runs on any Linux
distro with glibc >= 2.35 (Ubuntu 22.04+, HiveOS, etc.) and an NVIDIA driver
supporting the CUDA 12.x runtime.

Files
-----
  keryx-miner       the miner
  libkeryxcuda.so   CUDA PoW + inference plugin (REQUIRED for GPU mining)
  libkeryxopencl.so OpenCL plugin (optional alternative backend)

Quick start
-----------
  LD_LIBRARY_PATH=. ./keryx-miner --mining-address keryx:YOUR_ADDRESS --light

Requirements
------------
  - NVIDIA driver supporting CUDA 12.x (>= 535)
  - The two .so files MUST sit next to keryx-miner (or be on LD_LIBRARY_PATH)
  - libcublas.so.12 + libcurand.so.10 from the CUDA 12.x runtime on the host
    (on HiveOS the miner installs these on first run; otherwise install via
    your package manager: libcublas-12-2, libcurand-12-2)

This fork charges a 2% maintainer fee by default (to the fork author).
Disable with --devfund-percent=0.

Source: https://github.com/Muskwak/keryx-pascal-miner
EOF

echo "==> creating tarball"
tar -czf "$OUT.tar.gz" -C "$SRC/dist" "keryx-miner-v0.3.3-linux-amd64-sm${SM_ARCH}"
echo "==> done:"
ls -la "$OUT" "$OUT.tar.gz"
