#!/usr/bin/env bash
# One-time CUDA 12.x runtime + 580 driver installer for keryx-miner (Linux, Pascal/sm_61).
# NVIDIA driver 610+ dropped Pascal support — the 580 branch is the last compatible one.
# Run with sudo before first launch on Debian/Ubuntu systems.
set -e

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: this script needs sudo. Run: sudo bash install_cuda_deps.sh"
    exit 1
fi

# Already installed?
if ldconfig -p 2>/dev/null | grep -q libcublas.so.12 \
   && ldconfig -p 2>/dev/null | grep -q libcurand.so.12 \
   && ldconfig -p 2>/dev/null | grep -q libcudart.so.12; then
    echo "CUDA 12.x runtime libraries already present."
fi

command -v apt-get >/dev/null 2>&1 || { echo "apt-get required (Debian/Ubuntu)"; exit 1; }

cd /tmp
CUDA_KEYRING="cuda-keyring_1.1-1_all.deb"
wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/$CUDA_KEYRING" -O "$CUDA_KEYRING"
dpkg -i "$CUDA_KEYRING"
apt-get update -qq

echo "Installing Pascal-compatible NVIDIA 580 driver and CUDA 12.2 runtime..."
apt-get install -y -qq libcublas-12-2 libcurand-12-2 cuda-drivers-580 cuda-runtime-12-2

# Register the install path with the loader
CUBLAS_PATH="$(find /usr/local /usr/lib -name 'libcublas.so.12' 2>/dev/null | head -1)"
if [ -z "$CUBLAS_PATH" ]; then
    echo "ERROR: libcublas.so.12 not found after install."
    exit 1
fi
echo "$(dirname "$CUBLAS_PATH")" > /etc/ld.so.conf.d/keryx-cuda.conf
ldconfig
rm -f "$CUDA_KEYRING"
echo "Done. REBOOT your system to load the 580 driver safely."
echo "After reboot, run the miner with: LD_LIBRARY_PATH=. ./keryx-miner --mining-address YOUR_ADDRESS --light"
