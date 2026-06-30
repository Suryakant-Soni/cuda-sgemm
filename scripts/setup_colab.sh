#!/usr/bin/env bash
set -euo pipefail

echo "=== Verifying toolchain ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvcc --version | grep release
cmake --version | head -n1

echo "=== Toolchain OK ==="
