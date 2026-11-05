#!/usr/bin/env bash
set -euo pipefail

# Build or test mbed with in-process MLXSwiftLMEmbeddingGemma2Engine enabled.
# Requires:
#   1. /Users/fredbliss/mlx/mlx-swift submodules initialized (Source/Cmlx/mlx and Source/Cmlx/mlx-c)
#   2. xcodebuild (required by MLX Swift to compile Metal shaders into default.metallib)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLX_SWIFT_DIR="$(cd "${REPO_ROOT}/../../mlx/mlx-swift" 2>/dev/null && pwd || true)"

if [[ -z "${MLX_SWIFT_DIR}" || ! -f "${MLX_SWIFT_DIR}/Source/Cmlx/mlx/CMakeLists.txt" ]]; then
  echo "error: mlx-swift C++ submodules are not initialized at ${MLX_SWIFT_DIR}/Source/Cmlx/mlx." >&2
  echo "Run: git -C ../../mlx/mlx-swift submodule update --init --recursive" >&2
  exit 1
fi

ACTION="${1:-build}"
export MBED_ENABLE_NATIVE_MLX=1

cd "${REPO_ROOT}"
xcodebuild "${ACTION}" \
  -scheme mbed-Package \
  -destination 'platform=macOS' \
  -skipPackagePluginValidation
