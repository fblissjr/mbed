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
shift || true
export MBED_ENABLE_NATIVE_MLX=1

cd "${REPO_ROOT}"

find_products_dir() {
  local dir
  dir="$(ls -td "${HOME}"/Library/Developer/Xcode/DerivedData/mbed-*/Build/Products/Debug 2>/dev/null | head -n 1 || true)"
  if [[ -z "${dir}" || ! -d "${dir}" ]]; then
    dir="$(xcodebuild -scheme mbed-app -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation -showBuildSettings 2>/dev/null \
      | awk -F' = ' '/^[[:space:]]*BUILT_PRODUCTS_DIR = / {print $2; exit}' || true)"
  fi
  echo "${dir}"
}

needs_build() {
  local bin="$1"
  if [[ -z "${bin}" || ! -x "${bin}" ]]; then
    return 0
  fi
  if [[ -n "$(find "${REPO_ROOT}/Sources" "${REPO_ROOT}/Package.swift" -newer "${bin}" -print -quit 2>/dev/null)" ]]; then
    return 0
  fi
  return 1
}

case "${ACTION}" in
  build|test)
    xcodebuild "${ACTION}" \
      -scheme mbed-Package \
      -destination 'platform=macOS,arch=arm64' \
      -skipPackagePluginValidation \
      "$@"
    ;;
  cli)
    PRODUCTS_DIR="$(find_products_dir)"
    if needs_build "${PRODUCTS_DIR}/mbed-cli"; then
      xcodebuild build -scheme mbed-Package -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation
      PRODUCTS_DIR="$(find_products_dir)"
    fi
    exec "${PRODUCTS_DIR}/mbed-cli" "$@"
    ;;
  app)
    PRODUCTS_DIR="$(find_products_dir)"
    if needs_build "${PRODUCTS_DIR}/mbed-app"; then
      xcodebuild build -scheme mbed-Package -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation
      PRODUCTS_DIR="$(find_products_dir)"
    fi
    exec "${PRODUCTS_DIR}/mbed-app" "$@"
    ;;
  *)
    echo "Usage: $0 [build|test|cli <args...>|app]" >&2
    exit 1
    ;;
esac
