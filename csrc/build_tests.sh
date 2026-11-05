#!/usr/bin/env bash
# Build standalone C++ test binaries (allocator, thread-safety, fragmentation).
# Usage: build_tests.sh [--sanitize <asan|ubsan|tsan|none>]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INCLUDE_DIR="${ROOT}/csrc/include"
SRC_DIR="${ROOT}/csrc"
SAN="${1:-none}"

case "$SAN" in
    asan)  FLAGS=(-std=c++17 -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -pthread -I"${INCLUDE_DIR}" -I"${SRC_DIR}");;
    tsan)  FLAGS=(-std=c++17 -O1 -g -fsanitize=thread -pthread -I"${INCLUDE_DIR}" -I"${SRC_DIR}");;
    none)  FLAGS=(-std=c++17 -O2 -Wall -Wextra -pthread -I"${INCLUDE_DIR}" -I"${SRC_DIR}");;
    *) echo "[build_tests] unknown sanitizer: $SAN (use asan|tsan|none)"; exit 2;;
esac

echo "[build_tests] sanitizer=$SAN"
for src in tests/thread_safety_test.cpp benchmarks/fragmentation_bench.cpp; do
    name="${src##*/}"
    base="${name%.*}"
    out="${SRC_DIR}/${base}"
    echo "[build_tests] ${src} -> ${out}"
    "${CXX:-g++}" "${FLAGS[@]}" "${SRC_DIR}/${src}" -o "${out}"
done
echo "[build_tests] done"