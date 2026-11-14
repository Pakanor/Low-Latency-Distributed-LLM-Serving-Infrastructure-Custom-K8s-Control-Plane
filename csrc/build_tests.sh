#!/usr/bin/env bash
# Build standalone C++ test binaries directly with g++ (no CMake dependency).
#
# Sources built:
#   allocator_test        -> csrc/allocator_test
#   allocator_audit_test  -> csrc/allocator_audit_test
#   thread_safety_test    -> csrc/thread_safety_test
#   fragmentation_bench   -> csrc/fragmentation_bench
#
# Usage: build_tests.sh [--sanitize <asan|ubsan|tsan|none>] [--coverage] [--build-type <Debug|Release>]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CSRC_DIR="${ROOT}/csrc"
INCLUDE_DIR="${CSRC_DIR}/include"

SAN="none"
COVERAGE=0
BUILD_TYPE="Debug"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sanitize)
            SAN="${2:-none}"
            shift 2
            ;;
        --coverage)
            COVERAGE=1
            shift
            ;;
        --build-type)
            BUILD_TYPE="${2:-Debug}"
            shift 2
            ;;
        *) echo "[build_tests] unknown option: $1"; exit 2;;
    esac
done

CXX="${CXX:-g++}"

case "${BUILD_TYPE}" in
    Debug)   OPT=(-O1);;
    Release) OPT=(-O2);;
    *) echo "[build_tests] unknown build type: ${BUILD_TYPE} (use Debug|Release)"; exit 2;;
esac

FLAGS=(-std=c++17 -g -Wall -Wextra -Wpedantic -Werror -pthread -I"${INCLUDE_DIR}" "${OPT[@]}")

case "${SAN}" in
    asan)  FLAGS+=(-fsanitize=address,undefined -fno-omit-frame-pointer);;
    ubsan) FLAGS+=(-fsanitize=undefined);;
    tsan)  FLAGS+=(-fsanitize=thread);;   # TSan needs -O1, which Debug already uses
    none)  ;;
    *) echo "[build_tests] unknown sanitizer: ${SAN} (use asan|ubsan|tsan|none)"; exit 2;;
esac

if [[ ${COVERAGE} -eq 1 ]]; then
    FLAGS+=(--coverage)
fi

echo "[build_tests] cxx=${CXX} sanitizer=${SAN} coverage=${COVERAGE} build_type=${BUILD_TYPE}"

build() {
    local name="$1" src="$2"
    echo "[build_tests] ${src} -> ${CSRC_DIR}/${name}"
    "${CXX}" "${FLAGS[@]}" "${CSRC_DIR}/${src}" -o "${CSRC_DIR}/${name}"
}

build allocator_test       tests/allocator_test.cpp
build allocator_audit_test tests/allocator_audit_test.cpp
build thread_safety_test   tests/thread_safety_test.cpp
build fragmentation_bench  benchmarks/fragmentation_bench.cpp

echo "[build_tests] done"