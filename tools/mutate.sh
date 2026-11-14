#!/usr/bin/env bash
# Mutation testing for csrc/include/page_allocator.hpp.
#
# Every mutant is applied to a COPY of the header under /tmp (never to the repo)
# and the C++ test binaries are rebuilt against that copy. A mutant is KILLED if
# at least one test binary exits non-zero (a CHECK failed / a crash), SURVIVED if
# all tests still pass.
#
# Usage:
#   tools/mutate.sh                 # run the default mutant set (M1,M2,M5,M7,M8,M11)
#   tools/mutate.sh M7 M8 M11       # run specific mutants
#   tools/mutate.sh --list          # list available mutants
#   PA_MUT_KEEP=1 tools/mutate.sh   # keep the /tmp work dir for inspection
#
# The original header is never modified.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_HEADER="${ROOT}/csrc/include/page_allocator.hpp"
TESTS_DIR="${ROOT}/csrc/tests"
WORK="${PA_MUT_WORKDIR:-/tmp/pa-mut}"
CXX="${CXX:-g++}"
KEEP="${PA_MUT_KEEP:-0}"

CXXFLAGS=(-std=c++17 -O1 -g -Wall -Wextra -Wpedantic -Werror -pthread)

if [[ ! -f "${SRC_HEADER}" ]]; then
    echo "[mutate] missing ${SRC_HEADER}" >&2
    exit 2
fi

if [[ "${1:-}" == "--list" ]]; then
    grep -oE '^MUTANT_[A-Za-z0-9_]+=' "${BASH_SOURCE[0]}" | sed 's/^MUTANT_//; s/=$//'
    exit 0
fi

# Mutants to run (default set). Each entry: NAME
MUTANTS=("$@")
if [[ ${#MUTANTS[@]} -eq 0 ]]; then
    MUTANTS=(M1 M2 M5 M7 M8 M11)
fi

rm -rf "${WORK}"
mkdir -p "${WORK}/include" "${WORK}/tests" "${WORK}/bin"
cp "${ROOT}/csrc/include/"*.hpp "${WORK}/include/"
cp "${TESTS_DIR}/allocator_test.cpp" "${TESTS_DIR}/allocator_audit_test.cpp" "${WORK}/tests/"

# Compile + run a single test binary. Returns 0 if the mutant is killed by it.
run_one() {
    local src="$1" out="$2"
    if ! "${CXX}" "${CXXFLAGS[@]}" -I"${WORK}/include" "${WORK}/tests/${src}" -o "${WORK}/bin/${out}" 2>"${WORK}/bin/${out}.buildlog"; then
        echo "    [build-error] ${src} (see ${WORK}/bin/${out}.buildlog)"
        return 0  # a mutant that breaks the build is still a killed mutant
    fi
    if "${WORK}/bin/${out}" >"${WORK}/bin/${out}.runlog" 2>&1; then
        return 1
    fi
    return 0
}

overall=0
for m in "${MUTANTS[@]}"; do
    echo "== mutant ${m} =="
    cp "${SRC_HEADER}" "${WORK}/include/page_allocator.hpp"
    if ! python3 - "${WORK}/include/page_allocator.hpp" "${m}" <<'PY'
import sys
path, name = sys.argv[1], sys.argv[2]
with open(path) as f:
    text = f.read()

# Each mutant is a single, exact string replacement (old -> new) that MUST match
# exactly once. Names and replacements mirror the audit report.
MUTANTS = {
    # M1: free_block does not decrement ref_count
    "M1": ("        blocks_[block_id].ref_count--;\n",
           "        blocks_[block_id].ref_count -= 0;\n"),
    # M2: free_block does not return the block to free_blocks_
    "M2": ("        if (blocks_[block_id].ref_count == 0) {\n"
           "            free_blocks_.push(block_id);\n"
           "        }\n",
           "        if (blocks_[block_id].ref_count == 0) {\n"
           "            (void)block_id;\n"
           "        }\n"),
    # M5: allocate_block skips cleanup_stale_cpu_mapping
    "M5": ("        cleanup_stale_cpu_mapping(block_id);\n"
           "        blocks_[block_id].ref_count = 1;\n"
           "        return block_id;\n",
           "        blocks_[block_id].ref_count = 1;\n"
           "        return block_id;\n"),
    # M7: get_host_ptr returns the pool base instead of the per-block pointer
    "M7": ("        return reinterpret_cast<uintptr_t>(cpu_block_ptrs_[cpu_block_id]);\n",
           "        return reinterpret_cast<uintptr_t>(base_host_ptr_);\n"),
    # M8: swap_in forgets to mark the CPU block FREE (state/free-list diverge)
    "M8": ("        cpu_block_states_[cpu_block_id] = CpuBlockState::FREE;\n"
           "        cpu_free_blocks_.push_back(cpu_block_id);\n"
           "\n"
           "        // Find and remove the reverse mapping\n",
           "        cpu_free_blocks_.push_back(cpu_block_id);\n"
           "\n"
           "        // Find and remove the reverse mapping\n"),
    # M11: free_block's swapped path drops the ref_count == 0 guard
    "M11": ("            if (blocks_[block_id].ref_count == 0 &&\n"
            "                static_cast<size_t>(cpu_block_id) < total_cpu_blocks_ &&\n",
            "            if (static_cast<size_t>(cpu_block_id) < total_cpu_blocks_ &&\n"),
}

if name not in MUTANTS:
    sys.stderr.write("unknown mutant %s\n" % name)
    sys.exit(3)
old, new = MUTANTS[name]
count = text.count(old)
if count != 1:
    sys.stderr.write("mutant %s: expected exactly 1 match, found %d\n" % (name, count))
    sys.exit(4)
text = text.replace(old, new, 1)
with open(path, "w") as f:
    f.write(text)
PY
    then
        echo "  [apply-error] mutant ${m} could not be applied"
        overall=1
        continue
    fi

    killed=0
    for t in allocator_test allocator_audit_test; do
        if run_one "${t}.cpp" "${t}"; then
            echo "  KILLED by ${t}"
            killed=1
        fi
    done
    if [[ ${killed} -eq 1 ]]; then
        echo "  => ${m}: KILLED"
    else
        echo "  => ${m}: SURVIVED"
        overall=1
    fi
done

if [[ "${KEEP}" != "1" ]]; then
    rm -rf "${WORK}"
fi

exit ${overall}