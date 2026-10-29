# Codebase Audit — llm-infra

**Date:** 2026-10-29  
**State:** Reset to commit `c9bff53` (before CUDA/HIP/ROCm implementation)  
**Focus:** CPU-only paged KV cache — storage efficiency & no fragmentation, not native kernel speed

---

## 1. Current Slice Status

**You are in Slice 2** ("C++ DRAM KV-Cache Allocator & Fragmentation Benchmark").

### Slice completion status:

| Slice | Description | Status |
|-------|-------------|--------|
| 1 | End-to-End Baseline Pipeline (REST + TinyLlama + k3d) | **DONE** — FastAPI server, Dockerfile, k8s manifests exist |
| 2 | C++ DRAM KV-Cache Allocator & Fragmentation Benchmark | **IN PROGRESS** — C++ allocator + bindings exist (CPU-only now), needs cleanup + benchmark |
| 3 | gRPC Router & Custom Go Operator | **NOT STARTED** — No gRPC, no Go operator, no CRD |
| 4 | Degraded Mode Fallback & Load Testing | **NOT STARTED** — No fallback node, no k6 scripts |

---

## 2. What Was Removed (Reset to `c9bff53`)

The following CUDA/HIP/ROCm and native-kernel (speed-focused) code was removed:

| Removed | File(s) | Reason |
|---------|---------|--------|
| `paged_kernels.cpp` (CPU + HIP kernel) | `csrc/src/paged_kernels.cpp` | Speed-focused native kernel; not needed for storage/no-fragmentation |
| `paged_kernels.h` (C-ABI header) | `csrc/include/paged_kernels.h` | Speed-focused native kernel interface |
| `paged_attention` pybind11 binding | `csrc/src/bindings.cpp` | Removed from bindings; allocator only |
| `paged_attention.py` | `app/model/paged_attention.py` | Speed-focused attention backend integration |
| `attention_backends.py` | `app/model/attention_backends.py` | NativePagedBackend + TorchSdpBackend dispatch |
| CUDA/ROCm build logic | `setup.py` | Removed CUDA/ROCm extension paths |
| HIP/ROCm detection macros | `csrc/include/page_allocator.hpp` | `LLM_INFRA_HIP_ENABLED` defaults to 0 |

### What Remains (CPU-only, storage-focused):

- `csrc/include/page_allocator.hpp` — `PageAllocator` + `SequenceBlockTable` (CPU DRAM page management)
- `csrc/src/bindings.cpp` — pybind11 bindings for `PageAllocator` + `SequenceBlockTable` only
- `app/memory/manager.py` — `PagedKVCacheManager` (block table, slot mapping, scatter/gather, swap)
- `app/memory/strategies.py` — `KVCacheMemoryStrategy` + `CpuKVCacheStrategy` (+ leftover CUDA/HIP classes to remove)
- `app/scheduler/scheduler.py` — Prefill/decode scheduler with block allocation
- `app/scheduler/sequence.py` — `Sequence` + `SequenceStatus`
- `app/engine/llm_engine.py` — `LLMEngine` (step, add_request, KV write)
- `app/api/server.py` — FastAPI REST server
- `app/model/client.py` — `K8sModelClient` + `MockModelClient`
- `k8s/deployment.yaml`, `k8s/pvc.yaml` — K8s manifests
- `Dockerfile` — Container build
- `tests/` — Unit tests for sequence, KV cache, scheduler, engine, swapping
- `csrc/benchmarks/fragmentation_bench.cpp` — Fragmentation benchmark (needs rework for storage focus)

---

## 3. Remaining Issues to Fix (Slice 2 Completion)

### Critical

**3.1 `run_tests.py` hardcoded wrong path**
`run_tests.py:6-7` uses `/home/roberto/llm-infra` instead of `/home/roberto/Projects/llm-infra`. All tests fail.

**3.2 `CudaKVCacheStrategy` and `HipZeroCopyStrategy` still in `strategies.py`**
`app/memory/strategies.py:71-127` still contains CUDA/HIP strategy classes. For a CPU-only, storage-focused build these should be removed. `detect_strategy()` still checks `torch.cuda.is_available()`.

**3.3 `block_size` semantic mismatch**
Python passes `block_size` as **tokens per block**, but C++ `PageAllocator` treats it as `block_size_bytes` for the host swap pool sizing. With `block_size=4`, the host pool allocates `32 * 4 = 128 bytes` — far too small for actual KV data.

**3.4 KV cache data loss on preemption (scheduler OOM)**
`app/scheduler/scheduler.py:86-90` frees all blocks when a decode sequence needs a new block but none are free, then re-allocates from scratch. Generated tokens' KV cache is silently discarded.

**3.5 `swap_out_sequence` is a no-op for data copy**
`app/memory/manager.py` — `swap_out_sequence` has `if self.strategy.requires_copy_on_swap: pass`. No data is actually copied to the swap pool.

**3.6 `run_server.py` uses unrealistically small KV cache**
`TOTAL_BLOCKS = 10`, `BLOCK_SIZE = 4` — only 40 tokens total. SmolLM-135M needs at least 256 blocks.

### Medium

**3.7 Missing `__init__.py` in `app/api/`, `app/engine/`, `app/model/`, `app/scheduler/`**
Only `app/memory/__init__.py` and `app/__init__.py` exist.

**3.8 `fragmentation_bench.cpp` measures speed, not storage/fragmentation**
The benchmark compares allocation speed, not memory footprint or fragmentation under concurrent variable-length sequences.

**3.9 No metrics endpoint**
No `/metrics` endpoint exposing KV cache utilization (`% free pages`, active stream count) for Prometheus.

---

## 4. Missing Components (not yet implemented)

| Component | Status | Required for |
|-----------|--------|--------------|
| Prometheus metrics endpoint | NOT STARTED | Observability |
| Kubernetes CRD definitions | NOT STARTED | Module C (Operator) |
| Custom Go Kubernetes Operator | NOT STARTED | Module C |
| Grafana dashboards | NOT STARTED | Module C |
| gRPC proto definitions | NOT STARTED | Module B (Router) |
| gRPC router implementation (Go/Rust) | NOT STARTED | Module B |
| k6 load test scripts | NOT STARTED | Slice 4 |
| AWQ/GPTQ quantization pipeline | NOT STARTED | Module A (Quantization) |
| Perplexity validation tests | NOT STARTED | Module A |
| Fallback/degraded mode node | NOT STARTED | Slice 4 |
| Load-shedding logic | NOT STARTED | Module B |

---

## 5. Test Coverage After Reset

| Area | Coverage | Notes |
|------|----------|-------|
| KV cache allocation | Good | `test_kv_cache_manager.py` |
| KV cache geometry | Good | `test_kv_cache_geometry.py` |
| Scheduler logic | Partial | `test_scheduler.py` |
| Engine end-to-end | Partial | `test_llm_engine.py` |
| Swapping | Weak | `test_kv_swapping.py` checks pointers, not data integrity |
| OOM preemption | Weak | Tests check allocator state, not output correctness |
| K8s integration | None | `test_k8s_integration.py` is mislabeled mock test |
| Fragmentation benchmark | None | C++ bench exists but not integrated into test suite |
| Quantization | None | No AWQ/GPTQ tests |
| Load testing | None | `load_test.py` is raw httpx, not k6 |
| gRPC router | N/A | Not implemented |
| K8s operator | N/A | Not implemented |

---

## 6. Next Steps (Slice 2 — CPU-Only, Storage/No-Fragmentation Focus)

1. **Fix `run_tests.py` path** — DONE: replaced hardcoded `/home/roberto/llm-infra` with `REPO_ROOT = os.path.dirname(os.path.abspath(__file__))`
2. **Remove CUDA/HIP strategy classes** — delete `CudaKVCacheStrategy`, `HipZeroCopyStrategy` from `strategies.py`; simplify `detect_strategy()` to return `CpuKVCacheStrategy()`
3. **Fix `block_size` semantic** — document that C++ `block_size` is in tokens, not bytes; or size host pool as `total_cpu_blocks * block_size * num_heads * head_dim * dtype_size`
4. **Fix preemption data loss** — implement actual swap-out (copy KV data to host pool) before freeing blocks, or implement recomputation fallback
5. **Fix `swap_out_sequence` data copy** — implement copy logic instead of `pass`
6. **Wire `eos_token_id` through `add_request`** — pass tokenizer's EOS token ID so sequences can terminate via EOS
7. **Fix `run_server.py` block count** — increase to at least 256 for SmolLM-135M
8. **Add missing `__init__.py` files** — to `app/api/`, `app/engine/`, `app/model/`, `app/scheduler/`
9. **Rework fragmentation benchmark** — measure memory footprint and fragmentation under concurrent variable-length sequences, not just allocation speed
10. **Add metrics endpoint** — expose `/metrics` with KV cache utilization, free blocks, active sequences (Prometheus format)
11. **Add integration test for fragmentation benchmark** — run C++ bench as part of test suite

---

## 7. Baseline Frozen

**Baseline frozen at commit `c9bff53`** (before CUDA/HIP/ROCm removal).

**Command:** `venv/bin/python -m pytest tests/ -v --tb=short`

**Result: 21 passed, 0 failed, 0 errors (12.26s)**

Collected test files (all passing):
- `tests/test_sequence.py`
- `tests/test_kv_cache_manager.py`
- `tests/test_kv_cache_geometry.py`
- `tests/test_scheduler.py`
- `tests/test_llm_engine.py`
- `tests/test_api_server.py`
- `tests/test_k8s_integration.py`
- `tests/test_kv_swapping.py`
- `tests/stress_tests/test_01_memory_oom_preemption.py`
- `tests/stress_tests/test_02_state_concurrency.py`
- `tests/stress_tests/test_03_ragged_layout.py`
- `tests/stress_tests/test_04_edge_case.py`

**Removed (no longer exist):** `test_attention_backends.py`, `test_paged_attention.py`, `test_paged_kv_store.py`, `test_device_resolution.py` — deleted during reset to `c9bff53`.

**Directory structure:** Only `tests/stress_tests/` exists (no `tests/test_stress_tests/` — that was a typo in earlier drafts).

---

**Bottom line:** You are in **Slice 2**, now reset to CPU-only (no CUDA/HIP/native kernels). The focus is storage efficiency and zero fragmentation via the paged KV cache allocator. Fix the critical issues above before moving to Slice 3 (gRPC router + Go operator).
