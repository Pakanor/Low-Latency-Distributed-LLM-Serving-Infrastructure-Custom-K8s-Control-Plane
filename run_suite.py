#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import torch

from benchmarks import benchmark_runner
from app.config import DEFAULT_CONFIG

ROOT = Path(__file__).resolve().parent
PROFILE_NAMES = ("smoke", "full")
PROFILE_TESTS = {
    "smoke": ("fragmentation", "thread-safety", "throughput"),
    "full": ("fragmentation", "thread-safety", "engine-hf", "throughput"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILE_NAMES, default="smoke")
    parser.add_argument(
        "--tests",
        default=None,
        help="Opcjonalny wybór testów oddzielony przecinkami; nadpisuje profil.",
    )
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Wznów istniejący katalog wyników, np. results/20261113T120000Z.",
    )
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--skip-engine", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--throughput-requests", type=int, default=None)
    parser.add_argument("--throughput-output-dir", type=Path, default=None)
    return parser


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _metadata(
    profile: str,
    tests: list[str],
    args: argparse.Namespace,
    legacy: argparse.Namespace,
) -> dict[str, Any]:
    try:
        transformers_version = version("transformers")
    except PackageNotFoundError:
        transformers_version = "unknown"
    try:
        psutil_version = version("psutil")
    except PackageNotFoundError:
        psutil_version = "unknown"
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "tests": tests,
        "arguments": {
            "runner": {key: _json_value(value) for key, value in vars(args).items()},
            "benchmark": {
                key: _json_value(value) for key, value in vars(legacy).items()
            },
        },
        "git": {
            "commit": _git_value("rev-parse", "HEAD"),
            "dirty": bool(_git_value("status", "--porcelain")),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "threads": {
            "omp": os.getenv("OMP_NUM_THREADS"),
            "mkl": os.getenv("MKL_NUM_THREADS"),
        },
        "memory": {
            "available_bytes": _available_memory_bytes(),
        },
        "libraries": {
            "torch": torch.__version__,
            "transformers": transformers_version,
            "psutil": psutil_version,
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "config": {
            "model_name": os.getenv("MODEL_NAME", DEFAULT_CONFIG.model_name),
            "dtype": os.getenv("LLM_DTYPE", DEFAULT_CONFIG.dtype),
            "device": os.getenv("LLM_DEVICE", DEFAULT_CONFIG.device),
            "max_model_len": DEFAULT_CONFIG.max_model_len,
            "budget_mib": DEFAULT_CONFIG.budget_mib,
            "block_size": DEFAULT_CONFIG.block_size,
            "seed": DEFAULT_CONFIG.seed,
        },
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _available_memory_bytes() -> int | None:
    try:
        import psutil

        return psutil.virtual_memory().total
    except (ImportError, AttributeError):
        return None


def validate_result(result: dict[str, Any]) -> None:
    required = {"test", "elapsed", "returncode", "output"}
    missing = required.difference(result)
    if missing:
        raise ValueError(f"Benchmark result is missing fields: {sorted(missing)}")
    if not isinstance(result["test"], str) or not result["test"]:
        raise ValueError("Benchmark result field 'test' must be a non-empty string")
    if not isinstance(result["returncode"], int):
        raise ValueError("Benchmark result field 'returncode' must be an integer")
    if isinstance(result["elapsed"], bool) or not isinstance(result["elapsed"], (int, float)):
        raise ValueError("Benchmark result field 'elapsed' must be numeric")
    if isinstance(result["elapsed"], float) and result["elapsed"] != result["elapsed"]:
        raise ValueError("Benchmark result field 'elapsed' must not be NaN")


def _new_run_dir(results_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = results_dir / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = results_dir / f"{timestamp}-{suffix}"
        suffix += 1
    return run_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _legacy_args(args: argparse.Namespace, run_dir: Path) -> argparse.Namespace:
    legacy = benchmark_runner.build_parser().parse_args([])
    legacy.base_url = args.base_url
    legacy.skip_engine = args.skip_engine
    legacy.skip_baseline = args.skip_baseline
    legacy.concurrency = args.concurrency or 1
    legacy.max_tokens = args.max_tokens or 8
    legacy.throughput_requests = args.throughput_requests or 5
    legacy.throughput_output_dir = str(args.throughput_output_dir or run_dir)
    legacy.throughput_min_prompt = 8
    legacy.throughput_max_prompt = 8
    legacy.throughput_min_output = 8
    legacy.throughput_max_output = 8
    legacy.throughput_max_batch_size = 8
    legacy.throughput_max_batched_tokens = 256
    legacy.throughput_max_model_len = DEFAULT_CONFIG.max_model_len
    return legacy


def _select_tests(args: argparse.Namespace) -> list[str]:
    raw = args.tests or ",".join(PROFILE_TESTS[args.profile])
    return benchmark_runner.resolve_tests(raw)


def run_suite(args: argparse.Namespace) -> int:
    tests = _select_tests(args)
    run_dir = args.resume or _new_run_dir(args.results_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tests_dir = run_dir / "tests"
    tests_dir.mkdir(exist_ok=True)

    legacy = _legacy_args(args, run_dir)
    metadata_path = run_dir / "meta.json"
    if metadata_path.exists() and args.resume:
        metadata = json.loads(metadata_path.read_text())
    else:
        metadata = _metadata(args.profile, tests, args, legacy)
        _write_json(metadata_path, metadata)

    runners: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
        "fragmentation": benchmark_runner.run_fragmentation,
        "thread-safety": benchmark_runner.run_thread_safety,
        "engine-hf": benchmark_runner.run_engine_hf,
        "throughput": benchmark_runner.run_throughput,
    }
    results: list[dict[str, Any]] = []
    for test_name in tests:
        result_path = tests_dir / f"{test_name}.json"
        if args.resume and result_path.exists():
            result = json.loads(result_path.read_text())
            print(f"[POMIJAM] {test_name}: wynik już istnieje")
        else:
            result = runners[test_name](legacy)
            validate_result(result)
            _write_json(result_path, result)
        results.append(result)

    _write_json(run_dir / "results.json", {"results": results})
    benchmark_runner.print_summary(results)
    return 1 if any(result.get("returncode") != 0 for result in results) else 0


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(run_suite(args))


if __name__ == "__main__":
    main()
