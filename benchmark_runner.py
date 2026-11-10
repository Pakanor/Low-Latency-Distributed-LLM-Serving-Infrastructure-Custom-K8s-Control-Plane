#!/usr/bin/env python3
"""Unified benchmark runner for the LLM infrastructure project.

Jeden punkt wejścia dla wszystkich testów wydajnościowych:
  - fragmentation  - C++ PageAllocator fragmentation benchmark (csrc/fragmentation_bench)
  - thread-safety  - C++ PageAllocator thread-safety test (csrc/thread_safety_test)
  - engine-hf      - Python 1:1 comparison: Paged KV Cache engine (/generate)
                     vs standard HuggingFace baseline (model.generate)

Każdy test można wywołać osobno przez flagę --tests (np. --tests fragmentation
lub --tests engine-hf). Domyślnie uruchamiane jest --tests all.

Nazwa modelu dla testów opartych na modelu (engine-hf) pobierana jest wyłącznie
z konfiguracji projektu (app.config.DEFAULT_CONFIG.model_name) lub zmiennej
środowiskowej MODEL_NAME — bez hardkodowania.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_CONFIG  # noqa: E402

AVAILABLE_TESTS = ("fragmentation", "thread-safety", "engine-hf", "all")

FRAGMENTATION_BIN = ROOT / "csrc" / "fragmentation_bench"
THREAD_SAFETY_BIN = ROOT / "csrc" / "thread_safety_test"
UNIFIED_BENCH = ROOT / "benchmark_comparison_unified.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner: fragmentation, thread-safety, engine-vs-HF"
    )
    parser.add_argument(
        "--tests",
        default="all",
        help=(
            "Które testy uruchomić: fragmentation, thread-safety, engine-hf, all "
            "(domyślnie all). Można podać wiele oddzielonych przecinkami."
        ),
    )
    # ---- fragmentation options ----
    parser.add_argument("--frag-sequences", type=int, default=None,
                        help="NUM_SEQUENCES dla fragmentation_bench (jeśli binary wspiera)")
    parser.add_argument("--frag-block-size", type=int, default=None,
                        help="BLOCK_SIZE dla fragmentation_bench")
    parser.add_argument("--frag-max-tokens", type=int, default=None,
                        help="MAX_TOKENS_PER_REQ dla fragmentation_bench")
    parser.add_argument("--frag-total-blocks", type=int, default=None,
                        help="TOTAL_BLOCKS dla fragmentation_bench")
    parser.add_argument("--frag-steps", type=int, default=None,
                        help="STEPS dla fragmentation_bench")
    # ---- thread-safety options ----
    parser.add_argument("--ts-duration", type=float, default=None,
                        help="Czas trwania testu wątków (sekundy, jeśli binary wspiera)")
    # ---- engine-hf options ----
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Liczba równoległych zadań (engine-hf, domyślnie 4)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        dest="max_tokens",
                        help="Maksymalna liczba generowanych tokenów (domyślnie 128)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Alias dla --max-tokens (baseline HF)")
    parser.add_argument("--prompt", default=None,
                        help="Tekst wejściowy (engine-hf)")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Adres serwera FastAPI (engine-hf)")
    parser.add_argument("--skip-engine", action="store_true",
                        help="Pomiń scenariusz silnika (engine-hf)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Pomiń scenariusz baseline HF (engine-hf)")
    # ---- general ----
    parser.add_argument("--model", default=None,
                        help=argparse.SUPPRESS)  # model zawsze z konfiguracji/ENV
    parser.add_argument("--no-color", action="store_true",
                        help="Wyłącz kolory w outputcie")
    return parser


def resolve_tests(raw: str) -> list[str]:
    selected = [t.strip() for t in raw.split(",") if t.strip()]
    invalid = [t for t in selected if t not in AVAILABLE_TESTS]
    if invalid:
        raise SystemExit(
            f"Nieznane testy: {', '.join(invalid)}. Dostępne: {', '.join(AVAILABLE_TESTS)}"
        )
    if "all" in selected:
        return ["fragmentation", "thread-safety", "engine-hf"]
    return selected


def check_binary(path: Path, name: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"Nie znaleziono binarnego {name} pod ścieżką {path}. "
            f"Skompiluj go najpierw (np. csrc/build_tests.sh)."
        )
    if not os.access(path, os.X_OK):
        raise SystemExit(f"Binarny {path} nie ma uprawnień do wykonania.")


def run_fragmentation(args: argparse.Namespace) -> dict[str, Any]:
    check_binary(FRAGMENTATION_BIN, "fragmentation_bench")
    cmd = [str(FRAGMENTATION_BIN)]
    # fragmentation_bench.cpp używa stałych wbudowanych w kompilacji,
    # nie przyjmuje argumentów CLI — uruchamiamy go bez argumentów i raportujemy.
    print(f"\n[URUCHAMIAM] fragmentation_bench: {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()
    out = proc.stdout + proc.stderr
    print(out)
    if proc.returncode != 0:
        print(f"[fragmentation_bench] zakończony z kodem błędu {proc.returncode}")
    return {
        "test": "fragmentation",
        "elapsed": t1 - t0,
        "returncode": proc.returncode,
        "output": out,
    }


def run_thread_safety(args: argparse.Namespace) -> dict[str, Any]:
    check_binary(THREAD_SAFETY_BIN, "thread_safety_test")
    cmd = [str(THREAD_SAFETY_BIN)]
    print(f"\n[URUCHAMIAM] thread_safety_test: {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()
    out = proc.stdout + proc.stderr
    print(out)
    if proc.returncode != 0:
        print(f"[thread_safety_test] zakończony z kodem błędu {proc.returncode}")
    return {
        "test": "thread-safety",
        "elapsed": t1 - t0,
        "returncode": proc.returncode,
        "output": out,
    }


def run_engine_hf(args: argparse.Namespace) -> dict[str, Any]:
    if not UNIFIED_BENCH.exists():
        raise SystemExit(f"Nie znaleziono skryptu {UNIFIED_BENCH}")
    cmd = [
        sys.executable, str(UNIFIED_BENCH),
        "--concurrency", str(args.concurrency),
        "--max-tokens", str(args.max_tokens),
        "--base-url", str(args.base_url),
    ]
    if args.max_new_tokens is not None:
        cmd += ["--max-new-tokens", str(args.max_new_tokens)]
    if args.prompt is not None:
        cmd += ["--prompt", str(args.prompt)]
    if args.skip_engine:
        cmd.append("--skip-engine")
    if args.skip_baseline:
        cmd.append("--skip-baseline")

    print(f"\n[URUCHAMIAM] benchmark_comparison_unified.py: {' '.join(cmd[1:])}")
    model_name = os.getenv("MODEL_NAME", DEFAULT_CONFIG.model_name)
    print(f"[engine-hf] Model (z konfiguracji/ENV): {model_name}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()
    out = proc.stdout + proc.stderr
    print(out)
    if proc.returncode != 0:
        print(f"[benchmark_comparison_unified] zakończony z kodem błędu {proc.returncode}")
    return {
        "test": "engine-hf",
        "elapsed": t1 - t0,
        "returncode": proc.returncode,
        "output": out,
    }


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 70)
    print("PODSUMOWANIE WSZYSTKICH TESTÓW")
    print("=" * 70)
    header = f"{'Test':<20} | {'Czas [s]':>10} | {'Kod powrotu':>11} | {'Status'}"
    print(header)
    print("-" * len(header))
    for r in results:
        status = "PASS" if r["returncode"] == 0 else "FAIL"
        print(
            f"{r['test']:<20} | {r['elapsed']:>10.3f} | {r['returncode']:>11} | {status}"
        )
    print("=" * 70 + "\n")


def main() -> None:
    args = build_parser().parse_args()

    tests = resolve_tests(args.tests)
    print(f"Wybrane testy: {', '.join(tests)}")

    results: list[dict[str, Any]] = []

    if "fragmentation" in tests:
        results.append(run_fragmentation(args))
    if "thread-safety" in tests:
        results.append(run_thread_safety(args))
    if "engine-hf" in tests:
        results.append(run_engine_hf(args))

    print_summary(results)

    any_fail = any(r["returncode"] != 0 for r in results)
    if any_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
