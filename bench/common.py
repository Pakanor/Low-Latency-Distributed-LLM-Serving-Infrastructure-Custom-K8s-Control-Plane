#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

from app.config import RuntimeConfig, configure_runtime

ROOT = Path(__file__).resolve().parents[1]


def set_determinism(seed: int, num_threads: int | None = None) -> int:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Try deterministic algorithms where possible, but don't hard-fail on CPU ops.
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    cfg = RuntimeConfig(num_threads=num_threads, seed=seed)
    threads = configure_runtime(cfg)
    return threads


def log_cpu_freqs(path: Path, label: str) -> None:
    rows = []
    base = Path("/sys/devices/system/cpu")
    for cpu in sorted(base.glob("cpu[0-9]*")):
        gov = cpu / "cpufreq/scaling_governor"
        cur = cpu / "cpufreq/scaling_cur_freq"
        if gov.exists() or cur.exists():
            rows.append(
                {
                    "label": label,
                    "cpu": cpu.name,
                    "governor": gov.read_text().strip() if gov.exists() else "",
                    "cur_khz": int(cur.read_text().strip()) if cur.exists() else None,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "cpu", "governor", "cur_khz"])
        if mode == "w":
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(ordered[int(k)])
    return float(ordered[f] * (c - k) + ordered[c] * (k - f))


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


@dataclass
class TimingResult:
    samples_s: list[float]
    stats_s: dict[str, float]
    warmup_s: list[float]


def measure(fn: Callable[[], Any], repeats: int = 5, warmup: int = 1) -> TimingResult:
    warmup_s: list[float] = []
    for _ in range(warmup):
        t0 = time.perf_counter()
        fn()
        warmup_s.append(time.perf_counter() - t0)
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return TimingResult(samples_s=samples, stats_s=summarize(samples), warmup_s=warmup_s)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # union of keys, stable order preferring common keys
    keys: list[str] = []
    seen = set()
    preferred = ["variant", "scenario", "metric", "value", "median", "p50", "p95", "p99"]
    for k in preferred:
        for r in rows:
            if k in r and k not in seen:
                keys.append(k)
                seen.add(k)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
