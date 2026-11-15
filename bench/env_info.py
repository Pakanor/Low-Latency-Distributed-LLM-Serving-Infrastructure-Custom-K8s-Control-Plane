#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"<unavailable: {exc}>"


def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *cmd], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _cpu_info() -> dict[str, Any]:
    model_name = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if "model name" in line:
                model_name = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    governors = {}
    freqs = {}
    try:
        base = Path("/sys/devices/system/cpu")
        for cpu in sorted(base.glob("cpu[0-9]*")):
            gov = cpu / "cpufreq/scaling_governor"
            cur = cpu / "cpufreq/scaling_cur_freq"
            if gov.exists():
                governors[cpu.name] = gov.read_text().strip()
            if cur.exists():
                freqs[cpu.name] = int(cur.read_text().strip())
    except Exception:
        pass
    return {
        "model_name": model_name,
        "logical_cores": os.cpu_count(),
        "governors": governors,
        "current_freq_khz": freqs,
    }


def _mem_info() -> dict[str, Any]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "total_bytes": vm.total,
            "available_bytes": vm.available,
            "percent_used": vm.percent,
        }
    except Exception as exc:
        return {"error": str(exc)}


def collect() -> dict[str, Any]:
    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:
        transformers_version = "unknown"
    try:
        import psutil

        psutil_version = psutil.__version__
    except Exception:
        psutil_version = "unknown"
    try:
        import llm_allocator_cpp

        ext = getattr(llm_allocator_cpp, "__file__", "unknown")
    except Exception:
        ext = "unavailable"

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": _git(["rev-parse", "HEAD"]),
            "status_porcelain": _git(["status", "--porcelain"]),
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        },
        "platform": {
            "platform": platform.platform(),
            "python": sys.version,
            "machine": platform.machine(),
        },
        "cpu": _cpu_info(),
        "memory": _mem_info(),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers_version,
            "psutil": psutil_version,
            "compiler_gxx": _run(["g++", "--version"]).splitlines()[0] if _run(["which", "g++"]) != "" else "unknown",
        },
        "extension": {"llm_allocator_cpp": ext},
        "env": {
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
        },
    }


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    data = collect()
    text = json.dumps(data, indent=2, ensure_ascii=False)
    print(text)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
