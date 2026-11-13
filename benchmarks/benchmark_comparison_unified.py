#!/usr/bin/env python3
"""Unified comparison: Paged KV Cache engine (FastAPI /generate) vs standard HF baseline (model.generate).

Uruchamia dwa scenariusze 1:1 na tym samym modelu (nazwa modelu wyłącznie z
app.config.DEFAULT_CONFIG.model_name lub env MODEL_NAME) i drukuje tabelę
porównawczą: czas testu, średnia latency, przepustowość, błędy/OOM.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_CONFIG  # noqa: E402
from app.memory.strategies import resolve_device  # noqa: E402

DEFAULT_CONCURRENCY = 4
DEFAULT_MAX_TOKENS = 128
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_PROMPT = (
    "Explain Linux kernel memory management, page tables, and virtual memory mapping in complete depth:"
)
HEALTH_TIMEOUT = 5.0
REQUEST_TIMEOUT = 600.0
RSS_SAMPLE_INTERVAL = 0.05


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Porównanie 1:1 silnika Paged KV Cache vs standardowy baseline HF"
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="Liczba równoległych zadań (domyślnie 4)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        dest="max_tokens",
                        help="Maksymalna liczba generowanych tokenów (domyślnie 128)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Alias dla --max-tokens (baseline HF)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help="Tekst wejściowy używany we wszystkich zapytaniach")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="Adres serwera FastAPI (domyślnie http://localhost:8000)")
    parser.add_argument("--skip-engine", action="store_true",
                        help="Pomiń scenariusz A (własny silnik)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Pomiń scenariusz B (baseline HF)")
    return parser


def model_name() -> str:
    return os.getenv("MODEL_NAME", DEFAULT_CONFIG.model_name)


# ---------------------------------------------------------------- Scenariusz A

async def _fire_engine_request(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        started = time.perf_counter()
        try:
            resp = await client.post(
                url,
                json={"prompt": prompt, "max_tokens": max_tokens,
                      "temperature": 1.0, "top_p": 0.95},
            )
            latency = time.perf_counter() - started
            if resp.status_code == 200:
                payload = resp.json()
                return {"ok": True, "latency": latency,
                        "tokens": payload.get("tokens_generated", 0)}
            return {"ok": False, "latency": latency, "tokens": 0,
                    "error": f"HTTP {resp.status_code}"}
        except Exception as exc:
            return {"ok": False, "latency": time.perf_counter() - started,
                    "tokens": 0, "error": str(exc)[:200]}


async def run_engine_scenario(base_url: str, prompt: str, max_tokens: int,
                              concurrency: int) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/generate"
    timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=HEALTH_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(base_url.rstrip("/") + "/health",
                                timeout=HEALTH_TIMEOUT)
        resp.raise_for_status()
        health = resp.json()
        if health.get("model") and health["model"] != model_name():
            raise RuntimeError(
                f"Model na serwerze ({health.get('model')!r}) różni się od "
                f"konfiguracji ({model_name()!r})"
            )
        sem = asyncio.Semaphore(concurrency)
        started_at = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(_fire_engine_request(client, url, prompt, max_tokens, sem)
              for _ in range(concurrency))
        )
        t1 = time.perf_counter()
        ended_at = datetime.now(timezone.utc)

    ok = [r for r in results if r["ok"]]
    latencies = [r["latency"] for r in results]
    total_tokens = sum(r["tokens"] for r in ok)
    elapsed = t1 - t0
    return {
        "name": "Twój silnik (Paged KV Cache)",
        "elapsed": elapsed,
        "started_at": started_at,
        "ended_at": ended_at,
        "total": len(results),
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "avg_latency": statistics.mean(latencies) if latencies else 0.0,
        "total_tokens": total_tokens,
        "throughput": total_tokens / elapsed if elapsed else 0.0,
    }


# ---------------------------------------------------------------- Scenariusz B

def _load_hf_model(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name(), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name(),
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def _hf_generate_one(model: Any, tokenizer: Any, prompt: str,
                     max_new_tokens: int, device: torch.device) -> dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        latency = time.perf_counter() - started
        new_tokens = int(outputs.shape[1]) - prompt_tokens
        return {"ok": True, "latency": latency, "tokens": new_tokens}
    except torch.cuda.OutOfMemoryError as exc:
        latency = time.perf_counter() - started
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return {"ok": False, "latency": latency, "tokens": 0,
                "error": "OutOfMemoryError"}


def _sample_rss(process: psutil.Process, stop: threading.Event,
                bucket: list[float]) -> None:
    while not stop.is_set():
        try:
            bucket.append(process.memory_info().rss)
        except Exception:
            pass
        stop.wait(RSS_SAMPLE_INTERVAL)


def run_baseline_scenario(prompt: str, max_new_tokens: int,
                          concurrency: int, device: torch.device) -> dict[str, Any]:
    try:
        return _run_baseline_scenario_inner(prompt, max_new_tokens, concurrency, device)
    except Exception as exc:
        traceback.print_exc()
        return {
            "name": "Baseline HF (model.generate)",
            "error_msg": str(exc),
            "elapsed": 0.0, "avg_latency": 0.0, "throughput": 0.0,
            "total": concurrency, "ok": 0, "errors": concurrency,
            "oom": False, "peak_gb": None,
        }


def _run_baseline_scenario_inner(prompt: str, max_new_tokens: int,
                                 concurrency: int, device: torch.device) -> dict[str, Any]:
    model, tokenizer = _load_hf_model(device)

    process = psutil.Process()
    rss_samples: list[float] = []
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_rss, args=(process, stop, rss_samples),
                               daemon=True)
    sampler.start()

    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    oom = False
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(_hf_generate_one, model, tokenizer, prompt,
                            max_new_tokens, device)
            for _ in range(concurrency)
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if not result["ok"] and "OutOfMemory" in str(result.get("error", "")):
                oom = True
    t1 = time.perf_counter()
    ended_at = datetime.now(timezone.utc)

    stop.set()
    sampler.join(timeout=1.0)

    peak_gb = (max(rss_samples) if rss_samples else process.memory_info().rss) / (1024 ** 3)

    ok = [r for r in results if r["ok"]]
    latencies = [r["latency"] for r in results]
    total_tokens = sum(r["tokens"] for r in ok)
    elapsed = t1 - t0
    return {
        "name": "Baseline HF (model.generate)",
        "elapsed": elapsed,
        "started_at": started_at,
        "ended_at": ended_at,
        "total": len(results),
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "oom": oom,
        "peak_gb": peak_gb,
        "avg_latency": statistics.mean(latencies) if latencies else 0.0,
        "total_tokens": total_tokens,
        "throughput": total_tokens / elapsed if elapsed else 0.0,
    }


# ---------------------------------------------------------------- Raport

def _by_name(results: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for r in results:
        if r.get("name") == name:
            return r
    return {}


def _fmt_value(result: dict[str, Any], key: str, fmt: str) -> str:
    if result.get("error_msg"):
        return f"BŁĄD: {result['error_msg']}"
    val = result.get(key, "n/a")
    if isinstance(val, (int, float)):
        return f"{val:{fmt}}"
    return str(val)


def print_table(results: list[dict[str, Any]]) -> None:
    engine = _by_name(results, "Twój silnik (Paged KV Cache)")
    baseline = _by_name(results, "Baseline HF (model.generate)")

    print("\n" + "=" * 80)
    print("PORÓWNANIE 1:1: WŁASNY SILNIK (Paged KV Cache) vs BASELINE HF")
    print("=" * 80)
    header = f"{'Metryka':<26} | {'Twój silnik':>22} | {'Baseline HF':>22}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Całkowity czas testu [s]", "elapsed", ".3f"),
        ("Średnia latency zapytania [s]", "avg_latency", ".3f"),
        ("Przepustowość [tokeny/s]", "throughput", ".2f"),
        ("Liczba zadań", "total", ""),
        ("Zakończone pomyślnie", "ok", ""),
        ("Błędy / OOM", "errors", ""),
        ("Szczyt RAM/VRAM [GB]", "peak_gb", ".3f"),
    ]
    for metric, key, fmt in rows:
        eng = _fmt_value(engine, key, fmt) if engine else "n/a"
        base = _fmt_value(baseline, key, fmt) if baseline else "n/a"
        if key == "errors" and baseline and baseline.get("oom"):
            base = base + " [OOM]"
        print(f"{metric:<26} | {eng:>22} | {base:>22}")

    print("=" * 80)
    for r in results:
        print(f"\n[{r['name']}]")
        if r.get("error_msg"):
            print(f"  BŁĄD: {r['error_msg']}")
        else:
            print(f"  Całkowity czas:   {r['elapsed']:.3f} s")
            print(f"  Średnia latency:  {r['avg_latency']:.3f} s")
            print(f"  Przepustowość:    {r['throughput']:.2f} tokeny/s")
            print(f"  Sukcesy/Błędy:    {r['ok']} / {r['errors']}")
            if r.get("oom"):
                print(f"  OOM:              TAK")
            if r.get("peak_gb") is not None:
                print(f"  Szczyt pamięci:   {r['peak_gb']:.3f} GB")
    print("=" * 80 + "\n")


def main() -> None:
    args = build_parser().parse_args()

    if args.concurrency < 1:
        raise SystemExit("--concurrency musi być >= 1")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens musi być >= 1")

    max_tokens = args.max_new_tokens if args.max_new_tokens is not None else args.max_tokens
    name = model_name()
    device = resolve_device(DEFAULT_CONFIG.device)
    print(f"Model (z konfiguracji/ENV): {name}")
    print(f"Urządzenie: {device}, precyzja: float16, concurrency={args.concurrency}, "
          f"max_tokens={max_tokens}")

    results: list[dict[str, Any]] = []

    if not args.skip_engine:
        print(f">>> Scenariusz A: Twój silnik (Paged KV Cache) przez {args.base_url}/generate ...")
        try:
            res = asyncio.run(run_engine_scenario(args.base_url, args.prompt,
                                                  max_tokens, args.concurrency))
            results.append(res)
            print(f">>> Scenariusz A zakończony: {res['ok']}/{res['total']} sukcesów, "
                  f"{res['elapsed']:.3f} s")
        except Exception as exc:
            traceback.print_exc()
            results.append({
                "name": "Twój silnik (Paged KV Cache)",
                "error_msg": str(exc),
                "elapsed": 0.0, "avg_latency": 0.0, "throughput": 0.0,
                "total": args.concurrency, "ok": 0, "errors": args.concurrency,
                "peak_gb": None,
            })

    if not args.skip_baseline:
        print(f">>> Scenariusz B: Baseline HF (model.generate) na {device} ...")
        res = run_baseline_scenario(args.prompt, max_tokens,
                                    args.concurrency, device)
        results.append(res)
        if res.get("error_msg"):
            print(f">>> Scenariusz B zakończony błędem: {res['error_msg']}")
        else:
            print(f">>> Scenariusz B zakończony: {res['ok']}/{res['total']} sukcesów, "
                  f"{res['elapsed']:.3f} s, szczyt pamięci {res['peak_gb']:.3f} GB")

    if results:
        print_table(results)


if __name__ == "__main__":
    main()
