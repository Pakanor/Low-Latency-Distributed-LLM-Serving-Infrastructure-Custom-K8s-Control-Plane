#!/usr/bin/env python3
"""Generuje trace'y obciążenia (krok 2).

Każdy wiersz JSONL to jedno żądanie:
  id, distribution, prompt_token_ids, prompt_len, output_len,
  arrival_gap (bezwymiarowa przerwa Exp(1)), seed.

Czas przybycia NIE jest zapisywany w pliku. Runner odtwarza go z tempa:
  arrival_time[i] = sum(arrival_gap[:i+1]) / rate
gdzie rate to jedno z ARRIVAL_RATES (0.5, 1, 2, 4, 8, 16 req/s).
Pomocnicza funkcja: arrival_times(rows, rate).

Dystrybucje: siatka stała (grid), stała długość (const), lognormal, zipf
(dla fragmentacji) oraz burst (te same żądania co lognormal, arrival_gap=0,
maksymalna współbieżność). Długość wyjścia wymuszona: output_len ignoruje EOS
(min_tokens=max_tokens). Stały seed -> identyczny md5 plików.

Generator waliduje: prompt_len + output_len <= max_model_len,
len(prompt_token_ids) == prompt_len, brak tokenów specjalnych w promptach
oraz determinizm (ten sam seed -> ten sam md5 każdego pliku).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

ARRIVAL_RATES: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
MIN_REQUESTS_PER_CELL = 8

Trace = dict[str, dict[str, Any]]


def arrival_times(rows: Iterable[dict[str, Any]], rate: float) -> list[float]:
    """Odtwarza arrival_time z znormalizowanych gap'ów i tempa (req/s)."""
    if rate <= 0:
        return [0.0 for _ in rows]
    times: list[float] = []
    t = 0.0
    for row in rows:
        t += float(row.get("arrival_gap", 0.0)) / rate
        times.append(t)
    return times


def _sample_ids(vocab_ids: list[int], rng: random.Random, length: int) -> list[int]:
    return [vocab_ids[rng.randrange(len(vocab_ids))] for _ in range(length)]


def _non_special_ids(tokenizer) -> list[int]:
    vocab = tokenizer.get_vocab()
    special = set(tokenizer.all_special_ids)
    ids = [tid for tok, tid in vocab.items() if tid not in special]
    if not ids:
        ids = list(range(tokenizer.vocab_size))
    return sorted(ids)  # deterministyczna kolejność niezależna od hashowania dictów


def _exp_gaps(rng: random.Random, n: int) -> list[float]:
    # Znormalizowany proces Poissona: gap ~ Exp(1), średnia 1.0.
    return [round(rng.expovariate(1.0), 9) for _ in range(n)]


def _zipf_raw(rng: random.Random, a: float, vmin: int, vmax: int) -> int:
    # numpy-free zipf-ish: P(x) ~ x^-a, rejection sampling over [vmin, vmax]
    while True:
        x = vmin + int(rng.random() * (vmax - vmin + 1))
        if rng.random() < (x ** (-a)) / (vmin ** (-a)):
            return x


def _lognormal_raw(rng: random.Random, mu: float, sigma: float) -> int:
    return int(round(math.exp(rng.gauss(mu, sigma))))


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(ordered[f])
    return float(ordered[f] * (c - k) + ordered[c] * (k - f))


def _len_stats(values: list[int]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "max": max(values),
    }


def _row(
    rid: str,
    distribution: str,
    ids: list[int],
    output_len: int,
    gap: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "id": rid,
        "distribution": distribution,
        "prompt_token_ids": ids,
        "prompt_len": len(ids),
        "output_len": output_len,
        "arrival_gap": gap,
        "seed": seed,
    }


def _build(
    vocab_ids: list[int],
    input_grid: tuple[int, ...],
    output_grid: tuple[int, ...],
    max_model_len: int,
    num_requests: int,
    seed: int,
    lognormal_mu: float,
    lognormal_sigma: float,
    zipf_a: float,
    zipf_vmin: int,
    zipf_vmax: int,
) -> tuple[Trace, dict[int, tuple[int, int]]]:
    """Buduje wiersze wszystkich plików; zwraca też pominięte komórki grid."""
    rng = random.Random(seed)
    traces: Trace = {}

    def clamp(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    # --- grid: input x output, min. MIN_REQUESTS_PER_CELL żądań na komórkę ---
    cells = [(i, o) for i in input_grid for o in output_grid if i + o <= max_model_len]
    skipped = [(i, o) for i in input_grid for o in output_grid if i + o > max_model_len]
    per_cell = max(MIN_REQUESTS_PER_CELL, math.ceil(num_requests / len(cells))) if cells else 0
    grid_rows: list[dict[str, Any]] = []
    gaps = _exp_gaps(rng, len(cells) * per_cell)
    rid = 0
    for inp, outl in cells:
        for _ in range(per_cell):
            grid_rows.append(_row(f"grid_{rid:04d}", "grid", _sample_ids(vocab_ids, rng, inp),
                                  outl, gaps[rid], seed))
            rid += 1
    traces["grid.jsonl"] = {
        "rows": grid_rows,
        "truncated": {"prompt": 0, "output": 0, "requests": 0},
        "output_len_distribution": {"type": "grid", "values": list(output_grid)},
    }

    # --- const ---
    const_len = clamp(512, 1, max_model_len - clamp(128, 1, max_model_len // 2))
    out_const = clamp(128, 1, max_model_len - const_len)
    const_rows: list[dict[str, Any]] = []
    gaps = _exp_gaps(rng, num_requests)
    for i in range(num_requests):
        ids = _sample_ids(vocab_ids, rng, const_len)
        const_rows.append(_row(f"const_{i:04d}", "const", ids, out_const, gaps[i], seed))
    traces["const.jsonl"] = {
        "rows": const_rows,
        "truncated": {"prompt": 0, "output": 0, "requests": 0},
        "output_len_distribution": {"type": "const", "value": out_const},
        "const_definition": {
            "prompt_len": const_len,
            "output_len": out_const,
            "rule": "prompt_len=min(512, max_model_len-128), output_len=min(128, max_model_len-prompt_len)",
        },
    }

    # --- lognormal ---
    prompt_cap = max_model_len - min(output_grid)
    log_rows: list[dict[str, Any]] = []
    gaps = _exp_gaps(rng, num_requests)
    trunc_prompt = trunc_output = trunc_req = 0
    for i in range(num_requests):
        raw_in = _lognormal_raw(rng, lognormal_mu, lognormal_sigma)
        inp = clamp(raw_in, 1, prompt_cap)
        raw_out = _lognormal_raw(rng, lognormal_mu, lognormal_sigma)
        outl = clamp(raw_out, 1, max_model_len - inp)
        tp, to = raw_in > prompt_cap, raw_out > max_model_len - inp
        trunc_prompt += int(tp)
        trunc_output += int(to)
        trunc_req += int(tp or to)
        ids = _sample_ids(vocab_ids, rng, inp)
        log_rows.append(_row(f"lognormal_{i:04d}", "lognormal", ids, outl, gaps[i], seed))
    log_spec = {
        "type": "lognormal",
        "mu": lognormal_mu,
        "sigma": lognormal_sigma,
        "prompt_clamp": [1, prompt_cap],
        "output_clamp": [1, max_model_len],
    }
    traces["lognormal.jsonl"] = {
        "rows": log_rows,
        "truncated": {"prompt": trunc_prompt, "output": trunc_output, "requests": trunc_req},
        "output_len_distribution": log_spec,
    }

    # --- zipf ---
    prompt_cap = max_model_len - clamp(zipf_vmin, 1, max_model_len // 2)
    zipf_rows: list[dict[str, Any]] = []
    gaps = _exp_gaps(rng, num_requests)
    trunc_prompt = trunc_output = trunc_req = 0
    for i in range(num_requests):
        raw_in = _zipf_raw(rng, zipf_a, zipf_vmin, zipf_vmax)
        inp = clamp(raw_in, zipf_vmin, prompt_cap)
        raw_out = _zipf_raw(rng, zipf_a, zipf_vmin, zipf_vmax)
        outl = clamp(raw_out, zipf_vmin, max_model_len - inp)
        tp, to = raw_in > prompt_cap, raw_out > max_model_len - inp
        trunc_prompt += int(tp)
        trunc_output += int(to)
        trunc_req += int(tp or to)
        ids = _sample_ids(vocab_ids, rng, inp)
        zipf_rows.append(_row(f"zipf_{i:04d}", "zipf", ids, outl, gaps[i], seed))
    traces["zipf.jsonl"] = {
        "rows": zipf_rows,
        "truncated": {"prompt": trunc_prompt, "output": trunc_output, "requests": trunc_req},
        "output_len_distribution": {
            "type": "zipf",
            "a": zipf_a,
            "support": [zipf_vmin, zipf_vmax],
            "output_clamp": [zipf_vmin, max_model_len],
        },
    }

    # --- burst: te same żądania co lognormal, arrival_gap=0 (max współbieżność) ---
    burst_rows = [
        _row(f"burst_{i:04d}", "burst", r["prompt_token_ids"], r["output_len"], 0.0, seed)
        for i, r in enumerate(log_rows)
    ]
    traces["burst.jsonl"] = {
        "rows": burst_rows,
        "truncated": dict(traces["lognormal.jsonl"]["truncated"]),
        "output_len_distribution": log_spec,
        "arrival_mode": "burst",
    }

    return traces, skipped


def _validate(
    traces: Trace,
    special_ids: set[int],
    max_model_len: int,
) -> None:
    for name, payload in traces.items():
        for row in payload["rows"]:
            rid = row["id"]
            if row["prompt_len"] + row["output_len"] > max_model_len:
                raise ValueError(
                    f"{name}:{rid} prompt_len+output_len={row['prompt_len'] + row['output_len']} "
                    f"> max_model_len={max_model_len}")
            if len(row["prompt_token_ids"]) != row["prompt_len"]:
                raise ValueError(f"{name}:{rid} len(prompt_token_ids)!=prompt_len")
            bad = special_ids.intersection(row["prompt_token_ids"])
            if bad:
                raise ValueError(f"{name}:{rid} prompt zawiera tokeny specjalne: {sorted(bad)[:5]}")
            if row["arrival_gap"] < 0:
                raise ValueError(f"{name}:{rid} ujemny arrival_gap")


def _serialize(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def generate(
    model_name: str,
    out_dir: Path,
    seed: int = 42,
    max_model_len: int = 2048,
    num_requests: int = 300,
    input_grid: tuple[int, ...] = (128, 512, 1536),
    output_grid: tuple[int, ...] = (128, 512),
    lognormal_mu: float = 6.0,
    lognormal_sigma: float = 1.0,
    zipf_a: float = 1.5,
    zipf_vmin: int = 8,
    zipf_vmax: int = 2040,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_ids = _non_special_ids(tokenizer)
    special_ids = set(tokenizer.all_special_ids)

    build_args = dict(
        input_grid=input_grid,
        output_grid=output_grid,
        max_model_len=max_model_len,
        num_requests=num_requests,
        seed=seed,
        lognormal_mu=lognormal_mu,
        lognormal_sigma=lognormal_sigma,
        zipf_a=zipf_a,
        zipf_vmin=zipf_vmin,
        zipf_vmax=zipf_vmax,
    )
    traces, skipped = _build(vocab_ids, **build_args)
    _validate(traces, special_ids, max_model_len)

    payloads: dict[str, bytes] = {name: _serialize(p["rows"]) for name, p in traces.items()}
    digests: dict[str, str] = {name: hashlib.md5(b).hexdigest() for name, b in payloads.items()}

    # Deterministyczność: ten sam seed -> identyczny md5 każdego pliku.
    traces2, _ = _build(vocab_ids, **build_args)
    for name, payload in traces2.items():
        again = hashlib.md5(_serialize(payload["rows"])).hexdigest()
        if again != digests[name]:
            raise ValueError(f"niedeterministyczny trace {name}: {digests[name]} != {again}")

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, data in payloads.items():
        path = out_dir / name
        path.write_bytes(data)
        written.append(path)

    def file_meta(name: str, payload: dict[str, Any]) -> dict[str, Any]:
        rows = payload["rows"]
        return {
            "requests": len(rows),
            "md5": digests[name],
            "arrival_mode": payload.get("arrival_mode", "poisson-normalized"),
            "prompt_len": _len_stats([r["prompt_len"] for r in rows]),
            "output_len": _len_stats([r["output_len"] for r in rows]),
            "truncated_to_max_model_len": payload["truncated"],
            "output_len_distribution": payload["output_len_distribution"],
        }

    meta = {
        "model": model_name,
        "seed": seed,
        "max_model_len": max_model_len,
        "num_requests": num_requests,
        "arrival": {
            "field": "arrival_gap",
            "gap_distribution": "exp(1)",
            "replay_rates_req_s": list(ARRIVAL_RATES),
            "replay_formula": "arrival_time[i] = sum(arrival_gap[0:i+1]) / rate",
            "burst_file": "burst.jsonl",
        },
        "grid": {
            "input_grid": list(input_grid),
            "output_grid": list(output_grid),
            "requests_per_cell": max(
                MIN_REQUESTS_PER_CELL,
                math.ceil(num_requests / max(1, len(input_grid) * len(output_grid)
                                             - len(skipped)))) if input_grid else 0,
            "skipped_cells": [
                {"input_len": i, "output_len": o,
                 "reason": f"input+output={i + o} > max_model_len={max_model_len}"}
                for i, o in skipped
            ],
        },
        "const_definition": traces["const.jsonl"]["const_definition"],
        "distributions": {
            "lognormal": {"mu": lognormal_mu, "sigma": lognormal_sigma},
            "zipf": {"a": zipf_a, "vmin": zipf_vmin, "vmax": zipf_vmax},
        },
        "validation": {
            "prompt_plus_output_within_max_model_len": True,
            "prompt_len_matches_tokenized_length": True,
            "no_special_tokens_in_prompts": True,
            "same_seed_same_md5": True,
        },
        "files": {name: file_meta(name, p) for name, p in traces.items()},
    }
    (out_dir / "trace_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[make_trace] {len(written)} pliki w {out_dir}")
    print(f"{'plik':16s} {'req':>4s} {'in min/med/p95/max':>24s} "
          f"{'out min/med/p95/max':>24s} {'obcięte':>7s}  md5")
    for name in traces:
        m = meta["files"][name]
        pin, pout = m["prompt_len"], m["output_len"]
        print(f"{name:16s} {m['requests']:4d} "
              f"{pin['min']:6d}/{pin['median']:6.0f}/{pin['p95']:6.0f}/{pin['max']:5d} "
              f"{pout['min']:6d}/{pout['median']:6.0f}/{pout['p95']:6.0f}/{pout['max']:5d} "
              f"{m['truncated_to_max_model_len']['requests']:7d}  {m['md5']}")
    for cell in meta["grid"]["skipped_cells"]:
        print(f"[make_trace] pominięta komórka grid: {cell['input_len']}x{cell['output_len']} ({cell['reason']})")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM-135M-Instruct")
    ap.add_argument("--out-dir", default=str(ROOT / "bench" / "traces"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--num-requests", type=int, default=300)
    ap.add_argument("--arrival-rate", type=float, default=None,
                    help="ignorowane (zgodność): arrival_time liczy runner z arrival_gap i tempa")
    ap.add_argument("--closed-loop", action="store_true",
                    help="ignorowane (zgodność): użyj pliku burst.jsonl")
    ap.add_argument("--input-grid", type=str, default="128,512,1536")
    ap.add_argument("--output-grid", type=str, default="128,512")
    ap.add_argument("--lognormal-mu", type=float, default=6.0)
    ap.add_argument("--lognormal-sigma", type=float, default=1.0)
    ap.add_argument("--zipf-a", type=float, default=1.5)
    ap.add_argument("--zipf-vmin", type=int, default=8)
    ap.add_argument("--zipf-vmax", type=int, default=2040)
    args = ap.parse_args()

    if args.arrival_rate is not None or args.closed_loop:
        print("[make_trace] uwaga: --arrival-rate/--closed-loop są ignorowane; "
              "przybycia odtwarza runner z arrival_gap", file=sys.stderr)

    generate(
        model_name=args.model,
        out_dir=Path(args.out_dir),
        seed=args.seed,
        max_model_len=args.max_model_len,
        num_requests=args.num_requests,
        input_grid=tuple(int(x) for x in args.input_grid.split(",")),
        output_grid=tuple(int(x) for x in args.output_grid.split(",")),
        lognormal_mu=args.lognormal_mu,
        lognormal_sigma=args.lognormal_sigma,
        zipf_a=args.zipf_a,
        zipf_vmin=args.zipf_vmin,
        zipf_vmax=args.zipf_vmax,
    )


if __name__ == "__main__":
    main()
