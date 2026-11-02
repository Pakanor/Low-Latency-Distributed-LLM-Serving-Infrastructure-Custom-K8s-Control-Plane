#!/usr/bin/env python3
"""Compare paged and contiguous-style KV allocation at a fixed RAM budget."""

from __future__ import annotations

import argparse
import csv
import gc
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM

from app.engine.llm_engine import LLMEngine
from app.memory.manager import PagedKVCacheManager
from app.scheduler.scheduler import Scheduler


@dataclass(frozen=True)
class Request:
    prompt_token_ids: list[int]
    max_tokens: int


@dataclass(frozen=True)
class Result:
    variant: str
    tokens_per_block: int
    total_blocks: int
    blocks_per_seq_p95: float
    theoretical_max_seqs: int
    max_parallel: int
    output_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    preemptions: int
    outputs: list[list[int]]


def build_workload(model, count: int, seed: int, min_prompt: int, max_prompt: int,
                   min_output: int, max_output: int) -> list[Request]:
    rng = random.Random(seed)
    vocab_size = int(model.config.vocab_size)
    return [
        Request(
            prompt_token_ids=[rng.randrange(vocab_size) for _ in range(rng.randint(min_prompt, max_prompt))],
            max_tokens=rng.randint(min_output, max_output),
        )
        for _ in range(count)
    ]


def run_variant(model, workload: Iterable[Request], variant: str, tokens_per_block: int,
                total_blocks: int, max_batch_size: int, max_num_batched_tokens: int) -> Result:
    workload = list(workload)
    blocks_per_seq_p95 = sum(
        math.ceil((len(request.prompt_token_ids) + request.max_tokens) / tokens_per_block)
        for request in workload
    ) / len(workload)
    manager = PagedKVCacheManager.from_model_config(
        model.config,
        total_blocks=total_blocks,
        block_size=tokens_per_block,
        dtype=torch.float32,
        device="cpu",
        total_cpu_blocks=1,
    )
    scheduler = Scheduler(
        kv_cache_manager=manager,
        max_batch_size=max_batch_size,
        max_num_batched_tokens=max_num_batched_tokens,
        block_size=tokens_per_block,
        allocator=manager.get_allocator(),
    )
    engine = LLMEngine(kv_cache_manager=manager, scheduler=scheduler, model=model)
    sequences = [
        engine.add_request(request.prompt_token_ids, max_tokens=request.max_tokens)
        for request in workload
    ]

    started = time.perf_counter()
    max_parallel = 0
    steps = 0
    while engine.has_unfinished_requests():
        update = engine.step()
        scheduled = len(update["finished"]) + len(update["running"])
        max_parallel = max(max_parallel, scheduled, len(scheduler.running))
        steps += 1
        if steps > 100_000:
            raise RuntimeError(f"{variant}: exceeded the step limit with requests still queued")
    elapsed = time.perf_counter() - started

    outputs = [sequence.output_token_ids for sequence in sequences]
    output_tokens = sum(len(output) for output in outputs)
    if len(outputs) != len(workload):
        raise AssertionError(f"{variant}: request count changed during execution")
    if any(len(output) == 0 for output in outputs):
        raise AssertionError(f"{variant}: a request completed without output tokens")

    result = Result(
        variant=variant,
        tokens_per_block=tokens_per_block,
        total_blocks=total_blocks,
        blocks_per_seq_p95=blocks_per_seq_p95,
        theoretical_max_seqs=total_blocks // math.ceil(blocks_per_seq_p95),
        max_parallel=max_parallel,
        output_tokens=output_tokens,
        elapsed_seconds=elapsed,
        tokens_per_second=output_tokens / elapsed if elapsed else 0.0,
        preemptions=scheduler.preemption_count,
        outputs=outputs,
    )
    del engine, scheduler, manager
    gc.collect()
    return result


def write_csv(path: Path, results: list[Result]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "variant", "tokens_per_block", "total_blocks", "blocks_per_seq_p95",
            "theoretical_max_seqs", "max_parallel",
            "output_tokens", "elapsed_seconds", "tokens_per_second", "preemptions",
        ])
        for result in results:
            writer.writerow([
                result.variant, result.tokens_per_block, result.total_blocks,
                f"{result.blocks_per_seq_p95:.3f}", result.theoretical_max_seqs,
                result.max_parallel, result.output_tokens,
                f"{result.elapsed_seconds:.6f}", f"{result.tokens_per_second:.3f}",
                result.preemptions,
            ])


def write_svg(path: Path, results: list[Result]) -> None:
    width, height = 760, 420
    chart_left, chart_top, chart_width, chart_height = 80, 40, 640, 280
    max_value = max(result.max_parallel for result in results) or 1
    bar_width = 180
    gap = 120
    colors = {"paged": "#176b87", "contiguous": "#d97745"}
    bars = []
    for index, result in enumerate(results):
        bar_height = chart_height * result.max_parallel / max_value
        x = chart_left + 80 + index * (bar_width + gap)
        y = chart_top + chart_height - bar_height
        bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" '
            f'fill="{colors[result.variant]}" rx="4" />'
            f'<text x="{x + bar_width / 2}" y="{y - 10:.1f}" text-anchor="middle">'
            f'{result.max_parallel}</text>'
            f'<text x="{x + bar_width / 2}" y="{chart_top + chart_height + 28}" '
            f'text-anchor="middle">{result.variant}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>text {{ font: 16px sans-serif; fill: #25323a; }} .title {{ font-size: 20px; font-weight: 600; }}</style>
  <rect width="100%" height="100%" fill="#f7f4ee" />
  <text x="{width / 2}" y="25" text-anchor="middle" class="title">Maximum concurrent sequences at 512 MiB KV budget</text>
  <line x1="{chart_left}" y1="{chart_top + chart_height}" x2="{chart_left + chart_width}" y2="{chart_top + chart_height}" stroke="#25323a" />
  <line x1="{chart_left}" y1="{chart_top}" x2="{chart_left}" y2="{chart_top + chart_height}" stroke="#25323a" />
  <text x="25" y="{chart_top + chart_height / 2}" transform="rotate(-90 25 {chart_top + chart_height / 2})" text-anchor="middle">sequences</text>
  {''.join(bars)}
</svg>
'''
    path.write_text(svg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M-Instruct")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-mib", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--min-prompt", type=int, default=8)
    parser.add_argument("--max-prompt", type=int, default=48)
    parser.add_argument("--min-output", type=int, default=8)
    parser.add_argument("--max-output", type=int, default=24)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--max-batched-tokens", type=int, default=8192)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    if args.requests < 1:
        raise ValueError("--requests must be positive")
    if args.max_prompt + args.max_output > args.max_model_len:
        raise ValueError("prompt and output ranges must fit inside --max-model-len")

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    workload = build_workload(
        model, args.requests, args.seed, args.min_prompt, args.max_prompt,
        args.min_output, args.max_output,
    )

    bytes_per_token = (
        2 * model.config.num_hidden_layers
        * getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
        * getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
        * torch.tensor([], dtype=torch.float32).element_size()
    )
    budget_bytes = args.budget_mib * 1024 * 1024
    paged_blocks = budget_bytes // (16 * bytes_per_token)
    contiguous_blocks = budget_bytes // (args.max_model_len * bytes_per_token)
    if paged_blocks < 1 or contiguous_blocks < 1:
        raise ValueError("KV budget is too small for the selected model and context")

    paged = run_variant(
        model, workload, "paged", 16, paged_blocks,
        args.max_batch_size, args.max_batched_tokens,
    )
    contiguous = run_variant(
        model, workload, "contiguous", args.max_model_len, contiguous_blocks,
        args.max_batch_size, args.max_batched_tokens,
    )
    if paged.outputs != contiguous.outputs:
        raise AssertionError("paged and contiguous variants produced different tokens")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "throughput_results.csv", [paged, contiguous])
    write_svg(args.output_dir / "throughput_benchmark.svg", [paged, contiguous])
    print("variant,max_parallel,output_tokens,seconds,tokens_per_second,preemptions")
    for result in (paged, contiguous):
        print(
            f"{result.variant},{result.max_parallel},{result.output_tokens},"
            f"{result.elapsed_seconds:.3f},{result.tokens_per_second:.2f},{result.preemptions}"
        )
    print(f"identical_outputs=true requests={args.requests} seed={args.seed}")


if __name__ == "__main__":
    main()
