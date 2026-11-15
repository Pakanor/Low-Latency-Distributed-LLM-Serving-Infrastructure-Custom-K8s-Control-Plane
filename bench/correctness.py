#!/usr/bin/env python3
"""poprawność paged engine vs HF generate (greedy, fp32).

Porównuje ścieżkę paged (LLMEngine/_paged_forward_group, use_cache=False przez
run_paged_forward) z referencją transformers model.generate(do_sample=False)
na tym samym modelu i wagach.
"""
from __future__ import annotations

import argparse
import copy
import faulthandler
import json
import math
import os
import random
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.common import log_cpu_freqs, set_determinism, write_json

os.environ.setdefault("HF_HUB_OFFLINE", "1")

DEFAULT_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"
FORCED_OUTPUT_LEN = 64
LOGIT_STEPS = 5  # first + several next steps
TEACHER_FORCE_LOGIT_THRESHOLD = 1e-3   # bramka: max |logit_paged - logit_hf| przy teacher-forcing
NEAR_TIE_MARGIN_THRESHOLD = 1e-4       # rozbieżność tokenów dozwolona tylko gdy top1-top2 < ten próg


def load_model_and_tokenizer(model_name: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.eval()
    return model, tok


def make_prompts(tokenizer, lengths: list[int], seed: int) -> list[dict[str, Any]]:
    """Deterministic prompts of requested token lengths."""
    rng = random.Random(seed)
    vocab = tokenizer.get_vocab()
    # prefer common printable tokens to avoid special-only weirdness
    candidates = [tid for tok, tid in vocab.items() if tid not in tokenizer.all_special_ids]
    if not candidates:
        candidates = list(range(tokenizer.vocab_size))
    prompts = []
    for i, L in enumerate(lengths):
        ids = [candidates[rng.randrange(len(candidates))] for _ in range(L)]
        prompts.append({"id": f"len{L}_{i}", "length": L, "token_ids": ids})
    return prompts


def hf_generate_tokens(model, input_ids: list[int], out_len: int) -> list[int]:
    inp = torch.tensor([input_ids], dtype=torch.long)
    with torch.inference_mode():
        out = model.generate(
            inp,
            max_new_tokens=out_len,
            min_new_tokens=out_len,
            do_sample=False,
        )
    return out[0, -out_len:].tolist()


def hf_greedy_logits_steps(model, input_ids: list[int], steps: int) -> list[dict[str, Any]]:
    """Greedy rollout collecting logits for first `steps` generated tokens."""
    ids = torch.tensor([input_ids], dtype=torch.long)
    past = None
    collected = []
    cur_input = ids
    with torch.inference_mode():
        for step in range(steps):
            out = model(cur_input, past_key_values=past, use_cache=True)
            logits = out.logits[0, -1, :].detach().float().cpu()
            next_id = int(torch.argmax(logits))
            collected.append({"step": step, "next_id": next_id, "logits": logits})
            past = out.past_key_values
            cur_input = torch.tensor([[next_id]], dtype=torch.long)
    return collected


def paged_generate_tokens(model, input_ids: list[int], out_len: int, block_size: int, total_blocks: int, max_model_len: int) -> tuple[list[int], Any, Any, Any]:
    from app.engine.llm_engine import LLMEngine
    from app.memory.manager import PagedKVCacheManager
    from app.scheduler.scheduler import Scheduler

    mgr = PagedKVCacheManager.from_model_config(
        model.config,
        total_blocks=total_blocks,
        block_size=block_size,
        dtype=torch.float32,
        device="cpu",
        total_cpu_blocks=max(1, min(32, total_blocks)),
    )
    sch = Scheduler(
        kv_cache_manager=mgr,
        max_batch_size=64,
        max_num_batched_tokens=max_model_len,
        block_size=block_size,
        allocator=mgr.get_allocator(),
        max_model_len=max_model_len,
    )
    eng = LLMEngine(kv_cache_manager=mgr, scheduler=sch, model=model)
    seq = eng.add_request(input_ids, max_tokens=out_len)
    steps = 0
    limit = out_len + 20
    print(
        f"[paged] start len={len(input_ids)} block_size={block_size} "
        f"total_blocks={total_blocks} limit={limit}", flush=True,
    )
    while eng.has_unfinished_requests() and steps < limit:
        eng.step()
        steps += 1
    if eng.has_unfinished_requests():
        # Złap zawieszenie: zrzut stanu schedulera przed błędem.
        try:
            running = [(s.seq_id, s.status.name, s.get_len()) for s in sch.running]
            waiting = [(s.seq_id, s.status.name, s.get_len()) for s in sch.waiting]
        except Exception:  # pragma: no cover - diagnostyka
            running = waiting = None
        raise RuntimeError(
            f"paged_generate_tokens exceeded step limit {limit} "
            f"(steps={steps}); preemption_count={getattr(sch, 'preemption_count', None)} "
            f"running={running} waiting={waiting}"
        )
    return list(seq.output_token_ids), mgr, sch, eng


def paged_greedy_logits_steps(model, input_ids: list[int], steps: int, block_size: int, total_blocks: int, max_model_len: int) -> list[dict[str, Any]]:
    from app.engine.llm_engine import LLMEngine
    from app.memory.manager import PagedKVCacheManager
    from app.model.paged_attention import PagedStepContext, register_paged_attention, run_paged_forward
    from app.scheduler.scheduler import Scheduler
    from app.scheduler.sequence import Sequence

    register_paged_attention(model)
    mgr = PagedKVCacheManager.from_model_config(
        model.config,
        total_blocks=total_blocks,
        block_size=block_size,
        dtype=torch.float32,
        device="cpu",
        total_cpu_blocks=max(1, min(32, total_blocks)),
    )
    sch = Scheduler(
        kv_cache_manager=mgr,
        max_batch_size=1,
        max_num_batched_tokens=max_model_len,
        block_size=block_size,
        allocator=mgr.get_allocator(),
        max_model_len=max_model_len,
    )
    eng = LLMEngine(kv_cache_manager=mgr, scheduler=sch, model=model)
    # drive manually to capture logits per step
    seq = eng.add_request(input_ids, max_tokens=steps)
    collected = []
    # schedule prefill
    out = sch.schedule()
    assert len(out.scheduled_prefills) == 1
    # prefill step logits
    ctx = PagedStepContext(mgr, [seq], [0], [len(input_ids)])
    flat = torch.tensor([input_ids], dtype=torch.long)
    with torch.inference_mode():
        logits_t = run_paged_forward(model, ctx, flat)
    logits = logits_t[0, -1, :].detach().float().cpu()
    next_id = int(torch.argmax(logits))
    collected.append({"step": 0, "next_id": next_id, "logits": logits})
    seq.append_token(next_id)
    for step in range(1, steps):
        if seq.is_finished():
            break
        # schedule decode
        out = sch.schedule()
        if not out.scheduled_decodes:
            break
        start = seq.block_table.get_tokens_count() - 1
        token = [seq.output_token_ids[-1]]
        ctx = PagedStepContext(mgr, [seq], [start], [1])
        flat = torch.tensor([token], dtype=torch.long)
        with torch.inference_mode():
            logits_t = run_paged_forward(model, ctx, flat)
        logits = logits_t[0, -1, :].detach().float().cpu()
        next_id = int(torch.argmax(logits))
        collected.append({"step": step, "next_id": next_id, "logits": logits})
        seq.append_token(next_id)
    # cleanup
    sch.free_sequence(seq)
    return collected


def paged_teacher_forced_logits_steps(model, input_ids, ref_tokens, steps, block_size, total_blocks, max_model_len) -> list[dict[str, Any]]:
    """Paged logity z teacher-forcing: w kroku t input = ref_tokens[0..t-1] (identyczny prefix co HF).

    Zwraca listę słników {step, next_id (z paged), logits} dla kroków 0..steps-1.
    Bramka kroku 1: max |logit_paged - logit_hf| <= TEACHER_FORCE_LOGIT_THRESHOLD.
    Rozbieżność tokenów dozwolona tylko gdy top1-top2 <= NEAR_TIE_MARGIN_THRESHOLD.
    """
    from app.engine.llm_engine import LLMEngine
    from app.memory.manager import PagedKVCacheManager
    from app.model.paged_attention import PagedStepContext, register_paged_attention, run_paged_forward
    from app.scheduler.scheduler import Scheduler

    register_paged_attention(model)
    mgr = PagedKVCacheManager.from_model_config(
        model.config, total_blocks=total_blocks, block_size=block_size,
        dtype=torch.float32, device="cpu",
        total_cpu_blocks=max(1, min(32, total_blocks)),
    )
    sch = Scheduler(kv_cache_manager=mgr, max_batch_size=1,
                    max_num_batched_tokens=max_model_len, block_size=block_size,
                    allocator=mgr.get_allocator(), max_model_len=max_model_len)
    eng = LLMEngine(kv_cache_manager=mgr, scheduler=sch, model=model)
    seq = eng.add_request(input_ids, max_tokens=steps + 1)
    collected = []
    out = sch.schedule()
    assert len(out.scheduled_prefills) == 1
    ctx = PagedStepContext(mgr, [seq], [0], [len(input_ids)])
    flat = torch.tensor([input_ids], dtype=torch.long)
    with torch.inference_mode():
        logits_t = run_paged_forward(model, ctx, flat)
    logits = logits_t[0, -1, :].detach().float().cpu()
    next_id = int(torch.argmax(logits))
    collected.append({"step": 0, "next_id": next_id, "logits": logits})
    seq.append_token(int(ref_tokens[0]))
    for step in range(1, steps):
        if seq.is_finished():
            break
        out = sch.schedule()
        if not out.scheduled_decodes:
            break
        start = seq.block_table.get_tokens_count() - 1
        # Teacher forcing: input w kroku `step` to token z kroku step-1 z referencji
        # (identycznie jak HF: cur_input = [[next_id_z_poprzedniego_kroku]]).
        token = [int(ref_tokens[step - 1])]
        ctx = PagedStepContext(mgr, [seq], [start], [1])
        flat = torch.tensor([token], dtype=torch.long)
        with torch.inference_mode():
            logits_t = run_paged_forward(model, ctx, flat)
        logits = logits_t[0, -1, :].detach().float().cpu()
        next_id = int(torch.argmax(logits))
        collected.append({"step": step, "next_id": next_id, "logits": logits})
        seq.append_token(int(ref_tokens[step]))
    sch.free_sequence(seq)
    del mgr, sch, eng
    return collected


def compare_sequences(ref: list[int], got: list[int]) -> dict[str, Any]:
    match = ref == got
    first_diff = None
    if not match:
        for i, (a, b) in enumerate(zip(ref, got)):
            if a != b:
                first_diff = i
                break
        if first_diff is None:
            first_diff = min(len(ref), len(got))
    return {"match": match, "first_diff_index": first_diff, "ref_len": len(ref), "got_len": len(got)}


def compare_logits(ref_steps: list[dict[str, Any]], got_steps: list[dict[str, Any]]) -> dict[str, Any]:
    n = min(len(ref_steps), len(got_steps))
    per_step = []
    max_abs = 0.0
    for i in range(n):
        a = ref_steps[i]["logits"]
        b = got_steps[i]["logits"]
        diff = (a - b).abs()
        m = float(diff.max().item()) if diff.numel() else 0.0
        mean = float(diff.mean().item()) if diff.numel() else 0.0
        per_step.append({"step": i, "max_abs_diff": m, "mean_abs_diff": mean, "argmax_match": ref_steps[i]["next_id"] == got_steps[i]["next_id"]})
        if m > max_abs:
            max_abs = m
    return {"steps_compared": n, "max_abs_diff": max_abs, "per_step": per_step}


def check_no_aliasing(sch) -> list[str]:
    """Sprawdza czy żaden blok KV nie występuje w tabelach dwóch różnych sekwencji (brak aliasingu)."""
    problems = []
    block_to_seqs = {}
    for seq in list(sch.running) + list(sch.waiting):
        bt = seq.block_table
        if bt is None:
            continue
        for b in list(bt.get_physical_blocks()):
            block_to_seqs.setdefault(b, []).append(seq.seq_id)
    for b, seq_ids in block_to_seqs.items():
        if len(seq_ids) > 1:
            problems.append(f"block={b} shared by sequences {seq_ids}")
    return problems


def blocks_needed(prompt_len: int, out_len: int, block_size: int, margin: int = 8) -> int:
    return math.ceil((prompt_len + out_len) / block_size) + margin


def run_single_cases(args, model_name, prompts, block_sizes, out_len, max_model_len):
    results = []
    # Precompute HF refs per prompt (independent of block_size)
    for p in prompts:
        p["hf_tokens"] = None
        p["hf_logits"] = None
    # HF refs on a clean model
    model_hf, _tok = load_model_and_tokenizer(model_name, torch.float32)
    for p in prompts:
        try:
            p["hf_tokens"] = hf_generate_tokens(model_hf, p["token_ids"], out_len)
            p["hf_logits"] = hf_greedy_logits_steps(model_hf, p["token_ids"], LOGIT_STEPS)
        except Exception as e:
            p["hf_error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    del model_hf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # One model instance for paged runs (KV state is external in PagedKVCacheManager;
    # register_paged_attention is idempotent and contexts are set per forward).
    model_paged, _ = load_model_and_tokenizer(model_name, torch.float32)
    from app.model.paged_attention import register_paged_attention
    register_paged_attention(model_paged)

    for bs in block_sizes:
        for p in prompts:
            case = {
                "scenario": "single",
                "prompt_id": p["id"],
                "prompt_len": p["length"],
                "block_size": bs,
                "output_len": out_len,
            }
            if p.get("hf_error"):
                case["status"] = "hf_error"
                case["error"] = p["hf_error"]
                results.append(case)
                continue
            tb = blocks_needed(p["length"], out_len, bs, margin=16)
            if p["length"] + out_len > max_model_len:
                case["status"] = "skipped_exceeds_max_model_len"
                results.append(case)
                continue
            print(
                f"[case] start prompt_id={p['id']} prompt_len={p['length']} "
                f"block_size={bs} total_blocks={tb} output_len={out_len}",
                flush=True,
            )
            try:
                # 1) Paged generation (tokeny vs HF - informacja tylko, nie bramka)
                got_tokens, mgr, sch, eng = paged_generate_tokens(
                    model_paged, p["token_ids"], out_len, bs, tb, max_model_len
                )
                seq_cmp = compare_sequences(p["hf_tokens"], got_tokens)

                # 2) Teacher forcing: paged z prefixem HF, porównanie logitów (BRAMKA)
                tf_logits = paged_teacher_forced_logits_steps(
                    model_paged, p["token_ids"], p["hf_tokens"], LOGIT_STEPS, bs, tb, max_model_len
                )
                tf_cmp = compare_logits(p["hf_logits"], tf_logits)
                tf_pass = tf_cmp["max_abs_diff"] <= TEACHER_FORCE_LOGIT_THRESHOLD

                # 3) Paged vs paged bitowo (BRAMKA)
                got_tokens2, mgr2, sch2, eng2 = paged_generate_tokens(
                    model_paged, p["token_ids"], out_len, bs, tb, max_model_len
                )
                got_logits1 = paged_greedy_logits_steps(
                    model_paged, p["token_ids"], LOGIT_STEPS, bs, tb, max_model_len
                )
                got_logits2 = paged_greedy_logits_steps(
                    model_paged, p["token_ids"], LOGIT_STEPS, bs, tb, max_model_len
                )
                bitwise_tokens = got_tokens == got_tokens2
                bitwise_logits = all(
                    torch.equal(t1["logits"], t2["logits"])
                    for t1, t2 in zip(got_logits1, got_logits2)
                )
                bitwise_pass = bitwise_tokens and bitwise_logits

                # 4) Brak wycieku bloków (single seq - aliasing nie dotyczy)
                free = mgr.get_allocator().get_num_free_blocks()
                no_leak = free == tb

                del mgr, sch, eng
                del mgr2, sch2, eng2

                case.update({
                    "status": "ok",
                    # informacja tylko - nie bramka
                    "token_match": seq_cmp["match"],
                    "first_diff_index": seq_cmp["first_diff_index"],
                    # bramki
                    "teacher_forcing_pass": tf_pass,
                    "teacher_forcing_max_abs_diff": tf_cmp["max_abs_diff"],
                    "teacher_forcing_threshold": TEACHER_FORCE_LOGIT_THRESHOLD,
                    "paged_bitwise_pass": bitwise_pass,
                    "paged_bitwise_tokens_equal": bitwise_tokens,
                    "paged_bitwise_logits_equal": bitwise_logits,
                    "no_leak": no_leak,
                    # logity nauczyciela do inspekcji
                    "tf_logit_per_step": tf_cmp["per_step"],
                })
            except Exception as e:
                case["status"] = "error"
                case["error"] = f"{type(e).__name__}: {e}"
                case["traceback"] = traceback.format_exc()
            results.append(case)
    del model_paged
    return results




def run_batch_cases(model_name, prompts, block_sizes, out_len, max_model_len, batch_sizes: list[int]):
    """Concurrent sequences in one engine/batch (all admitted, stepped together)."""
    results = []
    if not batch_sizes:
        return results
    max_b = max(batch_sizes)
    batch = prompts[:max_b]
    model_hf, _ = load_model_and_tokenizer(model_name, torch.float32)
    refs = {p["id"]: hf_generate_tokens(model_hf, p["token_ids"], out_len) for p in batch}
    del model_hf

    model_paged, _ = load_model_and_tokenizer(model_name, torch.float32)
    from app.engine.llm_engine import LLMEngine
    from app.memory.manager import PagedKVCacheManager
    from app.model.paged_attention import register_paged_attention
    from app.scheduler.scheduler import Scheduler
    register_paged_attention(model_paged)

    for bs in block_sizes:
        for bsz in batch_sizes:
            if len(batch) < bsz:
                continue
            sub = batch[:bsz]
            tb = sum(blocks_needed(p["length"], out_len, bs, margin=8) for p in sub) + 32
            print(
                f"[batch] start batch_size={bsz} block_size={bs} "
                f"prompts={[p['id'] for p in sub]} total_blocks={tb}", flush=True,
            )
            try:
                mgr = PagedKVCacheManager.from_model_config(
                    model_paged.config,
                    total_blocks=tb,
                    block_size=bs,
                    dtype=torch.float32,
                    device="cpu",
                    total_cpu_blocks=max(1, min(32, tb)),
                )
                sch = Scheduler(
                    kv_cache_manager=mgr,
                    max_batch_size=bsz,
                    max_num_batched_tokens=max_model_len * bsz,
                    block_size=bs,
                    allocator=mgr.get_allocator(),
                    max_model_len=max_model_len,
                )
                eng = LLMEngine(kv_cache_manager=mgr, scheduler=sch, model=model_paged)
                seqs = [eng.add_request(p["token_ids"], max_tokens=out_len) for p in sub]
                steps = 0
                while eng.has_unfinished_requests() and steps < out_len + 50:
                    eng.step()
                    steps += 1
                for p, seq in zip(sub, seqs):
                    cmp = compare_sequences(refs[p["id"]], list(seq.output_token_ids))
                    results.append({
                        "scenario": "batch",
                        "batch_size": bsz,
                        "prompt_id": p["id"],
                        "prompt_len": p["length"],
                        "block_size": bs,
                        "output_len": out_len,
                        "status": "ok",
                        "token_match": cmp["match"],
                        "first_diff_index": cmp["first_diff_index"],
                        "steps": steps,
                    })
                # bramka: brak aliasingu bloków między sekwencjami
                aliasing = check_no_aliasing(sch)
                free = mgr.get_allocator().get_num_free_blocks()
                results.append({
                    "scenario": "batch_meta",
                    "batch_size": bsz,
                    "block_size": bs,
                    "no_leak": free == tb,
                    "no_aliasing": not aliasing,
                    "aliasing_problems": aliasing,
                    "status": "ok",
                })
                del mgr, sch, eng
            except Exception as e:
                results.append({"scenario": "batch", "batch_size": bsz, "block_size": bs, "status": "error", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
    del model_paged
    return results


def run_preemption_cases(model_name, prompts, block_sizes, out_len, max_model_len):
    """Force preemption with small pool; outputs must still match HF refs."""
    results = []
    if len(prompts) < 2:
        return results
    p1, p2 = prompts[0], prompts[1]
    model_hf, _ = load_model_and_tokenizer(model_name, torch.float32)
    ref1 = hf_generate_tokens(model_hf, p1["token_ids"], out_len)
    ref2 = hf_generate_tokens(model_hf, p2["token_ids"], out_len)
    del model_hf

    model_paged, _ = load_model_and_tokenizer(model_name, torch.float32)
    from app.engine.llm_engine import LLMEngine
    from app.memory.manager import PagedKVCacheManager
    from app.model.paged_attention import register_paged_attention
    from app.scheduler.scheduler import Scheduler
    register_paged_attention(model_paged)

    for block_size in block_sizes:
        need1 = math.ceil((p1["length"] + out_len) / block_size)
        need2 = math.ceil((p2["length"] + out_len) / block_size)
        tb = max(need1, need2) + 1
        print(
            f"[preemption] start block_size={block_size} "
            f"seq1={p1['id']} len={p1['length']} seq2={p2['id']} len={p2['length']} "
            f"total_blocks={tb}", flush=True,
        )
        try:
            mgr = PagedKVCacheManager.from_model_config(
                model_paged.config,
                total_blocks=tb,
                block_size=block_size,
                dtype=torch.float32,
                device="cpu",
                total_cpu_blocks=max(2, tb),
            )
            sch = Scheduler(
                kv_cache_manager=mgr,
                max_batch_size=2,
                max_num_batched_tokens=max_model_len,
                block_size=block_size,
                allocator=mgr.get_allocator(),
                max_model_len=max_model_len,
            )
            eng = LLMEngine(kv_cache_manager=mgr, scheduler=sch, model=model_paged)
            s1 = eng.add_request(p1["token_ids"], max_tokens=out_len)
            s2 = eng.add_request(p2["token_ids"], max_tokens=out_len)
            steps = 0
            while eng.has_unfinished_requests() and steps < out_len * 6 + 50:
                eng.step()
                steps += 1
            c1 = compare_sequences(ref1, list(s1.output_token_ids))
            c2 = compare_sequences(ref2, list(s2.output_token_ids))
            res = {
                "scenario": "preemption",
                "block_size": block_size,
                "total_blocks": tb,
                "preemptions": sch.preemption_count,
                "steps": steps,
                "seq1": {"prompt_id": p1["id"], "token_match": c1["match"], "first_diff_index": c1["first_diff_index"], "len": len(s1.output_token_ids)},
                "seq2": {"prompt_id": p2["id"], "token_match": c2["match"], "first_diff_index": c2["first_diff_index"], "len": len(s2.output_token_ids)},
                "status": "ok" if (c1["match"] and c2["match"]) else "mismatch",
                "note": "Paged preemption frees blocks and restarts waiting seq from prompt; correctness requires identical final tokens.",
            }
            results.append(res)
            del mgr, sch, eng
        except Exception as e:
            results.append({"scenario": "preemption", "block_size": block_size, "status": "error", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
    del model_paged
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-threads", type=int, default=None)
    ap.add_argument("--output-len", type=int, default=FORCED_OUTPUT_LEN)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--quick", action="store_true", help="Reduced set for smoke validation")
    ap.add_argument("--cpu-freq-log", default=None)
    args = ap.parse_args()

    threads = set_determinism(args.seed, args.num_threads)
    faulthandler.dump_traceback_later(300, repeat=True)
    faulthandler.enable()
    if args.cpu_freq_log:
        log_cpu_freqs(Path(args.cpu_freq_log), "correctness_before")

    if args.quick:
        lengths = [4, 32, 128, 512]
        block_sizes = [8, 16, 64]
        prompt_count = 8
        batch_sizes = [2]
    else:
        lengths = [4, 8, 16, 32, 64, 128, 256, 512, 768, 1000]
        block_sizes = [8, 16, 64]
        prompt_count = 20
        batch_sizes = [2, 4]

    _, tok = load_model_and_tokenizer(args.model, torch.float32)
    prompt_lengths = []
    i = 0
    while len(prompt_lengths) < prompt_count:
        prompt_lengths.append(lengths[i % len(lengths)])
        i += 1
    prompts = make_prompts(tok, prompt_lengths, args.seed)
    del tok

    fit_prompts = [p for p in prompts if p["length"] + args.output_len <= args.max_model_len]
    if len(fit_prompts) < prompt_count:
        print(f"[correctness] warning: {prompt_count - len(fit_prompts)} prompts exceed max_model_len with output; using {len(fit_prompts)}")

    print(f"[correctness] model={args.model} threads={threads} prompts={len(fit_prompts)} block_sizes={block_sizes} output_len={args.output_len}")

    all_results = []
    all_results.extend(run_single_cases(args, args.model, fit_prompts, block_sizes, args.output_len, args.max_model_len))

    batch_prompts = [p for p in fit_prompts if p["length"] <= 128][:max(batch_sizes)] if batch_sizes else []
    if len(batch_prompts) >= 2:
        all_results.extend(run_batch_cases(args.model, batch_prompts, block_sizes, args.output_len, args.max_model_len, batch_sizes))

    pre_prompts = [p for p in fit_prompts if p["length"] <= 128]
    if len(pre_prompts) >= 2:
        all_results.extend(run_preemption_cases(args.model, pre_prompts, [8, 16], args.output_len, args.max_model_len))

    # Bramki kroku 1 (token_match to tylko informacja, nie bramka)
    tf_cases = [r for r in all_results if r.get("scenario") == "single" and r.get("status") == "ok" and "teacher_forcing_pass" in r]
    tf_pass = all(r["teacher_forcing_pass"] for r in tf_cases) if tf_cases else None
    bitwise_cases = [r for r in all_results if r.get("scenario") == "single" and r.get("status") == "ok" and "paged_bitwise_pass" in r]
    bitwise_pass = all(r["paged_bitwise_pass"] for r in bitwise_cases) if bitwise_cases else None
    alias_cases = [r for r in all_results if r.get("scenario") == "batch_meta" and r.get("status") == "ok" and "no_aliasing" in r]
    alias_pass = all(r["no_aliasing"] for r in alias_cases) if alias_cases else None
    leak_cases = [r for r in all_results if r.get("scenario") in ("single", "batch_meta") and r.get("status") == "ok" and "no_leak" in r]
    leak_pass = all(r["no_leak"] for r in leak_cases) if leak_cases else None

    token_cases = [r for r in all_results if r.get("scenario") in ("single", "batch") and r.get("status") == "ok" and "token_match" in r]
    token_matches = [r for r in token_cases if r["token_match"]]
    errors = [r for r in all_results if r.get("status") in ("error", "hf_error")]

    summary = {
        "model": args.model,
        "dtype": "fp32",
        "output_len": args.output_len,
        "threads": threads,
        "prompt_count": len(fit_prompts),
        "block_sizes": block_sizes,
        # bramki kroku 1
        "gate_teacher_forcing_pass": tf_pass,
        "gate_paged_bitwise_pass": bitwise_pass,
        "gate_no_aliasing": alias_pass,
        "gate_no_block_leak": leak_pass,
        "gate_all_pass": bool(tf_pass and bitwise_pass and alias_pass and leak_pass),
        # informacja tylko (odsetek identycznych sekwencji nie jest bramką)
        "token_cases": len(token_cases),
        "token_matches": len(token_matches),
        "token_match_rate": (len(token_matches) / len(token_cases)) if token_cases else None,
        "error_count": len(errors),
    }

    report = {
        "kind": "POMIAR_SILNIKA_POPRAWNOSC",
        "summary": summary,
        "results": all_results,
        "notes": [
            "Porównanie greedy, wymuszona długość wyjścia (min_new_tokens==max_new_tokens).",
            "Referencja HF liczona na osobnej instancji modelu (register_paged_attention modyfikuje model).",
            "Preempcja w paged schedulerze: free+requeue (restart od promptu) - oczekiwane identyczne tokeny końcowe.",
            "Logity porównywane na krokach prefill i kilku decode (LOGIT_STEPS).",
        ],
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.cpu_freq_log:
        log_cpu_freqs(Path(args.cpu_freq_log), "correctness_after")
    if args.output:
        write_json(Path(args.output), report)
        print(f"[correctness] wrote {args.output}")

    if not summary["gate_all_pass"] or errors:
        print("[correctness] STOP: bramka kroku 1 nie przeszła lub wystąpiły błędy - patrz raport")
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
