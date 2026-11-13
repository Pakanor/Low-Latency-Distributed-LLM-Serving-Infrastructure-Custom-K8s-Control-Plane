import os
import sys
import json
import logging
import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sse_starlette.sse import EventSourceResponse

from app.config import DEFAULT_CONFIG, RuntimeConfig, configure_runtime
from app.engine.llm_engine import LLMEngine
from app.memory.strategies import resolve_device, resolve_dtype
from app.scheduler.sequence import Sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_CONFIG.model_name)

# Wall-clock timeout (seconds) for any single request. Sequences that exceed
# this are swept by the step loop and their KV blocks freed. Set to 0 to
# disable (not recommended — it is the only thing that bounds KV growth).
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "0"))

model = None
tokenizer = None
model_device: torch.device = torch.device("cpu")
_step_task: Optional[asyncio.Task] = None
step_event = asyncio.Event()


def _admit(prompt_tokens: List[int], max_tokens: int) -> Sequence:
    if engine is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="LLM Engine is not initialized")
    if engine.scheduler.is_full():
        raise HTTPException(
            status_code=503,
            detail="KV cache is full; wait for running requests to finish and retry.",
        )
    deadline = (time.monotonic() + REQUEST_TIMEOUT_SECONDS) if REQUEST_TIMEOUT_SECONDS > 0 else None
    sequence = engine.add_request(
        prompt_token_ids=prompt_tokens,
        max_tokens=max_tokens,
        deadline=deadline,
    )
    engine.update_event = step_event
    step_event.set()
    return sequence


def load_model(config: RuntimeConfig = DEFAULT_CONFIG) -> bool:
    global model, tokenizer, model_device
    try:
        configure_runtime(config)
        device = resolve_device(config.device)
        dtype = resolve_dtype(config.dtype)
        logger.info(f"Loading model: {config.model_name} on {device} ({dtype})...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=dtype,
            trust_remote_code=True,
        )
        model = model.to(device)
        model.eval()
        model_device = device
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


@asynccontextmanager
async def lifespan(app_):
    global model, tokenizer, _step_task, step_event
    step_event = asyncio.Event()
    if model is None:
        if not load_model(DEFAULT_CONFIG):
            logger.error("Fatal: Could not load model on startup")
            sys.exit(1)
    if engine is not None:
        engine.model = model
        engine.update_event = step_event
    _step_task = asyncio.create_task(_step_loop())
    yield
    if _step_task is not None:
        _step_task.cancel()
        try:
            await _step_task
        except asyncio.CancelledError:
            pass
    if model is not None:
        del model
        model = None
    if tokenizer is not None:
        del tokenizer
        tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Server shutdown complete.")


app = FastAPI(title="LLM Inference Engine API", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(default=50, ge=1, le=DEFAULT_CONFIG.max_model_len)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


class CompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(default=16, ge=1, le=DEFAULT_CONFIG.max_model_len)


engine: Optional[LLMEngine] = None


def set_engine(engine_instance: LLMEngine) -> None:
    global engine
    engine = engine_instance


async def _step_loop():
    while True:
        await step_event.wait()
        step_event.clear()
        if engine is None:
            continue
    
        try:
            engine.sweep_abandoned()
        except Exception as e:
            logger.warning(f"Sweep error: {e}")
        if not engine.has_unfinished_requests():
            continue
        try:
            await asyncio.to_thread(engine.step)
        except RuntimeError as e:
            logger.warning(f"Scheduler OOM, skipping step: {e}")
        except Exception as e:
            logger.error(f"Step loop execution error: {e}")
        if engine is not None and engine.has_unfinished_requests():
            step_event.set()


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "error",
        "model": MODEL_NAME,
        "device": str(model_device),
        "engine_initialized": engine is not None,
    }


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw: Request):
    prompt_tokens = tokenizer.encode(request.prompt, add_special_tokens=True)
    sequence = _admit(prompt_tokens, request.max_tokens)

    async def token_generator() -> AsyncGenerator[str, None]:
        last_yielded_count = 0
        try:
            while not sequence.is_finished():
                if await raw.is_disconnected():
                    sequence.abandon_requested = True
                    step_event.set()
                    yield json.dumps({"error": "client_disconnected", "finished": True})
                    return

                try:
                    await asyncio.wait_for(engine.update_event.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                engine.update_event.clear()

                if sequence.deadline is not None and time.monotonic() >= sequence.deadline:
                    sequence.abandon_requested = True
                    step_event.set()
                    yield json.dumps({"error": "timeout", "finished": True})
                    return

                current_outputs = sequence.output_token_ids
                if len(current_outputs) > last_yielded_count:
                    new_tokens = current_outputs[last_yielded_count:]
                    last_yielded_count = len(current_outputs)
                    is_finished = sequence.is_finished()

                    for token_id in new_tokens:
                        payload = {
                            "text": tokenizer.decode([token_id], skip_special_tokens=True),
                            "token_id": token_id,
                            "finished": is_finished,
                        }
                        yield json.dumps(payload)
        finally:
            if not sequence.is_finished():
                sequence.abandon_requested = True
                step_event.set()

        yield json.dumps({"finished": True, "text": ""})

    return EventSourceResponse(token_generator())


@app.post("/generate")
async def generate(req: GenerateRequest, raw: Request):
    prompt_tokens = tokenizer.encode(req.prompt, add_special_tokens=True)

    if len(prompt_tokens) > DEFAULT_CONFIG.max_model_len:
        raise HTTPException(status_code=400, detail="Prompt too long")

    sequence = _admit(prompt_tokens, req.max_tokens)

    try:
        while not sequence.is_finished():
            if await raw.is_disconnected():
                sequence.abandon_requested = True
                step_event.set()
                raise HTTPException(status_code=499, detail="Client disconnected")

            try:
                await asyncio.wait_for(step_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            step_event.clear()

            if sequence.deadline is not None and time.monotonic() >= sequence.deadline:
                sequence.abandon_requested = True
                step_event.set()
                raise HTTPException(status_code=504, detail="Request timed out")

        generated_tokens = sequence.output_token_ids
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return {
            "response": text,
            "tokens_generated": len(generated_tokens),
            "prompt_tokens": len(prompt_tokens),
        }
    except HTTPException:
        if not sequence.is_finished():
            sequence.abandon_requested = True
            step_event.set()
        raise
    except Exception as e:
        if not sequence.is_finished():
            sequence.abandon_requested = True
            step_event.set()
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/metrics")
async def metrics():
    lines = []
    if engine is not None:
        try:
            alloc = None
            if hasattr(engine.kv_cache_manager, "get_allocator"):
                alloc = engine.kv_cache_manager.get_allocator()
            elif hasattr(engine.scheduler, "allocator"):
                alloc = engine.scheduler.allocator
            if alloc is not None:
                free = alloc.get_num_free_blocks()
                if hasattr(alloc, "get_total_blocks"):
                    total = alloc.get_total_blocks()
                elif hasattr(alloc, "get_num_total_blocks"):
                    total = alloc.get_num_total_blocks()
                else:
                    total = getattr(engine.kv_cache_manager, "total_blocks", 1)
            else:
                free = 0
                total = 1
        except Exception:
            free = 0
            total = 1
        ratio = (free / total) if total else 0.0
        active = len(getattr(engine.scheduler, "running", []))
        preemptions = getattr(engine.scheduler, "preemption_count", 0)
        lines.append(f"kv_cache_free_pages_ratio {ratio}")
        lines.append(f"kv_cache_active_streams {active}")
        lines.append(f"kv_cache_preemptions_total {preemptions}")
    else:
        lines.append("kv_cache_free_pages_ratio 1.0")
        lines.append("kv_cache_active_streams 0")
        lines.append("kv_cache_preemptions_total 0")
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines) + "\n")
