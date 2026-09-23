import os
import sys
import json
import logging
import asyncio
from typing import AsyncGenerator, List, Optional, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sse_starlette.sse import EventSourceResponse

from app.engine.llm_engine import LLMEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Inference Engine API")

MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceTB/SmolLM-135M-Instruct")

model = None
tokenizer = None
kv_cache_store: dict = {}
engine: Optional[LLMEngine] = None
_step_task: Optional[asyncio.Task] = None


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_tokens: int = Field(default=50, ge=1, le=256)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


class StepGenerateRequest(BaseModel):
    seq_id: int
    input_ids: List[int]


class CompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(default=16, ge=1, le=2048)


def set_engine(engine_instance: LLMEngine) -> None:
    global engine
    engine = engine_instance


def load_model() -> bool:
    global model, tokenizer
    try:
        logger.info(f"Loading model: {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


async def _step_loop():
    while True:
        if engine is not None and engine.has_unfinished_requests():
            try:
                await asyncio.to_thread(engine.step)
                if getattr(engine, "update_event", None) is not None:
                    engine.update_event.set()
            except RuntimeError as e:
                logger.warning(f"Scheduler OOM, skipping step: {e}")
            except Exception as e:
                logger.error(f"Step loop execution error: {e}")
        await asyncio.sleep(0.005)


@app.on_event("startup")
async def startup():
    global _step_task
    if not load_model():
        logger.error("Fatal: Could not load model on startup")
        sys.exit(1)
    _step_task = asyncio.create_task(_step_loop())


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "error",
        "model": MODEL_NAME,
        "device": "cpu",
        "engine_initialized": engine is not None
    }


@app.post("/generate")
async def generate(req: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded")

    try:
        with torch.no_grad():
            inputs = tokenizer(req.prompt, return_tensors="pt")

            if inputs.input_ids.shape[1] > 2000:
                raise HTTPException(status_code=400, detail="Prompt too long")

            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {
            "response": text,
            "tokens_generated": outputs.shape[1] - inputs.input_ids.shape[1],
        }

    except torch.OutOfMemoryError:
        logger.error("CUDA OOM during generation")
        raise HTTPException(status_code=507, detail="Out of memory")
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate_step")
async def generate_step(req: StepGenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        with torch.no_grad():
            input_tensor = torch.tensor([req.input_ids], dtype=torch.long)
            past_kv = kv_cache_store.get(req.seq_id, None)
            outputs = model(input_tensor, past_key_values=past_kv, use_cache=True)
            next_token_id = int(torch.argmax(outputs.logits[0, -1, :]))
            kv_cache_store[req.seq_id] = outputs.past_key_values

        return {
            "seq_id": req.seq_id,
            "next_token_id": next_token_id,
            "is_eos": next_token_id == tokenizer.eos_token_id,
        }
    except Exception as e:
        logger.error(f"Step execution failed for seq_id {req.seq_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    if engine is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="LLM Engine is not initialized")

    prompt_tokens = tokenizer.encode(request.prompt, add_special_tokens=True)

    sequence = engine.add_request(
        prompt_token_ids=prompt_tokens,
        max_tokens=request.max_tokens,
    )

    if not hasattr(engine, "update_event") or engine.update_event is None:
        engine.update_event = asyncio.Event()

    async def token_generator() -> AsyncGenerator[str, None]:
        last_yielded_count = 0

        while not sequence.is_finished():
            try:
                await asyncio.wait_for(engine.update_event.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            engine.update_event.clear()

            current_outputs = sequence.output_token_ids
            if len(current_outputs) > last_yielded_count:
                new_tokens = current_outputs[last_yielded_count:]
                last_yielded_count = len(current_outputs)
                is_finished = sequence.is_finished()

                for token_id in new_tokens:
                    payload = {
                        "text": tokenizer.decode([token_id], skip_special_tokens=True),
                        "token_id": token_id,
                        "finished": is_finished
                    }
                    yield json.dumps(payload)

        yield json.dumps({"finished": True, "text": ""})

    return EventSourceResponse(token_generator())