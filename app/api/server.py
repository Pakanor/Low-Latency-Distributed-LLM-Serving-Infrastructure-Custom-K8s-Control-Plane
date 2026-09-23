import asyncio
import json
import logging
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.engine.llm_engine import LLMEngine

logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Inference Engine API")

engine: LLMEngine = None
_step_task: asyncio.Task | None = None


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 16


def set_engine(engine_instance: LLMEngine) -> None:
    global engine
    engine = engine_instance


async def _step_loop():
    while True:
        if engine is not None and engine.has_unfinished_requests():
            try:
                await asyncio.to_thread(engine.step)
            except Exception as e:
                logger.error(f"Step error: {e}")
        await asyncio.sleep(0.01)


@app.on_event("startup")
async def startup():
    global _step_task
    _step_task = asyncio.create_task(_step_loop())


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="LLM Engine is not initialized")

    prompt_tokens = [ord(c) for c in request.prompt]

    sequence = engine.add_request(
        prompt_token_ids=prompt_tokens,
        max_tokens=request.max_tokens,
    )

    async def token_generator() -> AsyncGenerator[str, None]:
        last_yielded_count = 0

        while not sequence.is_finished():
            await asyncio.sleep(0.01)

            current_outputs = sequence.output_token_ids
            if len(current_outputs) > last_yielded_count:
                new_tokens = current_outputs[last_yielded_count:]
                last_yielded_count = len(current_outputs)
                is_finished = sequence.is_finished()

                for token_id in new_tokens:
                    payload = {
                        "text": chr(token_id) if 0 <= token_id < 256 else f"[{token_id}]",
                        "token_id": token_id,
                        "finished": is_finished
                    }
                    yield json.dumps(payload)

        yield json.dumps({"finished": True, "text": ""})

    return EventSourceResponse(token_generator())
