import asyncio
import json
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.engine.llm_engine import LLMEngine
from app.scheduler.sequence import Sequence

app = FastAPI(title="LLM Inference Engine API")

# Globalna instancja silnika (inicjalizowana przy starcie)
engine: LLMEngine = None


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 16


def set_engine(engine_instance: LLMEngine) -> None:
    global engine
    engine = engine_instance


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="LLM Engine is not initialized")

    # Tworzymy nową sekwencję na podstawie zapytania
    # Zakładamy proste przekształcenie promptu na tokeny na potrzeby orkiestratora
    prompt_tokens = [ord(c) for c in request.prompt]  # Szybki mock tokenizacji
    seq_id = id(request)
    sequence = Sequence(seq_id=seq_id, prompt_token_ids=prompt_tokens, max_tokens=request.max_tokens)

    # Dodajemy sekwencję do kolejki Schedulera
    engine.add_sequence(sequence)

    async def token_generator() -> AsyncGenerator[str, None]:
        last_yielded_count = 0
        
        while not sequence.is_finished():
            # Dajemy szansę na wykonanie kroku silnika
            await asyncio.sleep(0.01)

            current_outputs = sequence.output_token_ids
            if len(current_outputs) > last_yielded_count:
                new_tokens = current_outputs[last_yielded_count:]
                last_yielded_count = len(current_outputs)

                for token_id in new_tokens:
                    payload = {
                        "text": chr(token_id) if token_id < 256 else f"[{token_id}]",
                        "token_id": token_id,
                        "finished": False
                    }
                    yield json.dumps(payload)

        # Wysyłamy sygnał zakończenia
        yield json.dumps({"finished": True, "text": ""})

    return EventSourceResponse(token_generator())