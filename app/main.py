import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Baseline Serving")

MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceTB/SmolLM-135M-Instruct")

model = None
tokenizer = None

def load_model():
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
        logger.info(" Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f" Failed to load model: {e}")
        return False

if not load_model():
    logger.error("Fatal: Could not load model on startup")
    sys.exit(1)

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_tokens: int = Field(default=50, ge=1, le=256)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)

@app.get("/health")
def health():
    """Health check endpoint."""
    status = "ok" if model is not None else "error"
    return {
        "status": status,
        "model": MODEL_NAME,
        "device": "cpu",
    }

@app.post("/generate")
async def generate(req: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        with torch.no_grad():
            inputs = tokenizer(req.prompt, return_tensors="pt")
            
            # Validate input length
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
        raise HTTPException(status_code=507, detail="Out of GPU memory")
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


class TensorGenerateRequest(BaseModel):
    input_ids: List[int]
    block_table: List[int]
    context_len: int

@app.post("/generate_step")
async def generate_step(req: TensorGenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        with torch.no_grad():
            inputs = torch.tensor([req.input_ids], dtype=torch.long)
            outputs = model(inputs)
            next_token_id = int(torch.argmax(outputs.logits[0, -1, :]))
            
        return {"next_token_id": next_token_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))