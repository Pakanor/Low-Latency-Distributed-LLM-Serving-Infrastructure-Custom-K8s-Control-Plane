import os
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="LLM Baseline Serving")

MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceTB/SmolLM-135M-Instruct")

print(f"Loading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float32, 
    device_map="cpu"
)
print("Model loaded successfully!")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/generate")
def generate(req: GenerateRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=req.max_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": text}