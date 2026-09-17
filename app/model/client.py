import urllib.request
import json
from typing import List, Optional


class K8sModelClient:
    def __init__(
        self, 
        endpoint_url: str = "http://llm-engine-service.default.svc.cluster.local/generate",
        model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct"
    ) -> None:
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        return self._tokenizer

    def generate_step(self, prompt_token_ids: List[int], output_token_ids: List[int], max_tokens: int = 1) -> int:
        all_tokens = prompt_token_ids + output_token_ids
        prompt_text = self.tokenizer.decode(all_tokens, skip_special_tokens=True)
        
        payload = {
            "prompt": prompt_text,
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "top_p": 0.95
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))
            generated_text = result.get("response", "")
            
            new_tokens = self.tokenizer.encode(generated_text, add_special_tokens=False)
            if len(new_tokens) > len(all_tokens):
                return new_tokens[len(all_tokens)]
            elif new_tokens:
                return new_tokens[-1]
            return self.tokenizer.eos_token_id


class MockModelClient:
    """Sztuczny klient do deterministycznych testów jednostkowych bez połączenia HTTP."""
    def generate_step(self, prompt_token_ids: List[int], output_token_ids: List[int], max_tokens: int = 1) -> int:
        return 100 + len(output_token_ids)