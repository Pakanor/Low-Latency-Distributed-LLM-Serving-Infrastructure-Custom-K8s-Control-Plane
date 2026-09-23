import urllib.request
import json
import torch
from typing import List, Optional


class K8sModelClient:
    def __init__(
        self,
        endpoint_url: str = "http://llm-engine-service.default.svc.cluster.local/generate_step",
        model_name: Optional[str] = None,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name or "HuggingFaceTB/SmolLM-135M-Instruct")
        except Exception as e:
            raise ImportError(f"Failed to import tokenizer: {e}")

    def generate_step_tensor(
        self,
        seq_id: int,
        input_ids: List[int],
    ) -> dict:

        payload = {
            "seq_id": seq_id,
            "input_ids": input_ids,
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
            return {
                "next_token_id": result["next_token_id"],
                "is_eos": result.get("is_eos", False),
            }


class MockModelClient:
    def generate_step_tensor(
        self,
        seq_id: int,
        input_ids: List[int],
        past_key_values: Optional[List] = None,
    ) -> dict:
        seq_len = len(input_ids)
        num_layers = 16

        if past_key_values is None:
            past_kv = []
            for _ in range(num_layers):
                key = torch.zeros(1, 2, seq_len, 8)
                value = torch.zeros(1, 2, seq_len, 8)
                past_kv.append((key, value))
        else:
            past_kv = []
            for layer in past_key_values:
                key = torch.cat([layer[0], torch.zeros(1, 2, 1, 8)], dim=2)
                value = torch.cat([layer[1], torch.zeros(1, 2, 1, 8)], dim=2)
                past_kv.append((key, value))

        return {
            "next_token_id": 100 + seq_len,
            "past_key_values": past_kv,
        }
