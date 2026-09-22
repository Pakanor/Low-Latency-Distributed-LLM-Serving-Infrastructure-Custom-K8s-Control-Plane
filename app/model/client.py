import urllib.request
import json
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
        except Exception:
            class _FallbackTokenizer:
                def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
                    return [ord(c) for c in text]

                @property
                def eos_token_id(self) -> int:
                    return 0

            self.tokenizer = _FallbackTokenizer()

    def generate_step_tensor(
        self,
        input_ids: List[int],
        block_table: List[int],
        context_len: int
    ) -> dict:
        
        payload = {
            "input_ids": input_ids,
            "block_table": block_table,
            "context_len": context_len
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
            keys = None
            values = None
            key = None
            value = None
            if result.get("keys") and result.get("values"):
                keys = torch.tensor(result["keys"])
                values = torch.tensor(result["values"])
            if result.get("key") and result.get("value"):
                key = torch.tensor(result["key"])
                value = torch.tensor(result["value"])
            return {
                "next_token_id": result["next_token_id"],
                "keys": keys,
                "values": values,
                "key": key,
                "value": value,
            }

class MockModelClient:
    def generate_step_tensor(
        self,
        input_ids: List[int],
        block_table: List[int],
        context_len: int
    ) -> dict:
        import torch as _torch
        seq_len = len(input_ids)
        keys = _torch.zeros(seq_len, 2, 8)
        values = _torch.zeros(seq_len, 2, 8)
        last_key = keys[-1:, :, :]
        last_value = values[-1:, :, :]
        return {
            "next_token_id": 100 + seq_len,
            "keys": keys,
            "values": values,
            "key": last_key.squeeze(0).squeeze(-2),
            "value": last_value.squeeze(0).squeeze(-2),
        }