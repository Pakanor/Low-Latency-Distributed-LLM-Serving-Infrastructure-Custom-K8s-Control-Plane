import urllib.request
import json
import base64
import os
import torch
from typing import List, Optional, Dict, Tuple

from app.config import DEFAULT_CONFIG


def _serialize_tensor(tensor: torch.Tensor) -> str:
    return base64.b64encode(tensor.cpu().numpy().tobytes()).decode("ascii")


def _deserialize_tensor(data: str, dtype: torch.dtype, shape: Tuple[int, ...]) -> torch.Tensor:
    buf = base64.b64decode(data.encode("ascii"))
    arr = torch.frombuffer(buf, dtype=dtype).reshape(shape)
    return arr


def _serialize_past_kv(past_key_values: Tuple) -> List[Dict]:
    serialized = []
    for layer_kv in past_key_values:
        key, value = layer_kv
        serialized.append({
            "key": _serialize_tensor(key),
            "value": _serialize_tensor(value),
            "key_shape": list(key.shape),
            "value_shape": list(value.shape),
            "dtype": str(key.dtype),
        })
    return serialized


def _deserialize_past_kv(data: List[Dict]) -> Tuple:
    past_kv = []
    for layer in data:
        dtype = getattr(torch, layer["dtype"].split(".")[-1])
        key = _deserialize_tensor(layer["key"], dtype, tuple(layer["key_shape"]))
        value = _deserialize_tensor(layer["value"], dtype, tuple(layer["value_shape"]))
        past_kv.append((key, value))
    return tuple(past_kv)


class K8sModelClient:
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.endpoint_url = endpoint_url or os.getenv(
            "LLM_MODEL_ENDPOINT",
            "http://llm-engine-service.default.svc.cluster.local:8000/generate_step",
        )
        self.model_name = model_name
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name or DEFAULT_CONFIG.model_name)
        except Exception as e:
            raise ImportError(f"Failed to import tokenizer: {e}")

    def generate_step_tensor(
        self,
        seq_id: int,
        input_ids: List[int],
        past_key_values: Optional[Tuple] = None,
    ) -> dict:

        payload: Dict = {
            "seq_id": seq_id,
            "input_ids": input_ids,
        }
        if past_key_values is not None:
            payload["past_key_values"] = _serialize_past_kv(past_key_values)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))
            response_dict = {
                "next_token_id": result["next_token_id"],
                "is_eos": result.get("is_eos", False),
            }
            if "past_key_values" in result:
                response_dict["past_key_values"] = _deserialize_past_kv(result["past_key_values"])
            return response_dict


class MockModelClient:
    def __init__(self, eos_token_id: int = 0, num_layers: int = 16,
                 num_heads: int = 2, head_dim: int = 8):
        self.eos_token_id = eos_token_id
        self._seq_len_cache: Dict[int, int] = {}
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

    def generate_step_tensor(
        self,
        seq_id: int,
        input_ids: List[int],
        past_key_values: Optional[Tuple] = None,
        is_prefill: bool = False,
    ) -> dict:
        has_prefill = past_key_values is None or is_prefill
        if has_prefill:
            seq_len = len(input_ids)
            past_kv = []
            for _ in range(self.num_layers):
                key = torch.zeros(1, self.num_heads, seq_len, self.head_dim)
                value = torch.zeros(1, self.num_heads, seq_len, self.head_dim)
                past_kv.append((key, value))
        else:
            seq_len = self._seq_len_cache.get(seq_id, 0) + len(input_ids)
            past_kv = []
            for layer in past_key_values:
                key = torch.cat([
                    layer[0],
                    torch.zeros(1, self.num_heads, len(input_ids), self.head_dim),
                ], dim=2)
                value = torch.cat([
                    layer[1],
                    torch.zeros(1, self.num_heads, len(input_ids), self.head_dim),
                ], dim=2)
                past_kv.append((key, value))
        self._seq_len_cache[seq_id] = seq_len

        next_token_id = 100 + seq_len
        is_eos = next_token_id == self.eos_token_id

        return {
            "next_token_id": next_token_id,
            "is_eos": is_eos,
            "past_key_values": past_kv,
        }
