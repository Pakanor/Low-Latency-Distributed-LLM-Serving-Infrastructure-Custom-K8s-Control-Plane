import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RuntimeConfig:
    model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct"
    dtype: str = "fp32"
    device: str = "cpu"
    budget_mib: int = 512
    max_model_len: int = 2048
    block_size: int = 16
    num_threads: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported device: {self.device}; expected cpu or cuda")
        if self.dtype not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"Unsupported dtype: {self.dtype}; expected fp32, fp16, or bf16")
        for name in ("budget_mib", "block_size", "max_model_len"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.num_threads is not None and self.num_threads <= 0:
            raise ValueError("num_threads must be greater than zero or None")


def configure_runtime(config: RuntimeConfig) -> int:
    if config.device != "cpu":
        raise NotImplementedError("CUDA device support is not implemented yet")
    num_threads = config.num_threads or (os.cpu_count() or 1)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)
    return num_threads


def calculate_kv_blocks(model_config, dtype: torch.dtype, budget_mib: int, block_size: int) -> int:
    num_kv_heads = getattr(model_config, "num_key_value_heads", model_config.num_attention_heads)
    head_dim = getattr(model_config, "head_dim", None)
    if head_dim is None:
        head_dim = model_config.hidden_size // model_config.num_attention_heads
    bytes_per_token = (
        2 * model_config.num_hidden_layers * num_kv_heads * head_dim * dtype.itemsize
    )
    blocks = (budget_mib * 1024 * 1024) // (block_size * bytes_per_token)
    if blocks <= 0:
        raise ValueError("KV budget is too small for one cache block")
    return blocks


DEFAULT_CONFIG = RuntimeConfig()
