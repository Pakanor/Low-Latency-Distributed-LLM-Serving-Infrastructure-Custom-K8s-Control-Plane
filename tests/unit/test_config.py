from types import SimpleNamespace

import pytest
import torch

from app.config import RuntimeConfig, calculate_kv_blocks


def test_runtime_configs_change_kv_block_count():
    model_config = SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    first = RuntimeConfig(budget_mib=512, block_size=16)
    second = RuntimeConfig(budget_mib=256, block_size=16)

    first_blocks = calculate_kv_blocks(model_config, torch.float32, first.budget_mib, first.block_size)
    second_blocks = calculate_kv_blocks(model_config, torch.float32, second.budget_mib, second.block_size)

    assert first_blocks != second_blocks


def test_calculate_kv_blocks_rejects_budget_without_one_block():
    model_config = SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )

    with pytest.raises(ValueError, match="KV budget"):
        calculate_kv_blocks(model_config, torch.float32, budget_mib=0, block_size=16)


def test_calculate_kv_blocks_handles_none_head_dim():
    model_config = SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=None,
        hidden_size=32,
    )

    assert calculate_kv_blocks(model_config, torch.float32, budget_mib=1, block_size=16) > 0


@pytest.mark.hf
def test_smollm_kv_formula_matches_allocated_tensor():
    transformers = pytest.importorskip("transformers")
    from app.memory.manager import PagedKVCacheManager

    model_config = transformers.AutoConfig.from_pretrained(
        "HuggingFaceTB/SmolLM-135M-Instruct", local_files_only=True
    )
    dtype = torch.float32
    block_size = 16
    manager = PagedKVCacheManager.from_model_config(
        model_config,
        total_blocks=1,
        block_size=block_size,
        dtype=dtype,
        device="cpu",
    )

    allocated_bytes_per_token = (
        manager.key_cache.numel() + manager.value_cache.numel()
    ) * dtype.itemsize / block_size
    expected_bytes_per_token = (
        2
        * model_config.num_hidden_layers
        * getattr(model_config, "num_key_value_heads", model_config.num_attention_heads)
        * (model_config.head_dim or model_config.hidden_size // model_config.num_attention_heads)
        * dtype.itemsize
    )

    assert allocated_bytes_per_token == expected_bytes_per_token


@pytest.mark.parametrize(
    ("field", "value"),
    [("dtype", "int8"), ("device", "tpu"), ("budget_mib", 0), ("block_size", 0), ("max_model_len", 0)],
)
def test_runtime_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        RuntimeConfig(**{field: value})
