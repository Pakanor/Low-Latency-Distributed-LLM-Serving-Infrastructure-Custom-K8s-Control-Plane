import pytest
import torch
from types import SimpleNamespace
from app.memory.manager import PagedKVCacheManager


def _config(**overrides):
    defaults = dict(
        num_hidden_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
        hidden_size=576,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_from_model_config_sizes_cache_with_kv_heads():
    manager = PagedKVCacheManager.from_model_config(
        _config(),
        total_blocks=8,
        block_size=4,
    )

    assert manager.num_layers == 30
    assert manager.num_heads == 3
    assert manager.head_dim == 64
    assert manager.key_cache.shape == (8, 4, 30, 3, 64)
    assert manager.value_cache.shape == (8, 4, 30, 3, 64)


def test_from_model_config_uses_explicit_head_dim():
    manager = PagedKVCacheManager.from_model_config(
        _config(head_dim=128),
        total_blocks=8,
        block_size=4,
    )

    assert manager.head_dim == 128
    assert manager.key_cache.shape == (8, 4, 30, 3, 128)


def test_from_model_config_falls_back_to_attention_heads():
    config = SimpleNamespace(
        num_hidden_layers=4,
        num_attention_heads=9,
        hidden_size=576,
    )
    manager = PagedKVCacheManager.from_model_config(
        config,
        total_blocks=8,
        block_size=4,
    )

    assert manager.num_heads == 9
