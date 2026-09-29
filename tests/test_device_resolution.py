import os

import pytest
import torch

from app.memory.strategies import resolve_device, resolve_dtype


def test_resolve_device_prefers_explicit_argument():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_reads_environment(monkeypatch):
    monkeypatch.setenv("LLM_DEVICE", "cpu")
    assert resolve_device() == torch.device("cpu")


def test_resolve_dtype_reads_environment(monkeypatch):
    monkeypatch.setenv("LLM_DTYPE", "bf16")
    assert resolve_dtype() == torch.bfloat16


def test_resolve_dtype_aliases():
    assert resolve_dtype("fp16") == torch.float16
    assert resolve_dtype("float32") == torch.float32
    assert resolve_dtype("BF16") == torch.bfloat16


def test_resolve_dtype_rejects_unknown(monkeypatch):
    monkeypatch.delenv("LLM_DTYPE", raising=False)
    with pytest.raises(ValueError):
        resolve_dtype("int4")


def test_resolve_dtype_auto_is_float32_without_gpu():
    if torch.cuda.is_available():
        pytest.skip("GPU present: auto dtype is a hardware dtype")
    assert resolve_dtype("auto") == torch.float32


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
