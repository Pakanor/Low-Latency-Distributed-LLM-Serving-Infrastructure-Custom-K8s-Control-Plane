import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

import llm_allocator_cpp

if TYPE_CHECKING:
    from app.memory.manager import PagedKVCacheManager
    from app.model.paged_attention import PagedStepContext

BACKEND_ENV_VAR = "LLM_PAGED_BACKEND"
TORCH_BACKEND = "torch"
NATIVE_BACKEND = "native"


def native_kernel_available() -> bool:
    return hasattr(llm_allocator_cpp, "paged_attention")


class AttentionBackend(ABC):

    name: str = ""

    @abstractmethod
    def supports(self, context: "PagedStepContext", num_query_tokens: int) -> bool:
        pass

    @abstractmethod
    def attend(
        self,
        query: torch.Tensor,
        store: "PagedKVCacheManager",
        layer_idx: int,
        context: "PagedStepContext",
        scaling: float,
        dropout: float,
    ) -> torch.Tensor:
        pass


class TorchSdpBackend(AttentionBackend):

    name = TORCH_BACKEND

    def supports(self, context: "PagedStepContext", num_query_tokens: int) -> bool:
        return True

    def attend(
        self,
        query: torch.Tensor,
        store: "PagedKVCacheManager",
        layer_idx: int,
        context: "PagedStepContext",
        scaling: float,
        dropout: float,
    ) -> torch.Tensor:
        context_keys = store.gather_layer_keys(layer_idx, context.gather_slots)
        context_values = store.gather_layer_values(layer_idx, context.gather_slots)
        context_keys = context_keys.unsqueeze(0).transpose(1, 2)
        context_values = context_values.unsqueeze(0).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            query,
            context_keys,
            context_values,
            attn_mask=context.attention_mask(query.device),
            dropout_p=dropout,
            scale=scaling,
            enable_gqa=context_keys.shape[1] != query.shape[1],
        )
        return attn_output.transpose(1, 2).contiguous()


class NativePagedBackend(AttentionBackend):

    name = NATIVE_BACKEND

    def supports(self, context: "PagedStepContext", num_query_tokens: int) -> bool:
        return all(count == 1 for count in context.new_token_counts) and num_query_tokens > 0

    def attend(
        self,
        query: torch.Tensor,
        store: "PagedKVCacheManager",
        layer_idx: int,
        context: "PagedStepContext",
        scaling: float,
        dropout: float,
    ) -> torch.Tensor:
        if not native_kernel_available():
            raise RuntimeError(
                "llm_allocator_cpp does not export paged_attention; "
                "rebuild the extension with a ROCm toolchain to enable the native backend"
            )
        return llm_allocator_cpp.paged_attention(
            query,
            store.key_cache,
            store.value_cache,
            layer_idx,
            context.gather_slots,
            context.cu_context_offsets,
            scaling,
        )


_BACKENDS = {
    TORCH_BACKEND: TorchSdpBackend,
    NATIVE_BACKEND: NativePagedBackend,
}


def select_backend(requested: Optional[str] = None) -> AttentionBackend:
    name = (requested or os.getenv(BACKEND_ENV_VAR) or "auto").lower()
    if name == "auto":
        return _BACKENDS[NATIVE_BACKEND]() if native_kernel_available() else _BACKENDS[TORCH_BACKEND]()

    if name == "hip":
        name = NATIVE_BACKEND
    if name not in _BACKENDS:
        raise ValueError(
            f"Unsupported {BACKEND_ENV_VAR} '{name}'. Use 'auto', 'torch' or 'native'."
        )

    backend = _BACKENDS[name]()
    if backend.name == NATIVE_BACKEND and not native_kernel_available():
        raise RuntimeError(
            f"{BACKEND_ENV_VAR}=native requires llm_allocator_cpp.paged_attention, "
            "which this build does not provide"
        )
    return backend
