from abc import ABC, abstractmethod
from typing import Tuple
import torch


class KVCacheMemoryStrategy(ABC):

    @abstractmethod
    def allocate_cache(
        self,
        total_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def copy_blocks(
        self,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        physical_blocks: list,
    ) -> None:
        pass

    @abstractmethod
    def sync(self) -> None:
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass


class CpuKVCacheStrategy(KVCacheMemoryStrategy):

    def __init__(self) -> None:
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def allocate_cache(self, total_blocks, block_size, num_layers, num_heads, head_dim, dtype):
        shape = (total_blocks, block_size, num_layers, num_heads, head_dim)
        key_cache = torch.empty(shape, dtype=dtype)
        value_cache = torch.empty(shape, dtype=dtype)
        return key_cache, value_cache

    def copy_blocks(self, src_cache, dst_cache, physical_blocks):
        for bid in physical_blocks:
            dst_cache[bid].copy_(src_cache[bid])

    def sync(self) -> None:
        pass


def detect_strategy() -> KVCacheMemoryStrategy:
    return CpuKVCacheStrategy()
