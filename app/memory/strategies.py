from abc import ABC, abstractmethod
from typing import Tuple, Union
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

    @property
    @abstractmethod
    def requires_copy_on_swap(self) -> bool:
        pass


class CpuKVCacheStrategy(KVCacheMemoryStrategy):

    def __init__(self) -> None:
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def allocate_cache(self, total_blocks, block_size, num_layers, num_heads, head_dim, dtype):
        shape = (num_layers, total_blocks, block_size, num_heads, head_dim)
        key_cache = torch.empty(shape, dtype=dtype)
        value_cache = torch.empty(shape, dtype=dtype)
        return key_cache, value_cache

    def copy_blocks(self, src_cache, dst_cache, physical_blocks):
        for bid in physical_blocks:
            dst_cache[bid].copy_(src_cache[bid])

    def sync(self) -> None:
        pass

    @property
    def requires_copy_on_swap(self) -> bool:
        return False


class CudaKVCacheStrategy(KVCacheMemoryStrategy):

    def __init__(self, device_id: int = 0) -> None:
        self._device = torch.device(f"cuda:{device_id}")
        self._stream = torch.cuda.Stream(device=self._device)

    @property
    def device(self) -> torch.device:
        return self._device

    def allocate_cache(self, total_blocks, block_size, num_layers, num_heads, head_dim, dtype):
        shape = (num_layers, total_blocks, block_size, num_heads, head_dim)
        with torch.cuda._device_ctx(self._device.index):
            key_cache = torch.empty(shape, dtype=dtype, device=self._device)
            value_cache = torch.empty(shape, dtype=dtype, device=self._device)
        return key_cache, value_cache

    def copy_blocks(self, src_cache, dst_cache, physical_blocks):
        with torch.cuda._device_ctx(self._device.index):
            with torch.cuda.stream(self._stream):
                for bid in physical_blocks:
                    dst_cache[bid].copy_(src_cache[bid], non_blocking=True)
            self._stream.synchronize()

    def sync(self) -> None:
        self._stream.synchronize()

    @property
    def requires_copy_on_swap(self) -> bool:
        return True


class HipZeroCopyStrategy(KVCacheMemoryStrategy):

    def __init__(self) -> None:
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def allocate_cache(self, total_blocks, block_size, num_layers, num_heads, head_dim, dtype):
        shape = (num_layers, total_blocks, block_size, num_heads, head_dim)
        key_cache = torch.empty(shape, dtype=dtype).pin_memory()
        value_cache = torch.empty(shape, dtype=dtype).pin_memory()
        return key_cache, value_cache

    def copy_blocks(self, src_cache, dst_cache, physical_blocks):
        pass

    def sync(self) -> None:
        pass

    @property
    def requires_copy_on_swap(self) -> bool:
        return False


def _is_rocm() -> bool:
    return getattr(torch.version, "hip", None) is not None


def strategy_from_device(device: Union[str, torch.device]) -> KVCacheMemoryStrategy:
    dev = torch.device(device)
    if dev.type != "cuda":
        return CpuKVCacheStrategy()
    if _is_rocm():
        return HipZeroCopyStrategy()
    return CudaKVCacheStrategy(device_id=dev.index or 0)


def detect_strategy() -> KVCacheMemoryStrategy:
    if not torch.cuda.is_available():
        return CpuKVCacheStrategy()
    return strategy_from_device("cuda")
