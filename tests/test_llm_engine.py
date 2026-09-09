import torch
import torch.nn as nn
from app.engine.llm_engine import LLMEngine
from app.memory.manager import PagedKVCacheManager
from app.scheduler.scheduler import Scheduler


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int = 100):
        super().__init__()
        self.vocab_size = vocab_size
        self.linear = nn.Linear(1, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        return torch.randn(batch_size, seq_len, self.vocab_size)


def test_llm_engine_execution():
    model = DummyModel(vocab_size=100)
    
    manager = PagedKVCacheManager(
        total_blocks=16,
        block_size=4,
        num_heads=2,
        head_dim=16
    )
    
    scheduler = Scheduler(kv_cache_manager=manager, max_batch_size=2)
    engine = LLMEngine(model=model, kv_cache_manager=manager, scheduler=scheduler)

    req1 = engine.add_request(prompt_token_ids=[10, 20, 30], max_tokens=3)
    req2 = engine.add_request(prompt_token_ids=[40, 50], max_tokens=2)

    assert engine.has_unfinished_requests() is True

    step_count = 0
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        step_count += 1
        print(f"Krok {step_count}: wygenerowano tokeny: {step_outputs}")
        
        if step_count > 10:
            break

    print("Silnik zakończył przetwarzanie pomyślnie!")


if __name__ == "__main__":
    test_llm_engine_execution()