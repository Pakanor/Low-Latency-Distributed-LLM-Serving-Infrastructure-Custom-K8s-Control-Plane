"""Probe: does engine stop/refuse when pool exhausted? Uses no model (model_client path) via MockModelClient? No - easier: inspect by constructing tiny real-model-free path with LLMEngine using a dummy model object that has forward."""
import torch
from types import SimpleNamespace
from app.memory.manager import PagedKVCacheManager
from app.scheduler.scheduler import Scheduler
from app.engine.llm_engine import LLMEngine
from app.scheduler.sequence import SequenceStatus

class DummyOut:
    def __init__(self, logits, past_key_values):
        self.logits = logits
        self.past_key_values = past_key_values

class DummyModel:
    def __init__(self, n_layers=2, n_kv=2, hd=8, vocab=32):
        self.n_layers=n_layers; self.n_kv=n_kv; self.hd=hd; self.vocab=vocab
        self.config = SimpleNamespace(_attn_implementation="eager")
        self.model = SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(layer_idx=i, paged_context=None, scaling=hd**-0.5)) for i in range(n_layers)])
        self._p = torch.nn.Parameter(torch.zeros(1))
    def parameters(self):
        yield self._p
    def __call__(self, input_ids, past_key_values=None, use_cache=True, position_ids=None):
        # input_ids: (1, S)
        S = input_ids.shape[1]
        total = (past_key_values[0][0].shape[2] if past_key_values is not None else 0) + S
        logits = torch.zeros(1, S, self.vocab)
        logits[0,-1,5]=1.0
        pkv = tuple((torch.zeros(1,self.n_kv,total,self.hd), torch.zeros(1,self.n_kv,total,self.hd)) for _ in range(self.n_layers))
        return DummyOut(logits, pkv)
    def forward(self, *a, **k): return self.__call__(*a, **k)

# very small pool: 2 blocks, block_size 4 => max 8 tokens total
mgr = PagedKVCacheManager(total_blocks=2, block_size=4, num_heads=2, head_dim=8, total_cpu_blocks=0, dtype=torch.float32, num_layers=2, device="cpu")
sched = Scheduler(kv_cache_manager=mgr, max_batch_size=4, max_num_batched_tokens=32, block_size=4, allocator=mgr.get_allocator(), max_model_len=64)
engine = LLMEngine(kv_cache_manager=mgr, scheduler=sched, model=DummyModel())
try:
    seq = engine.add_request(prompt_token_ids=[1,2,3,4,5], max_tokens=2) # needs 2 blocks for prompt+max_tokens? prompt 5 tokens -> 2 blocks
    print("add_ok total_len", seq.total_len, "free", mgr.get_allocator().get_num_free_blocks())
except Exception as e:
    print("add_err", type(e), e)
print("has_unfinished", engine.has_unfinished_requests())
try:
    out = engine.step()
    print("step1", [(s.seq_id, s.status.value, s.output_token_ids, s.block_table.get_tokens_count() if s.block_table else None) for s in out['running']+out['finished']])
except Exception as e:
    import traceback; traceback.print_exc()
    print("step_err", type(e), e)

# now try to overcommit: request needing more blocks than pool
mgr2 = PagedKVCacheManager(total_blocks=2, block_size=4, num_heads=2, head_dim=8, total_cpu_blocks=0, dtype=torch.float32, num_layers=2, device="cpu")
sched2 = Scheduler(kv_cache_manager=mgr2, max_batch_size=4, max_num_batched_tokens=64, block_size=4, allocator=mgr2.get_allocator(), max_model_len=64)
engine2 = LLMEngine(kv_cache_manager=mgr2, scheduler=sched2, model=DummyModel())
try:
    seq2 = engine2.add_request(prompt_token_ids=[1,2,3,4,5,6,7,8,9], max_tokens=1) # 10 tokens -> 3 blocks > 2
    print("add2_ok")
except Exception as e:
    print("add2_err", type(e).__name__, str(e))
