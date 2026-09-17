import urllib.request
import json
from typing import List, Optional


class K8sModelClient:
    def __init__(
        self, 
        endpoint_url: str = "http://llm-engine-service.default.svc.cluster.local/generate_step"
    ) -> None:
        self.endpoint_url = endpoint_url

    def generate_step_tensor(
        self, 
        input_ids: List[int], 
        block_table: List[int], 
        context_len: int
    ) -> int:
        
        payload = {
            "input_ids": input_ids,
            "block_table": block_table,
            "context_len": context_len
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["next_token_id"]

class MockModelClient:
    def generate_step_tensor(
        self, 
        input_ids: List[int], 
        block_table: List[int], 
        context_len: int
    ) -> int:
        return 100 + len(input_ids)