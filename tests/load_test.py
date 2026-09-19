import asyncio
import time
import httpx

URL = "http://172.22.0.2/generate"
PAYLOAD = {
    "prompt": "Explain page allocation and C++ memory management:",
    "max_tokens": 40
}
CONCURRENT_REQUESTS = 5

async def send_request(client, req_id):
    start = time.perf_counter()
    try:
        response = await client.post(URL, json=PAYLOAD, timeout=60.0)
        elapsed = time.perf_counter() - start
        print(f"[Req {req_id}] Status: {response.status_code} | Time: {elapsed:.2f}s")
        return elapsed
    except Exception as e:
        print(f"[Req {req_id}] Failed: {e}")
        return None

async def main():
    async with httpx.AsyncClient() as client:
        print(f"--- Starting load test: {CONCURRENT_REQUESTS} parallel requests ---")
        start_total = time.perf_counter()
        tasks = [send_request(client, i) for i in range(CONCURRENT_REQUESTS)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_total
        
        valid_times = [r for r in results if r is not None]
        if valid_times:
            print(f"\nTotal batch time: {total_time:.2f}s")
            print(f"Avg response time: {sum(valid_times)/len(valid_times):.2f}s")

if __name__ == "__main__":
    asyncio.run(main())