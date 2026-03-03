"""
CodeGenAgent — code generation with long output.
  temperature    = 0.2  (low for deterministic code)
  max_new_tokens = 2048 (long outputs)
  Prompt length  : medium (2-4 sentences each)
  Runs 10 varied tasks per invocation.
"""

from cerebrum.llm.apis import LLMQuery
from cerebrum.utils.communication import send_request
from cerebrum.config.config_manager import config as cerebrum_config
from cerebrum.tasks.task_bank import TaskBank
import os, json

aios_kernel_url = cerebrum_config.get_kernel_url()

_UNUSED_TASKS = [
    "Write a Python class implementing a thread-safe LRU cache with configurable max size. Include type hints, docstrings, and unit tests.",
    "Implement a binary search tree in Rust with insert, delete, search, and in-order traversal. Handle all ownership and borrowing correctly.",
    "Write a Go HTTP middleware that implements rate limiting using a token bucket algorithm. It should support per-IP and per-API-key limits with configurable burst sizes.",
    "Create a TypeScript React hook called useDebounce that accepts a value and delay, returns the debounced value, and properly cleans up timers. Include generic typing.",
    "Write a Python async web scraper using aiohttp that crawls up to 100 pages concurrently with respect for robots.txt, retry logic with exponential backoff, and structured output as JSON.",
    "Implement a simple key-value store in C that uses a hash table with separate chaining for collision resolution. Support get, put, delete, and resize operations. Handle memory allocation failures.",
    "Write a Kotlin coroutine-based data pipeline that reads CSV files, transforms rows in parallel, filters by a predicate, and writes results to a new file. Use Flow and proper cancellation handling.",
    "Create a SQL migration script that transforms a denormalized orders table into a normalized schema with orders, order_items, customers, and products tables. Include rollback statements and data migration.",
    "Write a Python decorator that adds automatic retry with configurable max attempts, exponential backoff, and jitter. It should work with both sync and async functions and log each retry attempt.",
    "Implement a simple Raft consensus protocol in Python with leader election and log replication. Use asyncio for communication between nodes. Include at least 3 nodes in the simulation.",
]


class CodeGenAgent:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.config = self._load_config()

    def _load_config(self) -> dict:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        with open(config_path, "r") as f:
            return json.load(f)

    def run(self, task_input: str):
        system_prompt = "".join(self.config["description"])
        tasks = TaskBank.get_batch("code_gen_agent", n=10)
        results = []

        for i, task in enumerate(tasks):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            query = LLMQuery(
                messages=messages,
                tools=None,
                action_type="chat",
                temperature=0.2,
                max_new_tokens=2048,
            )

            try:
                resp = send_request(self.agent_name, query, aios_kernel_url)
                answer = resp.get("response", {}).get("response_message", "")
            except Exception as e:
                answer = f"[error] {e}"

            results.append({"task_idx": i, "prompt": task, "answer": answer})

        return {
            "agent_name": self.agent_name,
            "tasks_completed": len(results),
            "results": results,
        }
