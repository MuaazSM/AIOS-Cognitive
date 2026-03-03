"""
ShortQAAgent — single-turn, short factual answers.
  temperature   = 0.3
  max_new_tokens = 100
  Prompt length  : 1 sentence each (short input_char_length)
  Runs 10 varied tasks per invocation.
"""

from cerebrum.llm.apis import LLMQuery
from cerebrum.utils.communication import send_request
from cerebrum.config.config_manager import config as cerebrum_config
from cerebrum.tasks.task_bank import TaskBank
import os, json

aios_kernel_url = cerebrum_config.get_kernel_url()


class ShortQAAgent:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.config = self._load_config()

    def _load_config(self) -> dict:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        with open(config_path, "r") as f:
            return json.load(f)

    def run(self, task_input: str):
        system_prompt = "".join(self.config["description"])
        tasks = TaskBank.get_batch("short_qa_agent", n=10)
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
                temperature=0.3,
                max_new_tokens=100,
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
