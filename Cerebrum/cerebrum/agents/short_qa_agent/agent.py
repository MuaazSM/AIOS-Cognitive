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
import os, json

aios_kernel_url = cerebrum_config.get_kernel_url()

TASKS = [
    "What is photosynthesis?",
    "Define the Heisenberg uncertainty principle.",
    "What causes thunder?",
    "Name the largest planet in our solar system.",
    "What is the speed of light in a vacuum?",
    "Who wrote Hamlet?",
    "What is the chemical formula for table salt?",
    "Define machine learning in one sentence.",
    "What year did the Berlin Wall fall?",
    "What is the boiling point of water at sea level?",
]


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
        results = []

        for i, task in enumerate(TASKS):
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
