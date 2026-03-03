"""
ToolUseAgent — queries that include tool schemas (search + calculator).
  temperature    = 0.5
  max_new_tokens = 512
  has_tools      = True  (every call sends tool definitions)
  Runs 10 varied tasks per invocation.
"""

from cerebrum.llm.apis import LLMQuery
from cerebrum.utils.communication import send_request
from cerebrum.config.config_manager import config as cerebrum_config
from cerebrum.tasks.task_bank import TaskBank
import os, json

aios_kernel_url = cerebrum_config.get_kernel_url()

# Tools sent with every query — makes has_tools=True in the syscall log
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information on any topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-10)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A mathematical expression, e.g. '(3.14 * 25) + 17'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": "Convert between units of measurement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Numeric value to convert"},
                    "from_unit": {"type": "string", "description": "Source unit"},
                    "to_unit": {"type": "string", "description": "Target unit"}
                },
                "required": ["value", "from_unit", "to_unit"]
            }
        }
    },
]


class ToolUseAgent:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.config = self._load_config()

    def _load_config(self) -> dict:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        with open(config_path, "r") as f:
            return json.load(f)

    def run(self, task_input: str):
        system_prompt = "".join(self.config["description"])
        tasks = TaskBank.get_batch("tool_use_agent", n=10)
        results = []

        for i, task in enumerate(tasks):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            query = LLMQuery(
                messages=messages,
                tools=TOOL_SCHEMAS,
                action_type="chat",
                temperature=0.5,
                max_new_tokens=512,
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
