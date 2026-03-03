"""
LongReasoningAgent — multi-turn chain-of-thought reasoning.
  temperature    = 0.7
  max_new_tokens = 1024
  Prompt length  : 5-10 sentences each (long input_char_length)
  Runs 10 varied tasks, each with multi-turn conversation.
"""

from cerebrum.llm.apis import LLMQuery
from cerebrum.utils.communication import send_request
from cerebrum.config.config_manager import config as cerebrum_config
import os, json

aios_kernel_url = cerebrum_config.get_kernel_url()

TASKS = [
    (
        "A farmer has a fox, a chicken, and a bag of grain. He needs to cross a "
        "river in a boat that can only carry him and one item at a time. If left "
        "alone, the fox will eat the chicken, and the chicken will eat the grain. "
        "How can the farmer get everything across safely? Think through every "
        "possible sequence of moves and explain why each works or fails."
    ),
    (
        "Consider the trolley problem: a runaway trolley is heading toward five "
        "people tied to the tracks. You can pull a lever to divert the trolley "
        "onto a side track where only one person is tied. Analyze this from "
        "utilitarian, deontological, and virtue ethics perspectives. Which "
        "framework provides the most satisfying answer and why? Consider edge "
        "cases like the fat man variant."
    ),
    (
        "A company has three servers: A processes 100 requests/sec with 2ms "
        "latency, B processes 80 requests/sec with 5ms latency, C processes "
        "150 requests/sec with 10ms latency. They need to handle 250 req/sec "
        "total with P99 latency under 15ms. Design a load balancing strategy. "
        "Consider weighted round-robin, least-connections, and consistent "
        "hashing. Show the math for each approach and recommend the best one."
    ),
    (
        "You have a 3-gallon jug and a 5-gallon jug with no measuring marks. "
        "How do you measure exactly 4 gallons of water? Enumerate all possible "
        "states (amount in each jug), build a state graph, and find the shortest "
        "path from (0,0) to any state containing 4 gallons. Explain each pour "
        "operation mathematically using modular arithmetic."
    ),
    (
        "Explain why P vs NP is considered the most important open problem in "
        "computer science. Define both complexity classes formally, give three "
        "examples of NP-complete problems, explain Cook's theorem conceptually, "
        "and discuss what the implications would be if P=NP were proved true. "
        "Also discuss why most researchers believe P≠NP and what evidence "
        "supports this belief."
    ),
    (
        "A startup has $500K in funding and needs to decide between three "
        "strategies: (A) hire 5 junior developers at $80K each and build an MVP "
        "in 6 months, (B) hire 2 senior developers at $150K each plus $200K on "
        "marketing for a launch in 4 months, (C) outsource development for $200K "
        "and spend $300K on customer acquisition. Analyze the expected ROI, risks, "
        "and opportunity costs for each strategy assuming a B2B SaaS product with "
        "$50/mo per seat pricing and a TAM of 100K potential customers."
    ),
    (
        "Prove that the square root of 2 is irrational using proof by "
        "contradiction. Then extend this to prove that the square root of any "
        "prime number p is irrational. Explain each step clearly, identify where "
        "the fundamental theorem of arithmetic is used, and discuss why this "
        "proof technique does not work for perfect squares. What is the "
        "historical significance of this result in ancient Greek mathematics?"
    ),
    (
        "Compare and contrast three sorting algorithms: quicksort, mergesort, "
        "and heapsort. For each, provide the best-case, average-case, and "
        "worst-case time complexity with proofs. Analyze space complexity and "
        "cache performance. Explain when each is preferred in practice, discuss "
        "the impact of partially sorted input, and analyze why quicksort with "
        "median-of-three pivot selection outperforms mergesort on arrays despite "
        "having worse worst-case complexity."
    ),
    (
        "Design a database schema for a social media platform that supports "
        "users, posts, comments, likes, followers, and hashtags. The platform "
        "has 10 million users and 50 million posts. Discuss normalization vs "
        "denormalization tradeoffs, explain your indexing strategy for the top "
        "5 most common queries, and design a sharding strategy. Consider "
        "eventual consistency requirements for the follower count and like count "
        "features. Estimate storage requirements for the first year."
    ),
    (
        "A patient presents with fatigue, weight gain, cold intolerance, and "
        "constipation for the past 3 months. Their TSH is elevated at 12 mIU/L "
        "and free T4 is low at 0.5 ng/dL. Construct a differential diagnosis "
        "with at least 4 possible conditions. For each, explain the "
        "pathophysiology, describe confirmatory tests, and outline treatment "
        "approaches. Discuss why Hashimoto's thyroiditis is the most likely "
        "diagnosis and what autoimmune markers would confirm it."
    ),
]

# Follow-up questions for multi-turn conversation
FOLLOW_UPS = [
    "Can you identify any flaws in your reasoning above?",
    "What assumptions did you make? Are they justified?",
    "Summarize your conclusion in exactly three bullet points.",
]


class LongReasoningAgent:
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

            # Turn 1: initial reasoning
            query = LLMQuery(
                messages=messages,
                tools=None,
                action_type="chat",
                temperature=0.7,
                max_new_tokens=1024,
            )

            try:
                resp = send_request(self.agent_name, query, aios_kernel_url)
                answer1 = resp.get("response", {}).get("response_message", "")
            except Exception as e:
                answer1 = f"[error] {e}"

            messages.append({"role": "assistant", "content": answer1})

            # Turn 2: follow-up to deepen reasoning
            follow_up = FOLLOW_UPS[i % len(FOLLOW_UPS)]
            messages.append({"role": "user", "content": follow_up})

            query2 = LLMQuery(
                messages=messages,
                tools=None,
                action_type="chat",
                temperature=0.7,
                max_new_tokens=1024,
            )

            try:
                resp2 = send_request(self.agent_name, query2, aios_kernel_url)
                answer2 = resp2.get("response", {}).get("response_message", "")
            except Exception as e:
                answer2 = f"[error] {e}"

            results.append({
                "task_idx": i,
                "prompt": task[:80] + "...",
                "turns": 2,
                "final_answer": answer2,
            })

        return {
            "agent_name": self.agent_name,
            "tasks_completed": len(results),
            "results": results,
        }
