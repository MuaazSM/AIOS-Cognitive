#!/usr/bin/env python3
"""
Run Diverse Workload v2 — feature-rich data collection for the cognitive scheduler.

Key changes from v1:
  - Randomised max_tokens and temperature per request (breaks multicollinearity)
  - Multi-turn conversations (message_count > 2)
  - Cross-pollinated prompt lengths (same agent gets short + long prompts)
  - Support for --llm-name flag to tag which model produced the data
  - Aligned with AIOS paper (COLM 2025) benchmarking methodology

    5 agents × 10 tasks × variable params = rich feature variance per round

Usage:
    python scripts/run_diverse_workload.py --rounds 10
    python scripts/run_diverse_workload.py --rounds 20 --llm-name ollama/mistral:7b
    python scripts/run_diverse_workload.py --rounds 5 --no-multiturn
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Ensure project roots are importable ────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CEREBRUM_ROOT = os.path.join(PROJECT_ROOT, "Cerebrum")

for _p in (PROJECT_ROOT, CEREBRUM_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cerebrum.llm.apis import LLMQuery                       # noqa: E402
from cerebrum.utils.communication import send_request         # noqa: E402
from cerebrum.config.config_manager import config as cerebrum_config  # noqa: E402
from cerebrum.tasks.task_bank import TaskBank                 # noqa: E402
from cerebrum.agents.tool_use_agent.agent import TOOL_SCHEMAS # noqa: E402

KERNEL_URL = cerebrum_config.get_kernel_url()


# ═══════════════════════════════════════════════════════════════════
# RANDOMISED PARAMETER RANGES
# ═══════════════════════════════════════════════════════════════════

# These ranges ensure each feature varies WITHIN every agent type,
# breaking the agent_type <-> max_tokens / temperature correlation
# that made dataset 1 a lookup table.

MAX_TOKENS_CHOICES = [64, 128, 256, 512, 1024, 2048]
TEMPERATURE_CHOICES = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

# Cross-pollination: short prompts that any agent can get (adds variance)
_CROSS_SHORT = [
    "Explain in one sentence.",
    "Give a yes or no answer with brief justification.",
    "Summarize your response in under 50 words.",
]

# Cross-pollination: long preambles to inflate input_char_length
_CROSS_LONG_PREFIX = (
    "The following is a complex, multi-faceted request that requires careful "
    "consideration of multiple factors, trade-offs, and edge cases. Please "
    "provide a thorough, well-structured response that addresses every aspect "
    "of the problem. Consider alternative approaches and explain why your "
    "chosen solution is optimal. Include concrete examples where helpful. "
    "Be sure to validate your reasoning at each step.\n\n"
)

# Multi-turn conversation templates
_FOLLOW_UPS = [
    "Can you elaborate on that?",
    "What are the trade-offs of this approach?",
    "Give me a concrete example.",
    "How would this change if we doubled the scale?",
    "What's the most common mistake people make here?",
    "Summarize the key points in bullet form.",
    "Now explain it as if I'm a beginner.",
    "What would a critic say about this approach?",
]


# ═══════════════════════════════════════════════════════════════════
# AGENT SPECS — now with randomised params per request
# ═══════════════════════════════════════════════════════════════════

AGENT_SPECS = [
    {
        "agent_type": "short_qa_agent",
        "system_prompt": "You are a concise factual QA assistant. Answer in one or two sentences.",
        "tools": None,
    },
    {
        "agent_type": "long_reasoning_agent",
        "system_prompt": (
            "You are a deep-reasoning assistant. Think step by step, consider "
            "multiple angles, and provide a thorough analysis."
        ),
        "tools": None,
    },
    {
        "agent_type": "tool_use_agent",
        "system_prompt": (
            "You are a tool-augmented assistant. Use the provided tools when "
            "appropriate to answer the user's question."
        ),
        "tools": TOOL_SCHEMAS,
    },
    {
        "agent_type": "code_gen_agent",
        "system_prompt": (
            "You are an expert programmer. Write clean, well-documented code "
            "with type hints and tests."
        ),
        "tools": None,
    },
    {
        "agent_type": "summarizer_agent",
        "system_prompt": (
            "You are a summarisation assistant. Condense the given text into a "
            "concise summary preserving all key points."
        ),
        "tools": None,
    },
]

TASKS_PER_AGENT = 10
SYSCALLS_PER_ROUND = TASKS_PER_AGENT * len(AGENT_SPECS)      # 50
LOG_FILE = os.path.join(PROJECT_ROOT, "aios", "logs", "llm_syscalls.jsonl")


# ═══════════════════════════════════════════════════════════════════
# PROMPT BUILDER — adds variance to input_char_length
# ═══════════════════════════════════════════════════════════════════

def build_messages(
    system_prompt: str,
    task: str,
    enable_multiturn: bool,
) -> list[dict]:
    """
    Build a message list with randomised complexity:
      - 40% chance: plain single-turn (message_count = 2)
      - 30% chance: long-prefix single-turn (message_count = 2, longer input)
      - 30% chance: multi-turn with 1-3 follow-ups (message_count = 4-8)
    """
    messages = [{"role": "system", "content": system_prompt}]

    roll = random.random()

    if roll < 0.40:
        # Plain single-turn
        messages.append({"role": "user", "content": task})

    elif roll < 0.70:
        # Long-prefix single-turn — inflates input_char_length
        padded = _CROSS_LONG_PREFIX + task
        messages.append({"role": "user", "content": padded})

    else:
        # Multi-turn conversation
        if not enable_multiturn:
            messages.append({"role": "user", "content": task})
            return messages

        messages.append({"role": "user", "content": task})
        # Add 1-3 follow-up exchanges
        n_turns = random.randint(1, 3)
        for _ in range(n_turns):
            # Simulated assistant response (short placeholder)
            messages.append({
                "role": "assistant",
                "content": "I'll address that. Let me think through the key aspects.",
            })
            messages.append({
                "role": "user",
                "content": random.choice(_FOLLOW_UPS),
            })

    return messages


# ═══════════════════════════════════════════════════════════════════
# WORKER — executed inside each thread
# ═══════════════════════════════════════════════════════════════════

def _run_agent_thread(
    spec: dict, round_num: int, enable_multiturn: bool, llm_tag: str
) -> dict:
    """
    Fetch 10 tasks via TaskBank, randomise params, send through AIOS SDK.
    """
    agent_type = spec["agent_type"]
    agent_name = f"{agent_type}_r{round_num}"
    system_prompt = spec["system_prompt"]
    tools = spec["tools"]

    completed = 0
    errors = 0
    t0 = time.time()

    for i in range(TASKS_PER_AGENT):
        task = TaskBank.get(agent_type)

        # ── Randomise per-request params ────────────────────────
        max_new_tokens = random.choice(MAX_TOKENS_CHOICES)
        temperature = random.choice(TEMPERATURE_CHOICES)

        # 20% chance: cross-pollinate with a short follow-up
        if random.random() < 0.20:
            task = task + "\n\n" + random.choice(_CROSS_SHORT)

        messages = build_messages(system_prompt, task, enable_multiturn)

        query = LLMQuery(
            messages=messages,
            tools=tools,
            action_type="chat",
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        # Stagger arrivals (1-5s random sleep)
        time.sleep(random.uniform(1.0, 5.0))

        try:
            send_request(agent_name, query, KERNEL_URL)
            completed += 1
        except Exception:
            errors += 1

    elapsed = time.time() - t0
    return {
        "agent_type": agent_type,
        "round": round_num,
        "completed": completed,
        "errors": errors,
        "elapsed_s": round(elapsed, 1),
    }


# ═══════════════════════════════════════════════════════════════════
# LOG HELPERS
# ═══════════════════════════════════════════════════════════════════

def _count_log_rows() -> int:
    if not os.path.exists(LOG_FILE):
        return 0
    with open(LOG_FILE) as f:
        return sum(1 for _ in f)


def _read_log_stats() -> dict | None:
    """Parse the JSONL log and return summary statistics."""
    if not os.path.exists(LOG_FILE):
        return None

    latencies: list[float] = []
    char_lens: list[int] = []
    msg_counts: list[int] = []
    max_toks: list[int] = []
    temps: list[float] = []
    tools_true = 0
    tools_false = 0
    total = 0

    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            lat = row.get("latency_ms")
            if isinstance(lat, (int, float)):
                latencies.append(float(lat))
            cl = row.get("input_char_length")
            if isinstance(cl, (int, float)):
                char_lens.append(int(cl))
            mc = row.get("message_count")
            if isinstance(mc, (int, float)):
                msg_counts.append(int(mc))
            mt = row.get("max_tokens")
            if isinstance(mt, (int, float)):
                max_toks.append(int(mt))
            t = row.get("temperature")
            if isinstance(t, (int, float)):
                temps.append(float(t))
            if row.get("has_tools"):
                tools_true += 1
            else:
                tools_false += 1

    if not latencies:
        return None

    return {
        "total_rows": total,
        "latency_min": round(min(latencies), 1),
        "latency_median": round(statistics.median(latencies), 1),
        "latency_max": round(max(latencies), 1),
        "char_len_min": min(char_lens) if char_lens else 0,
        "char_len_median": round(statistics.median(char_lens)) if char_lens else 0,
        "char_len_max": max(char_lens) if char_lens else 0,
        "msg_count_min": min(msg_counts) if msg_counts else 0,
        "msg_count_max": max(msg_counts) if msg_counts else 0,
        "max_tokens_unique": len(set(max_toks)),
        "temperature_unique": len(set(temps)),
        "has_tools_true": tools_true,
        "has_tools_false": tools_false,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run diverse agent workload for syscall logging (v2 — feature-rich)",
    )
    parser.add_argument(
        "--rounds", type=int, default=5,
        help="Repeat the full 5-agent batch N times (default: 5)",
    )
    parser.add_argument(
        "--llm-name", type=str, default="ollama/llama3.1:8b",
        help="LLM model tag for logging context (default: ollama/llama3.1:8b)",
    )
    parser.add_argument(
        "--no-multiturn", action="store_true",
        help="Disable multi-turn conversations (single-turn only)",
    )
    args = parser.parse_args()

    enable_multiturn = not args.no_multiturn
    n_agents = len(AGENT_SPECS)
    estimated_rows = SYSCALLS_PER_ROUND * args.rounds
    rows_before = _count_log_rows()

    # ── Pre-launch summary ─────────────────────────────────────────
    print("=" * 65)
    print("  Diverse Workload Runner v2 (feature-rich)")
    print("=" * 65)
    print(f"  LLM model        : {args.llm_name}")
    print(f"  Agents           : {n_agents}")
    for s in AGENT_SPECS:
        has_tools = "tools" if s["tools"] else "no-tools"
        print(f"    - {s['agent_type']:25s}  {has_tools}")
    print(f"  Tasks per agent  : {TASKS_PER_AGENT}")
    print(f"  Rounds           : {args.rounds}")
    print(f"  Multi-turn       : {'ON' if enable_multiturn else 'OFF'}")
    print(f"  max_tokens range : {MAX_TOKENS_CHOICES}")
    print(f"  temperature range: {TEMPERATURE_CHOICES}")
    print(f"  Syscalls / round : {SYSCALLS_PER_ROUND}")
    print(f"  Estimated rows   : {estimated_rows}")
    print(f"  Existing log rows: {rows_before}")
    print(f"  Log file         : {LOG_FILE}")
    print(f"  Kernel URL       : {KERNEL_URL}")
    print("=" * 65)
    print()

    t_start = time.time()
    all_results: list[dict] = []

    for r in range(1, args.rounds + 1):
        print(f"── Round {r}/{args.rounds} ──")
        round_t0 = time.time()

        with ThreadPoolExecutor(max_workers=n_agents) as pool:
            futures = {
                pool.submit(
                    _run_agent_thread, spec, r, enable_multiturn, args.llm_name
                ): spec["agent_type"]
                for spec in AGENT_SPECS
            }
            for future in as_completed(futures):
                res = future.result()
                all_results.append(res)
                err_tag = f"  err={res['errors']}" if res['errors'] else ""
                print(f"  {res['agent_type']:25s}  done={res['completed']:>2}  "
                      f"time={res['elapsed_s']:>6.1f}s{err_tag}")

        print(f"  Round {r} finished in {time.time() - round_t0:.1f}s\n")

    total_elapsed = time.time() - t_start
    rows_after = _count_log_rows()
    new_rows = rows_after - rows_before

    # ── Post-run statistics ────────────────────────────────────────
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  Wall time        : {total_elapsed:.1f}s")
    print(f"  New log rows     : {new_rows}  (expected {estimated_rows})")
    print(f"  Total log rows   : {rows_after}")
    print()

    stats = _read_log_stats()
    if stats:
        print(f"  {'Metric':<25} {'Value':>15}")
        print(f"  {'-'*25} {'-'*15}")
        print(f"  {'latency_ms (min)':25s} {stats['latency_min']:>15.1f}")
        print(f"  {'latency_ms (median)':25s} {stats['latency_median']:>15.1f}")
        print(f"  {'latency_ms (max)':25s} {stats['latency_max']:>15.1f}")
        print(f"  {'char_len (min)':25s} {stats['char_len_min']:>15}")
        print(f"  {'char_len (median)':25s} {stats['char_len_median']:>15}")
        print(f"  {'char_len (max)':25s} {stats['char_len_max']:>15}")
        print(f"  {'message_count range':25s} {stats['msg_count_min']:>6} - {stats['msg_count_max']}")
        print(f"  {'unique max_tokens':25s} {stats['max_tokens_unique']:>15}")
        print(f"  {'unique temperatures':25s} {stats['temperature_unique']:>15}")
        print(f"  {'has_tools = true':25s} {stats['has_tools_true']:>15}")
        print(f"  {'has_tools = false':25s} {stats['has_tools_false']:>15}")
    else:
        print("  (no log data found)")

    # ── Error summary ──────────────────────────────────────────────
    total_errors = sum(r["errors"] for r in all_results)
    if total_errors:
        print(f"\n  Total errors: {total_errors}")
        for res in all_results:
            if res["errors"]:
                print(f"    round={res['round']}  {res['agent_type']}: "
                      f"{res['errors']} failed")

    print("=" * 65)

    # ── Feature richness check ─────────────────────────────────────
    if stats:
        issues = []
        if stats["msg_count_min"] == stats["msg_count_max"]:
            issues.append("message_count is constant — multi-turn not working")
        if stats["max_tokens_unique"] < 3:
            issues.append("max_tokens has < 3 unique values — not enough variance")
        if stats["temperature_unique"] < 3:
            issues.append("temperature has < 3 unique values — not enough variance")
        if stats["char_len_max"] - stats["char_len_min"] < 500:
            issues.append("input_char_length range too narrow (< 500 chars)")

        if issues:
            print("\n  FEATURE RICHNESS WARNINGS:")
            for iss in issues:
                print(f"    - {iss}")
        else:
            print("\n  Feature richness: GOOD — all features have variance")
    print()


if __name__ == "__main__":
    main()
