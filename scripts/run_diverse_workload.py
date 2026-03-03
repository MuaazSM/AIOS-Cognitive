#!/usr/bin/env python3
"""
Run Diverse Workload — launches all 5 benchmark agents concurrently.

Each agent type runs in its own thread.  Within a thread the script
calls ``TaskBank.get(agent_type)`` ten times and sends each task through
the AIOS SDK via ``send_request``.

    5 threads × 10 tasks = 50 syscalls per round
    50 × --rounds N       = total log rows

Usage:
    python scripts/run_diverse_workload.py                 # 5 rounds → 250 rows
    python scripts/run_diverse_workload.py --rounds 10     # 10 rounds → 500 rows
    python scripts/run_diverse_workload.py --rounds 1      # quick smoke test
"""

import argparse
import json
import os
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

# ── Per-agent-type parameters ──────────────────────────────────────
# Each entry mirrors the LLMQuery settings from its real agent class.
AGENT_SPECS = [
    {
        "agent_type": "short_qa_agent",
        "system_prompt": "You are a concise factual QA assistant. Answer in one or two sentences.",
        "temperature": 0.3,
        "max_new_tokens": 100,
        "tools": None,
    },
    {
        "agent_type": "long_reasoning_agent",
        "system_prompt": (
            "You are a deep-reasoning assistant. Think step by step, consider "
            "multiple angles, and provide a thorough analysis."
        ),
        "temperature": 0.7,
        "max_new_tokens": 1024,
        "tools": None,
    },
    {
        "agent_type": "tool_use_agent",
        "system_prompt": (
            "You are a tool-augmented assistant. Use the provided tools when "
            "appropriate to answer the user's question."
        ),
        "temperature": 0.5,
        "max_new_tokens": 512,
        "tools": TOOL_SCHEMAS,
    },
    {
        "agent_type": "code_gen_agent",
        "system_prompt": (
            "You are an expert programmer. Write clean, well-documented code "
            "with type hints and tests."
        ),
        "temperature": 0.2,
        "max_new_tokens": 2048,
        "tools": None,
    },
    {
        "agent_type": "summarizer_agent",
        "system_prompt": (
            "You are a summarisation assistant. Condense the given text into a "
            "concise summary preserving all key points."
        ),
        "temperature": 0.4,
        "max_new_tokens": 256,
        "tools": None,
    },
]

TASKS_PER_AGENT = 10
SYSCALLS_PER_ROUND = TASKS_PER_AGENT * len(AGENT_SPECS)      # 50
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "llm_syscalls.jsonl")


# ── Worker executed inside each thread ─────────────────────────────

def _run_agent_thread(spec: dict, round_num: int) -> dict:
    """
    Fetch 10 tasks via TaskBank.get() and send each through the AIOS SDK.
    Returns a summary dict with counts and timing.
    """
    agent_type = spec["agent_type"]
    agent_name = f"{agent_type}_r{round_num}"
    system_prompt = spec["system_prompt"]
    temperature = spec["temperature"]
    max_new_tokens = spec["max_new_tokens"]
    tools = spec["tools"]

    completed = 0
    errors = 0
    t0 = time.time()

    for _ in range(TASKS_PER_AGENT):
        task = TaskBank.get(agent_type)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]
        query = LLMQuery(
            messages=messages,
            tools=tools,
            action_type="chat",
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
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


# ── Log-file helpers ───────────────────────────────────────────────

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
        "has_tools_true": tools_true,
        "has_tools_false": tools_false,
    }


# ── Main ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run diverse agent workload for syscall logging",
    )
    parser.add_argument(
        "--rounds", type=int, default=5,
        help="Repeat the full 5-agent batch N times (default: 5)",
    )
    args = parser.parse_args()

    n_agents = len(AGENT_SPECS)
    estimated_rows = SYSCALLS_PER_ROUND * args.rounds
    rows_before = _count_log_rows()

    # ── Pre-launch summary ─────────────────────────────────────────
    print("=" * 60)
    print("  Diverse Workload Runner")
    print("=" * 60)
    print(f"  Agents           : {n_agents}")
    for s in AGENT_SPECS:
        has_tools = "tools" if s["tools"] else "no-tools"
        print(f"    - {s['agent_type']:25s}  temp={s['temperature']}  "
              f"max_tok={s['max_new_tokens']:<5}  {has_tools}")
    print(f"  Tasks per agent  : {TASKS_PER_AGENT}")
    print(f"  Rounds           : {args.rounds}")
    print(f"  Syscalls / round : {SYSCALLS_PER_ROUND}")
    print(f"  Estimated rows   : {estimated_rows}")
    print(f"  Existing log rows: {rows_before}")
    print(f"  Log file         : {LOG_FILE}")
    print(f"  Kernel URL       : {KERNEL_URL}")
    print("=" * 60)
    print()

    t_start = time.time()
    all_results: list[dict] = []

    for r in range(1, args.rounds + 1):
        print(f"── Round {r}/{args.rounds} ──")
        round_t0 = time.time()

        with ThreadPoolExecutor(max_workers=n_agents) as pool:
            futures = {
                pool.submit(_run_agent_thread, spec, r): spec["agent_type"]
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
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Wall time        : {total_elapsed:.1f}s")
    print(f"  New log rows     : {new_rows}  (expected {estimated_rows})")
    print(f"  Total log rows   : {rows_after}")
    print()

    stats = _read_log_stats()
    if stats:
        hdr = f"  {'Metric':<25} {'Min':>10} {'Median':>10} {'Max':>10}"
        sep = f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}"
        print(hdr)
        print(sep)
        print(f"  {'latency_ms':<25} "
              f"{stats['latency_min']:>10.1f} "
              f"{stats['latency_median']:>10.1f} "
              f"{stats['latency_max']:>10.1f}")
        print(f"  {'input_char_length':<25} "
              f"{stats['char_len_min']:>10} "
              f"{stats['char_len_median']:>10} "
              f"{stats['char_len_max']:>10}")
        print()
        print(f"  has_tools = true  : {stats['has_tools_true']}")
        print(f"  has_tools = false : {stats['has_tools_false']}")
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

    print("=" * 60)


if __name__ == "__main__":
    main()
