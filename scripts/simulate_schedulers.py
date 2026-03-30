#!/usr/bin/env python3
"""
Simulate Scheduling Algorithms on existing JSONL log data.

Takes logged syscall data (collected under FIFO) and simulates what
would happen under different scheduling algorithms by re-ordering
the dispatch sequence within each batch window.

What CAN be simulated:
  - Dispatch order within each batch
  - Wait times (based on new dispatch order)
  - Average turnaround time

What CANNOT change:
  - Individual LLM inference times (these are fixed per request)

Usage:
    python scripts/simulate_schedulers.py --log llama.jsonl
    python scripts/simulate_schedulers.py --log llama.jsonl mistral.jsonl
"""

import argparse
import csv
import json
import statistics
import re
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_logs(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # Sort by created_time (arrival order)
    rows.sort(key=lambda r: r.get("created_time", 0))
    return rows


def group_into_batches(rows: list[dict], window_ms: float = 1000.0) -> list[list[dict]]:
    """
    Group syscalls into batches based on overlapping arrival windows.
    Requests arriving within window_ms of each other are in the same batch.
    This mimics how the AIOS scheduler collects requests before dispatching.
    """
    if not rows:
        return []

    batches = []
    current_batch = [rows[0]]
    batch_start = rows[0]["created_time"]

    for row in rows[1:]:
        if row["created_time"] - batch_start <= window_ms / 1000.0:
            current_batch.append(row)
        else:
            batches.append(current_batch)
            current_batch = [row]
            batch_start = row["created_time"]

    if current_batch:
        batches.append(current_batch)

    return batches


# ═══════════════════════════════════════════════════════════════════
# Scheduling algorithms — each takes a batch and returns reordered batch
# ═══════════════════════════════════════════════════════════════════

def fifo_order(batch: list[dict]) -> list[dict]:
    """FIFO: process in arrival order."""
    return sorted(batch, key=lambda r: r["created_time"])


def sjf_order(batch: list[dict]) -> list[dict]:
    """SJF: shortest max_tokens first."""
    return sorted(batch, key=lambda r: r.get("max_tokens", 512))


def ljf_order(batch: list[dict]) -> list[dict]:
    """LJF (Longest Job First): longest max_tokens first."""
    return sorted(batch, key=lambda r: r.get("max_tokens", 512), reverse=True)


def priority_order(batch: list[dict]) -> list[dict]:
    """Static priority: P0 (fast) > P1 (medium) > P2 (slow)."""
    def get_priority(r):
        mt = r.get("max_tokens", 512)
        mc = r.get("message_count", 2)
        tools = r.get("has_tools", False)
        if mt <= 128:
            return 0
        if mt <= 512 and not tools and mc <= 4:
            return 1
        return 2
    return sorted(batch, key=get_priority)


def hrrn_order(batch: list[dict]) -> list[dict]:
    """HRRN: highest response ratio next."""
    now = max(r["created_time"] for r in batch)
    def response_ratio(r):
        service = (r.get("max_tokens", 512)) / 100.0
        wait = now - r["created_time"]
        if service <= 0:
            service = 1.0
        return (wait + service) / service
    return sorted(batch, key=response_ratio, reverse=True)


def sjf_input_order(batch: list[dict]) -> list[dict]:
    """SJF by input_char_length (shorter prompts first)."""
    return sorted(batch, key=lambda r: r.get("input_char_length", 0))


def combined_score_order(batch: list[dict]) -> list[dict]:
    """Combined: weighted score of max_tokens + input_char_length."""
    def score(r):
        return (r.get("max_tokens", 512) * 0.7 +
                r.get("input_char_length", 0) * 0.3)
    return sorted(batch, key=score)


def lifo_order(batch: list[dict]) -> list[dict]:
    """LIFO (Last In First Out): most recent arrival first."""
    return sorted(batch, key=lambda r: r["created_time"], reverse=True)


def round_robin_order(batch: list[dict]) -> list[dict]:
    """Round Robin by agent_type: interleave one from each agent."""
    import re
    from collections import OrderedDict
    agent_queues = OrderedDict()
    for r in sorted(batch, key=lambda r: r["created_time"]):
        at = re.sub(r"_r\d+$", "", r["agent_name"])
        agent_queues.setdefault(at, []).append(r)

    result = []
    while any(agent_queues.values()):
        for at in list(agent_queues.keys()):
            if agent_queues[at]:
                result.append(agent_queues[at].pop(0))
            else:
                del agent_queues[at]
    return result


def edf_order(batch: list[dict]) -> list[dict]:
    """Earliest Deadline First: requests that have waited longest get priority.
    Deadline = created_time + max_tokens (shorter deadline = earlier execution)."""
    def deadline(r):
        return r["created_time"] + (r.get("max_tokens", 512) / 1000.0)
    return sorted(batch, key=deadline)


def aging_sjf_order(batch: list[dict]) -> list[dict]:
    """SJF with aging: adjusts priority based on wait time.
    Score = max_tokens - (wait_time_ms * 0.5). Lower score = higher priority."""
    now = max(r["created_time"] for r in batch)
    def aged_score(r):
        mt = r.get("max_tokens", 512)
        wait_ms = (now - r["created_time"]) * 1000
        return mt - (wait_ms * 0.5)
    return sorted(batch, key=aged_score)


def lottery_order(batch: list[dict]) -> list[dict]:
    """Lottery scheduling: random order (weighted by inverse max_tokens).
    Short jobs get more lottery tickets."""
    import random
    random.seed(42)  # deterministic for reproducibility
    def tickets(r):
        mt = r.get("max_tokens", 512)
        return max(1, 2048 - mt)  # more tickets for shorter jobs
    weighted = [(r, tickets(r)) for r in batch]
    result = []
    remaining = list(weighted)
    while remaining:
        total = sum(w for _, w in remaining)
        pick = random.uniform(0, total)
        cumulative = 0
        for i, (r, w) in enumerate(remaining):
            cumulative += w
            if cumulative >= pick:
                result.append(r)
                remaining.pop(i)
                break
    return result


def multilevel_queue_order(batch: list[dict]) -> list[dict]:
    """Multi-Level Queue: separate queues per agent_type, drain in fixed order.
    Order: short_qa > tool_use > summarizer > code_gen > long_reasoning."""
    import re
    AGENT_PRIORITY = {
        "short_qa_agent": 0,
        "tool_use_agent": 1,
        "summarizer_agent": 2,
        "code_gen_agent": 3,
        "long_reasoning_agent": 4,
    }
    def agent_pri(r):
        at = re.sub(r"_r\d+$", "", r["agent_name"])
        return AGENT_PRIORITY.get(at, 5)
    return sorted(batch, key=lambda r: (agent_pri(r), r["created_time"]))


def srtf_order(batch: list[dict]) -> list[dict]:
    """Shortest Remaining Time First (approximated as SJF by estimated total cost).
    Cost = max_tokens * input_char_length / 1000 (approximates total compute)."""
    def cost(r):
        return (r.get("max_tokens", 512) * r.get("input_char_length", 100)) / 1000.0
    return sorted(batch, key=cost)


def fair_share_order(batch: list[dict]) -> list[dict]:
    """Fair Share: agents that have used less total CPU time go first.
    Tracks cumulative latency per agent_type across the batch."""
    import re
    agent_used = {}
    for r in batch:
        at = re.sub(r"_r\d+$", "", r["agent_name"])
        agent_used.setdefault(at, 0)

    result = []
    remaining = list(sorted(batch, key=lambda r: r["created_time"]))
    while remaining:
        # Pick the request whose agent_type has used least time
        best_idx = 0
        best_agent_time = float("inf")
        for i, r in enumerate(remaining):
            at = re.sub(r"_r\d+$", "", r["agent_name"])
            if agent_used.get(at, 0) < best_agent_time:
                best_agent_time = agent_used.get(at, 0)
                best_idx = i
        chosen = remaining.pop(best_idx)
        at = re.sub(r"_r\d+$", "", chosen["agent_name"])
        agent_used[at] = agent_used.get(at, 0) + chosen["latency_ms"]
        result.append(chosen)
    return result


def temperature_order(batch: list[dict]) -> list[dict]:
    """Low temperature first: deterministic requests (temp close to 0)
    tend to generate shorter outputs, so prioritize them."""
    return sorted(batch, key=lambda r: r.get("temperature", 0.5))


ALGORITHMS = {
    "FIFO": fifo_order,
    "LIFO": lifo_order,
    "SJF (max_tokens)": sjf_order,
    "LJF (max_tokens)": ljf_order,
    "SJF (input_len)": sjf_input_order,
    "SRTF (cost)": srtf_order,
    "Priority (rules)": priority_order,
    "HRRN": hrrn_order,
    "EDF (deadline)": edf_order,
    "Aging SJF": aging_sjf_order,
    "Round Robin": round_robin_order,
    "Multi-Level Queue": multilevel_queue_order,
    "Fair Share": fair_share_order,
    "Lottery": lottery_order,
    "Temp-First": temperature_order,
    "Combined Score": combined_score_order,
}


# ═══════════════════════════════════════════════════════════════════
# Simulation engine
# ═══════════════════════════════════════════════════════════════════

def simulate_scheduler(batches: list[list[dict]], order_fn) -> dict:
    """
    Simulate a scheduling algorithm on batched syscalls.

    For each batch:
      1. Reorder using order_fn
      2. Simulate sequential execution: each request starts after the previous finishes
      3. Calculate simulated wait_time and turnaround_time

    Returns stats dict.
    """
    all_waits = []
    all_turnarounds = []
    all_latencies = []
    class_waits = defaultdict(list)
    class_turnarounds = defaultdict(list)

    for batch in batches:
        ordered = order_fn(batch)

        # The LLM processes one at a time (or in small groups).
        # Simulate: first request starts immediately, subsequent wait for prior to finish.
        # Each request's actual inference time (latency_ms) is fixed.
        batch_start = min(r["created_time"] for r in ordered)
        current_time = batch_start

        for r in ordered:
            arrival = r["created_time"]
            inference_ms = r["latency_ms"]

            # Request can't start before it arrives or before LLM is free
            sim_start = max(current_time, arrival)
            sim_wait = (sim_start - arrival) * 1000  # ms
            sim_end = sim_start + inference_ms / 1000.0
            sim_turnaround = (sim_end - arrival) * 1000  # ms

            current_time = sim_end

            all_waits.append(sim_wait)
            all_turnarounds.append(sim_turnaround)
            all_latencies.append(inference_ms)

            # Classify for per-class stats
            mt = r.get("max_tokens", 512)
            if mt <= 128:
                cls = "fast"
            elif mt <= 512:
                cls = "medium"
            else:
                cls = "large"
            class_waits[cls].append(sim_wait)
            class_turnarounds[cls].append(sim_turnaround)

    return {
        "n": len(all_waits),
        "avg_wait": statistics.mean(all_waits) if all_waits else 0,
        "median_wait": statistics.median(all_waits) if all_waits else 0,
        "p95_wait": float(np.percentile(all_waits, 95)) if all_waits else 0,
        "avg_turnaround": statistics.mean(all_turnarounds) if all_turnarounds else 0,
        "median_turnaround": statistics.median(all_turnarounds) if all_turnarounds else 0,
        "p95_turnaround": float(np.percentile(all_turnarounds, 95)) if all_turnarounds else 0,
        "avg_latency": statistics.mean(all_latencies) if all_latencies else 0,
        "class_waits": {k: statistics.mean(v) for k, v in class_waits.items()},
        "class_turnarounds": {k: statistics.mean(v) for k, v in class_turnarounds.items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Simulate scheduling algorithms on existing JSONL logs"
    )
    parser.add_argument(
        "--log", type=str, nargs="+", required=True,
        help="Path(s) to JSONL log files",
    )
    parser.add_argument(
        "--window", type=float, default=1000.0,
        help="Batch window in ms (requests arriving within this window form a batch)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models",
        help="Directory to save CSV results (default: models/)",
    )
    args = parser.parse_args()

    rows = load_logs(args.log)
    print(f"Loaded {len(rows)} syscalls")

    batches = group_into_batches(rows, window_ms=args.window)
    batch_sizes = [len(b) for b in batches]
    print(f"Grouped into {len(batches)} batches "
          f"(avg size: {statistics.mean(batch_sizes):.1f}, "
          f"max: {max(batch_sizes)}, "
          f"window: {args.window}ms)\n")

    # Run all algorithms
    results = {}
    for name, fn in ALGORITHMS.items():
        results[name] = simulate_scheduler(batches, fn)

    # ── Comparison table ────────────────────────────────────────
    print("=" * 95)
    print(f"  {'Algorithm':<22} {'Avg Wait':>10} {'Med Wait':>10} {'P95 Wait':>10} "
          f"{'Avg Turn':>10} {'Med Turn':>10} {'P95 Turn':>10}")
    print(f"  {'':22} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} "
          f"{'(ms)':>10} {'(ms)':>10} {'(ms)':>10}")
    print("=" * 95)

    # Sort by avg turnaround (best first)
    ranked = sorted(results.items(), key=lambda x: x[1]["avg_turnaround"])

    fifo_turn = results["FIFO"]["avg_turnaround"]

    for i, (name, s) in enumerate(ranked):
        delta = s["avg_turnaround"] - fifo_turn
        marker = " <-- best" if i == 0 else ""
        print(f"  {name:<22} "
              f"{s['avg_wait']:>10,.0f} {s['median_wait']:>10,.0f} {s['p95_wait']:>10,.0f} "
              f"{s['avg_turnaround']:>10,.0f} {s['median_turnaround']:>10,.0f} {s['p95_turnaround']:>10,.0f}"
              f"{marker}")

    # ── Improvement over FIFO ───────────────────────────────────
    print(f"\n  IMPROVEMENT OVER FIFO (avg turnaround):")
    print(f"  {'-'*50}")
    for name, s in ranked:
        delta = s["avg_turnaround"] - fifo_turn
        pct = (delta / fifo_turn * 100) if fifo_turn else 0
        direction = "faster" if delta < 0 else "slower" if delta > 0 else "same"
        print(f"  {name:<22} {delta:>+10,.0f}ms  ({pct:>+.1f}%)  {direction}")

    # ── Per-class breakdown for best algorithm ──────────────────
    best_name, best_stats = ranked[0]
    fifo_stats = results["FIFO"]

    print(f"\n  PER-CLASS WAIT TIME: {best_name} vs FIFO")
    print(f"  {'Class':<10} {'FIFO Wait':>12} {f'{best_name} Wait':>12} {'Delta':>12}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}")

    for cls in ["fast", "medium", "large"]:
        fifo_w = fifo_stats["class_waits"].get(cls, 0)
        best_w = best_stats["class_waits"].get(cls, 0)
        delta = best_w - fifo_w
        print(f"  {cls:<10} {fifo_w:>12,.0f} {best_w:>12,.0f} {delta:>+12,.0f}")

    print(f"\n  PER-CLASS TURNAROUND: {best_name} vs FIFO")
    print(f"  {'Class':<10} {'FIFO Turn':>12} {f'{best_name} Turn':>12} {'Delta':>12}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}")

    for cls in ["fast", "medium", "large"]:
        fifo_t = fifo_stats["class_turnarounds"].get(cls, 0)
        best_t = best_stats["class_turnarounds"].get(cls, 0)
        delta = best_t - fifo_t
        print(f"  {cls:<10} {fifo_t:>12,.0f} {best_t:>12,.0f} {delta:>+12,.0f}")

    # ── Save CSV results ────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    # 1. Summary CSV — one row per algorithm
    summary_path = out_dir / "simulation_results.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algorithm", "avg_wait_ms", "median_wait_ms", "p95_wait_ms",
            "avg_turnaround_ms", "median_turnaround_ms", "p95_turnaround_ms",
            "avg_latency_ms", "vs_fifo_ms", "vs_fifo_pct",
            "fast_avg_wait", "medium_avg_wait", "large_avg_wait",
            "fast_avg_turn", "medium_avg_turn", "large_avg_turn",
        ])
        for name, s in ranked:
            delta = s["avg_turnaround"] - fifo_turn
            pct = (delta / fifo_turn * 100) if fifo_turn else 0
            writer.writerow([
                name,
                round(s["avg_wait"], 1),
                round(s["median_wait"], 1),
                round(s["p95_wait"], 1),
                round(s["avg_turnaround"], 1),
                round(s["median_turnaround"], 1),
                round(s["p95_turnaround"], 1),
                round(s["avg_latency"], 1),
                round(delta, 1),
                round(pct, 2),
                round(s["class_waits"].get("fast", 0), 1),
                round(s["class_waits"].get("medium", 0), 1),
                round(s["class_waits"].get("large", 0), 1),
                round(s["class_turnarounds"].get("fast", 0), 1),
                round(s["class_turnarounds"].get("medium", 0), 1),
                round(s["class_turnarounds"].get("large", 0), 1),
            ])
    print(f"\n  Results saved to {summary_path}")
    print()


if __name__ == "__main__":
    main()
