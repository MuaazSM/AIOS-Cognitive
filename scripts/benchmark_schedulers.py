#!/usr/bin/env python3
"""
Benchmark Schedulers — compare FIFO vs CognitiveScheduler.

This is a post-hoc analysis script. It reads a JSONL log file,
classifies each row using the trained model, and computes per-class
latency/throughput stats. Run the workload twice (once with FIFO,
once with cognitive) and compare the output logs.

Usage:
    python scripts/benchmark_schedulers.py --fifo-log logs/fifo_run.jsonl --cognitive-log logs/cognitive_run.jsonl
    python scripts/benchmark_schedulers.py --log aios/logs/llm_syscalls.jsonl  # analyze single log
"""

import argparse
import json
import os
import sys
import re
import pickle
import statistics
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "complexity_classifier.pkl"


def load_log(path: str) -> list[dict]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows


def classify_rows(rows: list[dict], model_path: Path) -> list[str]:
    """Classify each row using the trained model. Returns list of predicted classes."""
    if not model_path.exists():
        print(f"  Model not found at {model_path}, defaulting all to 'medium'")
        return ["medium"] * len(rows)

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    pipeline = model_data["pipeline"]
    feature_names = model_data["feature_names"]

    predictions = []
    for row in rows:
        agent_type = re.sub(r"_r\d+$", "", row.get("agent_name", ""))
        model_name = row.get("model_name", "unknown")

        raw = {
            "input_char_length": row.get("input_char_length", 0),
            "message_count": row.get("message_count", 2),
            "has_tools": int(bool(row.get("has_tools", False))),
            "max_tokens": row.get("max_tokens", 512),
            "temperature": row.get("temperature", 0.5),
        }

        for fn in feature_names:
            if fn.startswith("agent_"):
                raw[fn] = 1 if agent_type == fn[len("agent_"):] else 0
            elif fn.startswith("model_"):
                raw[fn] = 1 if model_name == fn[len("model_"):] else 0

        vec = [float(raw.get(fn, 0)) for fn in feature_names]
        pred = pipeline.predict(np.array([vec]))[0]
        predictions.append(pred)

    return predictions


def analyze_log(rows: list[dict], predictions: list[str], label: str):
    """Print per-class and overall stats for a log."""
    lats = [r["latency_ms"] for r in rows]
    waits = [r["wait_ms"] for r in rows]

    # Overall
    print(f"\n{'='*60}")
    print(f"  {label} — {len(rows)} rows")
    print(f"{'='*60}")
    print(f"  Overall latency : median={statistics.median(lats):>8,.0f}ms  "
          f"mean={statistics.mean(lats):>8,.0f}ms  "
          f"p95={np.percentile(lats, 95):>8,.0f}ms")
    print(f"  Overall wait    : median={statistics.median(waits):>8,.0f}ms  "
          f"mean={statistics.mean(waits):>8,.0f}ms")

    # Time span for throughput
    timestamps = [r["timestamp"] for r in rows if r.get("timestamp")]
    if len(timestamps) >= 2:
        span_s = max(timestamps) - min(timestamps)
        throughput = len(rows) / span_s if span_s > 0 else 0
        print(f"  Throughput      : {throughput:.2f} syscalls/sec over {span_s:.0f}s")

    # Per-class
    print(f"\n  {'Class':<10} {'Count':>6} {'Median Lat':>12} {'Mean Lat':>12} {'P95 Lat':>12} {'Mean Wait':>12}")
    print(f"  {'-'*10} {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for cls in ["fast", "medium", "large"]:
        cls_lats = [lats[i] for i in range(len(rows)) if predictions[i] == cls]
        cls_waits = [waits[i] for i in range(len(rows)) if predictions[i] == cls]
        if not cls_lats:
            print(f"  {cls:<10} {0:>6}")
            continue
        print(f"  {cls:<10} {len(cls_lats):>6} "
              f"{statistics.median(cls_lats):>12,.0f} "
              f"{statistics.mean(cls_lats):>12,.0f} "
              f"{np.percentile(cls_lats, 95):>12,.0f} "
              f"{statistics.mean(cls_waits):>12,.0f}")

    # Class distribution
    from collections import Counter
    dist = Counter(predictions)
    print(f"\n  Class distribution: {dict(dist)}")


def compare_logs(fifo_rows, fifo_preds, cog_rows, cog_preds):
    """Side-by-side comparison of FIFO vs Cognitive."""
    fifo_lats = [r["latency_ms"] for r in fifo_rows]
    cog_lats = [r["latency_ms"] for r in cog_rows]
    fifo_waits = [r["wait_ms"] for r in fifo_rows]
    cog_waits = [r["wait_ms"] for r in cog_rows]

    print(f"\n{'='*60}")
    print(f"  COMPARISON: FIFO vs Cognitive")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'FIFO':>12} {'Cognitive':>12} {'Delta':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")

    metrics = [
        ("Median latency (ms)", statistics.median(fifo_lats), statistics.median(cog_lats)),
        ("Mean latency (ms)", statistics.mean(fifo_lats), statistics.mean(cog_lats)),
        ("P95 latency (ms)", np.percentile(fifo_lats, 95), np.percentile(cog_lats, 95)),
        ("Median wait (ms)", statistics.median(fifo_waits), statistics.median(cog_waits)),
        ("Mean wait (ms)", statistics.mean(fifo_waits), statistics.mean(cog_waits)),
    ]

    for name, fifo_val, cog_val in metrics:
        delta = cog_val - fifo_val
        direction = "faster" if delta < 0 else "slower"
        print(f"  {name:<25} {fifo_val:>12,.0f} {cog_val:>12,.0f} {delta:>+12,.0f} ({direction})")

    # Per-class comparison for fast tasks (the ones that should benefit most)
    fast_fifo = [fifo_lats[i] for i in range(len(fifo_rows)) if fifo_preds[i] == "fast"]
    fast_cog = [cog_lats[i] for i in range(len(cog_rows)) if cog_preds[i] == "fast"]

    if fast_fifo and fast_cog:
        print(f"\n  FAST tasks specifically:")
        print(f"    FIFO  median: {statistics.median(fast_fifo):>8,.0f}ms  (n={len(fast_fifo)})")
        print(f"    Cog   median: {statistics.median(fast_cog):>8,.0f}ms  (n={len(fast_cog)})")
        improvement = statistics.median(fast_fifo) - statistics.median(fast_cog)
        print(f"    Delta       : {improvement:>+8,.0f}ms")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark FIFO vs Cognitive scheduler")
    parser.add_argument("--log", type=str, help="Single log to analyze (classify + stats)")
    parser.add_argument("--fifo-log", type=str, help="FIFO scheduler log")
    parser.add_argument("--cognitive-log", type=str, help="Cognitive scheduler log")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Path to classifier pkl")
    args = parser.parse_args()

    model_path = Path(args.model)

    if args.log:
        rows = load_log(args.log)
        preds = classify_rows(rows, model_path)
        analyze_log(rows, preds, f"Analysis: {args.log}")

    elif args.fifo_log and args.cognitive_log:
        fifo_rows = load_log(args.fifo_log)
        cog_rows = load_log(args.cognitive_log)
        fifo_preds = classify_rows(fifo_rows, model_path)
        cog_preds = classify_rows(cog_rows, model_path)

        analyze_log(fifo_rows, fifo_preds, "FIFO Scheduler")
        analyze_log(cog_rows, cog_preds, "Cognitive Scheduler")
        compare_logs(fifo_rows, fifo_preds, cog_rows, cog_preds)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/benchmark_schedulers.py --log llama.jsonl")
        print("  python scripts/benchmark_schedulers.py --fifo-log logs/fifo.jsonl --cognitive-log logs/cognitive.jsonl")


if __name__ == "__main__":
    main()
