#!/usr/bin/env python3
"""
Diverse Workload Generator for LLM Syscall Logging
===================================================
Sends varied LLM queries directly to the AIOS kernel to produce
feature-diverse rows in logs/llm_syscalls.jsonl for Ridge regression.

Varies:
  - input_char_length  (~100 – 5000+)
  - message_count       (1 – 10)
  - has_tools           (True / False)
  - max_tokens          (50, 128, 256, 512, 1024, 2048)
  - temperature         (0.0, 0.3, 0.5, 0.7, 1.0, 1.5)

Usage:
  python scripts/diverse_workload.py --n 250 --concurrency 10
"""

import argparse
import json
import random
import string
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

KERNEL_URL = "http://127.0.0.1:8000"

# ── prompt templates (varied lengths) ────────────────────────────────
SHORT_PROMPTS = [
    "What is 2 + 2?",
    "Say hello.",
    "Name a color.",
    "Define AI.",
    "What is gravity?",
    "Translate 'hello' to French.",
    "What is the capital of Japan?",
    "Name three planets.",
]

MEDIUM_PROMPTS = [
    "Explain the difference between supervised and unsupervised machine learning in 3 sentences.",
    "Write a haiku about the ocean and explain the syllable structure.",
    "Compare and contrast Python and JavaScript for web development. List at least 3 pros and cons each.",
    "Describe the water cycle in detail, including evaporation, condensation, and precipitation.",
    "What are the main causes of climate change? Provide a brief summary with at least 5 contributing factors.",
    "Explain how a neural network learns, covering forward pass, loss, and backpropagation.",
    "Summarize the plot of Romeo and Juliet in exactly 100 words.",
    "What is the significance of the Turing Test in artificial intelligence research?",
]

LONG_PROMPTS = [
    (
        "You are a senior software architect. A client wants to migrate their monolithic "
        "Java application (500k LOC, Oracle DB, SOAP APIs) to a microservices architecture "
        "on Kubernetes. The application handles payroll processing for 10,000 employees, "
        "has strict compliance requirements (SOX, HIPAA), and must maintain 99.99% uptime. "
        "Provide a detailed migration plan covering: (1) service decomposition strategy, "
        "(2) data migration approach, (3) API gateway design, (4) authentication and "
        "authorization, (5) monitoring and observability, (6) rollback strategy, "
        "(7) timeline and team structure. Be specific about technology choices."
    ),
    (
        "Write a comprehensive analysis of the economic impacts of artificial intelligence "
        "on the global labor market over the next 20 years. Consider: (1) which industries "
        "will be most affected, (2) job displacement vs. job creation dynamics, (3) the role "
        "of education and retraining programs, (4) geographic disparities between developed "
        "and developing nations, (5) policy recommendations for governments, (6) the gig "
        "economy transformation, (7) universal basic income considerations. Support each "
        "point with reasoning and cite relevant economic theories where applicable."
    ),
    (
        "Design a distributed real-time recommendation system for an e-commerce platform "
        "serving 50 million daily active users. Requirements: sub-100ms latency for "
        "recommendations, personalization based on browsing history, purchase history, "
        "and collaborative filtering, support for A/B testing, handle cold-start problems "
        "for new users and items, scale horizontally, and comply with GDPR. Detail the "
        "architecture including data ingestion pipeline, feature store, model serving layer, "
        "caching strategy, and fallback mechanisms. Discuss tradeoffs between accuracy and "
        "latency, and explain your choice of ML algorithms."
    ),
    (
        "You are a medical research assistant. Provide a thorough literature review on "
        "the current state of mRNA vaccine technology beyond COVID-19 applications. Cover: "
        "(1) mechanism of action and immunological basis, (2) ongoing clinical trials for "
        "cancer vaccines (melanoma, pancreatic, lung), (3) applications in rare genetic "
        "diseases, (4) challenges in lipid nanoparticle delivery optimization, (5) cold "
        "chain logistics improvements, (6) regulatory pathways and accelerated approval "
        "mechanisms, (7) ethical considerations in human trials, (8) comparison with "
        "traditional vaccine platforms. Organize by disease area and technology readiness level."
    ),
    (
        "Create a detailed business plan for a startup that uses large language models to "
        "automate legal document review. Include: executive summary, market analysis with "
        "TAM/SAM/SOM calculations, competitive landscape (existing legal tech solutions), "
        "product description and technical architecture, go-to-market strategy targeting "
        "AmLaw 200 firms, pricing model (per-document vs. subscription vs. usage-based), "
        "team requirements and hiring plan for the first 18 months, financial projections "
        "with 3-year P&L, key risks and mitigation strategies, fundraising timeline and "
        "expected Series A terms. The document should be suitable for presenting to VCs."
    ),
]

# ── fake tool schemas for has_tools=True rows ────────────────────────
DUMMY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform arithmetic calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_executor",
            "description": "Execute Python code and return output",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to run"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds"}
                },
                "required": ["code"]
            }
        }
    },
]

# ── parameter distributions ──────────────────────────────────────────
TEMPERATURES = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5]
MAX_TOKENS   = [50, 100, 128, 200, 256, 384, 512, 768, 1024, 1500, 2048]
AGENT_NAMES  = [
    "research_agent", "code_reviewer", "math_tutor", "travel_planner",
    "health_advisor", "legal_analyst", "finance_bot", "creative_writer",
    "data_scientist", "devops_engineer", "product_manager", "ux_designer",
    "customer_support", "marketing_analyst", "qa_tester",
]

SYSTEM_PROMPTS = [
    "You are a helpful assistant.",
    "You are an expert software engineer. Be concise and precise.",
    "You are a creative writing coach. Use vivid language and metaphors.",
    "You are a math tutor. Show your work step by step.",
    "You are a medical information assistant. Always caveat with 'consult a doctor'.",
    "You are a legal research assistant. Cite relevant statutes when possible.",
    "You are a financial analyst. Use quantitative reasoning and data.",
    "You are a travel guide. Be enthusiastic and provide practical tips.",
    "",  # no system prompt
]

# ── query builder ────────────────────────────────────────────────────

def _random_filler(length: int) -> str:
    """Generate random text to pad messages to a target character length."""
    words = []
    cur = 0
    vocab = [
        "the", "system", "processes", "data", "efficiently", "using",
        "advanced", "algorithms", "for", "optimization", "and", "scaling",
        "distributed", "compute", "resources", "across", "multiple", "nodes",
        "while", "maintaining", "consistency", "throughput", "latency",
        "requirements", "are", "critical", "in", "production", "environments",
        "where", "reliability", "matters", "most", "therefore", "we", "need",
        "to", "carefully", "evaluate", "each", "component", "before", "deployment",
    ]
    while cur < length:
        w = random.choice(vocab)
        words.append(w)
        cur += len(w) + 1
    return " ".join(words)[:length]


def build_query(idx: int):
    """
    Build a single diverse LLM query dict.
    Returns (agent_name, payload_dict).
    """
    # Pick random parameters
    temp = random.choice(TEMPERATURES)
    max_tok = random.choice(MAX_TOKENS)
    agent = random.choice(AGENT_NAMES)
    use_tools = random.random() < 0.30  # ~30% have tools

    # Decide message count: 1-10
    # Weighted: mostly 1-4, sometimes more
    msg_count_weights = [25, 25, 15, 10, 7, 5, 4, 3, 3, 3]  # for 1..10
    n_user_msgs = random.choices(range(1, 11), weights=msg_count_weights, k=1)[0]

    # Pick prompt complexity
    prompt_type = random.choices(
        ["short", "medium", "long"],
        weights=[35, 40, 25],
        k=1
    )[0]

    if prompt_type == "short":
        base_prompt = random.choice(SHORT_PROMPTS)
    elif prompt_type == "medium":
        base_prompt = random.choice(MEDIUM_PROMPTS)
    else:
        base_prompt = random.choice(LONG_PROMPTS)

    # Build messages list
    messages = []

    # System prompt (80% chance)
    if random.random() < 0.80:
        sys_prompt = random.choice(SYSTEM_PROMPTS)
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

    # Build a multi-turn conversation
    for turn in range(n_user_msgs):
        if turn == 0:
            content = base_prompt
        else:
            # Simulate follow-up messages of varying length
            follow_up_len = random.choice([20, 50, 100, 200, 500])
            follow_ups = [
                f"Can you elaborate on point {turn}?",
                f"What about the implications for {random.choice(['industry', 'research', 'education', 'policy'])}?",
                f"Please provide more detail: {_random_filler(follow_up_len)}",
                f"How does this compare to {random.choice(['traditional approaches', 'recent developments', 'competing methods'])}?",
                f"Summarize the key takeaways so far in {random.randint(2, 5)} bullet points.",
            ]
            content = random.choice(follow_ups)

        messages.append({"role": "user", "content": content})

        # Add a fake assistant reply for multi-turn (except last turn)
        if turn < n_user_msgs - 1:
            reply_len = random.choice([30, 80, 150, 300])
            messages.append({
                "role": "assistant",
                "content": _random_filler(reply_len)
            })

    # Tools
    tools = None
    if use_tools:
        n_tools = random.randint(1, len(DUMMY_TOOLS))
        tools = random.sample(DUMMY_TOOLS, n_tools)

    # Build payload
    query_data = {
        "query_class": "llm",
        "messages": messages,
        "action_type": "chat",
        "temperature": temp,
        "max_new_tokens": max_tok,
        "message_return_type": "text",
    }
    if tools:
        query_data["tools"] = tools

    payload = {
        "query_type": "llm",
        "agent_name": agent,
        "query_data": query_data,
    }

    return agent, payload


# ── sender ───────────────────────────────────────────────────────────

_stats_lock = threading.Lock()
_stats = {"ok": 0, "err": 0, "total_ms": 0.0}


def send_one(idx: int, payload: dict, agent: str):
    """Send a single query to the kernel and record outcome."""
    t0 = time.time()
    try:
        r = requests.post(
            f"{KERNEL_URL}/query",
            json=payload,
            timeout=300,
        )
        elapsed = (time.time() - t0) * 1000
        ok = r.status_code == 200
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        ok = False

    with _stats_lock:
        if ok:
            _stats["ok"] += 1
        else:
            _stats["err"] += 1
        _stats["total_ms"] += elapsed

    done = _stats["ok"] + _stats["err"]
    if done % 10 == 0 or done == 1:
        print(f"  [{done}] agent={agent:20s}  status={'OK' if ok else 'ERR'}  {elapsed:.0f}ms")


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Diverse workload generator for AIOS kernel")
    parser.add_argument("--n", type=int, default=250, help="Total number of queries to send")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent requests")
    parser.add_argument("--url", type=str, default=KERNEL_URL, help="Kernel base URL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    global KERNEL_URL
    KERNEL_URL = args.url
    random.seed(args.seed)

    # Pre-generate all queries
    print(f"Generating {args.n} diverse queries (seed={args.seed})...")
    queries = [build_query(i) for i in range(args.n)]

    # Print feature distribution summary
    temps = [q[1]["query_data"]["temperature"] for q in queries]
    max_toks = [q[1]["query_data"]["max_new_tokens"] for q in queries]
    msg_counts = [len(q[1]["query_data"]["messages"]) for q in queries]
    has_tools_pct = sum(1 for q in queries if q[1]["query_data"].get("tools")) / len(queries) * 100
    char_lens = [len(json.dumps(q[1]["query_data"]["messages"])) for q in queries]

    print(f"\n{'Feature':<25} {'Min':>8} {'Max':>8} {'Mean':>8} {'Unique':>8}")
    print("-" * 60)
    print(f"{'input_char_length':<25} {min(char_lens):>8} {max(char_lens):>8} {sum(char_lens)/len(char_lens):>8.0f} {len(set(char_lens)):>8}")
    print(f"{'message_count':<25} {min(msg_counts):>8} {max(msg_counts):>8} {sum(msg_counts)/len(msg_counts):>8.1f} {len(set(msg_counts)):>8}")
    print(f"{'temperature':<25} {min(temps):>8.1f} {max(temps):>8.1f} {sum(temps)/len(temps):>8.2f} {len(set(temps)):>8}")
    print(f"{'max_tokens':<25} {min(max_toks):>8} {max(max_toks):>8} {sum(max_toks)/len(max_toks):>8.0f} {len(set(max_toks)):>8}")
    print(f"{'has_tools (%)':<25} {has_tools_pct:>8.1f}%")
    print(f"{'agents':<25} {len(set(q[0] for q in queries)):>8} unique")

    print(f"\nSending {args.n} queries with concurrency={args.concurrency} ...")
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(send_one, i, payload, agent)
            for i, (agent, payload) in enumerate(queries)
        ]
        for f in as_completed(futures):
            f.result()  # propagate exceptions

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  OK:  {_stats['ok']}")
    print(f"  ERR: {_stats['err']}")
    if _stats["ok"] > 0:
        print(f"  Avg latency: {_stats['total_ms'] / (_stats['ok'] + _stats['err']):.0f}ms")
    print(f"\nCheck: wc -l logs/llm_syscalls.jsonl")


if __name__ == "__main__":
    main()
