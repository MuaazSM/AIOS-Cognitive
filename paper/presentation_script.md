# Presentation Script — "When Scheduling Fails: Batched Execution in LLM Systems"

---

## MUAAZ: Section 1 — Introduction

Good morning ma'am. Our paper is titled "When Scheduling Fails: Batched Execution in LLM Systems."

AIOS is an LLM agent operating system that manages multiple AI agents competing for the same LLM resource. It uses FIFO scheduling by default. The natural question is — can we do better using classical OS scheduling algorithms like Shortest Job First, which provably minimizes average wait time when job sizes are known?

Our research question: Do classical scheduling assumptions — observable job sizes and sequential execution — hold in LLM agent systems? If not, what are the implications?

We hypothesized that request features like max_tokens and prompt length could approximate job size and enable scheduling improvements. We tested this through three phases and ultimately rejected the hypothesis. The root cause is that GPU batching destroys job-size observability, making classical scheduling theory inapplicable here.

Our contributions: empirical evidence of weak feature-latency correlation, identification of two violated assumptions, quantitative proof that FIFO is near-optimal, discovery of a fairness-latency tradeoff, and open-source release of six scheduler implementations.

---

## MUAAZ: Section 2 — Related Work

Briefly — AIOS by Mei et al. provides the architecture. MemGPT proposes LLM-as-OS. On scheduling theory, SJF, MLFQ, and HRRN are well-established but assume sequential execution and observable job sizes. Learned scheduling exists for databases and clusters but not within an LLM agent OS. Our work is the first to test these assumptions in this domain.

---

## DHRUVIL: Section 3 — System Architecture & Methodology

I'll cover our system architecture and methodology.

This diagram shows our instrumented AIOS pipeline. Five agent types — QA, Code, Tool, Reasoning, and Summarizer — submit queries through the AIOS SDK, which decomposes them into typed system calls. LLM syscalls enter a global request queue, the scheduler picks a batch and dispatches it to the Ollama GPU backend.

We instrumented the scheduler to log 15 features per syscall — request features like input_char_length, message_count, has_tools, max_tokens, temperature; context features being model_name and agent_name; timing features; and outcome features.

Our experiment has three phases shown in this pipeline diagram — Phase 1 is correlation analysis, Phase 2 is classification, Phase 3 is simulation. Each phase independently reaches the same conclusion.

For workload design, we randomized all parameters independently to avoid confounding. max_tokens sampled uniformly from 64 to 2048, temperature from 7 values, 30% multi-turn requests, prompt padding, and random 1-5 second arrival stagger. Five agent types run across 20 rounds per model.

---

## DHRUVIL: Section 4 — Experimental Setup

For the setup — we used Ollama serving Llama 3.1 8B and Mistral 7B, both instruction-tuned, running locally one at a time. We collected 1,000 syscalls per model, 2,000 total, with zero errors.

The dataset statistics: mean latency around 5,900ms, mean wait 1,800ms, average max_tokens of 684.

---

## MUAAZ: (adding depth on Section 4)

On dataset justification — 2,000 samples is modest for general ML, but this is a controlled experiment. We have exhaustive randomization across 5 agent types, 6 token budgets, 7 temperatures, and 2 models. Every configuration appears multiple times. We validate everything with stratified 5-fold cross-validation and report confidence intervals. The goal isn't building a production model — it's testing whether scheduling assumptions hold, and for that this coverage is sufficient.

---

## DHRUVIL: Section 5 — Phase 1: Feature-Latency Analysis

Phase 1 tests whether request features correlate with latency strongly enough for predictive scheduling.

The Spearman correlation results: the strongest predictor is max_tokens at rho equals 0.206. model_name is 0.132 — Llama is consistently 920ms slower than Mistral. Everything else — input_char_length, temperature, message_count — is below 0.05 and either negligible or not significant.

This correlation bar chart shows it clearly — only max_tokens crosses rho of 0.1. The features are simply too weak for predictive scheduling.

---

## MUAAZ: (adding depth on Section 5) — Structural Explanation

The key question is WHY these features fail. There are three structural reasons rooted in how LLM inference actually works.

First — max_tokens is a ceiling, not a target. The model generates until it emits an end-of-sequence token OR hits the limit. Setting max_tokens to 2048 doesn't produce 2048 tokens. The model stops early for most requests. Only very low values like 64 or 128 actively truncate output, which is why we see that weak positive correlation.

Second — GPU batching decouples individual latency from request properties. Ollama batches concurrent requests to the GPU. All requests in a batch complete at approximately the same time, because prefill runs in parallel and decode steps are dominated by the longest-generating request in the batch.

Third — actual output length is unobservable before execution. The true latency driver is how many tokens the model actually generates, which depends on semantic content and model state — not on any input feature we can see. This is the fundamental difference from classical OS scheduling.

So our hypothesis is rejected. The failure isn't due to bad data or poor modeling — it's structural.

---

## VAISHNAVI: Section 6 — Phase 2: Classification

Since continuous prediction failed, we attempted discrete classification into priority classes.

We defined three latency classes using percentile thresholds — fast below the 33rd percentile at 3,846ms, medium in the middle, and large above the 66th percentile at 6,993ms. The feature vector includes 5 numeric features, 5 agent-type one-hot indicators, and 2 model indicators — 12 features total. We evaluated six classifiers with stratified 5-fold CV.

The results: Gradient Boosting performed best with F1 of 0.438 before tuning. After hyperparameter tuning with a 432-candidate grid search, held-out F1 reached 0.47 with accuracy of 0.47. The random baseline for three classes is 0.33, so we're only marginally above chance.

Feature importances were: input_char_length at 0.36, max_tokens at 0.24, and agent_code_gen at 0.10. The medium class was poorly discriminated at F1 of 0.41.

---

## MUAAZ: (adding depth on Section 6) — Request Type Classifier

This is where we asked a different question. The latency classifier failed — but is the feature space itself unstructured, or is latency just the wrong prediction target?

We built a 4-class request type classifier with classes that represent real reasoning patterns: Simple QA for short single-turn requests, Conversational for multi-turn dialogue, Tool-Augmented for requests with tool calls, and Long Generation for extended output requests.

The critical design choice — we excluded max_tokens and has_tools from the feature vector entirely. The classifier only sees input_char_length, temperature, agent_type, model_name, and message_count. This forces it to learn from proxy signals. We used a proper 60-20-20 train-validation-test split.

Result: F1 of 0.75 on held-out test. Tool-Augmented and Conversational are classified perfectly — F1 of 1.0 — because agent_type and message_count are strong proxy signals. Simple QA gets F1 of 0.70. Long Generation is hardest at F1 of 0.28.

The confusion matrix tells the story — the only misclassification mass is 64 Long Generation requests predicted as Simple QA. Both are single-turn with no tools, so without max_tokens they're structurally identical to the classifier.

The insight: comparing F1 of 0.47 for latency prediction versus 0.75 for request type classification — the feature space has meaningful structure for categorizing WHAT a request is, but NOT for predicting HOW LONG it will take. You can identify request types for priority assignment, but you cannot predict runtime under GPU batching.

---

## VAISHNAVI: Section 7 — Phase 3: Simulation

For Phase 3, we simulated 16 scheduling algorithms on our logged data. Syscalls are grouped into batches by arrival window, and within each batch the algorithm determines dispatch order. We model sequential execution within each batch.

The 16 algorithms include FIFO, LIFO, SJF by max_tokens, LJF, SJF by input length, SRTF, Priority, HRRN, EDF, Aging SJF, Round Robin, Multi-Level Queue, Fair Share, Lottery, Temp-First, and Combined Score.

The key result: FIFO matches or outperforms all 16 algorithms on average turnaround. Round Robin and Fair Share tie with FIFO. SJF is 6.2% worse. LIFO is worst at 18.6% worse. A Welch's t-test between FIFO and SJF yields p equals 0.06 — not significant at alpha 0.05.

But the per-class analysis reveals a significant tradeoff. SJF reduces fast-task wait by 68% — from 5,400ms down to 1,711ms — but nearly doubles large-task wait from 6,200 to 12,000ms. Priority scheduling shows a similar pattern.

---

## MUAAZ: (adding depth on Section 7) — Why FIFO Wins

Let me explain the theory behind why FIFO wins. It contradicts the classical result that SJF minimizes average wait. Two assumptions are violated.

Violated assumption one — job-size observability. SJF requires knowing job completion time before execution. For LLM requests, the relevant job size is tokens the model will actually generate — which depends on semantic complexity, internal model state, and EOS probability at each decode step. None of this is in the request metadata. max_tokens gives only a loose upper bound.

Violated assumption two — sequential execution. SJF's optimality proof assumes jobs run one at a time. GPU inference batches requests together. The batch completion time equals prefill time plus the maximum decode time across all requests in the batch. Since the slowest request dominates, reordering within the batch doesn't change total execution time.

This diagram makes it concrete — on the left, sequential execution, SJF reduces average turnaround by 25% by putting short jobs first. On the right, GPU batched execution, all three jobs finish together at max of T_i. Order does not matter.

The implication: the real scheduling lever for LLM systems is batch composition — which requests to group together — not dispatch order within a batch.

---

## VAISHNAVI: Section 8 — Comparison with AIOS

Our results might seem to contradict AIOS's finding of 2.1x throughput improvement. But the comparison is different.

AIOS compared having a scheduler versus having no scheduler at all — raw agent frameworks flooding the GPU causing CUDA out-of-memory retries. The 2.1x improvement comes from having any scheduler, not from the algorithm choice.

We compared FIFO versus other algorithms, all within AIOS's scheduler framework. Given that a scheduler exists, the choice of algorithm has minimal impact on average turnaround.

These results are complementary: AIOS demonstrates the necessity of scheduling. Our work demonstrates the limits of scheduling under GPU batching. The gap between no scheduler and any scheduler is 2.1x. The gap between FIFO and the best alternative is less than 1%.

---

## MUAAZ: Section 9 — Implementation

We implemented six drop-in scheduler variants in the AIOS kernel: FIFO, SJF, Priority with three-level queues and aging, MLFQ with dynamic demotion and periodic boost, HRRN using response ratio, and a cognitive ML-based scheduler. Switching between them requires a single config change — no code modification.

---

## MUAAZ: Section 10 — Discussion

On when Priority scheduling is justified — FIFO wins on average but loses on fairness. If you have a real-time assistant agent needing sub-3-second responses alongside a batch code generator tolerating 15 seconds, Priority reduces the assistant's wait by 73% at acceptable cost. The choice between FIFO and Priority is an engineering decision driven by deployment requirements, not a universal optimum.

Limitations: single GPU backend, similar model sizes at 7B versus 8B, sequential simulation rather than true GPU batch simulation, and 2,000 syscalls — sufficient for controlled experiments but may not generalize to production.

Future work directions: live batch-composition experiments, output length prediction which could restore SJF's advantage, multi-GPU scheduling, and RL-based adaptive scheduling.

---

## MUAAZ: Section 11 — Conclusion

To wrap up — our three-phase study on 2,000 AIOS syscalls reveals four findings:

One — job-size observability fails. Request features are weak latency predictors, max rho is just 0.206, because max_tokens is a ceiling and actual output length is unobservable.

Two — ML cannot predict latency — F1 equals 0.47. But it CAN classify request types — F1 equals 0.75. The feature space has structure, but that structure doesn't predict runtime.

Three — FIFO is near-optimal. Among 16 simulated algorithms, none reduce average turnaround below FIFO, because GPU batching nullifies SJF's sequential execution assumption.

Four — a fairness tradeoff exists. SJF and Priority reduce fast-task wait by 68 to 73% at the cost of large-task degradation — a dimension the original AIOS design does not address.

The scheduling frontier for LLM agent systems lies not in dispatch order but in batch composition — deciding which requests to group for GPU execution.

Thank you.

---
