# Cognitive Scheduler Experiment

End-to-end guide for collecting syscall data, running EDA, and training the complexity classifier.

## Prerequisites

- Python 3.10–3.11 with venv
- [Ollama](https://ollama.com/) installed and running
- AIOS kernel + Cerebrum SDK installed (see README)

```bash
ollama pull llama3.1:8b
ollama pull mistral:7b    # optional, for multi-model experiments
```

## 1. Start Ollama

```bash
ollama serve
```

Leave running. You should see `Listening on 127.0.0.1:11434`.

## 2. Activate venv

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\activate
```

Set PYTHONPATH if modules aren't found:

```bash
# macOS / Linux
export PYTHONPATH="$(pwd):$(pwd)/Cerebrum"

# Windows (PowerShell)
$env:PYTHONPATH = "$(Get-Location);$(Get-Location)\Cerebrum"
```

## 3. Start the AIOS Kernel

```bash
python runtime/launch.py --llm-name ollama/llama3.1:8b --scheduler-log-mode file
```

Wait for the kernel ready message before proceeding.

## 4. Collect Data

In a separate terminal (with venv activated):

```bash
# Dataset with llama3.1:8b — 20 rounds = 1,000 rows
python scripts/run_diverse_workload.py --rounds 20 --llm-name ollama/llama3.1:8b
```

For multi-model experiments, restart the kernel with a different model and collect more:

```bash
# Restart kernel
python runtime/launch.py --llm-name ollama/mistral:7b --scheduler-log-mode file

# Collect 1,000 more rows
python scripts/run_diverse_workload.py --rounds 20 --llm-name ollama/mistral:7b
```

Log file: `aios/logs/llm_syscalls.jsonl`

### What the workload randomises

| Feature | Range |
|---|---|
| `max_tokens` | 64, 128, 256, 512, 1024, 2048 |
| `temperature` | 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0 |
| `message_count` | 2–8 (multi-turn conversations) |
| `input_char_length` | ~100–2,100 (cross-pollinated prompts) |
| `has_tools` | true/false (tool_use_agent only) |
| `model_name` | logged per request when multiple models used |

## 5. Run EDA

Open and run `notebooks/eda.ipynb` in Jupyter. It auto-resolves the log file path.

```bash
jupyter notebook notebooks/eda.ipynb
```

## 6. Train the Complexity Classifier

```bash
# Train on combined multi-model data
python scripts/train_complexity_classifier.py --log-paths llama.jsonl mistral.jsonl

# Compare only (skip tuning)
python scripts/train_complexity_classifier.py --log-paths llama.jsonl mistral.jsonl --skip-tuning
```

### Models compared

- Logistic Regression
- K-Nearest Neighbors
- SVM (RBF kernel)
- Decision Tree
- Random Forest
- Gradient Boosting

### Output artifacts (saved to `models/`)

| File | Description |
|---|---|
| `complexity_classifier.pkl` | Trained model pipeline |
| `comparison_results.csv` | All model scores |
| `model_comparison.png` | Test accuracy/F1 bar chart |
| `bias_variance_comparison.png` | Train vs test accuracy |
| `confusion_matrix.png` | Best model confusion matrix |
| `learning_curves.png` | Bias-variance diagnosis |
| `feature_importances.png` | Feature ranking (tree models) |

## 7. Run with CognitiveScheduler

After training, activate the ML-based scheduler:

```yaml
# In aios/config/config.yaml, change:
scheduler:
  log_mode: "file"
  scheduler_type: "cognitive"   # was "fifo"
```

Then restart the kernel and run the workload. The scheduler will classify each request into fast/medium/large priority queues.

## 8. Benchmark FIFO vs Cognitive

```bash
# Analyze a single log with per-class breakdown
python scripts/benchmark_schedulers.py --log llama.jsonl

# Compare two runs side-by-side
python scripts/benchmark_schedulers.py --fifo-log logs/fifo_run.jsonl --cognitive-log logs/cognitive_run.jsonl
```

## 9. Monitor Logs (live)

```bash
# macOS / Linux
tail -f aios/logs/llm_syscalls.jsonl

# Windows (PowerShell)
Get-Content aios\logs\llm_syscalls.jsonl -Wait -Tail 5
```

## Project Structure

```
scripts/
  run_diverse_workload.py          # Data collection (v2, feature-rich)
  train_complexity_classifier.py   # Model comparison + tuning
  benchmark_schedulers.py          # FIFO vs Cognitive comparison
notebooks/
  eda.ipynb                        # Exploratory data analysis
aios/
  scheduler/
    cognitive_scheduler.py         # ML-based priority queue scheduler
    fifo_scheduler.py              # Baseline FIFO scheduler
  hooks/modules/llm.py             # Syscall logging logic
  config/config.yaml               # scheduler_type: fifo | cognitive
models/
  complexity_classifier.pkl        # Trained Gradient Boosting model
```
