"""
TaskBank — randomised task pools for each benchmark agent type.

Usage:
    from cerebrum.tasks.task_bank import TaskBank
    task = TaskBank.get("short_qa_agent")
"""

import random
from typing import Dict, List


# ═══════════════════════════════════════════════════════════════════════
# SHORT QA — 20 one-line factual questions
# ═══════════════════════════════════════════════════════════════════════
_SHORT_QA: List[str] = [
    "What is the speed of light in meters per second?",
    "Who painted the Mona Lisa?",
    "What is the chemical symbol for gold?",
    "How many chromosomes do humans have?",
    "What is the largest ocean on Earth?",
    "In what year did World War II end?",
    "What is the powerhouse of the cell?",
    "Who developed the theory of general relativity?",
    "What is the boiling point of water in Celsius?",
    "Name the smallest planet in our solar system.",
    "What programming language was created by Guido van Rossum?",
    "What is the square root of 144?",
    "Which element has the atomic number 6?",
    "What is the capital of Australia?",
    "How many bits are in a byte?",
    "What organ in the human body produces insulin?",
    "Who wrote 'Pride and Prejudice'?",
    "What is the formula for the area of a circle?",
    "What gas do plants absorb during photosynthesis?",
    "In which decade was the internet invented?",
]

# ═══════════════════════════════════════════════════════════════════════
# LONG REASONING — 20 multi-step problems (4-6 sentences)
# ═══════════════════════════════════════════════════════════════════════
_LONG_REASONING: List[str] = [
    (
        "A train departs city A heading east at 90 km/h. Two hours later a second "
        "train departs city A on the same track at 120 km/h. How long after the "
        "second train departs will it overtake the first? Show the distance each "
        "train has covered at that time. Verify your answer by plugging back in."
    ),
    (
        "Alice, Bob, and Carol each have a bag of marbles. Alice has twice as many "
        "as Bob, and Carol has 5 fewer than Alice. Together they have 55 marbles. "
        "Set up a system of equations, solve for each person's count, and check "
        "that the totals are consistent."
    ),
    (
        "A factory produces widgets with a 3% defect rate. An inspector catches 90% "
        "of defective widgets but also incorrectly rejects 2% of good ones. If "
        "10,000 widgets are produced, how many are incorrectly rejected? How many "
        "defective widgets slip through? Calculate the precision and recall of the "
        "inspector. Show every step of Bayes' theorem."
    ),
    (
        "You have a 3-litre jug and a 5-litre jug, both empty, and an unlimited "
        "water supply. Describe the minimum sequence of pour, fill, and empty "
        "operations needed to measure exactly 4 litres. Prove that your sequence "
        "is optimal by listing all reachable states and transitions."
    ),
    (
        "Eight people need to cross a bridge at night with one flashlight. The "
        "bridge holds at most two people at once. Crossing times are 1, 2, 5, 8, "
        "12, 15, 20, and 25 minutes. Two people cross at the speed of the slower "
        "one. Find the minimum total crossing time. Show the schedule and prove "
        "optimality."
    ),
    (
        "A circular track is 400 metres. Runner X starts at the north point going "
        "clockwise at 5 m/s, runner Y starts at the south point going counter-"
        "clockwise at 3 m/s. When and where on the track do they first meet? "
        "Express the meeting point in degrees from north. Solve using relative "
        "velocity and verify with absolute positions."
    ),
    (
        "A company must schedule 6 tasks with precedence constraints: B depends on "
        "A, C depends on A, D depends on B and C, E depends on C, F depends on D "
        "and E. Task durations are A=3, B=4, C=2, D=5, E=3, F=2. Compute the "
        "critical path, project duration, and slack for each task. Draw the "
        "dependency DAG mentally and justify your answer."
    ),
    (
        "An investor puts $10,000 into account X earning 5% compounded annually "
        "and $10,000 into account Y earning 4.8% compounded monthly. After 10 "
        "years, which account has more money and by how much? Show the compound "
        "interest formula for each, compute exact values, and explain why "
        "compounding frequency matters."
    ),
    (
        "A 5x5 grid has 25 cells. You start at the top-left corner and must reach "
        "the bottom-right corner moving only right or down. How many distinct paths "
        "exist? Derive the answer using combinations and verify with Pascal's "
        "triangle. Then calculate how many paths pass through the centre cell (3,3)."
    ),
    (
        "There are 100 doors in a row, all initially closed. In round k (k=1..100) "
        "you toggle every k-th door. After all 100 rounds, which doors are open? "
        "Prove your answer using divisor counting and the properties of perfect "
        "squares. List all open door numbers explicitly."
    ),
    (
        "A recursive function is defined as f(1)=1, f(2)=1, f(n)=f(n-1)+2*f(n-2) "
        "for n>2. Compute f(8) step-by-step showing each intermediate value. Then "
        "derive a closed-form solution using the characteristic equation of the "
        "recurrence. Verify the closed form matches your computed f(8)."
    ),
    (
        "A box contains 4 red, 6 blue, and 5 green balls. You draw 3 balls without "
        "replacement. What is the probability of getting exactly 2 blue balls? "
        "Solve using the hypergeometric distribution. Then calculate the expected "
        "number of blue balls in the sample and its variance."
    ),
    (
        "A knight is placed on cell a1 of a standard 8x8 chessboard. Can it visit "
        "every cell exactly once (a knight's tour)? Describe Warnsdorff's heuristic, "
        "trace the first 10 moves from a1 using that heuristic, and explain why the "
        "heuristic works on most boards but is not guaranteed optimal."
    ),
    (
        "You must tile a 2×n board using 1×2 dominoes. Derive a recurrence relation "
        "for the number of tilings T(n). Show that T(n) equals the n-th Fibonacci "
        "number. Compute T(10) explicitly. Then explain how this relates to the "
        "golden ratio."
    ),
    (
        "Three friends split a dinner bill using the following rule: the oldest pays "
        "50% of what remains after the youngest pays, and the middle one pays the "
        "rest. The bill is $120. The youngest pays $30. Compute the exact amounts "
        "paid by each. Then generalise: if the youngest pays x dollars, express each "
        "person's share as a function of x and the total bill B."
    ),
    (
        "A DNA sequence is ATGCGATCGATCG. Find all palindromic subsequences of "
        "length 4 or more (where the complement also reads the same in reverse). "
        "Explain the biological significance of palindromic sequences in restriction "
        "enzyme recognition. List each palindrome with its position indices."
    ),
    (
        "A message is encoded using a Caesar cipher with an unknown shift. The "
        "ciphertext is 'KHOOR ZRUOG'. Determine the plaintext by frequency analysis "
        "and brute-force checking all 25 possible shifts. State which shift decodes "
        "it and explain why frequency analysis is more efficient for longer texts."
    ),
    (
        "A graph has vertices {A,B,C,D,E,F} and edges AB(2), AC(4), BD(7), BE(3), "
        "CD(1), CF(5), DE(2), EF(6). Find the shortest path from A to F using "
        "Dijkstra's algorithm. Show the priority queue state after each relaxation "
        "step. Then find the minimum spanning tree using Kruskal's algorithm."
    ),
    (
        "A fair six-sided die is rolled repeatedly until a 6 appears. Let X be the "
        "number of rolls needed. Derive the probability mass function, expected "
        "value, and variance of X from first principles using the geometric "
        "distribution. Then compute P(X > 10) and explain the memoryless property."
    ),
    (
        "A web server handles requests with service times following an exponential "
        "distribution with mean 50 ms. Requests arrive as a Poisson process at 15 "
        "per second. Model this as an M/M/1 queue. Compute the expected queue "
        "length, expected wait time, and server utilisation. Explain what happens "
        "as the arrival rate approaches the service rate."
    ),
]

# ═══════════════════════════════════════════════════════════════════════
# TOOL USE — 20 tasks requiring search + calculation
# ═══════════════════════════════════════════════════════════════════════
_TOOL_USE: List[str] = [
    "Search for the GDP of the top 5 economies in 2024 and calculate their combined share of world GDP.",
    "Find the distance from Earth to Mars at closest approach and calculate how long a radio signal takes to travel that distance.",
    "Search for the current price of Bitcoin and Ethereum. Calculate the ratio and how much 0.5 BTC is worth in ETH.",
    "Look up the population and area of the 5 largest US states. Calculate the population density of each and rank them.",
    "Search for the tallest 5 buildings in the world. Convert all heights from metres to feet and calculate the mean height.",
    "Find today's exchange rates for USD to EUR, GBP, JPY, and CHF. Calculate how much €500 and £300 combined are in USD.",
    "Search for the nutritional content of a Big Mac. Calculate what percentage of a 2000-calorie daily diet it represents for calories, fat, and sodium.",
    "Look up the orbital periods of all 8 planets. Calculate the ratio of each planet's period to Earth's and verify Kepler's third law using their semi-major axes.",
    "Search for the top 10 programming languages by popularity in 2025. Calculate the Herfindahl-Hirschman Index of market concentration from their percentage shares.",
    "Find the average annual rainfall in London, Tokyo, and Mumbai. Convert all to inches, calculate the total, and determine which city gets the most rain per day on average.",
    "Search for the speed records for the fastest car, train, plane, and rocket. Calculate how long each would take to travel from New York to Los Angeles (3,944 km).",
    "Look up the current US federal tax brackets for a single filer. Calculate the total tax owed on an income of $185,000 showing the tax at each bracket.",
    "Search for the specifications of the latest M-series Apple chip. Calculate the performance-per-watt improvement over the previous generation as a percentage.",
    "Find the CO2 emissions per km for a Tesla Model 3, a Toyota Camry, and a Boeing 737. Calculate the total emissions for a 500 km trip by each and the percentage savings of the Tesla vs the Camry.",
    "Search for the interest rates of 5 major central banks. Calculate the spread between the highest and lowest, and the geometric mean of all rates.",
    "Look up the marathon world records for men and women. Calculate the average speed in km/h and mph for each, and the percentage gap between them.",
    "Search for the market capitalisation of Apple, Microsoft, Google, Amazon, and Nvidia. Calculate each company's percentage of the total and the combined value.",
    "Find the literacy rates and GDP per capita of 5 BRICS nations. Calculate the Pearson correlation coefficient between literacy and GDP per capita.",
    "Search for the energy density (Wh/kg) of lithium-ion, solid-state, and hydrogen fuel cells. Calculate how much each weighs to store 100 kWh.",
    "Look up the bandwidth and latency of 5G, WiFi 7, and Starlink. Calculate the theoretical time to download a 50 GB file on each and the latency-bandwidth product.",
]

# ═══════════════════════════════════════════════════════════════════════
# CODE GEN — 20 coding problems with detailed spec paragraphs
# ═══════════════════════════════════════════════════════════════════════
_CODE_GEN: List[str] = [
    (
        "Write a Python function `merge_intervals(intervals: list[tuple[int,int]]) -> "
        "list[tuple[int,int]]` that takes a list of possibly overlapping (start, end) "
        "intervals and returns a sorted list of merged, non-overlapping intervals. Handle "
        "edge cases: empty input, single interval, fully nested intervals, and touching "
        "endpoints like (1,3) and (3,5). Include type hints, a docstring with examples, "
        "and at least 5 unit tests using pytest."
    ),
    (
        "Write a SQL query for a PostgreSQL database with tables `orders(id, customer_id, "
        "total_amount, created_at)` and `customers(id, name, country, created_at)`. The "
        "query should find the top 5 customers by total lifetime spend, but only include "
        "customers who have made at least 3 orders in the last 12 months. Return columns: "
        "customer name, country, order count, total spend, and average order value. Use "
        "CTEs for readability."
    ),
    (
        "Write a bash script that monitors a given directory for new `.csv` files. When a "
        "new file appears, validate that it has a header row, at least 10 data rows, and "
        "no empty cells in the first 3 columns. If valid, move it to a `processed/` "
        "subdirectory; if invalid, move it to `rejected/` and log the reason to "
        "`monitor.log` with a timestamp. Use inotifywait or a polling loop with 5-second "
        "intervals. Include proper error handling and a SIGTERM trap for clean shutdown."
    ),
    (
        "Write a Python class `RateLimiter` that implements a sliding-window rate limiter. "
        "The constructor takes `max_requests: int` and `window_seconds: float`. The method "
        "`allow(client_id: str) -> bool` returns True if the client is under the limit, "
        "False otherwise. Use `collections.deque` for per-client timestamp tracking. The "
        "implementation must be thread-safe using `threading.Lock`. Include a `reset(client_id)` "
        "method and a `get_remaining(client_id) -> int` method. Write tests demonstrating "
        "rate limiting across multiple clients."
    ),
    (
        "Write a SQL migration (up and down) that adds a `tags` system to a blog platform. "
        "The existing table is `posts(id SERIAL, title TEXT, body TEXT, author_id INT, "
        "created_at TIMESTAMP)`. Create a `tags(id SERIAL, name TEXT UNIQUE, slug TEXT UNIQUE)` "
        "table and a junction table `post_tags(post_id INT, tag_id INT, PRIMARY KEY(post_id, "
        "tag_id))`. Add indexes for common queries: finding posts by tag, finding tags by "
        "post, and full-text search on tag names. The down migration must drop everything in "
        "reverse order."
    ),
    (
        "Write a Python async function `fetch_all(urls: list[str], max_concurrent: int = 10) "
        "-> list[dict]` using aiohttp. It should fetch all URLs concurrently with a semaphore "
        "limiting concurrency. For each URL, return a dict with keys: url, status_code, "
        "content_length, elapsed_ms, and error (None if successful). Implement retry logic: "
        "up to 3 attempts with exponential backoff (1s, 2s, 4s) for 5xx errors and timeouts. "
        "Log each retry. Include a 30-second per-request timeout."
    ),
    (
        "Write a bash one-liner pipeline that processes a 1GB Apache access log file to find "
        "the top 20 IP addresses by request count, excluding requests to `/health` and "
        "`/favicon.ico`. Then expand it into a full script that also computes: requests per "
        "hour as a histogram, the 95th percentile response time, and a breakdown of HTTP "
        "status codes. Output results as a formatted text report. Use only standard Unix "
        "tools: awk, sort, uniq, head, tail, grep, sed."
    ),
    (
        "Write a Python implementation of Dijkstra's shortest path algorithm using a min-heap. "
        "The function signature should be `dijkstra(graph: dict[str, list[tuple[str, float]]], "
        "source: str) -> tuple[dict[str, float], dict[str, str | None]]` returning both "
        "distances and predecessors. Support reconstructing the actual path via a "
        "`reconstruct_path(predecessors, target)` function. Handle disconnected graphs by "
        "returning float('inf'). Include type hints and tests with at least 3 different graph "
        "topologies."
    ),
    (
        "Write a SQL analytics query for an e-commerce database with tables: `orders(id, "
        "customer_id, total, created_at)`, `order_items(id, order_id, product_id, quantity, "
        "price)`, `products(id, name, category)`. Compute a cohort retention analysis: for "
        "each monthly cohort (month of first purchase), calculate the percentage of customers "
        "who made a repeat purchase in months 1, 2, 3, 6, and 12 after their first purchase. "
        "Use window functions and CTEs. Format output as a pivot-style table."
    ),
    (
        "Write a Python context manager class `TempDatabase` that creates a temporary SQLite "
        "database with a given schema on enter and cleans it up on exit. The constructor takes "
        "a list of CREATE TABLE statements. Provide methods: `execute(sql, params)`, "
        "`query(sql, params) -> list[dict]`, and `bulk_insert(table, rows)`. All methods "
        "should handle transactions properly with automatic rollback on error. Include type "
        "hints and tests that create a users table, insert 100 rows, and query with filtering."
    ),
    (
        "Write a bash script that sets up a Python virtual environment, installs dependencies "
        "from `requirements.txt`, runs `pytest` with coverage, and fails the script (exit 1) "
        "if coverage is below 80%. Parse the coverage percentage from pytest-cov output using "
        "grep/awk. Support an optional `--skip-install` flag. Log all output to both stdout "
        "and a timestamped log file in a `ci_logs/` directory. Use `set -euo pipefail`."
    ),
    (
        "Write a Python function `parse_cron(expression: str) -> list[datetime]` that takes "
        "a standard 5-field cron expression (minute, hour, day-of-month, month, day-of-week) "
        "and returns the next 10 execution times from `datetime.now()`. Support `*`, ranges "
        "(1-5), lists (1,3,5), and step values (*/15). Raise `ValueError` for invalid "
        "expressions with descriptive messages. Include type hints and tests covering every "
        "field type plus edge cases like Feb 29 and month boundaries."
    ),
    (
        "Write a SQL query that detects fraudulent transactions in a `transactions(id, "
        "account_id, amount, merchant, timestamp, location)` table. A transaction is "
        "suspicious if: (a) the amount exceeds 3x the account's 30-day rolling average, "
        "or (b) two transactions from the same account occur in different cities within 1 "
        "hour, or (c) more than 5 transactions happen within 10 minutes from the same "
        "account. Use window functions with PARTITION BY and frame clauses. Return the "
        "transaction ID, rule violated, and a risk score."
    ),
    (
        "Write a Python generator function `stream_json(filepath: str, batch_size: int = 100) "
        "-> Generator[list[dict], None, None]` that reads a large JSON-Lines file without "
        "loading it entirely into memory. Yield batches of parsed dicts. Handle malformed "
        "lines by logging a warning and skipping them. Support gzip-compressed files "
        "automatically by detecting the `.gz` extension. Include a progress callback "
        "parameter. Write tests with a 10,000-line temp file."
    ),
    (
        "Write a bash script that performs zero-downtime deployment of a Docker container. "
        "Steps: pull the new image, start a new container on a different port, health-check "
        "the new container (retry 5 times with 3s intervals), update an nginx upstream config "
        "to point to the new container, reload nginx, wait 10s for connections to drain, then "
        "stop the old container. Accept image name and tag as arguments. Log each step with "
        "timestamps. Roll back automatically if the health check fails."
    ),
    (
        "Write a Python class `BloomFilter` implementing a probabilistic set membership test. "
        "The constructor takes `expected_items: int` and `false_positive_rate: float` and "
        "automatically calculates the optimal bit array size and number of hash functions. "
        "Use mmh3 (murmurhash3) for hashing. Implement `add(item: str)`, `contains(item: str) "
        "-> bool`, `__len__` (approximate count), and `union(other: BloomFilter)`. Include "
        "type hints and tests verifying the false positive rate empirically with 10,000 items."
    ),
    (
        "Write a SQL stored procedure `generate_report(start_date DATE, end_date DATE)` for a "
        "SaaS billing database with tables `subscriptions(id, customer_id, plan, mrr, "
        "started_at, cancelled_at)` and `invoices(id, subscription_id, amount, paid_at, "
        "status)`. The procedure should compute: total MRR, net new MRR (new - churned), "
        "churn rate, average revenue per user, and invoice collection rate. Return results "
        "as a single-row result set. Handle NULL cancelled_at as active subscription."
    ),
    (
        "Write a Python async WebSocket server using the `websockets` library that implements "
        "a chat room. Features: clients send JSON messages with `type` ('join', 'message', "
        "'leave'), the server broadcasts messages to all connected clients except the sender, "
        "maintains a list of connected usernames, and sends a 'user_list' event on join/leave. "
        "Implement a heartbeat ping every 30 seconds and disconnect idle clients after 2 "
        "minutes. Include graceful shutdown handling and tests using pytest-asyncio."
    ),
    (
        "Write a bash script that creates an encrypted backup of a PostgreSQL database. Use "
        "`pg_dump` with custom format, compress with `zstd` at level 10, encrypt with `gpg` "
        "using a symmetric passphrase from an environment variable, and upload to an S3 bucket "
        "using the AWS CLI. Implement rotation: keep the last 7 daily backups, 4 weekly (Sunday), "
        "and 12 monthly (1st). Verify the backup by downloading and decrypting a test restore. "
        "Send a Slack webhook notification on success or failure."
    ),
    (
        "Write a Python dataclass `Config` with a `@classmethod from_env()` that reads "
        "configuration from environment variables with type coercion, defaults, and validation. "
        "Support types: str, int, float, bool, list[str] (comma-separated), and Optional "
        "variants. Raise a `ConfigError` listing ALL missing/invalid variables at once, not "
        "just the first. Include a `to_dict()` method that redacts fields marked with "
        "`metadata={'secret': True}`. Write tests covering all types and error cases."
    ),
]

# ═══════════════════════════════════════════════════════════════════════
# SUMMARIZER — 20 long-form texts (3-5 paragraphs each)
# ═══════════════════════════════════════════════════════════════════════
_SUMMARIZER: List[str] = [
    # 0 — Quantum computing
    (
        "Summarize the following:\n\n"
        "Quantum computing leverages the principles of quantum mechanics—superposition, "
        "entanglement, and interference—to perform computations that are intractable for "
        "classical computers. A quantum bit (qubit) can exist in a superposition of 0 and 1, "
        "allowing a system of n qubits to represent 2^n states simultaneously. This exponential "
        "state space is the source of quantum speedups for specific problems.\n\n"
        "Shor's algorithm, published in 1994, demonstrated that a sufficiently large quantum "
        "computer could factor large integers in polynomial time, threatening RSA encryption. "
        "Grover's algorithm provides a quadratic speedup for unstructured search. More recently, "
        "variational quantum eigensolvers (VQEs) have shown promise for simulating molecular "
        "systems in chemistry and materials science, potentially accelerating drug discovery.\n\n"
        "Current quantum hardware faces significant challenges. Qubits are extremely fragile—"
        "thermal noise, electromagnetic interference, and cosmic rays can cause decoherence. "
        "Error rates of current superconducting qubits hover around 0.1-1%, requiring quantum "
        "error correction codes that add massive overhead: estimates suggest 1,000-10,000 "
        "physical qubits per logical qubit. IBM, Google, and startups like IonQ and Rigetti are "
        "competing to build fault-tolerant systems, with practical quantum advantage expected "
        "in the 2030s for specific applications."
    ),
    # 1 — CRISPR
    (
        "Summarize the following:\n\n"
        "CRISPR-Cas9 has revolutionized genetic engineering by providing a precise, programmable, "
        "and relatively inexpensive tool for editing DNA sequences. The system uses a guide RNA "
        "to direct the Cas9 nuclease to a specific genomic location, where it creates a double-"
        "strand break. The cell's repair machinery then introduces the desired changes through "
        "either non-homologous end joining (NHEJ) for gene knockouts or homology-directed repair "
        "(HDR) for precise edits.\n\n"
        "Clinical applications have expanded rapidly since the first human trials in 2019. Vertex "
        "Pharmaceuticals and CRISPR Therapeutics received FDA approval for Casgevy, a treatment "
        "for sickle cell disease and beta-thalassemia, marking the first approved CRISPR therapy. "
        "Ongoing trials target cancers through engineered CAR-T cells, inherited blindness via "
        "subretinal injection, and HIV by disrupting the CCR5 receptor that the virus uses for "
        "cell entry.\n\n"
        "Ethical concerns remain significant. Germline editing—modifying embryos so changes are "
        "heritable—was thrust into the spotlight when He Jiankui created the first gene-edited "
        "babies in 2018, drawing worldwide condemnation. The scientific community has called for "
        "a moratorium on clinical germline editing until safety, efficacy, and ethical frameworks "
        "are established. Off-target effects, mosaicism, and equitable access to gene therapies "
        "that can cost over $2 million per treatment are ongoing challenges."
    ),
    # 2 — Blockchain beyond crypto
    (
        "Summarize the following:\n\n"
        "Blockchain technology, originally designed as the ledger for Bitcoin, has found "
        "applications far beyond cryptocurrency. At its core, a blockchain is a distributed, "
        "append-only data structure where each block contains a cryptographic hash of the "
        "previous block, creating an immutable chain. This architecture provides transparency, "
        "tamper-resistance, and eliminates the need for a trusted central authority.\n\n"
        "Supply chain management has emerged as a compelling use case. Companies like Walmart "
        "and Maersk use blockchain to track products from origin to shelf, reducing food safety "
        "investigation times from days to seconds. In healthcare, blockchain-based systems "
        "enable patients to control access to their medical records across providers. Estonia "
        "has implemented a blockchain-based e-governance system covering voting, healthcare, "
        "and legal records for its 1.3 million citizens.\n\n"
        "Smart contracts on platforms like Ethereum enable programmable agreements that execute "
        "automatically when conditions are met. Decentralized finance (DeFi) protocols have "
        "created lending, borrowing, and trading systems without traditional intermediaries, "
        "managing over $50 billion in assets. However, smart contract vulnerabilities have led "
        "to significant losses—the DAO hack in 2016 resulted in $60 million stolen due to a "
        "reentrancy bug.\n\n"
        "Scalability remains the primary technical challenge. Bitcoin processes 7 transactions "
        "per second compared to Visa's 65,000. Layer-2 solutions like the Lightning Network and "
        "rollups on Ethereum aim to increase throughput while maintaining security. The "
        "environmental impact of proof-of-work consensus has led to alternatives like proof-of-"
        "stake, which Ethereum adopted in 2022, reducing its energy consumption by 99.95%."
    ),
    # 3 — Autonomous vehicles
    (
        "Summarize the following:\n\n"
        "The development of autonomous vehicles has progressed through the SAE's six levels of "
        "driving automation, from Level 0 (no automation) to Level 5 (full automation in all "
        "conditions). As of 2025, most commercial deployments operate at Level 4—fully autonomous "
        "within a defined operational design domain. Waymo operates robotaxi services in Phoenix, "
        "San Francisco, and Los Angeles, while Cruise, Zoox, and Chinese companies like Baidu's "
        "Apollo are expanding in other markets.\n\n"
        "The technology stack combines multiple sensor modalities: LiDAR provides precise 3D "
        "point clouds for obstacle detection, cameras enable traffic sign and signal recognition, "
        "radar handles velocity estimation in adverse weather, and ultrasonic sensors manage "
        "close-range parking scenarios. Deep neural networks fuse these inputs to create a "
        "unified world model, while planning algorithms generate safe trajectories considering "
        "traffic rules, pedestrian behaviour, and vehicle dynamics.\n\n"
        "Safety validation remains the greatest challenge. An autonomous vehicle must be at least "
        "as safe as a human driver, who averages one fatal accident per 100 million miles driven "
        "in the US. Proving this statistically requires billions of miles of testing—far more "
        "than any company has accumulated. Simulation environments, formal verification methods, "
        "and structured edge case testing are used to supplement real-world driving. Regulatory "
        "frameworks vary widely across jurisdictions, creating a patchwork of rules that "
        "companies must navigate."
    ),
    # 4 — Microbiome
    (
        "Summarize the following:\n\n"
        "The human microbiome comprises trillions of bacteria, viruses, fungi, and archaea that "
        "inhabit the gut, skin, mouth, and other body sites. The gut microbiome alone contains "
        "approximately 1,000 species and 3 million unique genes—150 times more than the human "
        "genome. Advances in metagenomic sequencing have enabled researchers to characterize "
        "these microbial communities in unprecedented detail, revealing their profound influence "
        "on human health.\n\n"
        "The gut-brain axis has emerged as a particularly active area of research. Gut bacteria "
        "produce neurotransmitters including serotonin, dopamine, and GABA, and communicate with "
        "the brain via the vagus nerve. Studies in germ-free mice show altered anxiety and "
        "stress responses that can be reversed by colonization with specific bacterial strains. "
        "Clinical trials are investigating fecal microbiota transplantation (FMT) for depression, "
        "autism spectrum disorder, and Parkinson's disease.\n\n"
        "Diet is the strongest modulator of microbiome composition. A diverse, fibre-rich diet "
        "promotes microbial diversity and the production of short-chain fatty acids like butyrate, "
        "which strengthen the intestinal barrier and reduce inflammation. In contrast, the Western "
        "diet—high in processed foods, sugar, and saturated fat—is associated with reduced "
        "diversity and increased prevalence of inflammatory conditions including IBD, obesity, "
        "and type 2 diabetes. Probiotic and prebiotic interventions aim to restore healthy "
        "microbiome composition, though evidence for specific strains remains mixed."
    ),
    # 5 — History of the internet
    (
        "Summarize the following:\n\n"
        "The internet originated from ARPANET, a US Department of Defense project launched in "
        "1969 to create a resilient communication network that could survive partial destruction. "
        "The adoption of TCP/IP in 1983 established the protocol foundation that still underpins "
        "the modern internet. Tim Berners-Lee's invention of the World Wide Web in 1989—combining "
        "HTML, URLs, and HTTP—transformed the internet from a specialist tool into a global "
        "information system accessible to non-technical users.\n\n"
        "The commercialization of the web in the mid-1990s triggered explosive growth. Netscape's "
        "IPO in 1995 ignited the dot-com bubble, during which internet companies attracted "
        "massive investment based on user growth rather than profitability. The bubble burst in "
        "2000, wiping out trillions in market capitalization, but the surviving companies—Amazon, "
        "eBay, Google—went on to reshape global commerce and information access.\n\n"
        "Web 2.0, characterized by user-generated content and social platforms, emerged in the "
        "mid-2000s. Facebook, YouTube, Twitter, and Wikipedia democratized content creation but "
        "also introduced challenges around misinformation, privacy, and market concentration. "
        "Today, the internet connects over 5 billion users and generates approximately 330 "
        "exabytes of data daily. Debates about net neutrality, content moderation, and digital "
        "sovereignty continue to shape internet governance worldwide."
    ),
    # 6 — Ocean acidification
    (
        "Summarize the following:\n\n"
        "Ocean acidification, often called 'the other CO2 problem,' occurs when seawater absorbs "
        "atmospheric carbon dioxide, forming carbonic acid and lowering pH. Since the industrial "
        "revolution, ocean pH has dropped from 8.2 to 8.1—a 26% increase in acidity on the "
        "logarithmic pH scale. Current absorption rates exceed 22 million tonnes of CO2 per day, "
        "and projections suggest a further 0.3-0.4 pH decline by 2100 under high-emission "
        "scenarios.\n\n"
        "The ecological consequences are severe. Calcifying organisms—corals, mollusks, sea "
        "urchins, and certain plankton—struggle to build and maintain calcium carbonate shells and "
        "skeletons in more acidic water. Pteropods, tiny sea snails that form the base of polar "
        "food webs, show shell dissolution at pH levels projected for 2050. Coral reef ecosystems, "
        "which support 25% of all marine species despite covering less than 1% of the ocean floor, "
        "face a double threat from acidification and thermal bleaching.\n\n"
        "Economic impacts ripple through fisheries and aquaculture. The global shellfish industry, "
        "valued at over $30 billion annually, is directly threatened as oyster and mussel larvae "
        "fail to form shells in acidified hatchery waters—a problem already affecting Pacific "
        "Northwest oyster farms. Coral reef degradation threatens the livelihoods of 500 million "
        "people who depend on reef fisheries and tourism. Mitigation requires rapid reduction of "
        "CO2 emissions, but ocean chemistry changes will persist for centuries even after "
        "atmospheric CO2 stabilizes."
    ),
    # 7 — Antibiotic resistance
    (
        "Summarize the following:\n\n"
        "Antimicrobial resistance (AMR) is one of the greatest threats to global public health. "
        "The WHO estimates that drug-resistant infections directly caused 1.27 million deaths in "
        "2019 and were associated with 4.95 million deaths. Without action, AMR could cause 10 "
        "million annual deaths by 2050, surpassing cancer. The problem is driven by overuse and "
        "misuse of antibiotics in human medicine and agriculture, where 73% of medically important "
        "antimicrobials are used in livestock.\n\n"
        "The pipeline for new antibiotics is alarmingly thin. Only 12 new antibiotics were "
        "approved between 2017 and 2021, and most are modifications of existing classes rather "
        "than novel mechanisms. The economics of antibiotic development are fundamentally broken: "
        "successful new antibiotics are reserved for resistant infections, limiting sales volumes, "
        "while development costs exceed $1 billion. Several companies developing antibiotics have "
        "gone bankrupt despite regulatory approval.\n\n"
        "Novel approaches to combat AMR include phage therapy—using bacteriophages (viruses that "
        "infect bacteria) to target specific pathogens—which has shown success in compassionate "
        "use cases against multidrug-resistant infections. CRISPR-based antimicrobials can "
        "selectively destroy resistance genes. Rapid diagnostic tools like MALDI-TOF mass "
        "spectrometry identify pathogens in hours rather than days, enabling targeted antibiotic "
        "prescribing and reducing broad-spectrum use. Policy reforms include delinking antibiotic "
        "revenue from sales volume through subscription models like the UK's five-year pilot."
    ),
    # 8 — Urban planning
    (
        "Summarize the following:\n\n"
        "The concept of the 15-minute city, championed by Carlos Moreno and adopted as policy "
        "by Paris, Melbourne, and other cities, reimagines urban planning around proximity. The "
        "goal is for every resident to access essential services—work, shopping, healthcare, "
        "education, and recreation—within a 15-minute walk or bike ride. This requires mixed-use "
        "zoning, decentralised public facilities, and investment in cycling and pedestrian "
        "infrastructure.\n\n"
        "Traffic reduction is a primary benefit. Private cars occupy 150-250 square feet of road "
        "space per person, compared to 5 square feet for a pedestrian and 10 for a cyclist. "
        "Barcelona's 'superblocks' program, which restricts through-traffic in 3x3 block areas, "
        "has reduced air pollution by 25%, noise by 5 dB, and increased retail activity by 30% "
        "in affected areas. Paris has removed 60,000 parking spaces since 2020, replacing them "
        "with bike lanes, parklets, and terraces.\n\n"
        "Critics argue the model works for dense European cities but is difficult to implement "
        "in sprawling North American suburbs built around car dependency. Equity concerns also "
        "arise: improving walkability tends to increase property values and rents, potentially "
        "displacing the lower-income residents who would benefit most. Successful implementation "
        "requires affordable housing protections alongside infrastructure changes.\n\n"
        "Digital twins—virtual replicas of cities using real-time sensor data—are enabling "
        "planners to simulate the impact of proposed changes before breaking ground. Singapore's "
        "Virtual Singapore platform models traffic flow, pedestrian movement, solar exposure, "
        "and wind patterns at building-level resolution, allowing evidence-based design decisions "
        "that balance competing priorities."
    ),
    # 9 — Sleep science
    (
        "Summarize the following:\n\n"
        "Sleep is regulated by two complementary systems: the circadian clock, a 24-hour "
        "internal rhythm driven by the suprachiasmatic nucleus in the hypothalamus, and sleep "
        "homeostasis, which builds sleep pressure through the accumulation of adenosine during "
        "waking hours. Caffeine works by blocking adenosine receptors, temporarily masking sleep "
        "pressure without eliminating it—leading to a 'sleep debt' when the caffeine wears off.\n\n"
        "The architecture of sleep consists of 4-6 cycles per night, each lasting approximately "
        "90 minutes. Each cycle progresses through three stages of non-REM sleep (N1, N2, N3) "
        "followed by REM sleep. N3 (slow-wave sleep) predominates in early cycles and is "
        "critical for physical restoration and immune function, while REM sleep increases in "
        "later cycles and plays a key role in memory consolidation, emotional regulation, and "
        "creative problem-solving.\n\n"
        "Chronic sleep deprivation—defined as consistently getting fewer than 7 hours per night "
        "for adults—has devastating health consequences. It increases the risk of cardiovascular "
        "disease by 48%, type 2 diabetes by 37%, and all-cause mortality by 12%. Cognitive "
        "impairment from 24 hours of sleep deprivation is equivalent to a blood alcohol content "
        "of 0.10%, above the legal driving limit. Despite this, one-third of adults in developed "
        "countries regularly sleep fewer than the recommended 7-9 hours."
    ),
    # 10 — Renewable hydrogen
    (
        "Summarize the following:\n\n"
        "Hydrogen is increasingly viewed as an essential component of the clean energy transition, "
        "particularly for sectors that are difficult to electrify directly: heavy industry "
        "(steel, cement, chemicals), long-haul shipping, aviation, and seasonal energy storage. "
        "The colour taxonomy distinguishes production methods: grey hydrogen from natural gas "
        "reforming (most common, with CO2 emissions), blue hydrogen with carbon capture, and "
        "green hydrogen from water electrolysis powered by renewables.\n\n"
        "Green hydrogen costs have declined from over $10/kg in 2010 to approximately $4-6/kg "
        "today, but remain well above grey hydrogen at $1-2/kg. Achieving cost parity requires "
        "scaling electrolyser manufacturing (currently ~1 GW/year globally, compared to a target "
        "of 100+ GW/year by 2030), reducing renewable electricity costs, and improving "
        "electrolyser efficiency from the current 60-70% to over 80%.\n\n"
        "Infrastructure is a major bottleneck. Hydrogen is difficult to store and transport due "
        "to its low volumetric energy density—it must be compressed to 700 bar, liquefied at "
        "-253°C, or converted to ammonia for long-distance shipping. Existing natural gas "
        "pipelines can handle 5-20% hydrogen blending, but dedicated hydrogen pipelines are "
        "needed for higher concentrations. The EU, Japan, South Korea, and Australia have all "
        "published national hydrogen strategies with combined investment commitments exceeding "
        "$100 billion through 2030."
    ),
    # 11 — Machine learning fairness
    (
        "Summarize the following:\n\n"
        "Algorithmic fairness has become a central concern as machine learning models are "
        "increasingly used for high-stakes decisions in hiring, criminal justice, lending, and "
        "healthcare. The fundamental challenge is that 'fairness' has multiple incompatible "
        "mathematical definitions. Demographic parity requires equal positive prediction rates "
        "across groups; equalized odds requires equal true positive and false positive rates; "
        "calibration requires that predicted probabilities match actual outcomes within each "
        "group. A landmark impossibility theorem proves that except in trivial cases, no "
        "classifier can simultaneously satisfy all three.\n\n"
        "The COMPAS recidivism prediction tool highlighted these tensions. ProPublica found that "
        "the tool was twice as likely to falsely label Black defendants as high-risk compared to "
        "white defendants (violating equalized odds). However, Northpointe countered that the "
        "tool was equally calibrated across racial groups—meaning a score of 7 had the same "
        "recidivism probability regardless of race. Both claims were true simultaneously, "
        "illustrating the impossibility of satisfying all fairness criteria.\n\n"
        "Mitigation strategies operate at three stages: pre-processing (rebalancing or "
        "transforming training data), in-processing (adding fairness constraints to the "
        "optimisation objective), and post-processing (adjusting predictions to satisfy fairness "
        "criteria). Each approach involves tradeoffs between accuracy, fairness, and "
        "interpretability. Increasingly, researchers advocate for participatory approaches that "
        "include affected communities in defining which fairness criteria should be prioritized "
        "for a given application."
    ),
    # 12 — Plate tectonics
    (
        "Summarize the following:\n\n"
        "Plate tectonics is the unifying theory of Earth science, explaining earthquakes, "
        "volcanic activity, mountain building, and continental drift through the movement of "
        "rigid lithospheric plates atop the convecting asthenosphere. Earth's surface is divided "
        "into approximately 15 major plates and several minor ones, ranging in size from the "
        "Pacific Plate (103 million km²) to the Juan de Fuca Plate (250,000 km²).\n\n"
        "Three types of plate boundaries produce distinct geological phenomena. Divergent "
        "boundaries, like the Mid-Atlantic Ridge, create new oceanic crust as magma rises to fill "
        "the gap between separating plates, at rates of 2-15 cm per year. Convergent boundaries "
        "produce subduction zones (where oceanic crust dives beneath continental crust, creating "
        "volcanic arcs like the Andes) or continental collision zones (building mountain ranges "
        "like the Himalayas). Transform boundaries, such as the San Andreas Fault, produce "
        "lateral sliding and powerful earthquakes.\n\n"
        "Modern GPS measurements have confirmed plate motion rates predicted by geological "
        "evidence. The theory elegantly explains the distribution of fossils—Glossopteris fern "
        "fossils on five continents indicate they were once connected as Pangaea. Looking forward, "
        "plate motion will eventually close the Atlantic Ocean in approximately 200-300 million "
        "years, forming a new supercontinent that geologists have named Pangaea Proxima."
    ),
    # 13 — Behavioural economics
    (
        "Summarize the following:\n\n"
        "Behavioural economics challenges the neoclassical assumption that economic agents are "
        "rational utility maximizers. Daniel Kahneman and Amos Tversky's prospect theory, "
        "published in 1979, demonstrated that people evaluate outcomes relative to a reference "
        "point and are loss-averse—the pain of losing $100 is approximately 2.5 times the "
        "pleasure of gaining $100. This asymmetry leads to predictable 'irrational' behaviour "
        "such as the endowment effect, where people value items they own more than identical "
        "items they don't.\n\n"
        "Cognitive biases systematically distort decision-making. The availability heuristic causes "
        "people to overestimate the probability of events that are easily recalled (plane crashes "
        "vs. car accidents). Anchoring biases initial estimates toward irrelevant numbers—judges "
        "who rolled a high number on a die before sentencing gave longer sentences. Confirmation "
        "bias leads people to seek and interpret evidence that supports existing beliefs while "
        "discounting contradictory information.\n\n"
        "Nudge theory, developed by Richard Thaler and Cass Sunstein, applies behavioural "
        "insights to policy design. By changing the 'choice architecture'—the context in which "
        "decisions are made—policymakers can steer behaviour without restricting options. Default "
        "opt-in for organ donation increases participation from 15% to over 90%. Automatic "
        "enrollment in retirement savings plans nearly doubles participation rates. The UK's "
        "Behavioural Insights Team and similar units worldwide have applied nudges to tax "
        "compliance, energy conservation, and public health with measurable impact."
    ),
    # 14 — Fusion energy
    (
        "Summarize the following:\n\n"
        "Nuclear fusion—the process that powers the Sun—promises nearly limitless, clean energy "
        "by fusing light atomic nuclei (typically deuterium and tritium) into helium, releasing "
        "enormous energy per unit mass. One kilogram of fusion fuel produces the energy equivalent "
        "of 10 million kilograms of fossil fuel. Unlike fission, fusion produces no long-lived "
        "radioactive waste and cannot cause a meltdown—the reaction stops if conditions deviate "
        "from the extreme temperatures and pressures required.\n\n"
        "ITER, the internationally funded tokamak under construction in France, aims to achieve "
        "Q=10 (producing 10 times more energy than used to heat the plasma) by the mid-2030s. "
        "The project has cost over $25 billion and involves contributions from 35 nations. In "
        "December 2022, the US National Ignition Facility achieved scientific breakeven (Q>1) "
        "using inertial confinement fusion—focusing 192 lasers on a fuel pellet—producing 3.15 MJ "
        "of energy from 2.05 MJ of laser input.\n\n"
        "Private fusion companies have raised over $6 billion in total. Commonwealth Fusion "
        "Systems is building SPARC, a compact tokamak using high-temperature superconducting "
        "magnets that enable a smaller, cheaper design. TAE Technologies uses field-reversed "
        "configuration with a different fuel (hydrogen-boron-11) that produces no neutron "
        "radiation. Helion Energy claims it will produce electricity by 2028 and has signed a "
        "power purchase agreement with Microsoft. Whether any of these timelines prove realistic "
        "remains to be seen."
    ),
    # 15 — Cognitive development
    (
        "Summarize the following:\n\n"
        "Jean Piaget's theory of cognitive development, published in the mid-20th century, "
        "proposed four sequential stages through which children construct understanding of the "
        "world. The sensorimotor stage (0-2 years) involves learning through physical interaction, "
        "culminating in object permanence. The preoperational stage (2-7) sees the development of "
        "symbolic thought but is limited by egocentrism and lack of conservation. The concrete "
        "operational stage (7-11) enables logical thinking about concrete events, and the formal "
        "operational stage (11+) introduces abstract and hypothetical reasoning.\n\n"
        "Lev Vygotsky offered a complementary perspective emphasizing social interaction as the "
        "primary driver of cognitive development. His concept of the Zone of Proximal Development "
        "(ZPD)—the gap between what a child can do alone and what they can do with guidance—"
        "suggests that learning is most effective when instruction targets this zone. Scaffolding, "
        "where an adult provides structured support that is gradually withdrawn, operationalises "
        "this concept in educational practice.\n\n"
        "Modern neuroscience has added biological detail to these frameworks. Synaptic density "
        "peaks at age 2-3 (approximately twice adult levels) and is pruned based on experience "
        "through adolescence—a process of 'use it or lose it.' Critical periods for language "
        "acquisition (birth to ~7 years) and binocular vision (birth to ~3 years) demonstrate "
        "that certain neural circuits require environmental input during specific windows for "
        "normal development. Adverse childhood experiences during these periods can have lasting "
        "effects on brain structure, stress response systems, and cognitive function."
    ),
    # 16 — Cryptocurrency regulation
    (
        "Summarize the following:\n\n"
        "The regulatory landscape for cryptocurrencies varies dramatically across jurisdictions, "
        "reflecting fundamentally different philosophies about financial innovation and consumer "
        "protection. The United States has taken a fragmented approach, with the SEC classifying "
        "many tokens as securities under the Howey test, the CFTC treating Bitcoin as a commodity, "
        "and individual states implementing their own requirements such as New York's BitLicense.\n\n"
        "The European Union's Markets in Crypto-Assets (MiCA) regulation, which came into full "
        "effect in 2024, represents the most comprehensive framework globally. MiCA establishes "
        "licensing requirements for crypto-asset service providers, reserves and redemption "
        "rules for stablecoins, and market abuse provisions. It explicitly classifies crypto-"
        "assets into three categories: utility tokens, asset-referenced tokens, and e-money "
        "tokens, each with tailored requirements.\n\n"
        "China has taken the most restrictive stance, banning cryptocurrency trading and mining "
        "entirely in 2021 while developing its central bank digital currency (CBDC), the digital "
        "yuan. El Salvador took the opposite approach, adopting Bitcoin as legal tender in 2021, "
        "though adoption has been limited and the IMF has urged the country to reverse course. "
        "Japan has implemented balanced regulations recognizing crypto as legal property while "
        "requiring exchanges to register and maintain client fund segregation.\n\n"
        "The collapse of FTX in November 2022, resulting in $8 billion in customer losses, "
        "accelerated regulatory action worldwide. Key debates centre on DeFi regulation (should "
        "protocols with no central operator be regulated like financial institutions?), stablecoin "
        "reserves (should they be required to hold dollar-for-dollar reserves?), and international "
        "coordination to prevent regulatory arbitrage."
    ),
    # 17 — Materials science
    (
        "Summarize the following:\n\n"
        "Graphene, a single layer of carbon atoms arranged in a hexagonal lattice, possesses "
        "extraordinary properties that have captivated materials scientists since its isolation "
        "in 2004 by Andre Geim and Konstantin Novoselov, who received the Nobel Prize in Physics "
        "in 2010. It is the strongest material ever measured (200 times stronger than steel), "
        "conducts electricity better than copper, conducts heat better than any known material, "
        "and is nearly transparent while being impermeable to gases.\n\n"
        "Despite two decades of research, commercial applications have been slower than initially "
        "predicted. Mass production of high-quality, defect-free graphene remains challenging. "
        "Chemical vapor deposition (CVD) produces excellent graphene but is expensive; liquid-"
        "phase exfoliation is cheap but produces flakes with variable quality. Current commercial "
        "uses are modest: graphene-enhanced concrete (30% stronger), composite materials in "
        "tennis rackets and bike frames, and improved thermal interface materials in electronics.\n\n"
        "The most promising near-term applications leverage graphene's electronic properties. "
        "Graphene-based sensors can detect individual gas molecules. Flexible graphene electrodes "
        "enable bendable displays and wearable health monitors. In energy storage, graphene "
        "coatings on lithium-ion battery cathodes improve charging speed and cycle life. Longer-"
        "term, graphene transistors could enable terahertz-frequency electronics beyond the "
        "limits of silicon, and graphene membranes with precisely controlled nanopores could "
        "revolutionize water desalination at a fraction of current energy costs."
    ),
    # 18 — Privacy and surveillance
    (
        "Summarize the following:\n\n"
        "The tension between privacy and surveillance has intensified in the digital age. "
        "Edward Snowden's 2013 revelations exposed the NSA's mass surveillance programs, "
        "including PRISM (collecting data from tech companies) and bulk collection of phone "
        "metadata. These disclosures triggered global debate about the balance between national "
        "security and civil liberties, leading to reforms including the USA FREEDOM Act, which "
        "ended bulk metadata collection, and the EU's General Data Protection Regulation (GDPR).\n\n"
        "Facial recognition technology has emerged as a particularly contentious surveillance "
        "tool. China has deployed extensive facial recognition networks with an estimated 626 "
        "million surveillance cameras, using the technology for law enforcement, social credit "
        "scoring, and tracking Uyghur minorities. In the West, cities including San Francisco, "
        "Portland, and Brussels have banned government use of facial recognition, citing "
        "accuracy disparities across racial groups and the chilling effect on free assembly.\n\n"
        "The 'nothing to hide' argument—that surveillance should only concern those engaged in "
        "wrongdoing—has been challenged on multiple grounds. Privacy enables political dissent, "
        "protects whistleblowers, and allows personal development free from judgment. The concept "
        "of 'chilling effects' describes how the awareness of being watched causes self-"
        "censorship even among law-abiding citizens. Legal scholars argue that privacy is not "
        "about hiding wrongdoing but about maintaining the power asymmetry between individuals "
        "and institutions that is essential for democracy."
    ),
    # 19 — Food systems
    (
        "Summarize the following:\n\n"
        "The global food system accounts for approximately 34% of greenhouse gas emissions, "
        "uses 50% of habitable land, and consumes 70% of freshwater withdrawals. Livestock "
        "production alone is responsible for 14.5% of global emissions through enteric "
        "fermentation (methane from ruminant digestion), manure management, and feed crop "
        "cultivation. Reducing the environmental footprint of food production while feeding a "
        "projected 10 billion people by 2050 is one of humanity's greatest challenges.\n\n"
        "Alternative proteins represent a rapidly growing solution space. Plant-based meat from "
        "companies like Beyond Meat and Impossible Foods has improved significantly in taste and "
        "texture, with market size reaching $8 billion in 2024. Cultivated (lab-grown) meat, "
        "produced by culturing animal cells in bioreactors, received its first regulatory "
        "approvals in Singapore (2020) and the US (2023), though production costs remain far "
        "above conventional meat at approximately $10-20 per kilogram.\n\n"
        "Precision agriculture technologies are improving the efficiency of conventional farming. "
        "GPS-guided tractors, drone-based crop monitoring, satellite imagery, and AI-powered "
        "disease detection can reduce fertiliser use by 20-30%, pesticide application by 50%, "
        "and water consumption by 25% while maintaining or increasing yields. Variable rate "
        "technology applies inputs at the optimal quantity for each square metre of a field "
        "rather than using uniform application rates.\n\n"
        "Food waste compounds the problem. One-third of all food produced globally is lost or "
        "wasted—1.3 billion tonnes per year. In developing countries, losses occur primarily "
        "post-harvest due to inadequate storage and transportation. In developed countries, "
        "waste occurs mainly at the retail and consumer level. Solutions include improved cold "
        "chain logistics, dynamic pricing to sell near-expiry products, and consumer education "
        "about date labelling (which causes 20% of household food waste in Europe)."
    ),
]

# ═══════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════
_POOLS: Dict[str, List[str]] = {
    "short_qa_agent": _SHORT_QA,
    "long_reasoning_agent": _LONG_REASONING,
    "tool_use_agent": _TOOL_USE,
    "code_gen_agent": _CODE_GEN,
    "summarizer_agent": _SUMMARIZER,
}


class TaskBank:
    """
    Returns randomised tasks for each agent type.

    Usage::

        task = TaskBank.get("short_qa_agent")   # random 1-liner
        task = TaskBank.get("summarizer_agent")  # random 3-5 para text
        tasks = TaskBank.get_batch("code_gen_agent", n=5)  # 5 random tasks
    """

    @staticmethod
    def get(agent_type: str) -> str:
        """Return one random task for *agent_type*."""
        pool = _POOLS.get(agent_type)
        if pool is None:
            raise ValueError(
                f"Unknown agent_type '{agent_type}'. "
                f"Valid types: {list(_POOLS.keys())}"
            )
        return random.choice(pool)

    @staticmethod
    def get_batch(agent_type: str, n: int = 10) -> List[str]:
        """
        Return *n* tasks for *agent_type*, sampled **with replacement**
        so n can exceed pool size.
        """
        pool = _POOLS.get(agent_type)
        if pool is None:
            raise ValueError(
                f"Unknown agent_type '{agent_type}'. "
                f"Valid types: {list(_POOLS.keys())}"
            )
        return [random.choice(pool) for _ in range(n)]

    @staticmethod
    def all_types() -> List[str]:
        """Return list of registered agent types."""
        return list(_POOLS.keys())

    @staticmethod
    def pool_size(agent_type: str) -> int:
        """Return number of tasks in the pool for *agent_type*."""
        pool = _POOLS.get(agent_type)
        if pool is None:
            raise ValueError(f"Unknown agent_type '{agent_type}'.")
        return len(pool)
