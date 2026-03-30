"""
Classical OS Scheduling Algorithms for AIOS.

Implements multiple well-known scheduling strategies:
  - SJF (Shortest Job First): sort by max_tokens as job size proxy
  - Priority: static rule-based priority assignment
  - MLFQ (Multi-Level Feedback Queue): dynamic demotion based on observed behavior
  - HRRN (Highest Response Ratio Next): balances short jobs + aging

All extend BaseScheduler and can be swapped via config.yaml scheduler_type.
"""

from aios.hooks.types.llm import LLMRequestQueueGetMessage
from aios.hooks.types.memory import MemoryRequestQueueGetMessage
from aios.hooks.types.tool import ToolRequestQueueGetMessage
from aios.hooks.types.storage import StorageRequestQueueGetMessage

from aios.memory.manager import MemoryManager
from aios.storage.storage import StorageManager
from aios.llm_core.adapter import LLMAdapter
from aios.tool.manager import ToolManager

from .base import BaseScheduler

from collections import deque, defaultdict
from queue import Empty

import threading
import traceback
import time
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Shared helpers (batch execution, single execution)
# ═══════════════════════════════════════════════════════════════════

class _SchedulerMixin:
    """Shared execution helpers for all classical schedulers."""

    def _execute_batch_syscalls(self, batch, executor, syscall_type):
        from aios.syscall.llm import LLMSyscall
        from aios.hooks.modules.llm import log_llm_syscall

        if not batch:
            return

        start_time = time.time()
        for syscall in batch:
            try:
                syscall.set_status("executing")
                syscall.set_start_time(start_time)
            except Exception as e:
                logger.error(f"Error preparing syscall: {e}")

        logger.info(f"Executing batch of {len(batch)} {syscall_type} syscalls.")

        try:
            executor(batch)
            for syscall in batch:
                syscall.set_end_time(time.time())
                syscall.set_status("done")
                if isinstance(syscall, LLMSyscall):
                    log_llm_syscall(syscall)
        except Exception as e:
            logger.error(f"Error executing {syscall_type} batch: {e}")
            traceback.print_exc()

    def _execute_syscall(self, syscall, executor, syscall_type):
        from aios.syscall.llm import LLMSyscall
        from aios.hooks.modules.llm import log_llm_syscall

        try:
            syscall.set_status("executing")
            self.logger.log(
                f"{syscall.agent_name} is executing {syscall_type} syscall.\n",
                "executing",
            )
            syscall.set_start_time(time.time())
            response = executor(syscall)
            syscall.set_response(response)
            syscall.event.set()
            syscall.set_status("done")
            syscall.set_end_time(time.time())
            if isinstance(syscall, LLMSyscall):
                log_llm_syscall(syscall)
            self.logger.log(
                f"Completed {syscall_type} syscall for {syscall.agent_name}.\n",
                "done",
            )
            return response
        except Exception as e:
            logger.error(f"Error executing {syscall_type} syscall: {e}")
            traceback.print_exc()
            return None

    # Non-LLM processors (identical across all schedulers)
    def process_memory_requests(self):
        while self.active:
            try:
                syscall = self.get_memory_syscall()
                self._execute_syscall(
                    syscall, self.memory_manager.address_request, "Memory"
                )
            except Empty:
                pass

    def process_storage_requests(self):
        while self.active:
            try:
                syscall = self.get_storage_syscall()
                self._execute_syscall(
                    syscall, self.storage_manager.address_request, "Storage"
                )
            except Empty:
                pass

    def process_tool_requests(self):
        while self.active:
            try:
                syscall = self.get_tool_syscall()
                self._execute_syscall(
                    syscall, self.tool_manager.address_request, "Tool"
                )
            except Empty:
                pass

    def start(self):
        self.active = True
        self.start_processing_threads([
            self.process_llm_requests,
            self.process_memory_requests,
            self.process_storage_requests,
            self.process_tool_requests,
        ])

    def stop(self):
        self.active = False
        self.stop_processing_threads()


# ═══════════════════════════════════════════════════════════════════
# 1. SJF — Shortest Job First
# ═══════════════════════════════════════════════════════════════════

class SJFScheduler(_SchedulerMixin, BaseScheduler):
    """
    Shortest Job First (non-preemptive).

    Sorts each batch by estimated job size (max_tokens) before dispatching.
    Provably optimal for minimizing average turnaround time in
    non-preemptive scheduling.

    Job size proxy: max_tokens (the single strongest latency correlate,
    Spearman rho = 0.206 in our dataset).
    """

    def __init__(self, *args, batch_interval: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_interval = batch_interval

    @staticmethod
    def _job_size(syscall) -> int:
        return syscall.max_tokens or 512

    def process_llm_requests(self):
        while self.active:
            time.sleep(self.batch_interval)

            batch = []
            while True:
                try:
                    batch.append(self.get_llm_syscall())
                except Empty:
                    break

            if batch:
                # Sort ascending by max_tokens (shortest first)
                batch.sort(key=self._job_size)
                logger.info(
                    f"SJF dispatching {len(batch)} syscalls "
                    f"(max_tokens range: {self._job_size(batch[0])}"
                    f"-{self._job_size(batch[-1])})"
                )
                self._execute_batch_syscalls(
                    batch, self.llm.execute_llm_syscalls, "LLM"
                )


# ═══════════════════════════════════════════════════════════════════
# 2. Priority — Static Rule-Based Priority Scheduling
# ═══════════════════════════════════════════════════════════════════

class PriorityScheduler(_SchedulerMixin, BaseScheduler):
    """
    Static priority scheduling with rule-based classification.

    Priority rules (no ML, deterministic):
      P0 (high):   max_tokens <= 128
      P1 (medium): max_tokens <= 512 AND no tools AND message_count <= 4
      P2 (low):    everything else (long generation, tools, multi-turn)

    Within each priority level, FIFO order is preserved.
    Includes aging: items waiting > aging_threshold_ms get promoted one level.
    """

    def __init__(self, *args, batch_interval: float = 0.5,
                 aging_threshold_ms: float = 3000.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_interval = batch_interval
        self.aging_threshold_ms = aging_threshold_ms
        # Each queue holds (syscall, enqueue_time)
        self._queues = {0: deque(), 1: deque(), 2: deque()}
        self._lock = threading.Lock()

    @staticmethod
    def _assign_priority(syscall) -> int:
        mt = syscall.max_tokens or 512
        mc = syscall.message_count or 2
        tools = bool(syscall.has_tools)

        if mt <= 128:
            return 0
        if mt <= 512 and not tools and mc <= 4:
            return 1
        return 2

    def _apply_aging(self):
        now = time.time()
        threshold_s = self.aging_threshold_ms / 1000.0

        with self._lock:
            # Promote P2 -> P1
            promoted = []
            remaining = deque()
            for entry in self._queues[2]:
                if now - entry[1] > threshold_s * 2:
                    promoted.append(entry)
                else:
                    remaining.append(entry)
            self._queues[2] = remaining
            self._queues[1].extend(promoted)

            # Promote P1 -> P0
            promoted = []
            remaining = deque()
            for entry in self._queues[1]:
                if now - entry[1] > threshold_s:
                    promoted.append(entry)
                else:
                    remaining.append(entry)
            self._queues[1] = remaining
            self._queues[0].extend(promoted)

    def process_llm_requests(self):
        while self.active:
            time.sleep(self.batch_interval)

            # Collect from global queue
            while True:
                try:
                    syscall = self.get_llm_syscall()
                    pri = self._assign_priority(syscall)
                    with self._lock:
                        self._queues[pri].append((syscall, time.time()))
                except Empty:
                    break

            # Apply aging
            self._apply_aging()

            # Drain in priority order
            batch = []
            with self._lock:
                for pri in (0, 1, 2):
                    while self._queues[pri]:
                        entry = self._queues[pri].popleft()
                        batch.append(entry[0])

            if batch:
                counts = {}
                for s in batch:
                    p = self._assign_priority(s)
                    counts[p] = counts.get(p, 0) + 1
                logger.info(
                    f"Priority dispatching {len(batch)} syscalls "
                    f"(P0={counts.get(0,0)}, P1={counts.get(1,0)}, P2={counts.get(2,0)})"
                )
                self._execute_batch_syscalls(
                    batch, self.llm.execute_llm_syscalls, "LLM"
                )


# ═══════════════════════════════════════════════════════════════════
# 3. MLFQ — Multi-Level Feedback Queue
# ═══════════════════════════════════════════════════════════════════

class MLFQScheduler(_SchedulerMixin, BaseScheduler):
    """
    Multi-Level Feedback Queue — the Linux-inspired scheduler.

    - 3 levels: Q0 (highest), Q1, Q2 (lowest)
    - All new requests start at Q0
    - After execution, if an agent_type's AVERAGE latency exceeds the
      time quantum for its current level, future requests from that
      agent_type are demoted one level
    - Periodic boost: every boost_interval, all items reset to Q0

    No prediction needed — adapts to observed runtime behavior.
    """

    def __init__(self, *args, batch_interval: float = 0.5,
                 time_quantum_ms: float = 5000.0,
                 boost_interval_s: float = 30.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_interval = batch_interval
        self.time_quantum_ms = time_quantum_ms
        self.boost_interval_s = boost_interval_s

        # Track which level each agent_type is at
        self._agent_level: dict[str, int] = defaultdict(int)  # 0 = highest
        # Track observed latencies per agent_type
        self._agent_latencies: dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()
        self._last_boost = time.time()

        # 3 queues
        self._queues = {0: deque(), 1: deque(), 2: deque()}

    def _get_agent_type(self, syscall) -> str:
        return re.sub(r"_r\d+$", "", syscall.agent_name)

    def _maybe_boost(self):
        """Periodic priority boost — reset all agent types to Q0."""
        now = time.time()
        if now - self._last_boost > self.boost_interval_s:
            with self._lock:
                self._agent_level.clear()
                self._last_boost = now
            logger.info("MLFQ: periodic boost — all agent types reset to Q0")

    def _record_latency(self, syscall):
        """After execution, record latency and maybe demote the agent type."""
        agent_type = self._get_agent_type(syscall)
        start = syscall.get_start_time()
        end = syscall.get_end_time()
        if not start or not end:
            return

        latency_ms = (end - start) * 1000
        with self._lock:
            self._agent_latencies[agent_type].append(latency_ms)
            # Keep only last 20 observations
            if len(self._agent_latencies[agent_type]) > 20:
                self._agent_latencies[agent_type] = self._agent_latencies[agent_type][-20:]

            avg_lat = sum(self._agent_latencies[agent_type]) / len(self._agent_latencies[agent_type])
            current_level = self._agent_level[agent_type]

            # Demote if average latency exceeds quantum for current level
            quantum = self.time_quantum_ms * (current_level + 1)
            if avg_lat > quantum and current_level < 2:
                self._agent_level[agent_type] = current_level + 1
                logger.info(
                    f"MLFQ: demoting {agent_type} to Q{current_level + 1} "
                    f"(avg_lat={avg_lat:.0f}ms > quantum={quantum:.0f}ms)"
                )

    def process_llm_requests(self):
        while self.active:
            time.sleep(self.batch_interval)
            self._maybe_boost()

            # Collect from global queue, assign to level queues
            while True:
                try:
                    syscall = self.get_llm_syscall()
                    agent_type = self._get_agent_type(syscall)
                    with self._lock:
                        level = self._agent_level.get(agent_type, 0)
                        self._queues[level].append(syscall)
                except Empty:
                    break

            # Drain in priority order
            batch = []
            with self._lock:
                for level in (0, 1, 2):
                    while self._queues[level]:
                        batch.append(self._queues[level].popleft())

            if batch:
                self._execute_batch_syscalls(
                    batch, self.llm.execute_llm_syscalls, "LLM"
                )
                # Record latencies for demotion decisions
                for syscall in batch:
                    self._record_latency(syscall)


# ═══════════════════════════════════════════════════════════════════
# 4. HRRN — Highest Response Ratio Next
# ═══════════════════════════════════════════════════════════════════

class HRRNScheduler(_SchedulerMixin, BaseScheduler):
    """
    Highest Response Ratio Next (non-preemptive).

    Response ratio = (wait_time + estimated_service_time) / estimated_service_time

    This naturally balances short jobs (low service time → high ratio)
    with long-waiting jobs (high wait → high ratio), preventing starvation
    while still favoring short tasks.

    Service time estimate: max_tokens (same proxy as SJF).
    """

    def __init__(self, *args, batch_interval: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_interval = batch_interval

    @staticmethod
    def _response_ratio(syscall, now: float) -> float:
        service_time = (syscall.max_tokens or 512) / 100.0  # normalize
        wait_time = now - (syscall.get_created_time() or now)
        if service_time <= 0:
            service_time = 1.0
        return (wait_time + service_time) / service_time

    def process_llm_requests(self):
        while self.active:
            time.sleep(self.batch_interval)

            batch = []
            while True:
                try:
                    batch.append(self.get_llm_syscall())
                except Empty:
                    break

            if batch:
                now = time.time()
                # Sort by response ratio (highest first)
                batch.sort(key=lambda s: self._response_ratio(s, now), reverse=True)
                logger.info(
                    f"HRRN dispatching {len(batch)} syscalls "
                    f"(top ratio: {self._response_ratio(batch[0], now):.2f}, "
                    f"bottom: {self._response_ratio(batch[-1], now):.2f})"
                )
                self._execute_batch_syscalls(
                    batch, self.llm.execute_llm_syscalls, "LLM"
                )
