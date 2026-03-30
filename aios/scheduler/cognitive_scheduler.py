"""
CognitiveScheduler — ML-based priority queue scheduler for AIOS.

Classifies incoming LLM syscalls into complexity classes (fast/medium/large)
using a trained sklearn pipeline, then routes them to priority queues.
Fast tasks are dispatched first, with an aging mechanism to prevent starvation.
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

from collections import deque
from queue import Empty
from pathlib import Path

import pickle
import numpy as np
import threading
import traceback
import time
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default model path relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "complexity_classifier.pkl"


class CognitiveScheduler(BaseScheduler):
    """
    ML-based priority queue scheduler.

    Incoming LLM syscalls are classified into fast/medium/large complexity
    classes using a trained model, then routed to priority queues.
    The scheduler drains fast > medium > large, with aging to prevent
    starvation of lower-priority tasks.
    """

    def __init__(
        self,
        llm: LLMAdapter,
        memory_manager: MemoryManager,
        storage_manager: StorageManager,
        tool_manager: ToolManager,
        log_mode: str,
        get_llm_syscall: LLMRequestQueueGetMessage,
        get_memory_syscall: MemoryRequestQueueGetMessage,
        get_storage_syscall: StorageRequestQueueGetMessage,
        get_tool_syscall: ToolRequestQueueGetMessage,
        model_path: str = None,
        aging_threshold_ms: float = 2000.0,
        batch_interval: float = 0.5,
    ):
        super().__init__(
            llm, memory_manager, storage_manager, tool_manager,
            log_mode, get_llm_syscall, get_memory_syscall,
            get_storage_syscall, get_tool_syscall,
        )
        self.batch_interval = batch_interval
        self.aging_threshold_ms = aging_threshold_ms

        # Priority queues: each item is (syscall, enqueue_time, predicted_class)
        self._fast_queue: deque = deque()
        self._medium_queue: deque = deque()
        self._large_queue: deque = deque()
        self._queue_lock = threading.Lock()

        # Load trained model
        model_path = model_path or str(_DEFAULT_MODEL_PATH)
        self._pipeline = None
        self._feature_names = []
        self._model_name_from_config = "unknown"
        self._load_model(model_path)

        # Get model name from AIOS config (static per session)
        try:
            from aios.config.config_manager import config
            cfg = getattr(config, "config", {}) or {}
            models_list = cfg.get("llms", {}).get("models", [])
            if models_list:
                self._model_name_from_config = models_list[0].get("name", "unknown")
        except Exception:
            pass

        logger.info(
            f"CognitiveScheduler initialized — model: {model_path}, "
            f"features: {len(self._feature_names)}, "
            f"aging: {aging_threshold_ms}ms, "
            f"llm: {self._model_name_from_config}"
        )

    def _load_model(self, model_path: str):
        """Load the trained classifier pipeline from disk."""
        path = Path(model_path)
        if not path.exists():
            logger.warning(
                f"Model not found at {model_path}. "
                "CognitiveScheduler will default all requests to 'medium'."
            )
            return

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self._pipeline = model_data["pipeline"]
        self._feature_names = model_data["feature_names"]
        logger.info(f"Loaded classifier: {model_data.get('best_model_name', '?')}")

    def _extract_features(self, syscall) -> np.ndarray:
        """Build a feature vector from a syscall, matching training column order."""
        # Extract agent_type from agent_name (strip _rN suffix)
        agent_type = re.sub(r"_r\d+$", "", syscall.agent_name)

        # Build feature dict
        raw = {
            "input_char_length": syscall.input_char_length or 0,
            "message_count": syscall.message_count or 2,
            "has_tools": int(bool(syscall.has_tools)),
            "max_tokens": syscall.max_tokens or 512,
            "temperature": syscall.temperature or 0.5,
        }

        # One-hot: agent_type dummies
        for fn in self._feature_names:
            if fn.startswith("agent_"):
                expected_type = fn[len("agent_"):]
                raw[fn] = 1 if agent_type == expected_type else 0

        # One-hot: model_name dummies
        for fn in self._feature_names:
            if fn.startswith("model_"):
                expected_model = fn[len("model_"):]
                raw[fn] = 1 if self._model_name_from_config == expected_model else 0

        # Build vector in exact column order
        vec = [float(raw.get(fn, 0)) for fn in self._feature_names]
        return np.array([vec])

    def _classify(self, syscall) -> str:
        """Predict complexity class for a syscall. Returns 'fast', 'medium', or 'large'."""
        if self._pipeline is None:
            return "medium"
        try:
            features = self._extract_features(syscall)
            prediction = self._pipeline.predict(features)[0]
            return prediction
        except Exception as e:
            logger.warning(f"Classification failed for {syscall.agent_name}: {e}")
            return "medium"

    def _enqueue(self, syscall):
        """Classify and route a syscall to the appropriate priority queue."""
        cls = self._classify(syscall)
        entry = (syscall, time.time(), cls)

        with self._queue_lock:
            if cls == "fast":
                self._fast_queue.append(entry)
            elif cls == "large":
                self._large_queue.append(entry)
            else:
                self._medium_queue.append(entry)

    def _apply_aging(self):
        """Promote starving items from lower queues to higher ones."""
        now = time.time()
        threshold_s = self.aging_threshold_ms / 1000.0

        with self._queue_lock:
            # Promote large -> medium
            promoted = []
            remaining = deque()
            for entry in self._large_queue:
                if now - entry[1] > threshold_s * 2:
                    promoted.append(entry)
                else:
                    remaining.append(entry)
            self._large_queue = remaining
            self._medium_queue.extend(promoted)

            # Promote medium -> fast
            promoted = []
            remaining = deque()
            for entry in self._medium_queue:
                if now - entry[1] > threshold_s:
                    promoted.append(entry)
                else:
                    remaining.append(entry)
            self._medium_queue = remaining
            self._fast_queue.extend(promoted)

    def _drain_queues(self) -> list:
        """Drain priority queues in order: fast > medium > large. Returns syscalls."""
        self._apply_aging()

        batch = []
        with self._queue_lock:
            # Fast first
            while self._fast_queue:
                batch.append(self._fast_queue.popleft())
            # Then medium
            while self._medium_queue:
                batch.append(self._medium_queue.popleft())
            # Then large
            while self._large_queue:
                batch.append(self._large_queue.popleft())

        return batch

    # ── Batch execution (reused from FIFO) ──────────────────────

    def _execute_batch_syscalls(self, batch, executor, syscall_type):
        """Execute a batch of syscalls with status tracking."""
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
                logger.info(
                    f"Completed {syscall_type} syscall for {syscall.agent_name}."
                )
        except Exception as e:
            logger.error(f"Error executing {syscall_type} batch: {e}")
            traceback.print_exc()

    def _execute_syscall(self, syscall, executor, syscall_type):
        """Execute a single syscall."""
        from aios.syscall.llm import LLMSyscall
        from aios.hooks.modules.llm import log_llm_syscall

        try:
            syscall.set_status("executing")
            self.logger.log(
                f"{syscall.agent_name} is executing {syscall_type} syscall.\n",
                "executing"
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
                "done"
            )
            return response
        except Exception as e:
            logger.error(f"Error executing {syscall_type} syscall: {e}")
            traceback.print_exc()
            return None

    # ── Request processors ──────────────────────────────────────

    def process_llm_requests(self):
        """Poll global queue, classify, route to priority queues, dispatch."""
        while self.active:
            time.sleep(self.batch_interval)

            # 1. Drain global queue into priority queues
            while True:
                try:
                    syscall = self.get_llm_syscall()
                    self._enqueue(syscall)
                except Empty:
                    break

            # 2. Drain priority queues and execute
            ordered = self._drain_queues()
            if ordered:
                syscalls = [entry[0] for entry in ordered]
                classes = [entry[2] for entry in ordered]
                n_fast = classes.count("fast")
                n_med = classes.count("medium")
                n_large = classes.count("large")
                logger.info(
                    f"Dispatching {len(syscalls)} syscalls "
                    f"(fast={n_fast}, medium={n_med}, large={n_large})"
                )
                self._execute_batch_syscalls(
                    syscalls, self.llm.execute_llm_syscalls, "LLM"
                )

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
