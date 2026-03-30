from random import randint

from typing import Tuple

# from aios.llm_core.llms import LLM
from aios.llm_core.adapter import LLMAdapter as LLM

from aios.hooks.types.llm import (
    LLMParams,
    LLMRequestQueue,
    LLMRequestQueueAddMessage,
    LLMRequestQueueGetMessage,
    LLMRequestQueueCheckEmpty
)
from aios.hooks.utils.validate import validate
from aios.hooks.stores import queue as QueueStore, processes as ProcessStore
import json
import os
import threading
from uuid import uuid4
from datetime import datetime

ids = []  # List to store process IDs

# Module-level lock for thread-safe file writes
_llm_syscall_log_lock = threading.Lock()


@validate(LLMParams)
def useCore(params: LLMParams) -> LLM:
    """
    Initialize and return a Language Learning Model (LLM) instance.

    Args:
        params (LLMParams): Parameters required for LLM initialization.

    Returns:
        LLM: An instance of the initialized LLM.
    """
    return LLM(**params.model_dump())


def useLLMRequestQueue() -> (
    Tuple[LLMRequestQueue, LLMRequestQueueGetMessage, LLMRequestQueueAddMessage, LLMRequestQueueCheckEmpty]
):
    """
    Creates and returns a queue for LLM requests along with helper methods to manage the queue.

    Returns:
        Tuple: A tuple containing the LLM request queue, get message function, add message function, and check queue empty function.
    """
    # r_str = (
    #     generate_random_string()
    # )  # Generate a random string for queue identification
    r_str = "llm"
    _ = LLMRequestQueue()

    # Store the LLM request queue in QueueStore
    QueueStore.REQUEST_QUEUE[r_str] = _

    # Function to get messages from the queue
    def getMessage():
        return QueueStore.getMessage(_)

    # Function to add messages to the queue
    def addMessage(message: str):
        return QueueStore.addMessage(_, message)

    # Function to check if the queue is empty
    def isEmpty():
        return QueueStore.isEmpty(_)

    return _, getMessage, addMessage, isEmpty


def log_llm_syscall(syscall) -> None:
    """
    Log an LLM syscall to a JSON-Lines file after it finishes executing.
    
    Args:
        syscall: The LLMSyscall object that has completed execution
        
    This function:
    - Checks if logging is enabled via config.get("log_syscalls")
    - Generates a UUID for the syscall
    - Extracts timing and execution metrics
    - Writes a single JSON line to logs/llm_syscalls.jsonl
    - Thread-safe file access via module-level lock
    - Silently catches and ignores all exceptions
    """
    try:
        from aios.config.config_manager import config

        config_dict = getattr(config, "config", {}) or {}
        if not config_dict.get("log_syscalls", False):
            return
        
        # Use absolute path anchored to this source tree
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        logs_dir = os.path.join(workspace_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        log_file = os.path.join(logs_dir, "llm_syscalls.jsonl")
        
        # Extract timing information
        created_time = syscall.get_created_time()
        start_time = syscall.get_start_time()
        end_time = syscall.get_end_time()
        
        # Calculate latency and wait time
        latency_ms = (end_time - start_time) * 1000 if start_time and end_time else 0
        wait_ms = (start_time - created_time) * 1000 if start_time and created_time else 0
        
        # Get model name from config (for multi-model experiments)
        llms_cfg = config_dict.get("llms", {})
        models_list = llms_cfg.get("models", [])
        model_name = models_list[0].get("name", "unknown") if models_list else "unknown"

        # Build the log record
        log_record = {
            "syscall_id": str(uuid4()),
            "agent_name": syscall.agent_name,
            "model_name": model_name,
            "timestamp": start_time,
            "input_char_length": syscall.input_char_length,
            "message_count": syscall.message_count,
            "has_tools": syscall.has_tools,
            "max_tokens": syscall.max_tokens,
            "temperature": syscall.temperature,
            "created_time": created_time,
            "start_time": start_time,
            "end_time": end_time,
            "latency_ms": latency_ms,
            "wait_ms": wait_ms,
            "was_interrupted": syscall.was_interrupted,
            "error": syscall.error,
        }
        
        # Thread-safe file write
        with _llm_syscall_log_lock:
            with open(log_file, "a") as f:
                f.write(json.dumps(log_record) + "\n")
                
    except Exception:
        pass
