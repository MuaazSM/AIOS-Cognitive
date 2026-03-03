# test_client.py
import unittest
import requests
import threading
import json
import time
import os
from typing import List, Dict, Any, Tuple

from cerebrum.llm.apis import llm_chat

from cerebrum.utils.communication import aios_kernel_url

# --- Helper function to send a single request ---
def send_request(payload: Dict[str, Any]) -> Tuple[Dict[str, Any] | None, float]:
    """Sends a POST request to the query endpoint and returns status code, response JSON, and duration."""
    start_time = time.time()
    try:
        response = llm_chat(
            agent_name=payload["agent_name"],
            messages=payload["query_data"]["messages"],
            llms=payload["query_data"]["llms"] if "llms" in payload["query_data"] else None,
        )
        end_time = time.time()
        duration = end_time - start_time
        return response["response"], duration
    
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        # General unexpected errors
        return {"error": f"An unexpected error occurred: {e}"}, duration


class TestConcurrentLLMQueries(unittest.TestCase):

    def _run_concurrent_requests(self, payloads: List[Dict[str, Any]]):
        results = [None] * len(payloads)
        threads = []

        def worker(index, payload):
            response_data, duration = send_request(payload)
            results[index] = {"data": response_data, "duration": duration}
            print(f"Thread {index}: Completed in {duration:.2f}s with response: {json.dumps(response_data)}")

        print(f"\n--- Running test: {self._testMethodName} ---")
        print(f"Sending {len(payloads)} concurrent requests to {aios_kernel_url}...")
        for i, payload in enumerate(payloads):
            thread = threading.Thread(target=worker, args=(i, payload))
            threads.append(thread)
            thread.start()

        for i, thread in enumerate(threads):
            thread.join()
            print(f"Thread {i} finished.")

        print("--- All threads completed ---")
        return results

    def test_both_llms(self):
        payload1 = {
            "agent_name": "test_agent_1",
            "query_type": "llm",
            "query_data": {
                "llms": [{"name": "gpt-4o", "backend": "openai"}],
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
                "action_type": "chat",
                "message_return_type": "text",
            }
        }
        payload2 = {
            "agent_name": "test_agent_2",
            "query_type": "llm",
            "query_data": {
                "llms": [{"name": "gpt-4o-mini", "backend": "openai"}], # Using a different model for variety
                "messages": [{"role": "user", "content": "What is the capital of the United States?"}],
                "action_type": "chat",
                "message_return_type": "text",
            }
        }
        results = self._run_concurrent_requests([payload1, payload2])
        
        for i, result in enumerate(results):
            print(f"Result {i} (No LLM): {result}")
            # Both should succeed using defaults
            status, response_message, error_message, finished = result["data"]["status_code"], result["data"]["response_message"], result["data"]["error"], result["data"]["finished"]
            
            self.assertEqual(status, 200, f"Request {i} (No LLM) should succeed, but failed with status {status}")
            self.assertIsNone(error_message, f"Request {i} (No LLM) returned an unexpected error: {error_message}")
            self.assertIsInstance(response_message, str, f"Request {i} (No LLM) result is not a string")
            self.assertTrue(finished, f"Request {i} (No LLM) result is empty") # Check not empty

    def test_one_llm_one_empty(self):
        payload_llm = {
            "agent_name": "test_agent_1",
            "query_type": "llm",
            "query_data": {
                "llms": [{"name": "gpt-4o", "backend": "openai"}],
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
                "action_type": "chat",
                "message_return_type": "text",
            }
        }
        payload_no_llm = {
            "agent_name": "test_agent_2",
            "query_type": "llm",
            "query_data": {
                # 'llms' key is omitted entirely
                "messages": [{"role": "user", "content": "What is the capital of the United States?"}],
                "action_type": "chat",
                "message_return_type": "text",
            }
        }
        results = self._run_concurrent_requests([payload_llm, payload_no_llm])
        
        for i, result in enumerate(results):
            print(f"Result {i} (No LLM): {result}")
            # Both should succeed using defaults
            status, response_message, error_message, finished = result["data"]["status_code"], result["data"]["response_message"], result["data"]["error"], result["data"]["finished"]
            
            self.assertEqual(status, 200, f"Request {i} (No LLM) should succeed, but failed with status {status}")
            self.assertIsNone(error_message, f"Request {i} (No LLM) returned an unexpected error: {error_message}")
            self.assertIsInstance(response_message, str, f"Request {i} (No LLM) result is not a string")
            self.assertTrue(finished, f"Request {i} (No LLM) result is empty") # Check not empty

    def test_no_llms(self):
        """Case 2: Both payloads have no LLMs defined. Should succeed using defaults."""
        payload1 = {
            "agent_name": "test_agent_1",
            "query_type": "llm",
            "query_data": {
                # 'llms' key is omitted
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
                "action_type": "chat",
                "message_return_type": "text",
            }
        }
        payload2 = {
            "agent_name": "test_agent_2",
            "query_type": "llm",
            "query_data": {
                # 'llms' key is omitted
                "messages": [{"role": "user", "content": "What is the capital of the United States?"}],
                "action_type": "chat",
                "message_return_type": "text",
            }
        }
        results = self._run_concurrent_requests([payload1, payload2])
        
        for i, result in enumerate(results):
            print(f"Result {i} (No LLM): {result}")
            # Both should succeed using defaults
            status, response_message, error_message, finished = result["data"]["status_code"], result["data"]["response_message"], result["data"]["error"], result["data"]["finished"]
            
            self.assertEqual(status, 200, f"Request {i} (No LLM) should succeed, but failed with status {status}")
            self.assertIsNone(error_message, f"Request {i} (No LLM) returned an unexpected error: {error_message}")
            self.assertIsInstance(response_message, str, f"Request {i} (No LLM) result is not a string")
            self.assertTrue(finished, f"Request {i} (No LLM) result is empty") # Check not empty

    def test_syscall_logging(self):
        """Test that LLM syscall logging captures all required metrics when enabled."""
        # Run a simple concurrent request
        payload1 = {
            "agent_name": "logging_test_agent_1",
            "query_type": "llm",
            "query_data": {
                "llms": [{"name": "gpt-4o-mini", "backend": "openai"}],
                "messages": [{"role": "user", "content": "Say hello"}],
                "action_type": "chat",
                "message_return_type": "text",
            }
        }
        payload2 = {
            "agent_name": "logging_test_agent_2",
            "query_type": "llm",
            "query_data": {
                "llms": [{"name": "gpt-4o-mini", "backend": "openai"}],
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "action_type": "chat",
                "message_return_type": "text",
            }
        }
        
        results = self._run_concurrent_requests([payload1, payload2])
        
        # Verify requests succeeded
        for i, result in enumerate(results):
            status = result["data"]["status_code"]
            self.assertEqual(status, 200, f"Request {i} should succeed for logging test")
        
        # Give time for logs to be written
        time.sleep(1)
        
        # Verify log file exists and contains entries
        log_file = "logs/llm_syscalls.jsonl"
        self.assertTrue(os.path.exists(log_file), f"Log file {log_file} does not exist")
        
        # Read and validate log entries
        with open(log_file, "r") as f:
            lines = f.readlines()
        
        self.assertGreater(len(lines), 0, f"Log file {log_file} is empty")
        
        required_keys = {"input_char_length", "message_count", "has_tools", "latency_ms", "was_interrupted"}
        
        for line_num, line in enumerate(lines[-2:]):  # Check last 2 entries (our test requests)
            line = line.strip()
            self.assertTrue(len(line) > 0, f"Line {line_num} is empty")
            
            try:
                log_entry = json.loads(line)
            except json.JSONDecodeError as e:
                self.fail(f"Line {line_num} is not valid JSON: {e}")
            
            # Verify required keys exist
            for key in required_keys:
                self.assertIn(key, log_entry, f"Log entry missing required key: {key}")
            
            # Verify latency_ms is a positive float
            latency_ms = log_entry["latency_ms"]
            self.assertIsInstance(latency_ms, (int, float), f"latency_ms should be numeric, got {type(latency_ms)}")
            self.assertGreater(latency_ms, 0, f"latency_ms should be positive, got {latency_ms}")


if __name__ == '__main__':
    unittest.main()