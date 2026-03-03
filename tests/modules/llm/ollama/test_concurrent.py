# test_client.py
import unittest
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


class TestConcurrentOllamaQueries(unittest.TestCase):

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
    
    def _verify_response(self, result, test_name, index):
        """Helper to verify response is successful"""
        print(f"Result {index} ({test_name}): {result}")
        
        # Common verification logic
        status = result["data"]["status_code"]
        response_message = result["data"]["response_message"]
        error_message = result["data"]["error"]
        finished = result["data"]["finished"]
        
        self.assertEqual(status, 200, f"Request {index} ({test_name}) should succeed, but failed with status {status}")
        self.assertIsNone(error_message, f"Request {index} ({test_name}) returned an unexpected error: {error_message}")
        self.assertIsInstance(response_message, str, f"Request {index} ({test_name}) result is not a string")
        self.assertTrue(finished, f"Request {index} ({test_name}) did not finish successfully")

    def test_ollama_same_model(self):
        """Case 1: All queries specify the same Ollama model (qwen3:1.7b)."""
        payloads = []
        questions = [
            "What is the capital of France?",
            "What is the largest planet in our solar system?",
            "What is the square root of 144?",
            "Who wrote the novel 'Pride and Prejudice'?"
        ]
        
        for i, question in enumerate(questions):
            payloads.append({
                "agent_name": f"test_agent_{i+1}",
                "query_type": "llm",
                "query_data": {
                    "llms": [{"name": "qwen3:1.7b", "backend": "ollama"}],
                    "messages": [{"role": "user", "content": question}],
                    "action_type": "chat",
                    "message_return_type": "text",
                }
            })
        
        results = self._run_concurrent_requests(payloads)
        
        for i, result in enumerate(results):
            self._verify_response(result, "Same Ollama Model", i)

    def test_ollama_different_models(self):
        """Case 2: Queries specify different Ollama models (mixing qwen3:1.7b and qwen3:4b)."""
        models = ["qwen3:1.7b", "qwen3:4b", "qwen3:1.7b", "qwen3:4b"]
        questions = [
            "What is the capital of Germany?",
            "What is the boiling point of water?",
            "Who invented the telephone?",
            "What is the tallest mountain in the world?"
        ]
        
        payloads = []
        for i, (model, question) in enumerate(zip(models, questions)):
            payloads.append({
                "agent_name": f"test_agent_{i+1}",
                "query_type": "llm",
                "query_data": {
                    "llms": [{"name": model, "backend": "ollama"}],
                    "messages": [{"role": "user", "content": question}],
                    "action_type": "chat",
                    "message_return_type": "text",
                }
            })
        
        results = self._run_concurrent_requests(payloads)
        
        for i, result in enumerate(results):
            self._verify_response(result, "Different Ollama Models", i)

    def test_ollama_some_specified_same_model(self):
        """Case 3: Some queries specify the same Ollama model, others don't specify any model."""
        questions = [
            "What is the capital of Japan?",
            "What is the speed of light?",
            "Who was the first person to walk on the moon?",
            "What is the chemical formula for water?"
        ]
        
        # First and third payloads specify the same model, others don't specify
        payloads = []
        for i, question in enumerate(questions):
            payload = {
                "agent_name": f"test_agent_{i+1}",
                "query_type": "llm",
                "query_data": {
                    "messages": [{"role": "user", "content": question}],
                    "action_type": "chat",
                    "message_return_type": "text",
                }
            }
            
            # Add LLM specification to first and third payloads
            if i == 0 or i == 2:
                payload["query_data"]["llms"] = [{"name": "qwen3:1.7b", "backend": "ollama"}]
                
            payloads.append(payload)
        
        results = self._run_concurrent_requests(payloads)
        
        for i, result in enumerate(results):
            self._verify_response(result, "Some Specified Same Model", i)

    def test_ollama_some_specified_different_models(self):
        """Case 4: Some queries specify different Ollama models, others don't specify any model."""
        questions = [
            "What is the distance from Earth to the Moon?",
            "Who painted the Mona Lisa?",
            "What is the main ingredient in bread?",
            "Who wrote 'Romeo and Juliet'?"
        ]
        
        payloads = []
        for i, question in enumerate(questions):
            payload = {
                "agent_name": f"test_agent_{i+1}",
                "query_type": "llm",
                "query_data": {
                    "messages": [{"role": "user", "content": question}],
                    "action_type": "chat",
                    "message_return_type": "text",
                }
            }
            
            # First payload uses qwen3:1.7b, third uses qwen3:4b
            if i == 0:
                payload["query_data"]["llms"] = [{"name": "qwen3:1.7b", "backend": "ollama"}]
            elif i == 2:
                payload["query_data"]["llms"] = [{"name": "qwen3:4b", "backend": "ollama"}]
                
            payloads.append(payload)
        
        results = self._run_concurrent_requests(payloads)
        
        for i, result in enumerate(results):
            self._verify_response(result, "Some Specified Different Models", i)

    def test_syscall_logging(self):
        """Test that LLM syscall logging captures all required metrics when enabled."""
        # Run a simple concurrent request
        payload1 = {
            "agent_name": "logging_test_agent_1",
            "query_type": "llm",
            "query_data": {
                "llms": [{"name": "qwen3:1.7b", "backend": "ollama"}],
                "messages": [{"role": "user", "content": "Say hi"}],
                "action_type": "chat",
                "message_return_type": "text",
            }
        }
        payload2 = {
            "agent_name": "logging_test_agent_2",
            "query_type": "llm",
            "query_data": {
                "llms": [{"name": "qwen3:1.7b", "backend": "ollama"}],
                "messages": [{"role": "user", "content": "What is one plus one?"}],
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