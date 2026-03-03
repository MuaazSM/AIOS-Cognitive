from cerebrum.llm.apis import llm_chat

from litellm import completion

class CoT:
    def __init__(self, on_aios: bool = True):
        self.agent_name = "llm"
        self.on_aios = on_aios

    def run_swebench(self, input_str: str):
        messages = [
            {"content": "You are a helpful assistant that can answer questions and help with tasks.", "role": "system"},
            {"content": input_str, "role": "user"}
        ]
        if self.on_aios:
            response = llm_chat(self.agent_name, messages)
        else:
            response = completion(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0,
            )
        result = response["response"]["response_message"]
        
        return result
    
    def run_humaneval(self, input_str: str):
        system_prompt = """You are an AI assistant good at coding. You will receive a function definition and
        comments. You need to help me complete this function. The completion should strictly follow the following format and requirements:
        
        Format:
        <FINAL_ANSWER>
        YOUR FINAL ANSWER
        </FINAL_ANSWER>
        
        Requirements: 
        1. YOUR FINAL ANSWER must be a piece of code that can be directly filled into the given code at the <CURRENT_CURSOR_POSITION> marker.
        2. Only include the code you're adding, don't include the original function definition or comments.
        3. Do not use extra code quotes like ```python``` to wrap the code.
        4. Make sure the syntax of the code is correct, especially pay attention to proper indentation.
        5. Maintain the same indentation level as the surrounding code.
        6. If you're completing a function body, ensure all code is properly indented inside the function.
        7. Check that all return statements, loops, and conditional blocks have correct indentation.
        8. Ensure your code aligns with the original code style and indentation pattern.
        
        Example of proper formatting:
        For a function like:
        ```
        def example_function(x):
            # Some comment
            # More comments
        ```
        
        Your answer should be:
        <FINAL_ANSWER>
            result = x * 2
            return result
        </FINAL_ANSWER>
        
        Notice how the code maintains proper indentation relative to the function definition.
        """
        messages = [
            {"content": system_prompt, "role": "system"},
            {"content": f"Given the following code: {input_str}, complete the function. ", "role": "user"}
        ]
        if self.on_aios:
            response = llm_chat(self.agent_name, messages)
            result = response["response"]["response_message"]
        else:
            response = completion(
                model="gpt-4o-mini",
                messages=messages,
                temperature=1.0,
            )
        return result
    
    def run_gaia(self, input_str: str):
        messages = [
            {"content": f"Given the following code: {input_str}, please provide the completion of the code to fulfill the functionality.", "role": "user"}
        ]
        if self.on_aios:
            response = llm_chat(self.agent_name, messages)
        else:
            response = completion(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0,
            )
        result = response["response"]["response_message"]
        return result