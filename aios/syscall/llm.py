from aios.syscall.syscall import Syscall

class LLMSyscall(Syscall):
    def __init__(self, agent_name: str, query):
        self.agent_name = agent_name
        self.query = query
        
        # Capture fields from LLMQuery object
        self.input_char_length: int = len(json.dumps(query.messages))
        self.message_count: int = len(query.messages)
        self.has_tools: bool = bool(getattr(query, "tools", None))
        self.max_tokens: int | None = getattr(query, "max_new_tokens", None)
        self.temperature: float | None = getattr(query, "temperature", None)
        
        # Outcome fields initialized to defaults
        self.was_interrupted: bool = False
        self.error: bool = False