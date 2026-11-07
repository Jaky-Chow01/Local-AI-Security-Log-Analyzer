import os
import threading
from typing import List, Dict, Any, Callable
from llama_cpp import Llama, LlamaGrammar
from app.prompts import build_chunk_prompt, LOG_CLASSIFICATION_SCHEMA, JSON_GBNF_GRAMMAR

class LLMManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_path: str = "models/qwen2.5-3b-instruct-q4_k_m.gguf", n_ctx: int = 2048):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.model = None
        self.grammar = None
        self._load_model()

    @classmethod
    def get_instance(cls, model_path: str = None) -> "LLMManager":
        with cls._lock:
            if cls._instance is None:
                path = model_path or os.getenv("MODEL_PATH", "models/qwen2.5-3b-instruct-q4_k_m.gguf")
                cls._instance = cls(model_path=path)
            return cls._instance

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. "
                "Please run scripts/download_model.py first."
            )
        
        # Initialize llama.cpp with offline settings, 2 vCPU threads, mmap enabled
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=2,
            use_mmap=True,
            verbose=False
        )
        self.grammar = LlamaGrammar.from_string(JSON_GBNF_GRAMMAR)

    def tokenize(self, text: str) -> List[int]:
        """Tokenizes text using the underlying GGUF model's tokenizer."""
        if not self.model:
            return list(text.encode("utf-8"))
        return self.model.tokenize(text.encode("utf-8"), add_bos=False)

    def classify_chunk(self, ledger_summary: str, compact_lines: str) -> Dict[str, Any]:
        """
        Executes temperature=0 grammar-constrained inference on a log chunk.
        Thread-safe under a global lock to ensure single-flight execution.
        """
        with self._lock:
            system_prompt, user_prompt = build_chunk_prompt(ledger_summary, compact_lines)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            try:
                response = self.model.create_chat_completion(
                    messages=messages,
                    temperature=0.0,
                    max_tokens=400,
                    grammar=self.grammar
                )
                
                content = response["choices"][0]["message"]["content"]
                import json
                return json.loads(content)
            except Exception as e:
                # Return empty findings container on parse or inference failure
                return {"findings": []}

def get_tokenizer_fn() -> Callable[[str], List[int]]:
    """Returns a helper tokenizer function for the chunker."""
    try:
        manager = LLMManager.get_instance()
        return manager.tokenize
    except Exception:
        # Fallback character estimation if model is not yet downloaded
        return lambda text: list(text.encode("utf-8"))