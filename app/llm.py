import json
import logging
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from llama_cpp import Llama, LlamaGrammar, StoppingCriteriaList

from app.prompts import (
    LOG_CLASSIFICATION_SCHEMA,
    build_chunk_prompt,
)

logger = logging.getLogger(__name__)


class ModelExecutionError(Exception):
    """Raised when the LLM fails to process a chunk."""


_INFERENCE_LOCK = threading.Lock()


class LLMManager:
    _instance: Optional["LLMManager"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_path: str = "models/qwen2.5-3b-instruct-q4_k_m.gguf",
        n_ctx: int = 2048,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.model: Optional[Llama] = None
        self.grammar: Optional[LlamaGrammar] = None
        self._load_model()

    @classmethod
    def get_instance(cls, model_path: Optional[str] = None) -> "LLMManager":
        with cls._lock:
            if cls._instance is None:
                path = model_path or os.getenv(
                    "MODEL_PATH", "models/qwen2.5-3b-instruct-q4_k_m.gguf"
                )
                cls._instance = cls(model_path=path)
            return cls._instance

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. "
                "Please run scripts/download_model.py first."
            )

        self.model = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=2,
            use_mmap=True,
            verbose=False,
        )
        self.grammar = LlamaGrammar.from_json_schema(
            json.dumps(LOG_CLASSIFICATION_SCHEMA), verbose=False
        )

    def tokenize(self, text: str) -> List[int]:
        """Tokenizes text using the underlying GGUF model's tokenizer."""
        if self.model is None:
            return list(text.encode("utf-8"))
        return self.model.tokenize(text.encode("utf-8"), add_bos=False)

    def classify_chunk(
        self,
        ledger_summary: str,
        compact_lines: str,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Classifies one chunk using schema-constrained, time-bounded inference."""
        model = self.model
        grammar = self.grammar
        if model is None or grammar is None:
            raise ModelExecutionError("LLM model or output grammar is unavailable.")

        system_prompt, user_prompt = build_chunk_prompt(ledger_summary, compact_lines)
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        response = _run_inference(
            lambda stopping_criteria: model(
                prompt,
                max_tokens=400,
                temperature=0.0,
                echo=False,
                grammar=grammar,
                stopping_criteria=stopping_criteria,
            ),
            timeout,
        )
        try:
            result = json.loads(response["choices"][0]["text"].strip())
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ModelExecutionError(f"Model output was not valid JSON: {exc}") from exc
        if not isinstance(result, dict) or not isinstance(result.get("findings"), list):
            raise ModelExecutionError("Model output does not match the findings schema.")
        return result


def _run_inference(
    invoke: Callable[[StoppingCriteriaList], Any], timeout: int
) -> Any:
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    deadline = time.monotonic() + timeout
    finished = threading.Event()
    result_queue: "queue.Queue[tuple[bool, Any]]" = queue.Queue(maxsize=1)

    def has_timed_out(_input_ids: Any, _logits: Any) -> bool:
        del _input_ids, _logits
        return time.monotonic() >= deadline

    def run() -> None:
        acquired = _INFERENCE_LOCK.acquire(blocking=False)
        if not acquired:
            result_queue.put(
                (False, ModelExecutionError("Previous model inference has not stopped."))
            )
            finished.set()
            return
        try:
            response = invoke(StoppingCriteriaList([has_timed_out]))
            if time.monotonic() >= deadline:
                raise ModelExecutionError(f"LLM inference timed out after {timeout} seconds.")
            result_queue.put((True, response))
        except Exception as exc:
            result_queue.put((False, exc))
        finally:
            _INFERENCE_LOCK.release()
            finished.set()

    threading.Thread(target=run, daemon=True).start()
    if not finished.wait(timeout):
        raise ModelExecutionError(f"LLM inference timed out after {timeout} seconds.")

    succeeded, result = result_queue.get_nowait()
    if not succeeded:
        if isinstance(result, ModelExecutionError):
            raise result
        raise ModelExecutionError(f"LLM execution failed: {result}") from result
    return result


def classify_chunk(llm_instance: Llama, prompt: str, timeout: int = 30) -> Dict[str, Any]:
    """Runs a raw completion with a hard caller timeout and validates its JSON."""
    if llm_instance is None:
        raise ModelExecutionError("LLM instance is uninitialized or unavailable.")

    response = _run_inference(
        lambda stopping_criteria: llm_instance(
            prompt,
            max_tokens=512,
            temperature=0.0,
            stop=["</ANALYSIS>"],
            echo=False,
            stopping_criteria=stopping_criteria,
        ),
        timeout,
    )
    try:
        parsed_data = json.loads(response["choices"][0]["text"].strip())
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ModelExecutionError(f"Model output failed JSON parsing: {exc}") from exc
    if not isinstance(parsed_data, dict) or not isinstance(parsed_data.get("findings"), list):
        raise ModelExecutionError("Model output does not match the findings schema.")
    return parsed_data


def get_tokenizer_fn() -> Callable[[str], List[int]]:
    """Returns the model tokenizer or a byte-count fallback before model setup."""
    try:
        manager = LLMManager.get_instance()
    except Exception:
        logger.warning(
            "Model tokenizer unavailable; using a conservative UTF-8 byte budget.",
            exc_info=True,
        )
        return lambda text: list(text.encode("utf-8"))
    return manager.tokenize
