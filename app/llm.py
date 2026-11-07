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


class InferenceTimeoutError(ModelExecutionError):
    """Raised when inference exceeds its caller deadline."""


class ModelBusyError(ModelExecutionError):
    """Raised when an earlier inference still owns the model."""


_INFERENCE_LOCK = threading.Lock()
DEFAULT_INFERENCE_TIMEOUT_SECONDS = 120
DEFAULT_MODEL_CONTEXT_SIZE = 2048
MAX_INFERENCE_TOKENS = 2048
MAX_OUTPUT_TOKENS = 400


def get_inference_timeout_seconds() -> int:
    """Reads the configured inference deadline, defaulting to a conservative 120 seconds."""
    raw_timeout = os.getenv("LLM_TIMEOUT_SECONDS")
    if raw_timeout is None:
        return DEFAULT_INFERENCE_TIMEOUT_SECONDS
    try:
        timeout = int(raw_timeout)
    except ValueError as exc:
        raise ValueError("LLM_TIMEOUT_SECONDS must be a positive integer.") from exc
    if timeout <= 0:
        raise ValueError("LLM_TIMEOUT_SECONDS must be a positive integer.")
    return timeout


def get_model_context_size() -> int:
    """Reads the model context size from the environment."""
    raw_context = os.getenv("MODEL_CONTEXT_SIZE")
    if raw_context is None:
        return DEFAULT_MODEL_CONTEXT_SIZE
    try:
        context_size = int(raw_context)
    except ValueError as exc:
        raise ValueError("MODEL_CONTEXT_SIZE must be a positive integer.") from exc
    if context_size < 1024:
        raise ValueError("MODEL_CONTEXT_SIZE must be at least 1024.")
    return context_size


class LLMManager:
    _instance: Optional["LLMManager"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_path: str = "models/qwen2.5-3b-instruct-q4_k_m.gguf",
        n_ctx: int = DEFAULT_MODEL_CONTEXT_SIZE,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.model: Optional[Llama] = None
        self.grammar: Optional[LlamaGrammar] = None
        self._load_model()

    @classmethod
    def get_instance(
        cls, model_path: Optional[str] = None, n_ctx: Optional[int] = None
    ) -> "LLMManager":
        with cls._lock:
            if cls._instance is None:
                path = model_path or os.getenv(
                    "MODEL_PATH", "models/qwen2.5-3b-instruct-q4_k_m.gguf"
                )
                cls._instance = cls(
                    model_path=path,
                    n_ctx=n_ctx if n_ctx is not None else get_model_context_size(),
                )
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
        timeout: Optional[float] = None,
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
        prompt_tokens = len(model.tokenize(prompt.encode("utf-8"), add_bos=False))
        token_limit = min(self.n_ctx, MAX_INFERENCE_TOKENS)
        if prompt_tokens + MAX_OUTPUT_TOKENS > token_limit:
            raise ModelExecutionError(
                f"LLM call requires {prompt_tokens} prompt tokens plus "
                f"{MAX_OUTPUT_TOKENS} output tokens, exceeding the "
                f"{token_limit}-token limit."
            )
        response = _run_inference(
            lambda stopping_criteria: model(
                prompt,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.0,
                echo=False,
                grammar=grammar,
                stopping_criteria=stopping_criteria,
            ),
            timeout if timeout is not None else get_inference_timeout_seconds(),
        )
        try:
            result = json.loads(response["choices"][0]["text"].strip())
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ModelExecutionError(f"Model output was not valid JSON: {exc}") from exc
        if not isinstance(result, dict) or not isinstance(result.get("findings"), list):
            raise ModelExecutionError("Model output does not match the findings schema.")
        return result


def _run_inference(
    invoke: Callable[[StoppingCriteriaList], Any], timeout: float
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
        remaining = max(0.0, deadline - time.monotonic())
        acquired = _INFERENCE_LOCK.acquire(timeout=remaining)
        if not acquired:
            result_queue.put(
                (
                    False,
                    InferenceTimeoutError(
                        f"Timed out waiting for the active model inference after "
                        f"{timeout} seconds."
                    ),
                )
            )
            finished.set()
            return
        try:
            if time.monotonic() >= deadline:
                result_queue.put(
                    (
                        False,
                        InferenceTimeoutError(
                            f"Timed out waiting for the active model inference after "
                            f"{timeout} seconds."
                        ),
                    )
                )
                return
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
        raise InferenceTimeoutError(
            f"LLM inference exceeded {timeout} seconds. "
            "The llama.cpp call may continue until prompt evaluation or generation returns."
        )

    succeeded, result = result_queue.get_nowait()
    if not succeeded:
        if isinstance(result, ModelExecutionError):
            raise result
        raise ModelExecutionError(f"LLM execution failed: {result}") from result
    return result


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
