import json
import logging
import os
import sys
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
DEFAULT_MODEL_PATH = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
MAX_INFERENCE_TOKENS = 2048
MAX_OUTPUT_TOKENS = 400
_METRICS_LOCK = threading.Lock()
_LAST_INFERENCE_SECONDS: Optional[float] = None


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
                path = model_path or cls.get_configured_model_path()
                cls._instance = cls(
                    model_path=path,
                    n_ctx=n_ctx if n_ctx is not None else get_model_context_size(),
                )
            return cls._instance

    @classmethod
    def get_configured_model_path(cls) -> str:
        """Returns the active model path or its configured default without loading it."""
        manager = cls._instance
        if manager is not None:
            return manager.model_path
        return os.getenv("MODEL_PATH") or DEFAULT_MODEL_PATH

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
            model_text = response["choices"][0]["text"]
            if not isinstance(model_text, str):
                raise TypeError("Model response text is not a string.")
            result = _parse_model_response(model_text)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ModelExecutionError(f"Model output was not valid JSON: {exc}") from exc
        if not isinstance(result, dict) or not isinstance(result.get("findings"), list):
            raise ModelExecutionError("Model output does not match the findings schema.")
        return result


def _parse_model_response(text: str) -> Dict[str, Any]:
    """Recovers a valid schema-shaped JSON object from otherwise noisy output."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.removeprefix("```json").removeprefix("```").strip()
        if candidate.endswith("```"):
            candidate = candidate[:-3].strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as original_error:
        parsed = None
        decoder = json.JSONDecoder()
        for offset, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                possible, _ = decoder.raw_decode(candidate, offset)
            except json.JSONDecodeError:
                continue
            if _matches_model_schema(possible):
                parsed = possible
                break
        if parsed is None:
            raise original_error
    if not _matches_model_schema(parsed):
        raise json.JSONDecodeError("Response does not match the findings schema", text, 0)
    return parsed


def _matches_model_schema(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"findings"}:
        return False
    findings = value["findings"]
    if not isinstance(findings, list):
        return False

    properties = LOG_CLASSIFICATION_SCHEMA["properties"]["findings"]["items"]["properties"]
    expected_fields = set(properties)
    for finding in findings:
        if not isinstance(finding, dict) or set(finding) != expected_fields:
            return False
        if (
            finding["attack_type"] not in properties["attack_type"]["enum"]
            or finding["severity"] not in properties["severity"]["enum"]
            or not isinstance(finding["identifier"], str)
            or not isinstance(finding["evidence_lines"], list)
            or any(
                not isinstance(line_no, int) or isinstance(line_no, bool)
                for line_no in finding["evidence_lines"]
            )
            or not isinstance(finding["reasoning"], str)
            or not isinstance(finding["suggested_actions"], list)
            or any(not isinstance(action, str) for action in finding["suggested_actions"])
            or not isinstance(finding["confidence"], (int, float))
            or isinstance(finding["confidence"], bool)
        ):
            return False
    return True


def get_runtime_metrics() -> Dict[str, Any]:
    """Returns lightweight process RSS and most recent native inference duration."""
    memory_mb, memory_kind = _get_process_memory_mb()
    with _METRICS_LOCK:
        inference_seconds = _LAST_INFERENCE_SECONDS
    return {
        "memory_mb": memory_mb,
        "memory_kind": memory_kind,
        "last_inference_seconds": inference_seconds,
    }


def _get_process_memory_mb() -> tuple[Optional[float], str]:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise OSError("GetProcessMemoryInfo failed.")
        return counters.WorkingSetSize / (1024 * 1024), "rss"

    statm_path = "/proc/self/statm"
    if os.path.exists(statm_path):
        with open(statm_path, encoding="ascii") as statm:
            resident_pages = int(statm.read().split()[1])
        return (
            resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024),
            "rss",
        )

    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = peak if sys.platform == "darwin" else peak * 1024
        return peak_bytes / (1024 * 1024), "peak-rss-estimate"
    except (ImportError, OSError):
        return None, "unavailable"


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
        global _LAST_INFERENCE_SECONDS
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
            inference_started = time.monotonic()
            try:
                response = invoke(StoppingCriteriaList([has_timed_out]))
            finally:
                with _METRICS_LOCK:
                    _LAST_INFERENCE_SECONDS = time.monotonic() - inference_started
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
