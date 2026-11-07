import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.llm import (
    LLMManager,
    MAX_INFERENCE_TOKENS,
    MAX_OUTPUT_TOKENS,
    _run_inference,
    get_inference_timeout_seconds,
)
from app.prompts import build_chunk_prompt

DEFAULT_MODEL = "models/qwen2.5-3b-instruct-q4_k_m.gguf"


def build_benchmark_chunk(manager: LLMManager, target_tokens: int) -> tuple[str, int]:
    if manager.model is None:
        raise RuntimeError("The benchmark model is not loaded.")
    unit = (
        "L1|10:00:00|workspace|GET /api/v1/status|200|192.0.2.10|"
        "message=normal health check "
    )
    unit_tokens = manager.tokenize(unit)
    if not unit_tokens:
        raise RuntimeError("The model tokenizer returned no tokens for benchmark data.")
    token_ids = (unit_tokens * ((target_tokens // len(unit_tokens)) + 2))[
        :target_tokens
    ]
    chunk_text = manager.model.detokenize(token_ids).decode("utf-8", errors="replace")
    actual_tokens = len(manager.tokenize(chunk_text))
    if actual_tokens < target_tokens - 8:
        raise RuntimeError(
            f"Could only synthesize {actual_tokens} of {target_tokens} requested chunk tokens."
        )
    return chunk_text, actual_tokens


def get_memory_usage() -> Dict[str, int]:
    """Returns current and peak resident memory in bytes, including native allocations."""
    if platform.system() == "Windows":
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
        process = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        ):
            raise OSError("GetProcessMemoryInfo failed.")
        return {
            "current": counters.WorkingSetSize,
            "peak": counters.PeakWorkingSetSize,
        }

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
    current_bytes = peak_bytes
    statm_path = Path("/proc/self/statm")
    if statm_path.exists():
        resident_pages = int(statm_path.read_text(encoding="ascii").split()[1])
        current_bytes = resident_pages * os.sysconf("SC_PAGE_SIZE")
    return {"current": current_bytes, "peak": peak_bytes}


def run_benchmark(
    model_path: str, chunk_tokens: int, context_size: int, timeout: int
) -> int:
    if chunk_tokens < 1 or context_size < 1024 or timeout < 1:
        print(
            "Chunk tokens, context size (at least 1024), and timeout must be positive.",
            file=sys.stderr,
        )
        return 2
    path = Path(model_path)
    if not path.exists():
        print(
            f"Model file not found at {path}. Run scripts/download_model.py first.",
            file=sys.stderr,
        )
        return 2

    print("=== Local AI Security Log Analyzer: Performance Benchmark ===")
    print(f"Target model: {path}")
    print(f"Target hardware budget: 2 vCPU, 4 GiB RAM (container limits required)")
    print(f"Model context: {context_size} tokens")
    print(f"Chunk target: {chunk_tokens} tokens")
    print(f"Inference timeout: {timeout} seconds")
    baseline = get_memory_usage()

    start = time.perf_counter()
    try:
        manager = LLMManager.get_instance(model_path=str(path), n_ctx=context_size)
    except Exception as exc:
        print(f"Model loading failed: {exc}", file=sys.stderr)
        return 1
    load_duration = time.perf_counter() - start
    if manager.model is None:
        print("Model manager initialized without a loaded model.", file=sys.stderr)
        return 1

    after_load = get_memory_usage()
    print(f"Model load time: {load_duration:.2f} seconds")
    print(f"Current RSS after load: {after_load['current'] / 2**20:.2f} MiB")
    print(f"Peak RSS after load: {after_load['peak'] / 2**20:.2f} MiB")

    if manager.grammar is None:
        print("Model JSON grammar was not loaded.", file=sys.stderr)
        return 1
    try:
        compact_lines, actual_chunk_tokens = build_benchmark_chunk(
            manager, chunk_tokens
        )
    except Exception as exc:
        print(f"Could not build benchmark chunk: {exc}", file=sys.stderr)
        return 1
    system_prompt, user_prompt = build_chunk_prompt("", compact_lines)
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    prompt_tokens = len(manager.tokenize(prompt))
    call_token_limit = min(context_size, MAX_INFERENCE_TOKENS)
    if prompt_tokens + MAX_OUTPUT_TOKENS > call_token_limit:
        print(
            f"Full prompt needs {prompt_tokens} input tokens plus "
            f"{MAX_OUTPUT_TOKENS} output tokens, exceeding the "
            f"{call_token_limit}-token inference-call limit.",
            file=sys.stderr,
        )
        return 1

    print(f"Actual compacted chunk tokens: {actual_chunk_tokens}")
    print(f"Full formatted prompt tokens: {prompt_tokens}")
    print(f"Total call token limit: {call_token_limit}")
    print(f"Running schema-constrained inference (up to {MAX_OUTPUT_TOKENS} output tokens)...")
    eval_start = time.perf_counter()
    try:
        response = _run_inference(
            lambda stopping_criteria: manager.model(
                prompt,
                temperature=0.0,
                max_tokens=MAX_OUTPUT_TOKENS,
                grammar=manager.grammar,
                stopping_criteria=stopping_criteria,
            ),
            timeout,
        )
    except Exception as exc:
        print(f"Inference test failed: {exc}", file=sys.stderr)
        return 1

    eval_duration = time.perf_counter() - eval_start
    usage = response.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    tokens_per_sec = completion_tokens / eval_duration if eval_duration else 0
    memory = get_memory_usage()
    try:
        model_output = json.loads(response["choices"][0]["text"].strip())
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        print(f"Model output was not valid JSON: {exc}", file=sys.stderr)
        return 1
    valid_output = isinstance(model_output, dict) and isinstance(
        model_output.get("findings"), list
    )

    print(f"Prompt tokens evaluated: {prompt_tokens}")
    print(f"Completion tokens generated: {completion_tokens}")
    print(f"Generation time: {eval_duration:.2f} seconds")
    print(f"Generation speed: {tokens_per_sec:.2f} tokens/second")
    print(f"Baseline current RSS: {baseline['current'] / 2**20:.2f} MiB")
    print(f"Current RSS after inference: {memory['current'] / 2**20:.2f} MiB")
    print(f"Peak RSS for process: {memory['peak'] / 2**20:.2f} MiB")
    print(f"Model result shape valid: {valid_output}")
    if not valid_output:
        print("Model result did not match the findings response shape.", file=sys.stderr)
        return 1
    if memory["peak"] > 4 * 1024**3:
        print("Result: peak RSS exceeded the 4 GiB budget.")
        return 1
    print("Result: peak RSS stayed within the 4 GiB budget.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-tokens", type=int, default=1000)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument(
        "--timeout-seconds", type=int, default=get_inference_timeout_seconds()
    )
    args = parser.parse_args()
    raise SystemExit(
        run_benchmark(
            args.model_path,
            args.chunk_tokens,
            args.context_size,
            args.timeout_seconds,
        )
    )
