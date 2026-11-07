import argparse
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.llm import LLMManager

DEFAULT_MODEL = "models/qwen2.5-3b-instruct-q4_k_m.gguf"


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


def run_benchmark(model_path: str) -> int:
    path = Path(model_path)
    if not path.exists():
        print(
            f"Model file not found at {path}. Run scripts/download_model.py first.",
            file=sys.stderr,
        )
        return 2

    print("=== Local AI Security Log Analyzer: Performance Benchmark ===")
    print(f"Target model: {path}")
    print("Target hardware budget: 2 vCPU, 4 GB RAM")
    baseline = get_memory_usage()

    start = time.perf_counter()
    try:
        manager = LLMManager.get_instance(model_path=str(path))
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

    prompt = (
        "<|im_start|>system\nYou are a log classifier. Return a JSON object.\n"
        "<|im_end|>\n<|im_start|>user\nAnalyze these two events:\n"
        "L1|10:00:00|workspace|POST /api/v1/auth/login|401|203.0.113.44|user=admin|fail\n"
        "L2|10:10:10|workspace|POST /api/v1/auth/login|200|203.0.113.44|user=admin|success"
        "\n<|im_end|>\n<|im_start|>assistant\n"
    )
    print("Running inference speed test...")
    eval_start = time.perf_counter()
    try:
        response = manager.model.create_completion(
            prompt=prompt,
            temperature=0.0,
            max_tokens=150,
        )
    except Exception as exc:
        print(f"Inference test failed: {exc}", file=sys.stderr)
        return 1

    eval_duration = time.perf_counter() - eval_start
    usage = response.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    tokens_per_sec = completion_tokens / eval_duration if eval_duration else 0
    memory = get_memory_usage()

    print(f"Prompt tokens evaluated: {prompt_tokens}")
    print(f"Completion tokens generated: {completion_tokens}")
    print(f"Generation time: {eval_duration:.2f} seconds")
    print(f"Generation speed: {tokens_per_sec:.2f} tokens/second")
    print(f"Baseline current RSS: {baseline['current'] / 2**20:.2f} MiB")
    print(f"Current RSS after inference: {memory['current'] / 2**20:.2f} MiB")
    print(f"Peak RSS for process: {memory['peak'] / 2**20:.2f} MiB")
    if memory["peak"] > 4 * 1024**3:
        print("Result: peak RSS exceeded the 4 GiB budget.")
        return 1
    print("Result: peak RSS stayed within the 4 GiB budget.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    args = parser.parse_args()
    raise SystemExit(run_benchmark(args.model_path))
