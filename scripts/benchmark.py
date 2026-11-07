import os
import time
import tracemalloc
from pathlib import Path
from app.llm import LLMManager

def run_benchmark():
    model_path = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
    if not Path(model_path).exists():
        print(f"[!] Error: Model file not found at {model_path}.")
        print("    Please run 'python scripts/download_model.py' first.")
        return

    print("=== Local AI Security Log Analyzer: Performance Benchmark ===")
    print(f"[*] Target Model: {model_path}")
    print("[*] Target Hardware Budget: 2 vCPU, 4 GB RAM\n")

    # Start memory tracing
    tracemalloc.start()
    start_time = time.time()

    print("[*] Loading model into memory (n_ctx=2048, threads=2)...")
    load_start = time.time()
    try:
        manager = LLMManager.get_instance(model_path=model_path)
    except Exception as e:
        print(f"[!] Model loading failed: {e}")
        return
    load_duration = time.time() - load_start

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"[+] Model loaded in {load_duration:.2f} seconds.")
    print(f"[+] Current Traced RAM: {current_mem / (1024*1024):.2f} MB")
    print(f"[+] Peak Traced RAM: {peak_mem / (1024*1024):.2f} MB\n")

    # Benchmark Prompt Evaluation and Generation Speed
    test_prompt = (
        "System: You are a log classifier. Output JSON.\n"
        "User: Analyze the following log chunk:\n"
        "L1|10:00:00|workspace|POST /api/v1/auth/login|401|203.0.113.44|user=admin|fail\n"
        "L2|10:10:10|workspace|POST /api/v1/auth/login|200|203.0.113.44|user=admin|success"
    )

    print("[*] Running inference speed test...")
    eval_start = time.time()
    
    try:
        response = manager.model.create_chat_completion(
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=150
        }
        eval_duration = time.time() - eval_start
        
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        tokens_per_sec = completion_tokens / eval_duration if eval_duration > 0 else 0

        print(f"[+] Prompt Tokens Evaluated: {prompt_tokens}")
        print(f"[+] Completion Tokens Generated: {completion_tokens}")
        print(f"[+] Generation Time: {eval_duration:.2f} seconds")
        print(f"[+] Generation Speed: {tokens_per_sec:.2f} tokens/sec\n")

    except Exception as e:
        print(f"[!] Inference test failed: {e}")

    print("=== Benchmark Complete ===")
    print("Recommendation: If peak RSS is under 3.2 GB and generation speed > 5 tokens/sec,")
    print("the 3B model is fully viable for your local deployment constraints.")

if __name__ == "__main__":
    run_benchmark()