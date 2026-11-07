import os
import hashlib
from pathlib import Path
from huggingface_hub import hf_hub_download

MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
MODEL_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
MODELS_DIR = Path("models")
SHA_PATH = MODELS_DIR / "MODEL.sha256"

def download_and_verify_model():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = MODELS_DIR / MODEL_FILENAME

    print(f"[*] Downloading {MODEL_FILENAME} from Hugging Face ({MODEL_REPO})...")
    print("    This may take a few minutes depending on your connection speed.")

    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        local_dir=str(MODELS_DIR),
    )

    print(f"[+] Model downloaded successfully to: {downloaded_path}")

    # Compute SHA-256 Checksum
    print("[*] Computing SHA-256 checksum...")
    sha256_hash = hashlib.sha256()
    with open(target_path, "rb") as f:
        for byte_block in iter(lambda: f.read(1024 * 1024), b""):
            sha256_hash.update(byte_block)
    
    digest = sha256_hash.hexdigest()
    with open(SHA_PATH, "w", encoding="utf-8") as f:
        f.write(f"{digest}  {MODEL_FILENAME}\n")

    print(f"[+] Locally computed SHA-256 saved to {SHA_PATH}")
    print(f"    SHA-256: {digest}")
    print("    This records the downloaded file's digest; it does not authenticate the publisher.")

if __name__ == "__main__":
    download_and_verify_model()