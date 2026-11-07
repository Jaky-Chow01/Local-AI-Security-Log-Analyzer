#!/usr/bin/env bash
set -e

echo "[*] Configuring environment for strict offline execution..."

# Force Hugging Face and Transformers libraries into offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Verify model file and checksum existence
if [ ! -f "models/qwen2.5-3b-instruct-q4_k_m.gguf" ]; then
    echo "[!] ERROR: Model file missing in models/ directory."
    echo "    Please run 'python scripts/download_model.py' with internet enabled first."
    exit 1
fi

if [ ! -f "models/MODEL.sha256" ]; then
    echo "[!] WARNING: Checksum file MODEL.sha256 missing."
else
    echo "[*] Verifying model SHA-256 checksum..."
    sha256sum -c models/MODEL.sha256 || { echo "[!] Checksum verification failed!"; exit 1; }
    echo "[+] Checksum verified successfully."
fi

echo "[+] Offline environment variables configured:"
echo "    HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "    TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "[*] Ready to launch server locally with: uvicorn app.main:app --host 127.0.0.1 --port 8000"