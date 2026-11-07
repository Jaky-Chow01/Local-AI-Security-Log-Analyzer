# Local AI Security Log Analyzer

An offline-first log analyzer that combines deterministic security detectors with
an optional local Qwen 2.5 3B model. Uploaded log content is processed locally;
the application does not send logs to a hosted AI service.

## Requirements

- Python 3.10 or newer
- 2 vCPU and 4 GB RAM recommended
- The Qwen2.5-3B-Instruct Q4_K_M GGUF model for AI-assisted classification

## Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/download_model.py
```

The model download is a setup-time network operation. Check the downloaded
model against the published checksum with `scripts/verify_offline.sh` in an
environment that provides a compatible shell. After setup, the analyzer runs
without a cloud connection.

To select another local GGUF file, set `MODEL_PATH` before starting the server.
If the model is missing or cannot load, deterministic detectors still analyze
the upload and the job/report explicitly identifies the run as detector-only.
Model failures, timeouts, and incomplete chunk ranges are reported as partial
coverage rather than as an empty successful AI result.

## Run

```powershell
python run.py
```

Open `http://127.0.0.1:8000`. The server binds to loopback by default. Upload
plain-text `.log` or `.txt` files up to 2 MB and 20,000 lines. A report is
available in the dashboard and is also written to `app/reports/`; job state is
stored in `app/db.sqlite`.

## Analysis behavior

- Input is split on whole log lines into token-budgeted chunks with up to two
  lines of context overlap. Long individual lines are truncated to fit and are
  listed in report coverage.
- Deterministic detectors run before local model classification. They identify
  common web, authentication, rate-abuse, and suspicious-input patterns.
- The model is called sequentially with a JSON-schema grammar, bounded retries,
  per-call timeout, and one split-chunk retry when a chunk fails.
- Findings are checked against the originating chunk's evidence line numbers
  and text before persistence. Invalid and benign model outputs are excluded
  and counted in report coverage.
- Reports correlate findings and expose failed ranges, truncated lines, and
  rejected model findings. A `partial` status means some analysis did not
  complete; it must not be interpreted as clean.

## Benchmark

Run the benchmark after downloading the model:

```powershell
python scripts/benchmark.py
```

It reports native process RSS, inference speed, and whether peak RSS remained
within 4 GiB. To enforce both the memory and CPU budget, run it inside a
container image that has this repository's Python dependencies installed and
the model mounted:

```sh
docker run --rm --memory=4g --cpus=2 \
  -v "$PWD:/workspace" -w /workspace \
  local-ai-security-log-analyzer:latest \
  python scripts/benchmark.py
```

The repository does not include a container image; `local-ai-security-log-analyzer:latest`
is an example image name. A benchmark result only applies to the hardware,
runtime, model, and container limits used for that run.

## Tests

```powershell
python -m pytest
```

## Limitations

Deterministic patterns and model classifications are heuristics, not proof of
malicious activity. False positives and missed detections are possible. Review
evidence and raw log lines before taking action. Analysis completeness is
reported separately from risk level.
