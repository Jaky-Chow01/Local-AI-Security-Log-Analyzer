# AI usage and failure handling

The optional language model runs locally through `llama-cpp-python`. Log lines
are untrusted input and are never sent to a cloud API. The application encloses
log data in per-request nonce markers, instructs the model to treat it as data,
constrains output to the classification JSON schema, and validates every
finding's type, identifier, and evidence against the source chunk.
The model context defaults to 2,048 tokens. Log chunks target at most 1,000
tokens, and each call measures the formatted prompt with the model tokenizer to
ensure prompt plus up to 400 generated tokens stays within the 2,048-token
limit. `LLM_TIMEOUT_SECONDS` configures the inference deadline. A timeout
cannot interrupt llama.cpp during prompt evaluation; the queue waits for the
active inference lock before starting another call.

The model is an assistive classifier, not an authority. Deterministic detectors
run independently, model output may be wrong or incomplete, and an unavailable
model does not mean that a log is safe. Reports expose model status and
coverage gaps; analysts should review the original evidence before responding
to an incident.

## Examples of AI-related mistakes fixed

1. **Inference failures looked like clean results.** The previous chunk
   classifier caught every exception and returned `{"findings": []}`. The queue
   therefore treated failures as successful empty classifications. It now
   raises a model execution error, retries with a timeout, tries a split chunk,
   marks unrecoverable chunks failed, and reports partial coverage.
2. **The untrusted-data nonce was not interpolated.** The system prompt
   previously contained the literal `{nonce}` while log data used the generated
   nonce. The prompt now interpolates the same random value into both boundary
   markers for each chunk.
3. **Repeated log lines were not deduplicated.** Compacted JSON lines included
   their changing `L<number>` prefix in the equality comparison. The chunker now
   compares content without that prefix and retains the original line numbers
   of collapsed entries, so detector evidence and raw-line inspection remain
   traceable.
4. **Finding IDs collided across jobs.** Deterministic detector IDs were based
   on the finding type and line number, while the database required IDs to be
   globally unique. IDs are now prefixed with the job ID so repeat uploads keep
   their findings.
5. **The chunk budget could exceed the per-call token limit.** Chunk size was
   treated as the whole-call budget even though prompts, ledger context, and
   generated output also use tokens. Chunks now target at most 1,000 tokens,
   and the real formatted prompt plus output allowance is checked against the
   2,048-token call limit before inference.

## Tools used

- Python with FastAPI, Pydantic, SQLite, and `llama-cpp-python` for the
  application and local inference.
- Pytest for deterministic unit and regression tests; tests use fake models
  where model weights are not required.
- Browser-native JavaScript and DOM APIs for the local dashboard. No remote
  AI or hosted inference tool is part of the application.
- A local GGUF model is required for end-to-end inference and performance
  benchmarking; that model was not present during the code changes, so no
  successful real-model run is claimed.

## Key coding prompts

The implementation work was guided by these prompt summaries:

1. Make model failures, timeouts, retries, unavailable-model runs, and failed
   chunk ranges explicit instead of treating errors as no findings.
2. Keep log content untrusted, constrain model output to the JSON schema, and
   validate evidence against the source lines.
3. Use audit findings to fix detector, context-ledger, chunking, reporting, and
   dashboard behavior, with regression tests for each changed behavior.
4. Keep model work out of the upload request, enforce upload limits while
   streaming, and validate resource usage with a representative full-size
   benchmark before making performance claims.

## Human review

- Verify findings against their evidence lines and the raw-line viewer.
- Treat `partial` reports, failed ranges, and detector-only reports as
  incomplete analysis, not a clean bill of health.
- Tune detector thresholds for the deployment's normal traffic and retain an
  independent incident-response process.
