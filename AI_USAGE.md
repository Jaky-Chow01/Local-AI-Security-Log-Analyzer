# AI usage and failure handling

The optional language model runs locally through `llama-cpp-python`. Log lines
are untrusted input and are never sent to a cloud API. The application encloses
log data in per-request nonce markers, instructs the model to treat it as data,
constrains output to the classification JSON schema, and validates every
finding's type, identifier, and evidence against the source chunk.

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

## Human review

- Verify findings against their evidence lines and the raw-line viewer.
- Treat `partial` reports, failed ranges, and detector-only reports as
  incomplete analysis, not a clean bill of health.
- Tune detector thresholds for the deployment's normal traffic and retain an
  independent incident-response process.
