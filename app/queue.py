import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.chunker import chunk_log_file
from app.db import get_db_connection, save_job_chunks
from app.detectors import run_deterministic_detectors
from app.llm import (
    InferenceTimeoutError,
    LLMManager,
    MAX_INFERENCE_TOKENS,
    ModelBusyError,
    get_tokenizer_fn,
    get_inference_timeout_seconds,
    get_model_context_size,
)
from app.prompts import LOG_CLASSIFICATION_SCHEMA

logger = logging.getLogger(__name__)
MAX_TOKENS_PER_CHUNK = 1000
MODEL_PROMPT_RESERVE_TOKENS = 600


class JobQueueManager:
    _instance: Optional["JobQueueManager"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.job_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self.cancel_flags: Dict[str, threading.Event] = {}
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.recover_pending_jobs()

    @classmethod
    def get_instance(cls) -> "JobQueueManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def enqueue_job(self, job_id: str, file_path: str) -> None:
        """Enqueues a job for background processing."""
        self.cancel_flags[job_id] = threading.Event()
        self.job_queue.put((job_id, file_path))

    def recover_pending_jobs(self) -> None:
        """Re-enqueues persisted partial jobs that have work left after a restart."""
        with get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT j.job_id, j.filename, j.model_name, j.model_status,
                    j.status, j.total_chunks,
                    SUM(CASE WHEN c.status = 'pending' THEN 1 ELSE 0 END) AS pending_count,
                    SUM(CASE WHEN c.status = 'failed' THEN 1 ELSE 0 END) AS failed_count
                FROM jobs AS j
                LEFT JOIN chunks AS c ON c.job_id = j.job_id
                WHERE j.status IN ('queued', 'partial')
                GROUP BY j.job_id
                """
            ).fetchall()
        for row in rows:
            filename = Path(row["filename"]).name
            file_path = Path("uploads") / f"{row['job_id']}_{filename}"
            needs_chunking = row["total_chunks"] == 0
            if not row["pending_count"] and not needs_chunking:
                recovered_status = "partial" if row["failed_count"] else "complete"
                recovered_model_status = row["model_status"]
                if recovered_model_status == "interrupted":
                    recovered_model_status = (
                        "degraded" if row["model_name"] else "unavailable"
                    )
                with get_db_connection() as conn:
                    conn.execute(
                        """
                        UPDATE jobs SET status = ?, model_status = ?,
                            updated_at = CURRENT_TIMESTAMP WHERE job_id = ?;
                        """,
                        (recovered_status, recovered_model_status, row["job_id"]),
                    )
                    conn.commit()
                continue

            if file_path.exists() or row["total_chunks"] > 0:
                self.enqueue_job(row["job_id"], str(file_path))
            else:
                logger.error(
                    "Cannot resume job %s: persisted upload is missing at %s",
                    row["job_id"],
                    file_path,
                )
                with get_db_connection() as conn:
                    conn.execute(
                        """
                        UPDATE jobs SET status = 'failed',
                            updated_at = CURRENT_TIMESTAMP WHERE job_id = ?;
                        """,
                        (row["job_id"],),
                    )
                    conn.commit()

    def cancel_job(self, job_id: str) -> None:
        """Triggers cooperative cancellation for a running job."""
        event = self.cancel_flags.get(job_id)
        if event is not None:
            event.set()

    def _worker_loop(self) -> None:
        """Single-flight background worker thread loop."""
        while True:
            try:
                job_id, file_path = self.job_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._process_job(job_id, file_path)
            except Exception:
                logger.exception("Unexpected failure processing job %s", job_id)
                with get_db_connection() as conn:
                    conn.execute(
                        "UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE job_id = ?;",
                        ("failed", job_id),
                    )
                    conn.commit()
            finally:
                self.job_queue.task_done()
                self.cancel_flags.pop(job_id, None)

    def _process_job(self, job_id: str, file_path: Optional[str] = None) -> None:
        """Runs deterministic detection and model analysis, preserving partial work."""
        cancel_event = self.cancel_flags.get(job_id)

        with get_db_connection() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE job_id = ?;",
                ("running", job_id),
            )
            conn.commit()
            all_chunk_rows = conn.execute(
                "SELECT * FROM chunks WHERE job_id = ? ORDER BY chunk_index ASC;",
                (job_id,),
            ).fetchall()

        if not all_chunk_rows and file_path is not None:
            chunks = chunk_log_file(
                file_path,
                get_tokenizer_fn(),
                max_tokens_per_chunk=min(
                    MAX_TOKENS_PER_CHUNK,
                    min(get_model_context_size(), MAX_INFERENCE_TOKENS)
                    - MODEL_PROMPT_RESERVE_TOKENS,
                ),
            )
            save_job_chunks(job_id, chunks)
            with get_db_connection() as conn:
                all_chunk_rows = conn.execute(
                    "SELECT * FROM chunks WHERE job_id = ? ORDER BY chunk_index ASC;",
                    (job_id,),
                ).fetchall()

        with get_db_connection() as conn:
            pending_chunk_rows = [
                row for row in all_chunk_rows if row["status"] == "pending"
            ]
            all_lines = []
            for row in all_chunk_rows:
                all_lines.extend(json.loads(row["compacted_data"]))

        deterministic_findings = run_deterministic_detectors(all_lines)
        self._persist_deterministic_findings(job_id, deterministic_findings)

        if pending_chunk_rows:
            self._rebuild_ledger(
                job_id, through_line=pending_chunk_rows[0]["start_line"] - 1
            )
        elif not all_chunk_rows:
            with get_db_connection() as conn:
                conn.execute(
                    """
                    UPDATE jobs SET status = 'failed', model_status = 'unavailable',
                        updated_at = CURRENT_TIMESTAMP WHERE job_id = ?;
                    """,
                    (job_id,),
                )
                conn.commit()
            return
        elif not any(row["status"] in {"running", "retrying"} for row in all_chunk_rows):
            final_status = (
                "partial"
                if any(row["status"] == "failed" for row in all_chunk_rows)
                else "complete"
            )
            with get_db_connection() as conn:
                conn.execute(
                    """
                    UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = ?;
                    """,
                    (final_status, job_id),
                )
                conn.commit()
            return

        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE jobs SET model_status = 'loading',
                    updated_at = CURRENT_TIMESTAMP WHERE job_id = ?;
                """,
                (job_id,),
            )
            conn.commit()

        try:
            llm_manager = LLMManager.get_instance()
        except Exception:
            logger.exception("Model unavailable for job %s", job_id)
            self._mark_model_unavailable(job_id, pending_chunk_rows)
            return

        model_name = Path(llm_manager.model_path).stem
        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE jobs SET model_status = 'available', model_name = ?,
                    updated_at = CURRENT_TIMESTAMP WHERE job_id = ?;
                """,
                (model_name, job_id),
            )
            conn.commit()

        for chunk_row in pending_chunk_rows:
            if cancel_event is not None and cancel_event.is_set():
                break

            chunk_id = chunk_row["chunk_id"]
            lines = json.loads(chunk_row["compacted_data"])
            model_findings, inference_complete = self._classify_with_retries(
                job_id, chunk_id, llm_manager, lines, cancel_event
            )
            rejected_count = self._persist_model_findings(
                job_id, chunk_id, lines, model_findings
            )

            chunk_status = "done" if inference_complete else "failed"
            with get_db_connection() as conn:
                conn.execute(
                    "UPDATE chunks SET status = ? WHERE chunk_id = ?;",
                    (chunk_status, chunk_id),
                )
                conn.execute(
                    """
                    UPDATE jobs
                    SET completed_chunks = (
                            SELECT COUNT(*) FROM chunks
                            WHERE job_id = ? AND status IN ('done', 'failed')
                        ),
                        rejected_model_findings = rejected_model_findings + ?,
                        model_status = CASE WHEN ? = 1 THEN model_status ELSE 'degraded' END,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = ?;
                    """,
                    (job_id, rejected_count, int(inference_complete), job_id),
                )
                conn.commit()
            self._rebuild_ledger(job_id, through_line=chunk_row["end_line"])

        with get_db_connection() as conn:
            states = conn.execute(
                "SELECT status, COUNT(*) AS count FROM chunks WHERE job_id = ? GROUP BY status;",
                (job_id,),
            ).fetchall()
            status_counts = {row["status"]: row["count"] for row in states}
            has_incomplete_chunks = any(
                status_counts.get(status, 0) > 0
                for status in ("pending", "running", "retrying", "failed")
            )
            final_status = (
                "partial"
                if has_incomplete_chunks
                or (cancel_event is not None and cancel_event.is_set())
                else "complete"
            )
            conn.execute(
                "UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE job_id = ?;",
                (final_status, job_id),
            )
            conn.commit()

    def _mark_model_unavailable(self, job_id: str, pending_chunks: List[Any]) -> None:
        with get_db_connection() as conn:
            for chunk in pending_chunks:
                conn.execute(
                    "UPDATE chunks SET status = 'failed' WHERE chunk_id = ?;",
                    (chunk["chunk_id"],),
                )
            conn.execute(
                """
                UPDATE jobs
                SET status = 'partial', model_status = 'unavailable',
                    model_name = NULL,
                    completed_chunks = (
                        SELECT COUNT(*) FROM chunks
                        WHERE job_id = ? AND status IN ('done', 'failed')
                    ),
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?;
                """,
                (job_id, job_id),
            )
            conn.commit()
        self._rebuild_ledger(job_id)

    def _classify_with_retries(
        self,
        job_id: str,
        chunk_id: str,
        manager: LLMManager,
        lines: List[Dict[str, Any]],
        cancel_event: Optional[threading.Event],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        inference_timeout = get_inference_timeout_seconds()
        tokenize = getattr(manager, "tokenize", None)

        def classify_segment(segment: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            compact_text = "\n".join(line["compacted"] for line in segment)
            last_error: Optional[Exception] = None
            for attempt in range(3):
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Job was cancelled.")
                if attempt:
                    with get_db_connection() as conn:
                        conn.execute(
                            "UPDATE chunks SET status = 'retrying' WHERE chunk_id = ?;",
                            (chunk_id,),
                        )
                        conn.commit()
                    if cancel_event is not None:
                        if cancel_event.wait(float(attempt)):
                            raise InterruptedError("Job was cancelled.")
                    else:
                        time.sleep(float(attempt))
                try:
                    return manager.classify_chunk(
                        self._get_ledger_summary(job_id, tokenize),
                        compact_text,
                        timeout=inference_timeout,
                    ).get("findings", [])
                except (InferenceTimeoutError, ModelBusyError):
                    raise
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "Model attempt %s failed for chunk %s: %s",
                        attempt + 1,
                        chunk_id,
                        exc,
                    )
            raise RuntimeError(f"Model failed after retries: {last_error}") from last_error

        try:
            return classify_segment(lines), True
        except InterruptedError:
            return [], False
        except (InferenceTimeoutError, ModelBusyError) as exc:
            logger.error("Model analysis stopped for chunk %s: %s", chunk_id, exc)
            return [], False
        except Exception as first_error:
            if len(lines) < 2:
                logger.error("Chunk %s failed model analysis: %s", chunk_id, first_error)
                return [], False

            midpoint = len(lines) // 2
            partial_findings: List[Dict[str, Any]] = []
            split_complete = True
            for segment in (lines[:midpoint], lines[midpoint:]):
                try:
                    partial_findings.extend(classify_segment(segment))
                except (InferenceTimeoutError, ModelBusyError) as exc:
                    logger.error("Split retry stopped for chunk %s: %s", chunk_id, exc)
                    split_complete = False
                    break
                except InterruptedError:
                    split_complete = False
                    break
                except Exception as exc:
                    logger.error("Split retry failed for chunk %s: %s", chunk_id, exc)
                    split_complete = False
            return partial_findings, split_complete

    def _persist_model_findings(
        self,
        job_id: str,
        chunk_id: str,
        lines: List[Dict[str, Any]],
        model_findings: List[Dict[str, Any]],
    ) -> int:
        valid_line_numbers = {
            line_no
            for line in lines
            for line_no in line.get("repeat_line_numbers", [line["line_no"]])
        }
        attack_types = set(LOG_CLASSIFICATION_SCHEMA["properties"]["findings"]["items"]["properties"]["attack_type"]["enum"])
        severities = set(LOG_CLASSIFICATION_SCHEMA["properties"]["findings"]["items"]["properties"]["severity"]["enum"])
        line_text = "\n".join(
            f"{line.get('raw', '')} {line.get('compacted', '')}" for line in lines
        )
        rejected_count = 0
        accepted = []

        for finding in model_findings:
            if not isinstance(finding, dict):
                rejected_count += 1
                continue
            attack_type = finding.get("attack_type")
            severity = finding.get("severity")
            evidence = finding.get("evidence_lines")
            identifier = finding.get("identifier")
            confidence = finding.get("confidence")
            valid = (
                isinstance(attack_type, str)
                and attack_type in attack_types
                and isinstance(severity, str)
                and severity in severities
                and isinstance(identifier, str)
                and bool(identifier.strip())
                and identifier.casefold() in line_text.casefold()
                and isinstance(evidence, list)
                and bool(evidence)
                and all(
                    isinstance(number, int)
                    and not isinstance(number, bool)
                    and number in valid_line_numbers
                    for number in evidence
                )
                and isinstance(finding.get("reasoning"), str)
                and bool(finding["reasoning"].strip())
                and isinstance(finding.get("suggested_actions"), list)
                and all(
                    isinstance(action, str)
                    for action in finding["suggested_actions"]
                )
                and isinstance(confidence, (int, float))
                and not isinstance(confidence, bool)
                and 0 <= confidence <= 1
            )
            if not valid or attack_type == "benign":
                rejected_count += 1
                continue
            accepted.append(finding)

        with get_db_connection() as conn:
            for index, finding in enumerate(accepted):
                finding_id = f"{job_id}:LLM-{chunk_id[:8]}-{index}"
                conn.execute(
                    """
                    INSERT OR IGNORE INTO findings
                    (id, job_id, attack_type, severity, identifier, evidence_lines,
                     reasoning, suggested_actions, confidence, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        finding_id,
                        job_id,
                        finding["attack_type"],
                        finding["severity"],
                        finding["identifier"],
                        json.dumps(finding["evidence_lines"]),
                        finding["reasoning"],
                        json.dumps(finding["suggested_actions"]),
                        float(finding["confidence"]),
                        "model",
                    ),
                )
            conn.commit()
        return rejected_count

    def _persist_deterministic_findings(
        self, job_id: str, findings: List[Dict[str, Any]]
    ) -> None:
        with get_db_connection() as conn:
            for finding in findings:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO findings
                    (id, job_id, attack_type, severity, identifier, evidence_lines,
                     reasoning, suggested_actions, confidence, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        f"{job_id}:{finding['id']}",
                        job_id,
                        finding["attack_type"],
                        finding["severity"],
                        finding["identifier"],
                        json.dumps(finding["evidence_lines"]),
                        finding["reasoning"],
                        json.dumps(finding["suggested_actions"]),
                        1.0,
                        finding["source"],
                    ),
                )
            conn.commit()

    def _get_ledger_summary(
        self, job_id: str, tokenize: Optional[Any] = None, max_tokens: int = 200
    ) -> str:
        """Returns a tokenizer-bounded summary of recent suspicious entities."""
        with get_db_connection() as conn:
            rows = conn.execute(
                """
                SELECT entity_key, suspicion_score, summary_text
                FROM ledger
                WHERE job_id = ? AND suspicion_score > 0
                ORDER BY suspicion_score DESC, last_updated_line DESC
                LIMIT 5;
                """,
                (job_id,),
            ).fetchall()
        entries = [
            f"- {row['entity_key'][:64]} (score {row['suspicion_score']:.1f}): "
            f"{row['summary_text'][:240]}"
            for row in rows
        ]
        while entries:
            summary = "\n".join(entries)
            token_count = (
                len(tokenize(summary))
                if tokenize is not None
                else len(summary.encode("utf-8"))
            )
            if token_count <= max_tokens:
                return summary
            entries.pop()
        return ""

    def _rebuild_ledger(
        self, job_id: str, through_line: Optional[int] = None
    ) -> None:
        with get_db_connection() as conn:
            chunk_rows = conn.execute(
                "SELECT compacted_data FROM chunks WHERE job_id = ?;",
                (job_id,),
            ).fetchall()
            finding_rows = conn.execute(
                """
                SELECT identifier, attack_type, severity, evidence_lines
                FROM findings WHERE job_id = ?;
                """,
                (job_id,),
            ).fetchall()

            entities: Dict[str, Dict[str, Any]] = {}
            seen_lines = set()
            for row in chunk_rows:
                for line in json.loads(row["compacted_data"]):
                    for line_no in line.get(
                        "repeat_line_numbers", [line["line_no"]]
                    ):
                        if through_line is not None and line_no > through_line:
                            continue
                        if line_no in seen_lines:
                            continue
                        seen_lines.add(line_no)
                        try:
                            data = json.loads(line.get("raw", ""))
                        except json.JSONDecodeError:
                            continue
                        ip = str(data.get("client_ip", "unknown"))
                        if not ip or ip == "unknown":
                            continue
                        entity = entities.setdefault(
                            ip,
                            {
                                "requests": 0,
                                "login_failures": 0,
                                "findings": [],
                                "evidence_lines": [],
                                "last_line": line_no,
                            },
                        )
                        entity["requests"] += 1
                        entity["last_line"] = max(entity["last_line"], line_no)
                        path = str(data.get("path", "")).lower()
                        auth_result = str(
                            data.get("auth_result", data.get("auth", ""))
                        ).lower()
                        if "/auth/login" in path and (
                            auth_result in {"fail", "failure", "invalid_password", "401"}
                            or str(data.get("status", "")) in {"401", "403"}
                        ):
                            entity["login_failures"] += 1

            severity_scores = {"low": 1.0, "medium": 2.0, "high": 3.0, "critical": 4.0}
            for row in finding_rows:
                identifier = row["identifier"]
                evidence_lines = json.loads(row["evidence_lines"])
                if through_line is not None and (
                    not evidence_lines or max(evidence_lines) > through_line
                ):
                    continue
                entity = entities.setdefault(
                    identifier,
                    {
                        "requests": 0,
                        "login_failures": 0,
                        "findings": [],
                        "evidence_lines": [],
                        "last_line": 0,
                    },
                )
                entity["findings"].append(row["attack_type"])
                for evidence_line in evidence_lines:
                    sampled_lines = sorted(
                        set(entity["evidence_lines"]) | {evidence_line}
                    )
                    entity["evidence_lines"] = (
                        sampled_lines
                        if len(sampled_lines) <= 4
                        else sampled_lines[:2] + sampled_lines[-2:]
                    )
                if evidence_lines:
                    entity["last_line"] = max(entity["last_line"], max(evidence_lines))
                entity["score"] = entity.get("score", 0.0) + severity_scores.get(
                    row["severity"], 1.0
                )

            conn.execute("DELETE FROM ledger WHERE job_id = ?;", (job_id,))
            for identifier, entity in entities.items():
                score = entity.get("score", 0.0) + max(
                    0, entity["login_failures"] - 2
                ) * 0.5
                if score <= 0 and entity["login_failures"] == 0:
                    continue
                attack_types = sorted(set(entity["findings"]))
                evidence = sorted(set(entity["evidence_lines"]))
                if len(evidence) > 4:
                    evidence_summary = (
                        f"{','.join(map(str, evidence[:2]))},...,"
                        f"{','.join(map(str, evidence[-2:]))}"
                    )
                else:
                    evidence_summary = ",".join(map(str, evidence))
                summary = (
                    f"req={entity['requests']}; "
                    f"fails={entity['login_failures']}; "
                    f"attacks={','.join(attack_types[:3]) or 'none'}; "
                    f"lines={evidence_summary or 'none'}"
                )
                conn.execute(
                    """
                    INSERT INTO ledger
                    (job_id, entity_key, suspicion_score, summary_text, last_updated_line)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (job_id, identifier, score, summary, entity["last_line"]),
                )
            conn.commit()
