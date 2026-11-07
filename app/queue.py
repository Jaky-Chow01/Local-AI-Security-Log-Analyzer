import threading
import queue
import time
import json
from typing import Dict, Any, List
from app.db import get_db_connection
from app.detectors import run_deterministic_detectors
from app.llm import LLMManager

class JobQueueManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.job_queue = queue.Queue()
        self.cancel_flags: Dict[str, threading.Event] = {}
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    @classmethod
    def get_instance(cls) -> "JobQueueManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def enqueue_job(self, job_id: str, file_path: str) -> None:
        """Enqueues a job for background processing."""
        event = threading.Event()
        self.cancel_flags[job_id] = event
        self.job_queue.put((job_id, file_path))

    def cancel_job(self, job_id: str) -> None:
        """Triggers cooperative cancellation for a running job."""
        if job_id in self.cancel_flags:
            self.cancel_flags[job_id].set()

    def _worker_loop(self):
        """Single-flight background worker thread loop."""
        while True:
            try:
                job_id, file_path = self.job_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._process_job(job_id)
            except Exception as e:
                with get_db_connection() as conn:
                    conn.execute("UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE job_id = ?;", ("failed", job_id))
                    conn.commit()
            finally:
                self.job_queue.task_done()
                self.cancel_flags.pop(job_id, None)

    def _process_job(self, job_id: str):
        """Executes hybrid deterministic detection, chunk-by-chunk LLM analysis, and merging."""
        cancel_event = self.cancel_flags.get(job_id)
        
        with get_db_connection() as conn:
            conn.execute("UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE job_id = ?;", ("running", job_id))
            conn.commit()

            chunks_rows = conn.execute("SELECT * FROM chunks WHERE job_id = ? ORDER BY chunk_index ASC;", (job_id,)).fetchall()
            
            # 1. Run full-file deterministic detectors first across all chunk lines
            all_lines = []
            for c_row in chunks_rows:
                lines = json.loads(c_row["compacted_data"])
                all_lines.extend(lines)

        deterministic_findings = run_deterministic_detectors(all_lines)
        
        # Persist deterministic findings immediately
        with get_db_connection() as conn:
            for f in deterministic_findings:
                conn.execute(
                    "INSERT OR IGNORE INTO findings (id, job_id, attack_type, severity, identifier, evidence_lines, reasoning, suggested_actions, confidence, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                    (
                        f["id"], job_id, f["attack_type"], f["severity"], f["identifier"],
                        json.dumps(f["evidence_lines"]), f["reasoning"], json.dumps(f["suggested_actions"]), 1.0, f["source"]
                    )
                )
            conn.commit()

        llm_manager = None
        try:
            llm_manager = LLMManager.get_instance()
        except Exception:
            pass  # Fallback to detector-only mode if model is unavailable

        completed_count = 0
        failed_chunks = 0

        for c_row in chunks_rows:
            if cancel_event and cancel_event.is_set():
                break

            chunk_id = c_row["chunk_id"]
            
            with get_db_connection() as conn:
                conn.execute("UPDATE chunks SET status = ? WHERE chunk_id = ?;", ("running", chunk_id))
                conn.commit()

            lines = json.loads(c_row["compacted_data"])
            compact_text = "\n".join([l["compacted"] for l in lines])

            # Retrieve current ledger context (~200 tokens summary)
            ledger_summary = self._get_ledger_summary(job_id)

            model_findings = []
            if llm_manager:
                # Retry loop with backoff & timeout handling per chunk
                for attempt in range(3):
                    try:
                        res = llm_manager.classify_chunk(ledger_summary, compact_text)
                        model_findings = res.get("findings", [])
                        break
                    except Exception:
                        if attempt == 2:
                            failed_chunks += 1
                        time.sleep(1.0 * (attempt + 1))

            # Validate and persist model findings
            valid_line_numbers = {l["line_no"] for l in lines}
            with get_db_connection() as conn:
                for idx, mf in enumerate(model_findings):
                    # Strict validation: verify evidence lines exist in this chunk
                    ev_lines = mf.get("evidence_lines", [])
                    if all(el in valid_line_numbers for el in ev_lines):
                        finding_id = f"LLM-{chunk_id[:8]}-{idx}"
                        conn.execute(
                            "INSERT OR IGNORE INTO findings (id, job_id, attack_type, severity, identifier, evidence_lines, reasoning, suggested_actions, confidence, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                            (
                                finding_id, job_id, mf.get("attack_type", "benign"), mf.get("severity", "low"),
                                mf.get("identifier", "unknown"), json.dumps(ev_lines), mf.get("reasoning", ""),
                                json.dumps(mf.get("suggested_actions", [])), mf.get("confidence", 0.5), "model"
                            )
                        )
                
                conn.execute("UPDATE chunks SET status = ? WHERE chunk_id = ?;", ("done", chunk_id))
                completed_count += 1
                conn.execute("UPDATE jobs SET completed_chunks = ?, updated_at = CURRENT_TIMESTAMP WHERE job_id = ?;", (completed_count, job_id))
                conn.commit()

            # Update ledger state from entity activity in this chunk
            self._update_ledger(job_id, lines)

        # Final job status determination
        final_status = "complete" if failed_chunks == 0 else "partial"
        if cancel_event and cancel_event.is_set():
            final_status = "partial"

        with get_db_connection() as conn:
            conn.execute("UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE job_id = ?;", (final_status, job_id))
            conn.commit()

    def _get_ledger_summary(self, job_id: str) -> str:
        """Retrieves top suspicious entities from ledger, capped at ~200 tokens."""
        with get_db_connection() as conn:
            rows = conn.execute("SELECT entity_key, suspicion_score, summary_text FROM ledger WHERE job_id = ? ORDER BY suspicion_score DESC LIMIT 8;", (job_id,)).fetchall()
            if not rows:
                return ""
            return "\n".join([f"- {r['entity_key']} (score {r['suspicion_score']}): {r['summary_text']}" for r in rows])

    def _update_ledger(self, job_id: str, lines: List[Dict[str, Any]]):
        """Updates entity state scores in the SQLite ledger table."""
        with get_db_connection() as conn:
            for l in lines:
                raw = l.get("compacted", "")
                # Simple extraction of IP entities for ledger tracking
                parts = raw.split("|")
                for p in parts:
                    if "." in p and any(char.isdigit() for char in p):
                        ip = p.strip()
                        conn.execute("""
                            INSERT INTO ledger (job_id, entity_key, suspicion_score, summary_text, last_updated_line)
                            VALUES (?, ?, 1.0, ?, ?)
                            ON CONFLICT(job_id, entity_key) DO UPDATE SET
                                suspicion_score = suspicion_score + 0.5,
                                last_updated_line = excluded.last_updated_line;
                        """, (job_id, ip, f"Observed at line {l['line_no']}", l['line_no']))
            conn.commit()