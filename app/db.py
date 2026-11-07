import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

DB_PATH = Path("app/db.sqlite")

def get_db_connection() -> sqlite3.Connection:
    """Returns a SQLite connection with WAL mode enabled and row factory configured."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db() -> None:
    """Initializes SQLite schema for jobs, chunks, ledger state, and findings."""
    with get_db_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                status TEXT NOT NULL, -- queued, running, complete, partial, failed
                total_lines INTEGER DEFAULT 0,
                total_chunks INTEGER DEFAULT 0,
                completed_chunks INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                status TEXT NOT NULL, -- pending, running, done, retrying, failed
                compacted_data TEXT NOT NULL, -- JSON serialized lines
                FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS findings (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                attack_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                identifier TEXT NOT NULL,
                evidence_lines TEXT NOT NULL, -- JSON array of line numbers
                reasoning TEXT NOT NULL,
                suggested_actions TEXT NOT NULL, -- JSON array of strings
                confidence REAL NOT NULL,
                source TEXT NOT NULL, -- detector, model, both
                FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS ledger (
                job_id TEXT NOT NULL,
                entity_key TEXT NOT NULL,
                suspicion_score REAL NOT NULL,
                summary_text TEXT NOT NULL,
                last_updated_line INTEGER NOT NULL,
                PRIMARY KEY (job_id, entity_key),
                FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
            );
        """)
        
        # Crash recovery: on boot, reset any running jobs/chunks to pending or failed
        conn.execute("""
            UPDATE jobs 
            SET status = 'partial', updated_at = CURRENT_TIMESTAMP 
            WHERE status = 'running';
        """)
        conn.execute("""
            UPDATE chunks 
            SET status = 'pending' 
            WHERE status = 'running';
        """)
        conn.commit()

def save_job_record(job_id: str, filename: str, total_lines: int, chunks: List[Dict[str, Any]]) -> None:
    """Persists a new job and its associated chunks into SQLite in a single transaction."""
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, filename, status, total_lines, total_chunks, completed_chunks) VALUES (?, ?, ?, ?, ?, 0);",
            (job_id, filename, "queued", total_lines, len(chunks))
        )
        for idx, chunk in enumerate(chunks):
            conn.execute(
                "INSERT INTO chunks (chunk_id, job_id, chunk_index, start_line, end_line, token_count, status, compacted_data) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (
                    chunk["chunk_id"],
                    job_id,
                    idx,
                    chunk["start_line"],
                    chunk["end_line"],
                    chunk["token_count"],
                    "pending",
                    json.dumps(chunk["lines"])
                )
            )
        conn.commit()

def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves current job status and completion metrics."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?;", (job_id,)).fetchone()
        if not row:
            return None
        
        chunks = conn.execute("SELECT chunk_id, chunk_index, status FROM chunks WHERE job_id = ? ORDER BY chunk_index ASC;", (job_id,)).fetchall()
        return {
            "job_id": row["job_id"],
            "filename": row["filename"],
            "status": row["status"],
            "total_lines": row["total_lines"],
            "total_chunks": row["total_chunks"],
            "completed_chunks": row["completed_chunks"],
            "chunks": [dict(c) for c in chunks]
        }