import sqlite3
from pathlib import Path

DB_PATH = Path("data/analyzer.db")

def init_ledger_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT,
            details TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_event(event_type: str, details: str):
    try:
        init_ledger_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO audit_ledger (event_type, details) VALUES (?, ?)",
            (event_type, details)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[!] Ledger logging failed: {e}")