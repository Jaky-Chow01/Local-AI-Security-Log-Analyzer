import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from app.db import get_db_connection
from app.merge import merge_and_correlate_findings

class ReportSchema(BaseModel):
    report_version: str
    job_id: str
    generated_at: str
    source: Dict[str, Any]
    status: str
    summary: Dict[str, Any]
    incidents: List[Dict[str, Any]]
    findings: List[Dict[str, Any]]
    coverage: Dict[str, Any]
    limitations: List[str]

def generate_json_report(job_id: str) -> Dict[str, Any]:
    """
    Assembles, validates via Pydantic, and writes out the final JSON report atomically.
    """
    with get_db_connection() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE job_id = ?;", (job_id,)).fetchone()
        if not job:
            raise ValueError(f"Job {job_id} not found.")

        findings_rows = conn.execute("SELECT * FROM findings WHERE job_id = ?;", (job_id,)).fetchall()
        raw_findings = [dict(row) for row in findings_rows]

    merged_findings, incidents = merge_and_correlate_findings(raw_findings)

    # Calculate severity counts
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for f in merged_findings:
        sev = f.get("severity", "low")
        if sev in counts:
            counts[sev] += 1

    overall_risk = "low"
    if counts["critical"] > 0:
        overall_risk = "critical"
    elif counts["high"] > 0:
        overall_risk = "high"
    elif counts["medium"] > 0:
        overall_risk = "medium"

    executive_summary = (
        f"Analyzed {job['total_lines']} lines across {job['total_chunks']} chunks. "
        f"Identified {len(merged_findings)} distinct security findings and {len(incidents)} multi-stage incident chains. "
        f"Overall assessed risk level is {overall_risk.upper()}."
    )

    report_data = {
        "report_version": "1.0",
        "job_id": job_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": {
            "filename": job["filename"],
            "total_lines": job["total_lines"],
            "chunks": job["total_chunks"],
            "model": "qwen2.5-3b-q4_k_m"
        },
        "status": job["status"],
        "summary": {
            "overall_risk": overall_risk,
            "counts_by_severity": counts,
            "executive_summary": executive_summary
        },
        "incidents": incidents,
        "findings": merged_findings,
        "coverage": {
            "failed_ranges": [],
            "truncated_lines": [],
            "rejected_model_findings": 0
        },
        "limitations": [
            "Analysis performed offline with 2 vCPU budget constraints.",
            "Heuristic thresholds tuned for standard web server access and audit logs."
        ]
    }

    # Validate against Pydantic schema
    validated = ReportSchema(**report_data)
    report_dict = validated.model_dump()

    # Atomic write to disk
    report_dir = Path("app/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{job_id}.json"
    temp_path = report_path.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    temp_path.replace(report_path)

    return report_dict