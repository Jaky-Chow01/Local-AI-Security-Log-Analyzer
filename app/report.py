import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel

from app.db import get_db_connection
from app.merge import merge_and_correlate_findings

logger = logging.getLogger(__name__)


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


def _validated_findings(rows: List[Any]) -> tuple[List[Dict[str, Any]], int]:
    valid_findings = []
    rejected_count = 0
    for row in rows:
        finding = dict(row)
        try:
            evidence_lines = json.loads(finding["evidence_lines"])
            suggested_actions = json.loads(finding["suggested_actions"])
            confidence = finding["confidence"]
            if (
                not isinstance(evidence_lines, list)
                or not evidence_lines
                or any(
                    not isinstance(line_no, int)
                    or isinstance(line_no, bool)
                    or line_no < 1
                    for line_no in evidence_lines
                )
                or not isinstance(suggested_actions, list)
                or any(not isinstance(action, str) for action in suggested_actions)
                or not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
                or not all(
                    isinstance(finding.get(field), str) and finding[field]
                    for field in (
                        "id",
                        "attack_type",
                        "severity",
                        "identifier",
                        "reasoning",
                        "source",
                    )
                )
            ):
                raise ValueError("stored finding fields are incomplete or invalid")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.error(
                "Excluding malformed stored finding %s: %s",
                finding.get("id", "<unknown>"),
                exc,
            )
            rejected_count += 1
            continue

        finding["evidence_lines"] = evidence_lines
        finding["suggested_actions"] = suggested_actions
        valid_findings.append(finding)
    return valid_findings, rejected_count


def generate_json_report(job_id: str) -> Dict[str, Any]:
    """Builds, validates, and atomically writes a report with measured coverage."""
    with get_db_connection() as conn:
        job = conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?;", (job_id,)
        ).fetchone()
        if not job:
            raise ValueError(f"Job {job_id} not found.")

        finding_rows = conn.execute(
            "SELECT * FROM findings WHERE job_id = ?;", (job_id,)
        ).fetchall()
        chunk_rows = conn.execute(
            """
            SELECT start_line, end_line, status, compacted_data
            FROM chunks WHERE job_id = ? ORDER BY chunk_index;
            """,
            (job_id,),
        ).fetchall()

    raw_findings, rejected_report_findings = _validated_findings(finding_rows)
    merged_findings, incidents = merge_and_correlate_findings(raw_findings)

    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for finding in merged_findings:
        severity = finding.get("severity", "low")
        if severity in counts:
            counts[severity] += 1

    overall_risk = next(
        (severity for severity in ("critical", "high", "medium") if counts[severity]),
        "low",
    )
    executive_summary = (
        f"Analyzed {job['total_lines']} lines across {job['total_chunks']} chunks. "
        f"Identified {len(merged_findings)} distinct security findings and "
        f"{len(incidents)} multi-stage incident chains. "
        f"Overall assessed risk level is {overall_risk.upper()}."
    )

    failed_ranges = []
    truncated_lines = set()
    rejected_chunk_data = 0
    for chunk in chunk_rows:
        failed_range = None
        if chunk["status"] != "done":
            failed_range = {
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "status": chunk["status"],
            }
            failed_ranges.append(failed_range)
        try:
            chunk_lines = json.loads(chunk["compacted_data"])
            if not isinstance(chunk_lines, list) or any(
                not isinstance(line, dict)
                or not isinstance(line.get("line_no"), int)
                or isinstance(line.get("line_no"), bool)
                or not isinstance(line.get("raw"), str)
                or (
                    "repeat_line_numbers" in line
                    and (
                        not isinstance(line["repeat_line_numbers"], list)
                        or any(
                            not isinstance(line_no, int)
                            or isinstance(line_no, bool)
                            for line_no in line["repeat_line_numbers"]
                        )
                    )
                )
                for line in chunk_lines
            ):
                raise ValueError("chunk data does not contain valid log-line records")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.error(
                "Excluding malformed chunk data for job %s, lines %s-%s: %s",
                job_id,
                chunk["start_line"],
                chunk["end_line"],
                exc,
            )
            if failed_range is None:
                failed_range = {
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "status": "invalid_chunk_data",
                }
                failed_ranges.append(failed_range)
            else:
                failed_range["data_error"] = "invalid_chunk_data"
            rejected_chunk_data += 1
            continue
        for line in chunk_lines:
            if line.get("truncated"):
                truncated_lines.update(
                    line.get("repeat_line_numbers", [line["line_no"]])
                )

    limitations = [
        "Analysis runs locally and is subject to the configured model context and CPU limits.",
        "Deterministic detector thresholds are heuristics and require analyst review.",
    ]
    if job["model_status"] == "unavailable":
        limitations.append(
            "The local model was unavailable; this report contains deterministic detector results only."
        )
    elif job["model_status"] == "degraded":
        limitations.append(
            "One or more model chunks failed; findings and coverage may be incomplete."
        )
    if failed_ranges:
        limitations.append(
            "Failed or unfinished chunks are listed in coverage.failed_ranges."
        )
    if rejected_report_findings or rejected_chunk_data:
        limitations.append(
            "Malformed persisted analysis data was excluded; see coverage rejection counts."
        )

    report_status = job["status"]
    if failed_ranges and report_status == "complete":
        report_status = "partial"
    report_data = {
        "report_version": "1.0",
        "job_id": job_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "filename": job["filename"],
            "total_lines": job["total_lines"],
            "chunks": job["total_chunks"],
            "model": job["model_name"],
            "model_status": job["model_status"],
        },
        "status": report_status,
        "summary": {
            "overall_risk": overall_risk,
            "counts_by_severity": counts,
            "executive_summary": executive_summary,
        },
        "incidents": incidents,
        "findings": merged_findings,
        "coverage": {
            "failed_ranges": failed_ranges,
            "truncated_lines": sorted(truncated_lines),
            "rejected_model_findings": job["rejected_model_findings"],
            "rejected_report_findings": rejected_report_findings,
            "rejected_chunk_data": rejected_chunk_data,
        },
        "limitations": limitations,
    }

    validated = ReportSchema.model_validate(report_data)
    report_dict = validated.model_dump()

    report_dir = Path("app/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{job_id}.json"
    temp_path = report_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as report_file:
        json.dump(report_dict, report_file, indent=2)
    temp_path.replace(report_path)
    return report_dict
