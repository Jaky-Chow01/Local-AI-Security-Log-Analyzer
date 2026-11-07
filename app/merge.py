import json
from typing import List, Dict, Any

def merge_and_correlate_findings(findings: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Merges duplicate findings by (attack_type, identifier), unions evidence lines,
    takes maximum severity, and correlates multi-stage incident chains (e.g., brute force -> takeover -> exfiltration).
    """
    merged_map: Dict[str, Dict[str, Any]] = {}
    severity_rank = {"low": 1, "medium": 2, "high": 3, "critical": 4}

    for f in findings:
        key = f"{f['attack_type']}:{f['identifier']}"
        ev_lines = json.loads(f["evidence_lines"]) if isinstance(f["evidence_lines"], str) else f["evidence_lines"]
        
        if key not in merged_map:
            merged_map[key] = {
                "id": f["id"],
                "attack_type": f["attack_type"],
                "severity": f["severity"],
                "identifier": f["identifier"],
                "evidence_lines": set(ev_lines),
                "reasoning": f["reasoning"],
                "suggested_actions": set(json.loads(f["suggested_actions"]) if isinstance(f["suggested_actions"], str) else f["suggested_actions"]),
                "confidence": f["confidence"],
                "source": f["source"]
            }
        else:
            existing = merged_map[key]
            existing["evidence_lines"].update(ev_lines)
            if severity_rank.get(f["severity"], 1) > severity_rank.get(existing["severity"], 1):
                existing["severity"] = f["severity"]
                existing["reasoning"] = f["reasoning"]
            existing["suggested_actions"].update(json.loads(f["suggested_actions"]) if isinstance(f["suggested_actions"], str) else f["suggested_actions"])

    # Convert sets back to sorted lists
    merged_findings = []
    for f in merged_map.values():
        f["evidence_lines"] = sorted(list(f["evidence_lines"]))
        f["suggested_actions"] = list(f["suggested_actions"])
        merged_findings.append(f)

    # Correlate multi-stage incident chains
    # Rule: brute_force or account_takeover + data_exfiltration from the same IP -> Critical Incident Chain
    incidents = []
    ip_findings: Dict[str, List[Dict[str, Any]]] = {}
    for f in merged_findings:
        ip_findings.setdefault(f["identifier"], []).append(f)

    for ip, items in ip_findings.items():
        attack_types = {item["attack_type"] for item in items}
        
        if "account_takeover" in attack_types and "data_exfiltration" in attack_types:
            all_lines = []
            for item in items:
                all_lines.extend([{"line": l, "event": item["reasoning"]} for l in item["evidence_lines"]])
            all_lines = sorted(all_lines, key=lambda x: x["line"])
            
            incidents.append({
                "title": f"Account takeover with data exfiltration ({ip})",
                "identifier": ip,
                "severity": "critical",
                "timeline": all_lines,
                "finding_ids": [item["id"] for item in items]
            })
        elif len(items) >= 2:
            all_lines = []
            for item in items:
                all_lines.extend([{"line": l, "event": item["reasoning"]} for l in item["evidence_lines"]])
            all_lines = sorted(all_lines, key=lambda x: x["line"])
            
            incidents.append({
                "title": f"Multi-vector attack campaign from {ip}",
                "identifier": ip,
                "severity": "high",
                "timeline": all_lines,
                "finding_ids": [item["id"] for item in items]
            })

    return merged_findings, incidents