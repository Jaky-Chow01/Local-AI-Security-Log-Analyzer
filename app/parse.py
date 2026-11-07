import json
from typing import Dict, Any, Tuple

def parse_log_line(raw_line: str, line_no: int) -> Tuple[int, str, Dict[str, Any]]:
    """
    Parses a single log line. Attempts JSON parsing; falls back to raw text.
    Strips '<<<' tags to prevent prompt injection breakouts.
    Returns (line_no, compacted_text, metadata_dict).
    """
    cleaned_line = raw_line.rstrip("\r\n")
    # Structural security: strip <<< to prevent log text from forging marker boundaries
    cleaned_line = cleaned_line.replace("<<<", "###")
    
    try:
        data = json.loads(cleaned_line)
    except json.JSONDecodeError:
        # Fallback for plain text lines
        return line_no, cleaned_line, {"is_json": False, "raw": cleaned_line}

    # Extract and compact essential fields to cut roughly 50% of tokens
    time_full = data.get("time", "")
    time_short = time_full.split("T")[-1].split(".")[0] if "T" in time_full else (time_full[-8:] if len(time_full) >= 8 else time_full)
    
    service = data.get("service", data.get("app", "app"))
    client_ip = data.get("client_ip", "unknown")
    method = data.get("method", "")
    path = data.get("path", "")
    status = data.get("status", "")
    username = data.get("username", "")
    auth_result = data.get("auth_result", data.get("auth", ""))
    security_rule = data.get("security_rule", "")
    message = data.get("message", "")
    submitted_text = data.get("submitted_text", "")
    requests_in_window = data.get("requests_in_window", "")

    # Build pipe-delimited compact format for the model window
    parts = [f"L{line_no}", time_short, service]
    if method or path:
        parts.append(f"{method} {path}".strip())
    if status:
        parts.append(str(status))
    if client_ip:
        parts.append(client_ip)
    if username:
        parts.append(f"user={username}")
    if auth_result:
        parts.append(auth_result)
    if security_rule:
        parts.append(f"rule={security_rule}")
    if requests_in_window:
        parts.append(f"reqs_win={requests_in_window}")
    if message:
        parts.append(f"msg={message}")
    if submitted_text:
        sub_txt = (submitted_text[:80] + "...") if len(submitted_text) > 80 else submitted_text
        parts.append(f"sub={sub_txt}")

    compacted = "|".join([p for p in parts if p])
    return line_no, compacted, {
        "is_json": True, 
        "raw": cleaned_line, 
        "ip": client_ip, 
        "status": status,
        "username": username
    }