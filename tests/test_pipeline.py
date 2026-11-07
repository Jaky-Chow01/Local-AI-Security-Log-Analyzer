import pytest
from pathlib import Path
from app.validate import validate_upload_file, ValidationError
from app.parse import parse_log_line
from app.chunker import chunk_log_file
from app.detectors import run_deterministic_detectors

def test_validation_invalid_extension():
    fake_file = Path("test.pdf")
    with pytest.raises(ValidationError) as exc:
        validate_upload_file(fake_file, "test.pdf")
    assert exc.value.code == "INVALID_TYPE"

def test_parse_json_line():
    raw = '{"time":"2026-11-01T08:02:30.000Z","client_ip":"203.0.113.44","method":"POST","path":"/api/v1/auth/login","status":401,"username":"workspace.admin","auth_result":"invalid_password"}'
    line_no, compacted, meta = parse_log_line(raw, 5)
    
    assert line_no == 5
    assert meta["is_json"] is True
    assert meta["ip"] == "203.0.113.44"
    assert "POST /api/v1/auth/login" in compacted
    assert "401" in compacted

def test_deterministic_detectors_on_fixture():
    fixture_path = Path("fixtures/Cultive8_Workspace_Agrigen_Server.log")
    if not fixture_path.exists():
        pytest.skip("Fixture log file not found.")

    entries = []
    with open(fixture_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            _, compacted, meta = parse_log_line(line, idx)
            entries.append({
                "line_no": idx,
                "compacted": compacted,
                "raw": meta["raw"],
                "is_json": meta["is_json"]
            })

    findings = run_deterministic_detectors(entries)
    
    # Verify that key attack vectors from the reference log are caught deterministically
    attack_types = {f["attack_type"] for f in findings}
    
    assert "sql_injection" in attack_types
    assert "path_traversal" in attack_types
    assert "prompt_injection" in attack_types
    assert "account_takeover" in attack_types or "brute_force" in attack_types