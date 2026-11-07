import uuid
from typing import List, Dict, Any, Callable
from app.parse import parse_log_line

def chunk_log_file(
    file_path: str, 
    tokenize_fn: Callable[[str], List[int]], 
    max_tokens_per_chunk: int = 1000
) -> List[Dict[str, Any]]:
    """
    Reads a log file, parses and compacts lines, deduplicates consecutive identical lines,
    and packs them into token-budgeted chunks with a 2-line context overlap.
    """
    raw_entries = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f, start=1):
            line_no, compacted, meta = parse_log_line(line, idx)
            raw_entries.append({
                "line_no": line_no,
                "compacted": compacted,
                "raw": meta["raw"],
                "is_json": meta["is_json"]
            })

    if not raw_entries:
        return []

    # 1. Deduplicate consecutive identical lines (e.g., repeating SQL injection or error bursts)
    deduped_entries = []
    i = 0
    while i < len(raw_entries):
        current = raw_entries[i]
        count = 1
        end_line_no = current["line_no"]
        while (i + count < len(raw_entries) and 
               raw_entries[i + count]["compacted"] == current["compacted"] and
               raw_entries[i + count]["is_json"] == current["is_json"]):
            end_line_no = raw_entries[i + count]["line_no"]
            count += 1
        
        entry_copy = current.copy()
        if count > 1:
            entry_copy["compacted"] = f"{current['line_no']}-{end_line_no} x{count}|{current['compacted']}"
        
        deduped_entries.append(entry_copy)
        i += count

    # Helper for accurate token counting using the model's tokenizer
    def count_tokens(text: str) -> int:
        try:
            return len(tokenize_fn(text))
        except Exception:
            # Fallback estimation if model/tokenizer isn't initialized yet
            return max(1, len(text) // 4)

    # 2. Greedy pack into token-budget chunks with 2-line overlap
    chunks = []
    current_chunk_lines = []
    current_tokens = 0
    chunk_start_line = deduped_entries[0]["line_no"]

    for entry in deduped_entries:
        line_text = entry["compacted"]
        line_tokens = count_tokens(line_text)

        # Handle oversized single lines exceeding budget
        if line_tokens > max_tokens_per_chunk:
            excess = line_tokens - max_tokens_per_chunk
            chars_to_cut = excess * 4
            if len(line_text) > chars_to_cut + 20:
                mid = len(line_text) // 2
                line_text = line_text[:mid - chars_to_cut//2] + "…[truncated]…" + line_text[mid + chars_to_cut//2:]
                entry["compacted"] = line_text
                line_tokens = count_tokens(line_text)

        if current_tokens + line_tokens > max_tokens_per_chunk and current_chunk_lines:
            # Commit current chunk
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "start_line": chunk_start_line,
                "end_line": current_chunk_lines[-1]["line_no"],
                "token_count": current_tokens,
                "lines": current_chunk_lines,
                "truncated": False
            })
            # Carry over last 2 lines as context overlap
            overlap = current_chunk_lines[-2:] if len(current_chunk_lines) >= 2 else current_chunk_lines
            current_chunk_lines = [dict(l, compacted=f"(context) {l['compacted']}") for l in overlap]
            current_tokens = sum(count_tokens(l["compacted"]) for l in current_chunk_lines)
            chunk_start_line = current_chunk_lines[0]["line_no"] if current_chunk_lines else entry["line_no"]

        current_chunk_lines.append(entry)
        current_tokens += line_tokens

    # Flush final chunk
    if current_chunk_lines:
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "start_line": chunk_start_line,
            "end_line": current_chunk_lines[-1]["line_no"],
            "token_count": current_tokens,
            "lines": current_chunk_lines,
            "truncated": False
        })

    return chunks