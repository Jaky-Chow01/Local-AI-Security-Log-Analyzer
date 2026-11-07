import uuid
import re
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
    if max_tokens_per_chunk < 1:
        raise ValueError("max_tokens_per_chunk must be greater than zero")

    raw_entries = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f, start=1):
            line_no, compacted, meta = parse_log_line(line, idx)
            raw_entries.append({
                "line_no": line_no,
                "compacted": compacted,
                "raw": meta["raw"],
                "is_json": meta["is_json"],
                "truncated": False,
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
        repeated_line_numbers = [current["line_no"]]
        current_without_line_number = re.sub(r"^L\d+\|", "", current["compacted"])
        while (i + count < len(raw_entries) and 
               re.sub(r"^L\d+\|", "", raw_entries[i + count]["compacted"]) == current_without_line_number and
               raw_entries[i + count]["is_json"] == current["is_json"]):
            end_line_no = raw_entries[i + count]["line_no"]
            repeated_line_numbers.append(end_line_no)
            count += 1
        
        entry_copy = current.copy()
        if count > 1:
            entry_copy["compacted"] = f"{current['line_no']}-{end_line_no} x{count}|{current['compacted']}"
            entry_copy["repeat_line_numbers"] = repeated_line_numbers
        
        deduped_entries.append(entry_copy)
        i += count

    # Helper for accurate token counting using the model's tokenizer
    def count_tokens(text: str) -> int:
        try:
            return len(tokenize_fn(text))
        except Exception:
            return max(1, len(text.encode("utf-8")))

    def truncate_line(text: str) -> str:
        marker = "...[truncated]..."
        low, high = 0, len(text)
        best = marker
        while low <= high:
            kept = (low + high) // 2
            left = (kept + 1) // 2
            right = kept // 2
            candidate = text[:left] + marker + (text[-right:] if right else "")
            if count_tokens(candidate) <= max_tokens_per_chunk:
                best = candidate
                low = kept + 1
            else:
                high = kept - 1
        if count_tokens(best) > max_tokens_per_chunk:
            best = text[:max_tokens_per_chunk]
            while best and count_tokens(best) > max_tokens_per_chunk:
                best = best[:-1]
        return best

    # 2. Greedy pack into token-budget chunks with 2-line overlap
    chunks = []
    current_chunk_lines = []
    current_tokens = 0
    chunk_start_line = deduped_entries[0]["line_no"]

    def get_end_line(entry: Dict[str, Any]) -> int:
        return entry.get("repeat_line_numbers", [entry["line_no"]])[-1]

    for entry in deduped_entries:
        line_text = entry["compacted"]
        line_tokens = count_tokens(line_text)

        # Handle oversized single lines exceeding budget
        if line_tokens > max_tokens_per_chunk:
            line_text = truncate_line(line_text)
            entry["compacted"] = line_text
            entry["truncated"] = True
            line_tokens = count_tokens(line_text)

        if current_tokens + line_tokens > max_tokens_per_chunk and current_chunk_lines:
            # Commit current chunk
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "start_line": chunk_start_line,
                "end_line": get_end_line(current_chunk_lines[-1]),
                "token_count": current_tokens,
                "lines": current_chunk_lines,
                "truncated": any(line.get("truncated", False) for line in current_chunk_lines)
            })
            # Carry over last 2 lines as context overlap
            overlap = current_chunk_lines[-2:] if len(current_chunk_lines) >= 2 else current_chunk_lines
            context_lines = [
                dict(l, compacted=f"(context) {l['compacted']}")
                for l in overlap
            ]
            while context_lines and (
                sum(count_tokens(l["compacted"]) for l in context_lines) + line_tokens
                > max_tokens_per_chunk
            ):
                context_lines.pop(0)
            current_chunk_lines = context_lines
            current_tokens = sum(count_tokens(l["compacted"]) for l in current_chunk_lines)
            chunk_start_line = current_chunk_lines[0]["line_no"] if current_chunk_lines else entry["line_no"]

        current_chunk_lines.append(entry)
        current_tokens += line_tokens

    # Flush final chunk
    if current_chunk_lines:
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "start_line": chunk_start_line,
            "end_line": get_end_line(current_chunk_lines[-1]),
            "token_count": current_tokens,
            "lines": current_chunk_lines,
            "truncated": any(line.get("truncated", False) for line in current_chunk_lines)
        })

    return chunks