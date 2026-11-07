import re
import uuid
from collections.abc import Iterator
from typing import Any, Callable, Dict, List

from app.parse import parse_log_line


def iter_log_chunks(
    file_path: str,
    tokenize_fn: Callable[[str], List[int]],
    max_tokens_per_chunk: int = 1000,
) -> Iterator[Dict[str, Any]]:
    """Streams parsed lines into bounded token chunks with a two-line overlap."""
    if max_tokens_per_chunk < 1:
        raise ValueError("max_tokens_per_chunk must be greater than zero")

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

    def build_entry(line_no: int, raw_line: str) -> Dict[str, Any]:
        parsed_line_no, compacted, metadata = parse_log_line(raw_line, line_no)
        return {
            "line_no": parsed_line_no,
            "compacted": compacted,
            "raw": metadata["raw"],
            "is_json": metadata["is_json"],
            "truncated": False,
        }

    def end_line(entry: Dict[str, Any]) -> int:
        return entry.get("repeat_line_numbers", [entry["line_no"]])[-1]

    current_chunk: List[Dict[str, Any]] = []
    current_tokens = 0
    chunk_start_line = 1
    overlap_line_numbers: List[int] = []
    pending_entry: Dict[str, Any] | None = None

    def pack(entry: Dict[str, Any]) -> Dict[str, Any] | None:
        nonlocal current_chunk, current_tokens, chunk_start_line
        nonlocal overlap_line_numbers
        line_text = entry["compacted"]
        line_tokens = count_tokens(line_text)
        if line_tokens > max_tokens_per_chunk:
            line_text = truncate_line(line_text)
            entry["compacted"] = line_text
            entry["truncated"] = True
            line_tokens = count_tokens(line_text)

        emitted = None
        if current_tokens + line_tokens > max_tokens_per_chunk and current_chunk:
            emitted = {
                "chunk_id": str(uuid.uuid4()),
                "start_line": chunk_start_line,
                "end_line": end_line(current_chunk[-1]),
                "token_count": current_tokens,
                "lines": current_chunk,
                "overlap_line_numbers": overlap_line_numbers,
                "truncated": any(line.get("truncated", False) for line in current_chunk),
            }
            overlap = current_chunk[-2:]
            context_lines = [
                dict(line, compacted=f"(context) {line['compacted']}", is_context=True)
                for line in overlap
            ]
            while context_lines and (
                sum(count_tokens(line["compacted"]) for line in context_lines)
                + line_tokens
                > max_tokens_per_chunk
            ):
                context_lines.pop(0)
            current_chunk = context_lines
            current_tokens = sum(count_tokens(line["compacted"]) for line in current_chunk)
            overlap_line_numbers = [
                line_no
                for line in context_lines
                for line_no in line.get("repeat_line_numbers", [line["line_no"]])
            ]
            chunk_start_line = (
                current_chunk[0]["line_no"] if current_chunk else entry["line_no"]
            )
        if not current_chunk:
            chunk_start_line = entry["line_no"]
        current_chunk.append(entry)
        current_tokens += line_tokens
        return emitted

    with open(file_path, "r", encoding="utf-8", errors="replace") as source:
        for line_no, raw_line in enumerate(source, start=1):
            entry = build_entry(line_no, raw_line)
            if pending_entry is None:
                pending_entry = entry
                pending_entry["repeat_line_numbers"] = [line_no]
                pending_entry["_base_compacted"] = entry["compacted"]
                continue

            same_content = (
                re.sub(r"^L\d+\|", "", pending_entry["_base_compacted"])
                == re.sub(r"^L\d+\|", "", entry["compacted"])
                and pending_entry["is_json"] == entry["is_json"]
            )
            if same_content:
                pending_entry["repeat_line_numbers"].append(line_no)
                pending_entry["compacted"] = (
                    f"{pending_entry['line_no']}-{line_no} "
                    f"x{len(pending_entry['repeat_line_numbers'])}|"
                    f"{pending_entry['_base_compacted']}"
                )
                continue

            pending_entry.pop("_base_compacted")
            emitted = pack(pending_entry)
            if emitted is not None:
                yield emitted
            pending_entry = entry
            pending_entry["repeat_line_numbers"] = [line_no]
            pending_entry["_base_compacted"] = entry["compacted"]

    if pending_entry is not None:
        pending_entry.pop("_base_compacted")
        emitted = pack(pending_entry)
        if emitted is not None:
            yield emitted
    if current_chunk:
        yield {
            "chunk_id": str(uuid.uuid4()),
            "start_line": chunk_start_line,
            "end_line": end_line(current_chunk[-1]),
            "token_count": current_tokens,
            "lines": current_chunk,
            "overlap_line_numbers": overlap_line_numbers,
            "truncated": any(line.get("truncated", False) for line in current_chunk),
        }


def chunk_log_file(
    file_path: str,
    tokenize_fn: Callable[[str], List[int]],
    max_tokens_per_chunk: int = 1000,
) -> List[Dict[str, Any]]:
    """Compatibility wrapper returning streamed chunks as a list."""
    return list(iter_log_chunks(file_path, tokenize_fn, max_tokens_per_chunk))
