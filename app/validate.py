import os
from pathlib import Path

MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB
MAX_LINES = 20000

class ValidationError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(self.message)

def validate_upload_file(file_path: Path, filename: str) -> None:
    """
    Validates the uploaded log file against extension, size, binary content, and line limits.
    Fails fast with specific error codes.
    """
    # 1. Extension Check
    ext = Path(filename).suffix.lower()
    if ext not in [".log", ".txt"]:
        raise ValidationError("INVALID_TYPE", "Only .log and .txt files are permitted.")

    # 2. Size Check
    if not file_path.exists():
        raise ValidationError("EMPTY_FILE", "The uploaded file does not exist.")
    
    file_size = file_path.stat().st_size
    if file_size == 0:
        raise ValidationError("EMPTY_FILE", "The uploaded file is empty.")
    if file_size > MAX_FILE_SIZE:
        raise ValidationError("FILE_TOO_LARGE", f"File size exceeds the 2 MB limit (got {file_size / (1024*1024):.2f} MB).")

    # 3. Binary Sniffing & Encoding Check
    try:
        with open(file_path, "rb") as f:
            header = f.read(65536) # Read first 64 KB
            
        if b"\x00" in header:
            raise ValidationError("NOT_TEXT", "Binary content detected (null bytes found).")
        
        # Decode chunk to test UTF-8 validity
        decoded = header.decode("utf-8", errors="replace")
        replacement_count = decoded.count("\ufffd")
        if len(header) > 0 and (replacement_count / len(header)) > 0.01:
            raise ValidationError("NOT_TEXT", "File appears to be binary or contains invalid UTF-8 encoding (>1% replacement chars).")

    except UnicodeDecodeError:
        raise ValidationError("NOT_TEXT", "File encoding could not be decoded as UTF-8.")

    # 4. Line Count Check
    line_count = 0
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_count += 1
            if line_count > MAX_LINES:
                raise ValidationError("TOO_MANY_LINES", f"Line count exceeds maximum allowed limit of {MAX_LINES} lines.")
                
    if line_count == 0:
        raise ValidationError("EMPTY_FILE", "The file contains no readable lines.")