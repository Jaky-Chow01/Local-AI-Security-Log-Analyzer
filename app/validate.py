import codecs
from pathlib import Path
from fastapi import UploadFile

MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB
MAX_LINES = 20000

class ValidationError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(self.message)


def validate_upload_filename(filename: str) -> None:
    if Path(filename).suffix.lower() not in {".log", ".txt"}:
        raise ValidationError("INVALID_TYPE", "Only .log and .txt files are permitted.")


class _UploadContentValidator:
    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.total_bytes = 0
        self._replacement_count = 0
        self._line_break_count = 0
        self._previous_was_cr = False
        self._last_byte = b""

    def consume(self, chunk: bytes) -> None:
        self.total_bytes += len(chunk)
        if b"\x00" in chunk:
            raise ValidationError("NOT_TEXT", "Binary content detected (null bytes found).")

        decoded = self._decoder.decode(chunk)
        self._replacement_count += decoded.count("\ufffd")

        for byte in chunk:
            if byte == 10:
                if not self._previous_was_cr:
                    self._line_break_count += 1
                self._previous_was_cr = False
            elif byte == 13:
                self._line_break_count += 1
                self._previous_was_cr = True
            else:
                self._previous_was_cr = False

        self._last_byte = chunk[-1:]
        if self.line_count > MAX_LINES:
            raise ValidationError(
                "TOO_MANY_LINES",
                f"Line count exceeds maximum allowed limit of {MAX_LINES} lines.",
            )

    @property
    def line_count(self) -> int:
        return (
            self._line_break_count
            + int(bool(self.total_bytes) and self._last_byte not in (b"\r", b"\n"))
        )

    def finish(self) -> int:
        decoded = self._decoder.decode(b"", final=True)
        self._replacement_count += decoded.count("\ufffd")
        if self.total_bytes == 0:
            raise ValidationError("EMPTY_FILE", "The uploaded file is empty.")
        if self._replacement_count / self.total_bytes > 0.01:
            raise ValidationError(
                "NOT_TEXT",
                "File appears to be binary or contains invalid UTF-8 encoding (>1% replacement chars).",
            )
        if self.line_count == 0:
            raise ValidationError("EMPTY_FILE", "The file contains no readable lines.")
        return self.line_count


async def save_upload_stream(upload: UploadFile, file_path: Path, filename: str) -> int:
    """Writes only within upload limits and returns the validated text line count."""
    validate_upload_filename(filename)
    validator = _UploadContentValidator()
    try:
        with open(file_path, "wb") as target:
            while chunk := await upload.read(64 * 1024):
                total_bytes = validator.total_bytes + len(chunk)
                if total_bytes > MAX_FILE_SIZE:
                    raise ValidationError(
                        "FILE_TOO_LARGE",
                        f"File size exceeds the 2 MB limit ({total_bytes / (1024 * 1024):.2f} MB).",
                    )
                validator.consume(chunk)
                target.write(chunk)
        return validator.finish()
    except BaseException:
        file_path.unlink(missing_ok=True)
        raise


def validate_upload_file(file_path: Path, filename: str) -> None:
    """
    Validates the uploaded log file against extension, size, binary content, and line limits.
    Fails fast with specific error codes.
    """
    validate_upload_filename(filename)
    if not file_path.exists():
        raise ValidationError("EMPTY_FILE", "The uploaded file does not exist.")

    file_size = file_path.stat().st_size
    if file_size == 0:
        raise ValidationError("EMPTY_FILE", "The uploaded file is empty.")
    if file_size > MAX_FILE_SIZE:
        raise ValidationError(
            "FILE_TOO_LARGE",
            f"File size exceeds the 2 MB limit (got {file_size / (1024 * 1024):.2f} MB).",
        )

    validator = _UploadContentValidator()
    with open(file_path, "rb") as source:
        while chunk := source.read(64 * 1024):
            validator.consume(chunk)
    validator.finish()