from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.db import get_job_line, get_job_status, init_db, save_job_record
from app.validate import validate_upload_file, ValidationError
from app.chunker import chunk_log_file
from app.llm import get_tokenizer_fn
from app.queue import JobQueueManager
from app.report import generate_json_report

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    init_db()
    JobQueueManager.get_instance()
    yield


app = FastAPI(
    title="Local AI Security Log Analyzer",
    description="Offline Hybrid Deterministic-LLM SIEM Log Analyzer",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static frontend directory
STATIC_DIR = Path("static")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
def read_index():
    """Serves the frontend dashboard interface."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Local AI Security Log Analyzer API is running. Upload logs via POST /api/upload."}

@app.post("/api/upload")
async def upload_log_file(file: UploadFile = File(...)):
    """
    Securely receives a log file upload, validates extension, size, binary content,
    and line limits, chunks the file, and enqueues it for background processing.
    """
    job_id = str(uuid.uuid4())
    safe_filename = f"{job_id}_{Path(file.filename).name}"
    file_path = UPLOADS_DIR / safe_filename

    try:
        # Stream upload to disk
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)

        # Validate upload
        validate_upload_file(file_path, file.filename)

        # Tokenize and chunk log file
        tokenizer_fn = get_tokenizer_fn()
        chunks = chunk_log_file(str(file_path), tokenizer_fn, max_tokens_per_chunk=1000)

        # Count total lines across chunks
        with open(file_path, "r", encoding="utf-8", errors="replace") as log_file:
            total_lines = sum(1 for _ in log_file)

        # Save job and chunks to SQLite
        save_job_record(job_id, file.filename, total_lines, chunks)

        # Enqueue job for background processing
        queue_manager = JobQueueManager.get_instance()
        queue_manager.enqueue_job(job_id, str(file_path))

        return JSONResponse(status_code=202, content={
            "job_id": job_id,
            "filename": file.filename,
            "total_lines": total_lines,
            "total_chunks": len(chunks),
            "status": "queued"
        })

    except ValidationError as ve:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=400, detail={"code": ve.code, "message": ve.message})
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR", "message": str(e)})

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    """Retrieves current job status, completion progress, and chunk states."""
    status = get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found.")
    return status

@app.get("/api/jobs/{job_id}/lines/{line_no}")
def get_job_raw_line(job_id: str, line_no: int):
    """Returns one original upload line for report evidence inspection."""
    if line_no < 1:
        raise HTTPException(status_code=422, detail="Line number must be positive.")
    line = get_job_line(job_id, line_no)
    if line is None:
        raise HTTPException(status_code=404, detail="Log line not found.")
    return line

@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Triggers cooperative cancellation for a running job."""
    status = get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found.")
    
    queue_manager = JobQueueManager.get_instance()
    queue_manager.cancel_job(job_id)
    return {"job_id": job_id, "status": "cancelling"}

@app.get("/api/jobs/{job_id}/report")
def get_job_report(job_id: str):
    """Assembles and returns the final validated JSON report."""
    try:
        report_data = generate_json_report(job_id)
        return report_data
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")