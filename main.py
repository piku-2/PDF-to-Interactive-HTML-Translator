"""FastAPI entrypoint for the PDF-to-Interactive-HTML Translator."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from app.config import MAX_UPLOAD_BYTES
from app.jobs import create_job, get_job, process_job, too_many_jobs
from app.models import JobStatus, TranslateAccepted
from app.parser import ensure_nltk
from app.providers import log_active_providers

logger = logging.getLogger(__name__)
_INDEX_PATH = Path(__file__).resolve().parent / "app" / "templates" / "index.html"


def configure_console_logging() -> None:
    """Show provider order, the attempt, and fallback switches on the console."""
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    for name in ("app", "main"):
        named = logging.getLogger(name)
        named.setLevel(logging.INFO)
        if not any(isinstance(handler, logging.StreamHandler) for handler in named.handlers):
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            named.addHandler(handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    configure_console_logging()
    await asyncio.to_thread(ensure_nltk)
    log_active_providers("サーバー起動 アクティブなプロバイダー順序")
    yield


app = FastAPI(
    title="PDF-to-Interactive-HTML Translator",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(_INDEX_PATH.read_text(encoding="utf-8"))


@app.post("/api/translate", response_model=TranslateAccepted)
async def translate(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> TranslateAccepted:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDFファイルをアップロードしてください。")
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="ファイルサイズが100MBを超えています。")
        chunks.append(chunk)
    payload = b"".join(chunks)
    if b"%PDF" not in payload[:1024]:
        raise HTTPException(status_code=400, detail="PDFとして読み取れません。")
    if too_many_jobs():
        raise HTTPException(status_code=429, detail="処理中のジョブが上限に達しています。しばらく待ってから再度お試しください。")
    job = create_job(filename)
    background_tasks.add_task(process_job, job.id, payload)
    logger.info("queued job=%s filename=%s bytes=%s", job.id, job.filename, len(payload))
    return TranslateAccepted(job_id=job.id)


@app.get("/api/status/{job_id}", response_model=JobStatus)
async def job_status(job_id: str) -> JobStatus:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    return JobStatus(
        job_id=job.id,
        status=job.status,  # type: ignore[arg-type]
        progress=job.progress,
        error=job.error,
        filename=job.filename,
    )


@app.get("/api/download/{job_id}")
async def download(job_id: str) -> Response:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    if job.status != "completed" or not job.html:
        raise HTTPException(status_code=409, detail="翻訳はまだ完了していません。")
    encoded = quote(job.download_name, safe="")
    disposition = f"attachment; filename=\"translation.html\"; filename*=UTF-8''{encoded}"
    return Response(
        content=job.html,
        media_type="text/html; charset=utf-8",
        headers={
            "Content-Disposition": disposition,
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
        },
    )


if __name__ == "__main__":
    import uvicorn

    configure_console_logging()
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
