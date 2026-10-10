"""In-memory async jobs and the polling lifecycle (SPEC.md section 6)."""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from app.config import JOB_TTL_SECONDS, MAX_ACTIVE_JOBS
from app.html_builder import apply_translations, build_html
from app.models import SentenceTranslation, TextBlock
from app.parser import parse_pdf_to_blocks
from app.providers import available_providers
from app.translator import translate_blocks

# Identity of the real translator, so a test double patched over the name
# does not trip the API-key check.
_DEFAULT_TRANSLATE = translate_blocks

logger = logging.getLogger(__name__)

Translator = Callable[..., Awaitable[dict[str, SentenceTranslation]]]

_FILENAME_SAFE = re.compile(r"[^\w.\-]+", re.UNICODE)


@dataclass
class Job:
    id: str
    filename: str
    download_name: str
    title: str
    status: str = "queued"
    progress: int = 0
    error: Optional[str] = None
    html: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def touch(self) -> None:
        self.updated_at = time.time()
        self.progress = max(0, min(100, int(self.progress)))


_jobs: dict[str, Job] = {}


def reset_jobs() -> None:
    _jobs.clear()


def active_job_count() -> int:
    return sum(1 for job in _jobs.values() if job.status in {"queued", "parsing", "translating"})


def create_job(filename: str) -> Job:
    _evict_expired()
    job_id = str(uuid.uuid4())
    safe_name = PathName(filename)
    job = Job(
        id=job_id,
        filename=safe_name.original,
        download_name=safe_name.download,
        title=safe_name.stem,
        created_at=time.time(),
        updated_at=time.time(),
    )
    _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)


def _evict_expired() -> None:
    now = time.time()
    stale = [
        job_id
        for job_id, job in _jobs.items()
        if job.status in {"completed", "failed"} and now - job.updated_at > JOB_TTL_SECONDS
    ]
    for job_id in stale:
        del _jobs[job_id]


class PathName:
    def __init__(self, filename: str) -> None:
        raw = (filename or "document.pdf").replace("\\", "/").split("/")[-1]
        raw = raw.replace("\r", "").replace("\n", "").strip() or "document.pdf"
        self.original = raw
        stem = raw[:-4] if raw.lower().endswith(".pdf") else raw
        stem = _FILENAME_SAFE.sub("_", stem).strip("._") or "translation"
        self.stem = stem
        self.download = f"{stem}-ja.html"


async def process_job(
    job_id: str,
    pdf_bytes: bytes,
    translator: Optional[Translator] = None,
) -> None:
    """Parse, translate, annotate, and store one HTML document.

    ``translator`` defaults to the priority chain. Tests pass a fake.
    A patched translator is a different function, so it skips the key check.
    """
    job = _jobs.get(job_id)
    if job is None:
        return
    translate = translator or translate_blocks
    try:
        if translate is _DEFAULT_TRANSLATE and not available_providers():
            raise RuntimeError(
                "有効なLLMのAPIキーが1つも設定されていません。LLM_PROVIDER_PRIORITY と各APIキーを確認してください。"
            )
        job.status = "parsing"
        job.progress = 5
        job.touch()
        blocks = await _parse(pdf_bytes)
        job.progress = 10
        job.touch()
        if blocks:
            job.status = "translating"
            job.touch()

            def on_progress(done: int, total: int) -> None:
                job.progress = 10 + int(85 * done / max(total, 1))
                job.touch()

            results = await translate(blocks, on_progress=on_progress)
            apply_translations(blocks, results)
        job.progress = 96
        job.touch()
        job.html = build_html(blocks, title=job.title)
        job.status = "completed"
        job.progress = 100
        job.error = None
        job.touch()
        logger.info("job %s completed blocks=%s", job_id, len(blocks))
    except Exception as exc:
        logger.exception("job %s failed", job_id)
        job.status = "failed"
        job.error = public_error(exc)
        job.touch()


async def _parse(pdf_bytes: bytes) -> list[TextBlock]:
    import asyncio

    return await asyncio.to_thread(parse_pdf_to_blocks, pdf_bytes)


_SECRET_RE = re.compile(
    r"sk-ant-[A-Za-z0-9_\-]+|"
    r"sk-[A-Za-z0-9_\-]{8,}|"
    r"AIza[0-9A-Za-z_\-]{10,}|"
    r"xai-[A-Za-z0-9_\-]{8,}|"
    r"x-api-key\s*[:=]\s*\S+",
    re.IGNORECASE,
)


def public_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    if _SECRET_RE.search(message):
        return "翻訳APIの呼び出しに失敗しました。APIキーとモデル名を確認してください。"
    if len(message) > 500:
        return message[:500] + "…"
    return message


def too_many_jobs() -> bool:
    return active_job_count() >= MAX_ACTIVE_JOBS
