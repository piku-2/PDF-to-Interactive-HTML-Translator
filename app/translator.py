"""Translation batches with priority fallback (SPEC.md sections 2, 4, and 6)."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

import anthropic

from app.config import BATCH_SIZE, MAX_CONCURRENCY, MAX_RETRIES
from app.llm_errors import OutputTooLarge
from app.models import SentenceTranslation, TextBlock
from app.providers import call_with_fallback, log_active_providers, reset_provider_state

logger = logging.getLogger(__name__)

_client: anthropic.AsyncAnthropic | None = None
_semaphore: asyncio.Semaphore | None = None

_NO_KEY = "有効なLLMのAPIキーが1つも設定されていません。LLM_PROVIDER_PRIORITY と各APIキーを確認してください。"


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        # SPEC 7: do not retry a 429 or 5xx here. Fallback switches provider.
        _client = anthropic.AsyncAnthropic(max_retries=MAX_RETRIES)
    return _client


def get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        # SPEC 6: at most five translation requests in flight.
        _semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    return _semaphore


def reset_client_state() -> None:
    """Drop the cached client and semaphore. Used by tests."""
    global _client, _semaphore
    _client = None
    _semaphore = None
    reset_provider_state()


async def translate_blocks(
    blocks: list[TextBlock],
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> dict[str, SentenceTranslation]:
    """Translate blocks in groups of 25, with at most five calls at once.

    SPEC section 6 asks for 25-block batches, ``asyncio.Semaphore(5)``, and
    progress on ``/api/status``. Those constraints are concurrent provider
    calls. The highest-priority provider with an API key goes first. A rate
    limit, server error, timeout, or auth failure is remembered in ``failed``
    and the next provider continues the same job. Prompt caching stays on the
    Anthropic system prompt and tool. The asynchronous Message Batches
    endpoint is not used: it cannot be paced with a local semaphore, and its
    queue can sit for a long time while the polling API is expected to move.
    """
    if not blocks:
        return {}
    chain = log_active_providers("翻訳リクエスト開始 アクティブなプロバイダー順序")
    if not chain:
        raise RuntimeError(_NO_KEY)

    failed: set[str] = set()
    batches = [
        blocks[index : index + BATCH_SIZE]
        for index in range(0, len(blocks), BATCH_SIZE)
    ]
    total = len(batches)
    completed = 0
    merged: dict[str, SentenceTranslation] = {}

    async def run(batch: list[TextBlock]) -> list[SentenceTranslation]:
        nonlocal completed
        items = await translate_batch_resilient(batch, failed)
        completed += 1
        if on_progress is not None:
            on_progress(completed, total)
        return items

    groups = await asyncio.gather(*(run(batch) for batch in batches))
    for items in groups:
        for item in items:
            merged[item.id] = item
    return merged


async def translate_batch_resilient(
    blocks: list[TextBlock],
    failed: Optional[set[str]] = None,
) -> list[SentenceTranslation]:
    """Call the provider chain, splitting the batch when the output is truncated."""
    if failed is None:
        failed = set()
    if not blocks:
        return []
    try:
        items = await _request_translation(blocks, failed)
    except OutputTooLarge:
        if len(blocks) == 1:
            raise RuntimeError(
                f"ブロック {blocks[0].id} の翻訳結果が出力上限を超えました。"
            )
        middle = len(blocks) // 2
        left = await translate_batch_resilient(blocks[:middle], failed)
        right = await translate_batch_resilient(blocks[middle:], failed)
        return left + right

    by_id = {item.id: item for item in items}
    missing = [block for block in blocks if block.id not in by_id]
    if not missing:
        return [by_id[block.id] for block in blocks]
    if len(blocks) == 1:
        logger.warning("translation missing for %s", blocks[0].id)
        return [SentenceTranslation(id=blocks[0].id, translated="", words=[])]
    if len(missing) < len(blocks):
        extra = await translate_batch_resilient(missing, failed)
        for item in extra:
            by_id[item.id] = item
    else:
        middle = len(blocks) // 2
        left = await translate_batch_resilient(blocks[:middle], failed)
        right = await translate_batch_resilient(blocks[middle:], failed)
        for item in left + right:
            by_id[item.id] = item
    return [
        by_id.get(block.id) or SentenceTranslation(id=block.id, translated="", words=[])
        for block in blocks
    ]


async def _request_translation(
    blocks: list[TextBlock],
    failed: set[str],
) -> list[SentenceTranslation]:
    # The semaphore covers the whole fallback walk, so one batch holds one slot.
    async with get_semaphore():
        return await call_with_fallback(blocks, failed)
