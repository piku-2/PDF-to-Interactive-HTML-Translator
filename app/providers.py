"""LLM provider priority, clients, and fallback.

The active order comes from LLM_PROVIDER_PRIORITY. Only providers with a
non-empty API key are used. A rate limit, server error, timeout, or auth
failure logs a warning and tries the next provider. Every successful call
returns BatchTranslationResponse items.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from app.config import MAX_RETRIES, MODEL_NAME
from app.llm_errors import AllProvidersFailed, OutputTooLarge, ProviderCallError
from app.models import BatchTranslationResponse, SentenceTranslation, TextBlock
from app.prompts import (
    ANTHROPIC_TOOL,
    JSON_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    anthropic_user_text,
    json_user_text,
)

logger = logging.getLogger(__name__)

DEFAULT_PRIORITY = "anthropic,gemini,openai,grok,deepseek,kimi,glm,openrouter,nvidia_nim"

_PLACEHOLDERS = {
    "changeme",
    "your-api-key",
    "your_api_key",
    "your-key-here",
    "replace-me",
    "xxx",
    "todo",
}

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    model: str
    env_vars: tuple[str, ...]
    kind: str
    base_url: Optional[str] = None
    max_output_tokens: int = 8192


PROVIDERS: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec("anthropic", MODEL_NAME, ("ANTHROPIC_API_KEY",), "anthropic", max_output_tokens=16000),
    "gemini": ProviderSpec("gemini", "gemini-2.5-flash-lite", ("GEMINI_API_KEY",), "gemini"),
    "openai": ProviderSpec("openai", "gpt-4o-mini", ("OPENAI_API_KEY",), "openai"),
    "grok": ProviderSpec(
        "grok",
        "grok-2-latest",
        ("GROK_API_KEY", "XAI_API_KEY"),
        "openai",
        "https://api.x.ai/v1",
    ),
    "deepseek": ProviderSpec(
        "deepseek",
        "deepseek-chat",
        ("DEEPSEEK_API_KEY",),
        "openai",
        "https://api.deepseek.com",
    ),
    "kimi": ProviderSpec(
        "kimi",
        "moonshot-v1-8k",
        ("KIMI_API_KEY",),
        "openai",
        "https://api.moonshot.cn/v1",
        max_output_tokens=4096,
    ),
    "glm": ProviderSpec(
        "glm",
        "glm-4-flash",
        ("GLM_API_KEY",),
        "openai",
        "https://open.bigmodel.cn/api/paas/v4/",
    ),
    "openrouter": ProviderSpec(
        "openrouter",
        "anthropic/claude-3.5-haiku",
        ("OPENROUTER_API_KEY",),
        "openai",
        "https://openrouter.ai/api/v1",
    ),
    "nvidia_nim": ProviderSpec(
        "nvidia_nim",
        "meta/llama-3.3-70b-instruct",
        ("NVIDIA_NIM_API_KEY",),
        "openai",
        "https://integrate.api.nvidia.com/v1",
    ),
}

_dotenv_loaded = False
_unknown_warned: set[str] = set()
_openai_clients: dict[tuple[str, str, str], Any] = {}
_gemini_clients: dict[str, Any] = {}


def load_env() -> None:
    """Load the project .env without overriding variables already in the process."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    _dotenv_loaded = True


def reset_provider_state() -> None:
    """Drop cached SDK clients. Tests call this between cases."""
    global _dotenv_loaded
    _openai_clients.clear()
    _gemini_clients.clear()
    _unknown_warned.clear()
    _dotenv_loaded = False


def parse_priority(raw: Optional[str]) -> list[str]:
    """Split a comma-separated priority list. Blank means the default order."""
    text = raw if raw is not None and raw.strip() else DEFAULT_PRIORITY
    names: list[str] = []
    for part in text.split(","):
        name = part.strip().lower()
        if not name or name in names:
            continue
        names.append(name)
    return names


def _usable_key(*env_names: str) -> Optional[str]:
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value is None:
            continue
        stripped = value.strip()
        if not stripped or stripped.lower() in _PLACEHOLDERS or stripped.endswith("..."):
            continue
        return stripped
    return None


def available_providers() -> list[ProviderSpec]:
    """Providers that have a key, in LLM_PROVIDER_PRIORITY order."""
    load_env()
    selected: list[ProviderSpec] = []
    for name in parse_priority(os.environ.get("LLM_PROVIDER_PRIORITY")):
        spec = PROVIDERS.get(name)
        if spec is None:
            if name not in _unknown_warned:
                _unknown_warned.add(name)
                logger.warning("未知のプロバイダー名を無視します: %s", name)
            continue
        if _usable_key(*spec.env_vars) is None:
            continue
        selected.append(spec)
    return selected


def describe_chain(chain: Optional[list[ProviderSpec]] = None) -> str:
    specs = available_providers() if chain is None else chain
    if not specs:
        return "(APIキーが設定されたプロバイダーはありません)"
    return ", ".join(f"{spec.name} ({spec.model})" for spec in specs)


def log_active_providers(prefix: str = "現在アクティブなプロバイダー順序") -> list[ProviderSpec]:
    chain = available_providers()
    logger.info("%s: %s", prefix, describe_chain(chain))
    if chain:
        logger.info("メインで使用するプロバイダー: %s model=%s", chain[0].name, chain[0].model)
    else:
        logger.warning("有効なAPIキーがないため、翻訳リクエストは失敗します。")
    return chain


def openai_request_kwargs(spec: ProviderSpec, blocks: list[TextBlock]) -> dict[str, Any]:
    """Body of an OpenAI-compatible chat call. The prompt contains 'json'."""
    return {
        "model": spec.model,
        "messages": [
            {"role": "system", "content": JSON_SYSTEM_PROMPT},
            {"role": "user", "content": json_user_text(blocks)},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": spec.max_output_tokens,
    }


async def call_with_fallback(
    blocks: list[TextBlock],
    failed: set[str],
) -> list[SentenceTranslation]:
    """Try providers in order. Fallback errors are remembered in ``failed``."""
    chain = available_providers()
    if not chain:
        raise RuntimeError(
            "有効なLLMのAPIキーが1つも設定されていません。LLM_PROVIDER_PRIORITY と各APIキーを確認してください。"
        )

    errors: list[tuple[str, str]] = []
    candidates = [spec for spec in chain if spec.name not in failed]
    if not candidates:
        skipped = [(spec.name, "このリクエストでは既に失敗") for spec in chain]
        logger.error(
            "全プロバイダーが失敗しました: %s",
            "; ".join(f"{name}: {message}" for name, message in skipped),
        )
        raise AllProvidersFailed(skipped)

    for index, spec in enumerate(candidates):
        logger.info(
            "試行中のプロバイダー: %s model=%s blocks=%s",
            spec.name,
            spec.model,
            len(blocks),
        )
        try:
            return await _dispatch(spec, blocks)
        except OutputTooLarge:
            raise
        except Exception as exc:
            if not _should_fallback(exc):
                raise
            reason = _short_error(exc)
            failed.add(spec.name)
            errors.append((spec.name, reason))
            nxt = candidates[index + 1].name if index + 1 < len(candidates) else None
            if nxt is None:
                rest = [item.name for item in chain if item.name not in failed]
                nxt = rest[0] if rest else None
            logger.warning(
                "フォールバック: %s が失敗したため %s へ切り替えます (%s)",
                spec.name,
                nxt or "残りのプロバイダーなし",
                reason,
            )

    logger.error(
        "全プロバイダーが失敗しました: %s",
        "; ".join(f"{name}: {message}" for name, message in errors),
    )
    raise AllProvidersFailed(errors)


async def _dispatch(spec: ProviderSpec, blocks: list[TextBlock]) -> list[SentenceTranslation]:
    api_key = _usable_key(*spec.env_vars)
    if not api_key:
        raise ProviderCallError(f"{spec.name} のAPIキーがありません。")
    if spec.kind == "anthropic":
        return await _anthropic_translate(spec, blocks)
    if spec.kind == "gemini":
        return await _gemini_translate(spec, api_key, blocks)
    return await _openai_translate(spec, api_key, blocks)


async def _anthropic_translate(spec: ProviderSpec, blocks: list[TextBlock]) -> list[SentenceTranslation]:
    from app.translator import get_client

    try:
        response = await get_client().messages.create(
            model=spec.model,
            max_tokens=spec.max_output_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[ANTHROPIC_TOOL],
            tool_choice={"type": "tool", "name": "submit_translations"},
            messages=[{"role": "user", "content": anthropic_user_text(blocks)}],
        )
    except Exception as exc:
        if _should_fallback(exc):
            raise ProviderCallError(_short_error(exc)) from exc
        raise

    if getattr(response, "stop_reason", None) == "max_tokens":
        raise OutputTooLarge(blocks[0].id if blocks else spec.name)

    tool_input = None
    for part in response.content:
        if getattr(part, "type", None) == "tool_use" and getattr(part, "name", None) == "submit_translations":
            tool_input = part.input
            break
    if tool_input is None:
        raise OutputTooLarge("tool_use missing")

    try:
        parsed = BatchTranslationResponse.model_validate(tool_input)
    except Exception as exc:
        raise OutputTooLarge("invalid tool payload") from exc

    usage = getattr(response, "usage", None)
    if usage is not None:
        logger.info(
            "応答 provider=anthropic blocks=%s cache_read=%s cache_write=%s",
            len(blocks),
            getattr(usage, "cache_read_input_tokens", None),
            getattr(usage, "cache_creation_input_tokens", None),
        )
    return list(parsed.items)


async def _openai_translate(
    spec: ProviderSpec,
    api_key: str,
    blocks: list[TextBlock],
) -> list[SentenceTranslation]:
    client = _openai_client(spec, api_key)
    kwargs = openai_request_kwargs(spec, blocks)
    try:
        response = await client.chat.completions.create(**kwargs)
    except Exception as exc:
        if _should_fallback(exc):
            raise ProviderCallError(_short_error(exc)) from exc
        raise

    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ProviderCallError(f"{spec.name} が空の応答を返しました。")
    choice = choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        raise OutputTooLarge(spec.name)
    content = choice.message.content or ""
    parsed = parse_batch_json(content)
    logger.info("応答 provider=%s model=%s blocks=%s", spec.name, spec.model, len(blocks))
    return list(parsed.items)


async def _gemini_translate(
    spec: ProviderSpec,
    api_key: str,
    blocks: list[TextBlock],
) -> list[SentenceTranslation]:
    from google.genai import types

    client = _gemini_client(api_key)
    try:
        response = await client.aio.models.generate_content(
            model=spec.model,
            contents=json_user_text(blocks),
            config=types.GenerateContentConfig(
                system_instruction=JSON_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=BatchTranslationResponse,
                max_output_tokens=spec.max_output_tokens,
            ),
        )
    except Exception as exc:
        if _should_fallback(exc):
            raise ProviderCallError(_short_error(exc)) from exc
        raise

    if _gemini_hit_token_limit(response):
        raise OutputTooLarge(spec.name)

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, BatchTranslationResponse):
        items = list(parsed.items)
    elif isinstance(parsed, dict):
        items = list(BatchTranslationResponse.model_validate(parsed).items)
    else:
        text = getattr(response, "text", None) or ""
        if not text.strip():
            raise ProviderCallError("gemini が空の応答を返しました。")
        items = list(parse_batch_json(text).items)
    logger.info("応答 provider=gemini model=%s blocks=%s", spec.model, len(blocks))
    return items


def _gemini_hit_token_limit(response: Any) -> bool:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return False
    reason = getattr(candidates[0], "finish_reason", None)
    label = getattr(reason, "name", None) or str(reason or "")
    return "MAX_TOKEN" in label.upper()


def _openai_client(spec: ProviderSpec, api_key: str) -> Any:
    cache_key = (spec.name, spec.base_url or "", api_key)
    client = _openai_clients.get(cache_key)
    if client is None:
        from openai import AsyncOpenAI

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "max_retries": MAX_RETRIES,
            "timeout": 120.0,
        }
        if spec.base_url:
            kwargs["base_url"] = spec.base_url
        client = AsyncOpenAI(**kwargs)
        _openai_clients[cache_key] = client
    return client


def _gemini_client(api_key: str) -> Any:
    client = _gemini_clients.get(api_key)
    if client is None:
        from google import genai

        client = genai.Client(api_key=api_key)
        _gemini_clients[api_key] = client
    return client


def parse_batch_json(raw: str) -> BatchTranslationResponse:
    text = _JSON_FENCE.sub("", raw.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderCallError(f"json の解析に失敗しました: {exc}") from exc
    try:
        return BatchTranslationResponse.model_validate(data)
    except Exception as exc:
        raise ProviderCallError(f"json の形が仕様と違います: {exc}") from exc


def _should_fallback(exc: BaseException) -> bool:
    if isinstance(exc, OutputTooLarge):
        return False
    if isinstance(exc, ProviderCallError):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status in {400, 401, 403, 408, 409, 429} or status >= 500):
        return True
    code = getattr(exc, "code", None)
    if isinstance(code, int) and (code in {400, 401, 403, 408, 409, 429} or code >= 500):
        return True
    typename = type(exc).__name__.lower()
    markers = ("timeout", "connection", "ratelimit", "authentication", "servererror", "clienterror")
    return any(marker in typename for marker in markers)


def _short_error(exc: BaseException) -> str:
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        code = getattr(exc, "code", None)
        status = code if isinstance(code, int) else None
    message = str(exc).replace("\n", " ").strip()
    if len(message) > 180:
        message = message[:180] + "…"
    if status is not None:
        return f"{type(exc).__name__} {status}: {message}"
    return f"{type(exc).__name__}: {message}"
