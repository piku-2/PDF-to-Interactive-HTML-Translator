"""Priority parsing and provider fallback. No live API calls."""

from __future__ import annotations

import asyncio
import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from app.config import MODEL_NAME
from app.jobs import public_error, reset_jobs
from app.llm_errors import AllProvidersFailed, OutputTooLarge
from app.models import SentenceTranslation, TextBlock
from app.providers import (
    DEFAULT_PRIORITY,
    PROVIDERS,
    _usable_key,
    available_providers,
    call_with_fallback,
    openai_request_kwargs,
    parse_batch_json,
    parse_priority,
    reset_provider_state,
)

PROVIDER_ENV = (
    "LLM_PROVIDER_PRIORITY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "GROK_API_KEY",
    "XAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "KIMI_API_KEY",
    "GLM_API_KEY",
    "OPENROUTER_API_KEY",
    "NVIDIA_NIM_API_KEY",
)


@contextmanager
def isolated_env(**overrides: str):
    """Hide developer keys and skip .env so tests pick the keys they set."""
    import app.providers as providers

    reset_provider_state()
    providers._dotenv_loaded = True
    values = {name: "" for name in PROVIDER_ENV}
    values.update(overrides)
    with patch.dict(os.environ, values, clear=False):
        yield


def _block() -> list[TextBlock]:
    return [
        TextBlock(id="b-1", type="sentence", original_text="Attention is all you need.")
    ]


class PriorityTests(unittest.TestCase):
    def test_parse_strips_case_and_dupes(self) -> None:
        self.assertEqual(
            parse_priority(" Gemini, OPENAI ,gemini, grok "),
            ["gemini", "openai", "grok"],
        )

    def test_blank_uses_default_order(self) -> None:
        expected = [part.strip() for part in DEFAULT_PRIORITY.split(",")]
        self.assertEqual(parse_priority(None), expected)
        self.assertEqual(parse_priority("   "), expected)
        self.assertEqual(
            expected,
            [
                "anthropic",
                "gemini",
                "openai",
                "grok",
                "deepseek",
                "kimi",
                "glm",
                "openrouter",
                "nvidia_nim",
            ],
        )

    def test_only_keyed_providers_follow_priority(self) -> None:
        with isolated_env(
            LLM_PROVIDER_PRIORITY="openai, gemini, anthropic, nope",
            OPENAI_API_KEY="test-key",
            ANTHROPIC_API_KEY="test-key",
            GEMINI_API_KEY="",
        ):
            with self.assertLogs("app.providers", level="WARNING") as logs:
                names = [spec.name for spec in available_providers()]
                available_providers()
        self.assertEqual(names, ["openai", "anthropic"])
        self.assertEqual(sum("未知のプロバイダー" in line for line in logs.output), 1)

    def test_grok_accepts_either_key_and_prefers_grok(self) -> None:
        with isolated_env(GROK_API_KEY="grok-secret", XAI_API_KEY="xai-secret"):
            self.assertEqual(_usable_key("GROK_API_KEY", "XAI_API_KEY"), "grok-secret")
        with isolated_env(LLM_PROVIDER_PRIORITY="grok", XAI_API_KEY="xai-secret"):
            chain = available_providers()
        self.assertEqual([spec.name for spec in chain], ["grok"])
        self.assertEqual(chain[0].model, "grok-2-latest")
        self.assertEqual(chain[0].base_url, "https://api.x.ai/v1")

    def test_placeholders_are_not_keys(self) -> None:
        with isolated_env(
            ANTHROPIC_API_KEY="your-api-key",
            OPENAI_API_KEY="sk-live...",
            GEMINI_API_KEY="  test-key  ",
            LLM_PROVIDER_PRIORITY="anthropic,openai,gemini",
        ):
            names = [spec.name for spec in available_providers()]
            self.assertEqual(_usable_key("GEMINI_API_KEY"), "test-key")
        self.assertEqual(names, ["gemini"])

    def test_catalog_matches_the_requested_models(self) -> None:
        expected = {
            "anthropic": (MODEL_NAME, ("ANTHROPIC_API_KEY",), None, "anthropic"),
            "gemini": ("gemini-2.5-flash-lite", ("GEMINI_API_KEY",), None, "gemini"),
            "openai": ("gpt-4o-mini", ("OPENAI_API_KEY",), None, "openai"),
            "grok": (
                "grok-2-latest",
                ("GROK_API_KEY", "XAI_API_KEY"),
                "https://api.x.ai/v1",
                "openai",
            ),
            "deepseek": (
                "deepseek-chat",
                ("DEEPSEEK_API_KEY",),
                "https://api.deepseek.com",
                "openai",
            ),
            "kimi": (
                "moonshot-v1-8k",
                ("KIMI_API_KEY",),
                "https://api.moonshot.cn/v1",
                "openai",
            ),
            "glm": (
                "glm-4-flash",
                ("GLM_API_KEY",),
                "https://open.bigmodel.cn/api/paas/v4/",
                "openai",
            ),
            "openrouter": (
                "anthropic/claude-3.5-haiku",
                ("OPENROUTER_API_KEY",),
                "https://openrouter.ai/api/v1",
                "openai",
            ),
            "nvidia_nim": (
                "meta/llama-3.3-70b-instruct",
                ("NVIDIA_NIM_API_KEY",),
                "https://integrate.api.nvidia.com/v1",
                "openai",
            ),
        }
        self.assertEqual(MODEL_NAME, "claude-haiku-5-5")
        self.assertEqual(set(PROVIDERS), set(expected))
        for name, (model, env_vars, base_url, kind) in expected.items():
            spec = PROVIDERS[name]
            self.assertEqual(spec.model, model)
            self.assertEqual(spec.env_vars, env_vars)
            self.assertEqual(spec.base_url, base_url)
            self.assertEqual(spec.kind, kind)

    def test_openai_json_object_prompt_contains_json(self) -> None:
        kwargs = openai_request_kwargs(PROVIDERS["deepseek"], _block())
        self.assertEqual(kwargs["model"], "deepseek-chat")
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        blob = "\n".join(message["content"] for message in kwargs["messages"])
        self.assertIn("json", blob)
        parsed = parse_batch_json(
            '```json\n{"items":[{"id":"b-1","translated":"訳","words":[]}]}\n```'
        )
        self.assertEqual(parsed.items[0].translated, "訳")


class FallbackTests(unittest.TestCase):
    def test_rate_limit_falls_back_and_stays_failed(self) -> None:
        calls: list[str] = []

        async def fake_dispatch(spec, blocks):
            calls.append(spec.name)
            if spec.name == "anthropic":
                exc = RuntimeError("slow down")
                exc.status_code = 429
                raise exc
            return [
                SentenceTranslation(id=block.id, translated="ジェミニ", words=[])
                for block in blocks
            ]

        async def run():
            from app import providers

            failed: set[str] = set()
            with patch.object(providers, "_dispatch", fake_dispatch):
                first = await call_with_fallback(_block(), failed)
                second = await call_with_fallback(_block(), failed)
            return first, second, set(failed)

        with isolated_env(
            LLM_PROVIDER_PRIORITY="anthropic,gemini",
            ANTHROPIC_API_KEY="test-key",
            GEMINI_API_KEY="test-key",
        ):
            with self.assertLogs("app.providers", level="INFO") as logs:
                first, second, failed = asyncio.run(run())
        self.assertEqual(calls, ["anthropic", "gemini", "gemini"])
        self.assertEqual(failed, {"anthropic"})
        self.assertEqual(first[0].translated, "ジェミニ")
        self.assertEqual(second[0].translated, "ジェミニ")
        self.assertTrue(
            any("フォールバック" in line and "gemini" in line for line in logs.output)
        )
        self.assertTrue(
            any("試行中のプロバイダー: anthropic" in line for line in logs.output)
        )

    def test_truncation_does_not_switch_provider(self) -> None:
        calls: list[str] = []

        async def fake_dispatch(spec, blocks):
            calls.append(spec.name)
            raise OutputTooLarge("too big")

        async def run():
            from app import providers

            with patch.object(providers, "_dispatch", fake_dispatch):
                await call_with_fallback(_block(), set())

        with isolated_env(
            LLM_PROVIDER_PRIORITY="anthropic,gemini",
            ANTHROPIC_API_KEY="test-key",
            GEMINI_API_KEY="test-key",
        ):
            with self.assertRaises(OutputTooLarge):
                asyncio.run(run())
        self.assertEqual(calls, ["anthropic"])

    def test_all_providers_fail_logs_and_raises(self) -> None:
        async def fake_dispatch(spec, blocks):
            exc = RuntimeError("nope")
            exc.status_code = 500
            raise exc

        async def run():
            from app import providers

            with patch.object(providers, "_dispatch", fake_dispatch):
                await call_with_fallback(_block(), set())

        with isolated_env(
            LLM_PROVIDER_PRIORITY="anthropic,openai",
            ANTHROPIC_API_KEY="test-key",
            OPENAI_API_KEY="test-key",
        ):
            with self.assertLogs("app.providers", level="INFO") as logs:
                with self.assertRaises(AllProvidersFailed) as caught:
                    asyncio.run(run())
        self.assertEqual(
            [name for name, _message in caught.exception.errors],
            ["anthropic", "openai"],
        )
        self.assertIn("全プロバイダーが失敗しました", str(caught.exception))
        self.assertTrue(any("フォールバック" in line for line in logs.output))
        self.assertTrue(
            any(
                line.startswith("ERROR:") and "全プロバイダーが失敗しました" in line
                for line in logs.output
            )
        )

    def test_programming_error_does_not_fall_back(self) -> None:
        calls: list[str] = []

        async def fake_dispatch(spec, blocks):
            calls.append(spec.name)
            raise ValueError("bug")

        async def run():
            from app import providers

            with patch.object(providers, "_dispatch", fake_dispatch):
                await call_with_fallback(_block(), set())

        with isolated_env(
            LLM_PROVIDER_PRIORITY="anthropic,gemini",
            ANTHROPIC_API_KEY="test-key",
            GEMINI_API_KEY="test-key",
        ):
            with self.assertRaises(ValueError):
                asyncio.run(run())
        self.assertEqual(calls, ["anthropic"])

    def test_job_fails_clearly_when_no_key_is_set(self) -> None:
        from fastapi.testclient import TestClient

        from main import app

        reset_jobs()
        try:
            with isolated_env():
                with TestClient(app) as client:
                    accepted = client.post(
                        "/api/translate",
                        files={"file": ("notes.pdf", b"%PDF-1.4\n", "application/pdf")},
                    )
                    self.assertEqual(accepted.status_code, 200)
                    body = client.get(f"/api/status/{accepted.json()['job_id']}").json()
            self.assertEqual(body["status"], "failed")
            self.assertIn("APIキー", body["error"])
            self.assertIn("LLM_PROVIDER_PRIORITY", body["error"])
        finally:
            reset_jobs()

    def test_public_error_redacts_provider_keys(self) -> None:
        message = public_error(
            RuntimeError("rejected sk-ant-abc_DEF and sk-proj-abcdef12")
        )
        self.assertNotIn("sk-ant", message)
        self.assertNotIn("sk-proj", message)


if __name__ == "__main__":
    unittest.main()
