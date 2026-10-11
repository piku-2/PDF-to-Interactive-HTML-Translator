"""Spec checks for annotation, parsing, batching, and the polling API."""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.annotator import safe_word_annotation
from app.config import BATCH_SIZE, MAX_CONCURRENCY, MAX_RETRIES, MODEL_NAME
from app.html_builder import apply_translations, build_html
from app.jobs import create_job, get_job, process_job, reset_jobs
from app.models import SentenceTranslation, TextBlock, WordTranslation
from app.parser import markdown_to_blocks
from app.translator import get_client, reset_client_state, translate_blocks

SAMPLE_MARKDOWN = """
# Attention Is All You Need

This paper presents a new architecture. It relies solely on attention.

| Model | BLEU |
| --- | --- |
| Transformer | 28.4 |

## Conclusion

Future work remains.
"""


class AnnotationTests(unittest.TestCase):
    def test_escapes_before_inserting_tags(self) -> None:
        result = safe_word_annotation('<script>alert("x")</script>', [])
        self.assertNotIn("<script>", result)
        self.assertIn("&lt;script&gt;", result)

    def test_word_boundary_skips_partial_words(self) -> None:
        result = safe_word_annotation(
            "The cat sat in the catalog.",
            [WordTranslation(word="cat", context_meaning="猫")],
        )
        self.assertEqual(result.count('class="word-hover"'), 1)
        self.assertIn(">cat</span>", result)
        self.assertIn("catalog", result)
        self.assertNotIn(">cat</span>alog", result)

    def test_longest_phrase_wins_and_does_not_nest(self) -> None:
        result = safe_word_annotation(
            "machine learning is a field.",
            [
                WordTranslation(word="learning", context_meaning="学習"),
                WordTranslation(word="machine learning", context_meaning="機械学習"),
            ],
        )
        self.assertEqual(result.count('class="word-hover"'), 1)
        self.assertIn('data-meaning="機械学習"', result)
        self.assertIn(">machine learning</span>", result)

    def test_shorter_word_still_matches_outside_the_longer_phrase(self) -> None:
        result = safe_word_annotation(
            "Deep learning uses machine learning.",
            [
                WordTranslation(word="learning", context_meaning="学習"),
                WordTranslation(word="machine learning", context_meaning="機械学習"),
            ],
        )
        self.assertEqual(result.count('class="word-hover"'), 2)
        self.assertIn("Deep <span", result)
        self.assertIn(">machine learning</span>", result)

    def test_attribute_quotes_and_embedded_word_do_not_break_markup(self) -> None:
        result = safe_word_annotation(
            "Use the API carefully.",
            [WordTranslation(word="API", context_meaning='APIの"呼び出し"')],
        )
        self.assertEqual(result.count("<span"), 1)
        self.assertEqual(result.count("</span>"), 1)
        self.assertIn("&quot;", result)
        self.assertIn(">API</span>", result)

    def test_keeps_original_casing(self) -> None:
        result = safe_word_annotation(
            "The Transformer works.",
            [WordTranslation(word="transformer", context_meaning="変換器")],
        )
        self.assertIn(">Transformer</span>", result)

    def test_apostrophe_matches_after_escape(self) -> None:
        result = safe_word_annotation(
            "Please don't stop.",
            [WordTranslation(word="don't", context_meaning="しないで")],
        )
        self.assertEqual(result.count('class="word-hover"'), 1)
        self.assertIn("しないで", result)


class ParserTests(unittest.TestCase):
    def test_paragraph_rules(self) -> None:
        blocks = markdown_to_blocks(SAMPLE_MARKDOWN)
        self.assertEqual(
            [block.type for block in blocks],
            ["heading", "sentence", "sentence", "table", "heading", "sentence"],
        )
        self.assertEqual(
            [block.id for block in blocks], ["b-1", "b-2", "b-3", "b-4", "b-5", "b-6"]
        )
        self.assertEqual(blocks[0].original_text, "Attention Is All You Need")
        self.assertEqual(blocks[0].level, 1)
        self.assertEqual(
            blocks[1].original_text, "This paper presents a new architecture."
        )
        self.assertEqual(blocks[2].original_text, "It relies solely on attention.")
        self.assertTrue(blocks[3].original_text.startswith("| Model |"))
        self.assertIn("| --- |", blocks[3].original_text)
        self.assertEqual(blocks[4].level, 2)

    def test_real_pdf_roundtrip(self) -> None:
        import pymupdf

        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 80), "Attention Is All You Need", fontsize=22)
        page.insert_text(
            (72, 130), "This paper presents a new architecture.", fontsize=11
        )
        payload = document.tobytes()
        document.close()

        from app.parser import parse_pdf_to_blocks

        blocks = parse_pdf_to_blocks(payload)
        text = "\n".join(block.original_text for block in blocks)
        self.assertTrue(blocks)
        self.assertIn("architecture", text)


class HtmlTests(unittest.TestCase):
    def test_single_file_document_has_required_features(self) -> None:
        blocks = markdown_to_blocks(SAMPLE_MARKDOWN)
        results = {
            "b-2": SentenceTranslation(
                id="b-2",
                translated="本論文は新しい構造を提案する。",
                words=[
                    WordTranslation(word="architecture", context_meaning="構造"),
                    WordTranslation(
                        word="new architecture", context_meaning="新しい構造"
                    ),
                ],
            ),
            "b-3": SentenceTranslation(
                id="b-3", translated="注意機構だけに依存する。", words=[]
            ),
            "b-4": SentenceTranslation(
                id="b-4",
                translated="| モデル | BLEU |\n| --- | --- |\n| Transformer | 28.4 |",
                words=[WordTranslation(word="Transformer", context_meaning="変換器")],
            ),
            "b-1": SentenceTranslation(
                id="b-1", translated="注意機構こそがすべて", words=[]
            ),
            "b-5": SentenceTranslation(id="b-5", translated="結論", words=[]),
            "b-6": SentenceTranslation(
                id="b-6", translated="今後の課題が残る。", words=[]
            ),
        }
        apply_translations(blocks, results)
        document = build_html(blocks, title="paper")

        self.assertIn('class="mode-default"', document)
        self.assertIn("mode-full-ja", document)
        self.assertIn("mode-block", document)
        self.assertIn('data-theme="system"', document)
        self.assertIn("--base-font-size", document)
        self.assertIn('class="table-container"', document)
        self.assertIn("overflow-x: auto", document)
        self.assertIn('class="word-hover"', document)
        self.assertIn('data-meaning="新しい構造"', document)
        self.assertIn('data-meaning="変換器"', document)
        self.assertIn("<table>", document)
        self.assertIn("本論文は新しい構造を提案する。", document)
        self.assertIn('href="#b-1"', document)
        self.assertIn("注意機構こそがすべて", document)
        self.assertNotIn("<script src", document)
        self.assertNotIn("<link", document)
        self.assertNotIn("http://", document)
        self.assertNotIn("https://", document)
        self.assertEqual(document.count("<span"), document.count("</span>"))


class TranslatorTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_client_state()

    def tearDown(self) -> None:
        reset_client_state()

    def test_client_retries(self) -> None:
        client = get_client()
        self.assertEqual(client.max_retries, MAX_RETRIES)

    def test_batches_of_25_and_semaphore(self) -> None:
        state = {"current": 0, "peak": 0, "sizes": []}
        release = asyncio.Event()

        class FakeMessages:
            async def create(self, **kwargs):
                payload = json.loads(kwargs["messages"][0]["content"].split("\n", 1)[1])
                state["sizes"].append(len(payload))
                state["current"] += 1
                state["peak"] = max(state["peak"], state["current"])
                if state["current"] >= MAX_CONCURRENCY:
                    release.set()
                await release.wait()
                state["current"] -= 1
                self.assert_shape(kwargs, payload)
                items = [
                    {"id": row["id"], "translated": "訳", "words": []}
                    for row in payload
                ]
                block = SimpleNamespace(
                    type="tool_use", name="submit_translations", input={"items": items}
                )
                return SimpleNamespace(
                    stop_reason="end_turn", content=[block], usage=None
                )

            @staticmethod
            def assert_shape(kwargs, payload):
                assert kwargs["model"] == MODEL_NAME
                assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
                assert kwargs["tools"][0]["cache_control"] == {"type": "ephemeral"}
                assert kwargs["tool_choice"] == {
                    "type": "tool",
                    "name": "submit_translations",
                }
                assert payload

        fake = SimpleNamespace(messages=FakeMessages())
        blocks = [
            TextBlock(
                id=f"b-{index}",
                type="sentence",
                original_text=f"Sentence {index} mentions attention.",
            )
            for index in range(1, BATCH_SIZE * 5 + 6)
        ]

        async def run():
            from app import translator

            translator._client = fake
            with patch.dict(
                os.environ,
                {"ANTHROPIC_API_KEY": "test-key", "LLM_PROVIDER_PRIORITY": "anthropic"},
            ):
                return await translate_blocks(blocks)

        result = asyncio.run(run())
        self.assertEqual(set(result), {block.id for block in blocks})
        self.assertEqual(state["sizes"], [25, 25, 25, 25, 25, 5])
        self.assertEqual(state["peak"], MAX_CONCURRENCY)

    def test_splits_when_output_is_truncated(self) -> None:
        class FakeMessages:
            async def create(self, **kwargs):
                payload = json.loads(kwargs["messages"][0]["content"].split("\n", 1)[1])
                if len(payload) > 1:
                    return SimpleNamespace(
                        stop_reason="max_tokens", content=[], usage=None
                    )
                items = [{"id": payload[0]["id"], "translated": "分割訳", "words": []}]
                block = SimpleNamespace(
                    type="tool_use", name="submit_translations", input={"items": items}
                )
                return SimpleNamespace(
                    stop_reason="end_turn", content=[block], usage=None
                )

        fake = SimpleNamespace(messages=FakeMessages())
        blocks = [
            TextBlock(id=f"b-{index}", type="sentence", original_text=f"Line {index}.")
            for index in range(1, 5)
        ]

        async def run():
            from app import translator

            translator._client = fake
            with patch.dict(
                os.environ,
                {"ANTHROPIC_API_KEY": "test-key", "LLM_PROVIDER_PRIORITY": "anthropic"},
            ):
                return await translate_blocks(blocks)

        result = asyncio.run(run())
        self.assertEqual(set(result), {"b-1", "b-2", "b-3", "b-4"})
        self.assertTrue(all(item.translated == "分割訳" for item in result.values()))


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_jobs()

    def tearDown(self) -> None:
        reset_jobs()

    def test_polling_flow(self) -> None:
        from fastapi.testclient import TestClient

        from main import app

        def fake_parse(pdf_bytes: bytes):
            self.assertTrue(pdf_bytes.startswith(b"%PDF"))
            return [
                TextBlock(id="b-1", type="heading", original_text="Results", level=1),
                TextBlock(
                    id="b-2",
                    type="sentence",
                    original_text="The model uses attention.",
                    level=0,
                ),
                TextBlock(
                    id="b-3",
                    type="table",
                    original_text="| Model | BLEU |\n| --- | --- |\n| Transformer | 28.4 |",
                ),
            ]

        async def fake_translate(blocks, on_progress=None):
            if on_progress:
                on_progress(1, 1)
            rendered = {}
            for block in blocks:
                words = []
                if "attention" in block.original_text:
                    words.append(
                        WordTranslation(word="attention", context_meaning="注意")
                    )
                if block.type == "table":
                    words.append(
                        WordTranslation(word="Transformer", context_meaning="変換器")
                    )
                rendered[block.id] = SentenceTranslation(
                    id=block.id, translated=f"訳:{block.type}", words=words
                )
            return rendered

        with (
            patch("app.jobs.parse_pdf_to_blocks", fake_parse),
            patch("app.jobs.translate_blocks", fake_translate),
        ):
            with TestClient(app) as client:
                rejected = client.post(
                    "/api/translate",
                    files={"file": ("notes.txt", b"hello", "text/plain")},
                )
                self.assertEqual(rejected.status_code, 400)

                not_pdf = client.post(
                    "/api/translate",
                    files={"file": ("notes.pdf", b"not-a-pdf", "application/pdf")},
                )
                self.assertEqual(not_pdf.status_code, 400)

                self.assertEqual(client.get("/api/status/missing").status_code, 404)
                self.assertEqual(client.get("/api/download/missing").status_code, 404)

                pending = create_job("pending.pdf")
                pending.status = "translating"
                self.assertEqual(
                    client.get(f"/api/download/{pending.id}").status_code, 409
                )

                created = client.post(
                    "/api/translate",
                    files={"file": ("論文.pdf", b"%PDF-1.4\n%", "application/pdf")},
                )
                self.assertEqual(created.status_code, 200)
                job_id = created.json()["job_id"]
                status = client.get(f"/api/status/{job_id}")
                self.assertEqual(status.status_code, 200)
                body = status.json()
                self.assertEqual(body["status"], "completed")
                self.assertEqual(body["progress"], 100)
                self.assertEqual(body["job_id"], job_id)

                downloaded = client.get(f"/api/download/{job_id}")
                self.assertEqual(downloaded.status_code, 200)
                self.assertIn("text/html", downloaded.headers["content-type"])
                self.assertIn(
                    "filename*=UTF-8''", downloaded.headers["content-disposition"]
                )
                html = downloaded.text
                self.assertIn('class="word-hover"', html)
                self.assertIn('data-meaning="注意"', html)
                self.assertIn('class="table-container"', html)
                self.assertIn("訳:sentence", html)
                self.assertIn("mode-block", html)
                self.assertIn("--base-font-size", html)

        stored = get_job(job_id)
        self.assertIsNotNone(stored)
        self.assertEqual(stored.download_name, "論文-ja.html")

    def test_failure_is_visible_on_the_status_api(self) -> None:
        from fastapi.testclient import TestClient

        from main import app

        def fake_parse(pdf_bytes: bytes):
            return [TextBlock(id="b-1", type="sentence", original_text="Hello world.")]

        async def fake_translate(blocks, on_progress=None):
            raise RuntimeError("モデルが応答しませんでした。")

        with (
            patch("app.jobs.parse_pdf_to_blocks", fake_parse),
            patch("app.jobs.translate_blocks", fake_translate),
        ):
            with TestClient(app) as client:
                created = client.post(
                    "/api/translate",
                    files={"file": ("paper.pdf", b"%PDF-1.4\n", "application/pdf")},
                )
                job_id = created.json()["job_id"]
                body = client.get(f"/api/status/{job_id}").json()
                self.assertEqual(body["status"], "failed")
                self.assertIn("モデルが応答しませんでした", body["error"])
                self.assertEqual(client.get(f"/api/download/{job_id}").status_code, 409)


class JobDirectTests(unittest.TestCase):
    def tearDown(self) -> None:
        reset_jobs()

    def test_process_job_records_queued_then_completed(self) -> None:
        reset_jobs()
        job = create_job("demo.pdf")
        self.assertEqual(job.status, "queued")
        self.assertEqual(job.progress, 0)

        def fake_parse(pdf_bytes: bytes):
            return [
                TextBlock(id="b-1", type="sentence", original_text="Attention matters.")
            ]

        async def fake_translate(blocks, on_progress=None):
            if on_progress:
                on_progress(1, 1)
            return {
                "b-1": SentenceTranslation(
                    id="b-1",
                    translated="注意が重要だ。",
                    words=[WordTranslation(word="Attention", context_meaning="注意")],
                )
            }

        with (
            patch("app.jobs.parse_pdf_to_blocks", fake_parse),
            patch("app.jobs.translate_blocks", fake_translate),
        ):
            asyncio.run(process_job(job.id, b"%PDF-1.4"))
        self.assertEqual(job.status, "completed")
        self.assertIn("注意が重要だ。", job.html or "")
        self.assertIn('data-meaning="注意"', job.html or "")


if __name__ == "__main__":
    unittest.main()
