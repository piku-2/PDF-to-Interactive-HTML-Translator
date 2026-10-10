"""Single-file HTML synthesis (SPEC.md sections 5.3 and 7)."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from app.annotator import safe_word_annotation
from app.models import SentenceTranslation, TextBlock
from app.tables import markdown_table_to_html

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_env = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    autoescape=select_autoescape(enabled_extensions=("html", "j2"), default=True),
)
_TAG_SPLIT = re.compile(r"(<[^>]*>)")


@dataclass
class ViewBlock:
    id: str
    type: str
    level: int
    tag: str
    original_html: Markup
    translated_html: Markup
    toc_original: str
    toc_translated: str


def apply_translations(
    blocks: list[TextBlock],
    results: dict[str, SentenceTranslation],
) -> None:
    """Fill translated text and original word-hover HTML for every block."""
    for block in blocks:
        item: Optional[SentenceTranslation] = results.get(block.id)
        words = list(item.words) if item else []
        block.translated_text = item.translated if item and item.translated else ""
        if block.type == "table":
            block.annotated_html = markdown_table_to_html(block.original_text, words)
        else:
            block.annotated_html = safe_word_annotation(block.original_text, words)


def build_html(blocks: list[TextBlock], title: str) -> str:
    """Render one self-contained HTML document."""
    view_blocks = [_view_block(block) for block in blocks]
    document_title = _document_title(blocks, title)
    template = _env.get_template("document.html.j2")
    return template.render(title=document_title, blocks=view_blocks)


def _document_title(blocks: list[TextBlock], fallback: str) -> str:
    for block in blocks:
        if block.type == "heading" and block.original_text.strip():
            return block.original_text.strip()
    return fallback or "対訳ドキュメント"


def _view_block(block: TextBlock) -> ViewBlock:
    level = block.level if block.type == "heading" else 0
    tag = f"h{min(max(level, 1), 6)}" if block.type == "heading" else "div"
    if block.type == "table":
        original = Markup(block.annotated_html or markdown_table_to_html(block.original_text, []))
        translated_source = (block.translated_text or "").strip()
        if translated_source:
            translated = Markup(markdown_table_to_html(translated_source, []))
        else:
            translated = Markup('<div class="table-container"><p class="missing">（訳なし）</p></div>')
        return ViewBlock(
            id=block.id,
            type=block.type,
            level=level,
            tag=tag,
            original_html=original,
            translated_html=translated,
            toc_original="",
            toc_translated="",
        )

    original_fragment = block.annotated_html
    if original_fragment is None:
        original_fragment = html.escape(block.original_text)
    translated_text = (block.translated_text or "").strip()
    if translated_text:
        translated = Markup(html.escape(translated_text).replace("\n", "<br>"))
    else:
        translated = Markup('<span class="missing">（訳なし）</span>')
    return ViewBlock(
        id=block.id,
        type=block.type,
        level=level or 1,
        tag=tag,
        original_html=Markup(_breaks_outside_tags(original_fragment)),
        translated_html=translated,
        toc_original=block.original_text if block.type == "heading" else "",
        toc_translated=translated_text if block.type == "heading" else "",
    )


def _breaks_outside_tags(fragment: str) -> str:
    parts = _TAG_SPLIT.split(fragment)
    rendered: list[str] = []
    for part in parts:
        if part.startswith("<") and part.endswith(">"):
            rendered.append(part)
        else:
            rendered.append(part.replace("\n", "<br>"))
    return "".join(rendered)
