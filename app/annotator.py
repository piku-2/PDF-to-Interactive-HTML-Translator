"""Safe word-hover annotation (SPEC.md section 5.2)."""

from __future__ import annotations

import html
import re
from typing import List

from app.models import WordTranslation

# A span already inserted for a longer phrase, including the text inside it.
# Later, shorter words must not match inside that phrase or its attributes.
_PROTECTED_SPAN_RE = re.compile(
    r'(<span class="word-hover" [^>]*>.*?</span>)',
    re.DOTALL,
)


def _substitute_outside_tags(text: str, pattern: re.Pattern[str], replacement: str) -> str:
    parts = _PROTECTED_SPAN_RE.split(text)
    for index, part in enumerate(parts):
        if part.startswith('<span class="word-hover"'):
            continue
        parts[index] = pattern.sub(replacement, part)
    return "".join(parts)


def safe_word_annotation(original_text: str, words: List[WordTranslation]) -> str:
    """Inject word-hover spans without breaking the surrounding text.

    Escape order and the ``\\b`` boundary match follow SPEC.md section 5.2:
    escape the whole source first, then replace longer phrases first.
    Each replacement uses ``html.escape(..., quote=True)`` for the attribute
    and ``re.IGNORECASE`` so the original casing is kept via the ``\\1`` group.
    Matches are applied only outside tags already inserted, so a shorter word
    cannot land inside ``data-meaning`` or split an earlier span.
    """
    # 1. 原文全体を先にエスケープ
    escaped_text = html.escape(original_text)

    # 2. 単語長が長い順（熟語優先）にソート
    sorted_words = sorted(words, key=lambda item: len(item.word), reverse=True)

    seen: set[str] = set()
    for item in sorted_words:
        if not item.word or not item.word.strip():
            continue
        # The same surface form twice would nest a span inside itself.
        dedupe_key = item.word.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        target_word = html.escape(item.word)
        # 属性値用に引用符等を二重エスケープ
        meaning = html.escape(item.context_meaning, quote=True)
        # re.sub treats backslashes in the replacement template as group refs.
        meaning_for_repl = meaning.replace("\\", "\\\\")

        # \b による単語境界マッチ（英数字以外の境界も考慮）
        pattern = re.compile(rf"\b({re.escape(target_word)})\b", re.IGNORECASE)
        replacement = f'<span class="word-hover" data-meaning="{meaning_for_repl}">\\1</span>'
        escaped_text = _substitute_outside_tags(escaped_text, pattern, replacement)

    return escaped_text
