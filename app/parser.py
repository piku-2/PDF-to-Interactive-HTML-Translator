"""PDF to block parsing (SPEC.md section 5.1)."""

from __future__ import annotations

import re

from app.models import TextBlock

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_RULE_RE = re.compile(r"[-*_]{3,}")
_nltk_ready = False


def ensure_nltk() -> None:
    """Load NLTK's English sentence tokenizer, downloading punkt_tab if needed."""
    global _nltk_ready
    if _nltk_ready:
        return
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        nltk.data.find("tokenizers/punkt_tab")
    _nltk_ready = True


def parse_pdf_to_blocks(pdf_bytes: bytes) -> list[TextBlock]:
    """Convert PDF bytes to ordered heading, sentence, and table blocks."""
    ensure_nltk()
    import pymupdf
    import pymupdf4llm

    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"PDFを開けませんでした: {exc}") from exc

    try:
        if document.is_encrypted and not document.authenticate(""):
            raise ValueError("パスワード付きPDFには対応していません。")
        if document.page_count == 0:
            raise ValueError("PDFにページがありません。")
        markdown = pymupdf4llm.to_markdown(document, show_progress=False)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"PDFを解析できませんでした: {exc}") from exc
    finally:
        document.close()

    if isinstance(markdown, list):
        parts: list[str] = []
        for chunk in markdown:
            if isinstance(chunk, dict) and "text" in chunk:
                parts.append(str(chunk["text"]))
            else:
                parts.append(str(chunk))
        markdown = "\n\n".join(parts)
    if not isinstance(markdown, str):
        raise ValueError("PDFのMarkdown変換結果を解釈できませんでした。")
    return markdown_to_blocks(markdown)


def markdown_to_blocks(markdown: str) -> list[TextBlock]:
    """Split Markdown into blocks.

    Blank lines separate paragraphs (SPEC 5.1). A paragraph whose lines start
    with ``|`` or an HTML ``<table>`` becomes one table block. A line that
    starts with ``#`` becomes a heading. Everything else is split into
    sentences with ``nltk.sent_tokenize``.
    """
    ensure_nltk()
    from nltk.tokenize import sent_tokenize

    normalized = markdown.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n[ \t]*\n", normalized)
    blocks: list[TextBlock] = []

    def add(block_type: str, content: str, level: int = 0) -> None:
        content = content.strip()
        if not content:
            return
        blocks.append(
            TextBlock(
                id=f"b-{len(blocks) + 1}",
                type=block_type,  # type: ignore[arg-type]
                original_text=content,
                level=level,
            )
        )

    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
        if not lines:
            continue
        if len(lines) == 1 and _RULE_RE.fullmatch(lines[0]):
            continue

        if any("<table" in line.lower() for line in lines) or all(_is_table_line(line) for line in lines):
            add("table", "\n".join(lines))
            continue

        text_buffer: list[str] = []
        table_buffer: list[str] = []

        def flush_text() -> None:
            if not text_buffer:
                return
            chunk = "\n".join(text_buffer).strip()
            text_buffer.clear()
            if not chunk or _RULE_RE.fullmatch(chunk):
                return
            for sentence in sent_tokenize(chunk, language="english"):
                add("sentence", sentence)

        def flush_table() -> None:
            if not table_buffer:
                return
            add("table", "\n".join(table_buffer))
            table_buffer.clear()

        for line in lines:
            heading = _HEADING_RE.match(line)
            if heading and not line.startswith("|"):
                flush_text()
                flush_table()
                add("heading", heading.group(2), level=len(heading.group(1)))
                continue
            if _is_table_line(line):
                flush_text()
                table_buffer.append(line)
                continue
            flush_table()
            text_buffer.append(line)

        flush_text()
        flush_table()

    return blocks


def _is_table_line(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("|") or stripped.lower().startswith("<table")
