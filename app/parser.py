"""PDF to block parsing (SPEC.md section 5.1)."""

from __future__ import annotations

import re

from app.models import TextBlock

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_RULE_RE = re.compile(r"[-*_]{3,}")
_FENCE_RE = re.compile(r"^`{3,}")
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
    """Split Markdown into blocks (SPEC 5.1).

    ``#`` / ``##`` / ``###`` become headings at levels 1-3 (deeper marks stay
    headings too). Lines that start with ``|`` stay one table. A fenced code
    block stays one ``code`` block, including blank lines inside the fence.
    Lines that start with ``>`` become ``quote`` blocks, split into sentences.
    Other prose is split with ``nltk.sent_tokenize``.
    """
    ensure_nltk()
    from nltk.tokenize import sent_tokenize

    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[TextBlock] = []
    index = 0

    def add(block_type: str, content: str, level: int = 0, *, keep_indent: bool = False) -> None:
        if keep_indent:
            content = content.strip("\n")
            if not content.strip():
                return
        else:
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

    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if _FENCE_RE.match(stripped):
            index += 1
            body: list[str] = []
            while index < len(lines) and not _FENCE_RE.match(lines[index].strip()):
                body.append(lines[index])
                index += 1
            if index < len(lines):
                index += 1
            add("code", "\n".join(body), keep_indent=True)
            continue
        if _RULE_RE.fullmatch(stripped):
            index += 1
            continue

        heading = _HEADING_RE.match(stripped)
        if heading and not stripped.startswith("|"):
            level = min(len(heading.group(1)), 6)
            add("heading", heading.group(2), level=level)
            index += 1
            continue

        if stripped.lower().startswith("<table") or _is_table_line(stripped):
            collected: list[str] = []
            if stripped.lower().startswith("<table"):
                while index < len(lines):
                    collected.append(lines[index].strip())
                    closed = "</table>" in lines[index].lower()
                    index += 1
                    if closed:
                        break
            else:
                while index < len(lines) and _is_table_line(lines[index].strip()):
                    collected.append(lines[index].strip())
                    index += 1
            add("table", "\n".join(collected))
            continue

        if stripped.startswith(">"):
            quote_lines: list[str] = []
            while index < len(lines):
                current = lines[index].strip()
                if current.startswith(">"):
                    quote_lines.append(_strip_quote(lines[index]))
                    index += 1
                    continue
                next_index = index + 1
                if (
                    not current
                    and next_index < len(lines)
                    and lines[next_index].strip().startswith(">")
                ):
                    index += 1
                    continue
                break
            quote_text = " ".join(part.strip() for part in quote_lines if part.strip())
            for sentence in sent_tokenize(quote_text, language="english"):
                add("quote", sentence)
            continue

        paragraph: list[str] = []
        while index < len(lines) and not _is_structural(lines[index]):
            paragraph.append(lines[index].strip())
            index += 1
        chunk = "\n".join(paragraph).strip()
        if chunk and not _RULE_RE.fullmatch(chunk):
            for sentence in sent_tokenize(chunk, language="english"):
                add("sentence", sentence)

    return blocks


def _is_structural(line: str) -> bool:
    stripped = line.strip()
    if not stripped or _RULE_RE.fullmatch(stripped):
        return True
    if _FENCE_RE.match(stripped) or stripped.startswith(">"):
        return True
    if _is_table_line(stripped) or stripped.lower().startswith("<table"):
        return True
    return _HEADING_RE.match(stripped) is not None and not stripped.startswith("|")


def _strip_quote(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith(">"):
        stripped = stripped[1:]
        if stripped.startswith(" "):
            stripped = stripped[1:]
    return stripped


def _is_table_line(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("|") or stripped.lower().startswith("<table")
