# PDF-to-Interactive-HTML Translator 仕様書

## 1. プロジェクト概要
アップロードされた英語PDF（論文・技術文書・レポート等）を解析し、インタラクティブ機能（文脈単語ホバー、3表示モード、テーマ切り替え、目次サイドバー）が組み込まれた**1ファイル完結型の日本語対訳HTML**を生成するFastAPIバックエンドWebアプリケーション。

---

## 2. 技術スタック & 主要ライブラリ
- **Language:** Python 3.10+
- **Framework:** FastAPI, Uvicorn
- **PDF Parsing (Non-AI):** `pymupdf4llm`
- **Text Splitting:** `nltk` (`nltk.sent_tokenize`)
- **LLM Engine:** Anthropic Claude Haiku 5.5 (`claude-haiku-5-5`) via `anthropic` Python SDK
  - `max_retries` 有効化（指数バックオフ）
  - Message Batches API & Prompt Caching 活用
- **Template Engine:** Jinja2 (HTML/CSS/JS 合成)

---

## 3. 全体アーキテクチャ & データフロー

```
[ PDF Upload ]
     │
     ▼
[ Step 1: PDF to Structured Text (PyMuPDF4LLM) ]
     │  - Markdown変換 (見出し, 本文, 表)
     │
     ▼
[ Step 2: Block & Sentence Parsing ]
     │  - 表 (| ...) はそのまま1ブロック化
     │  - 通常段落は NLTK で1文ごとに分割して軽量JSON化
     │
     ▼
[ Step 3: LLM Batch Processing (Claude Haiku 5.5) ]
     │  - 25文単位のバッチ処理 (BackgroundTasks)
     │  - Structured Output (Tool Use) で和訳 & 文脈単語抽出
     │
     ▼
[ Step 4: HTML Safe Annotation ]
     │  - html.escape() 通過後の単語境界 (\b) 正規表現置換
     │  - Longest-first 順置換で部分一致・崩壊を完全防止
     │
     ▼
[ Step 5: HTML Synthesis & Output ]
     └─ インラインCSS/JS組み込み単一 `.html` ファイル生成
```

---

## 4. データ構造定義 (Pydantic / JSON Schema)

### 4.1 LLM 出力スキーマ (Tool Use / Structured Output)
```python
from pydantic import BaseModel, Field
from typing import List

class WordTranslation(BaseModel):
    word: str = Field(description="本文中に登場しているそのままの単語・熟語（原形ではなく本文内の完全一致表記）")
    context_meaning: str = Field(description="この文脈における具体的な日本語の意味・ニュアンス")

class SentenceTranslation(BaseModel):
    id: str = Field(description="入力ブロックのID (例: 'b-12')")
    translated: str = Field(description="1文の自然な日本語訳")
    words: List[WordTranslation] = Field(default_factory=list, description="専門用語や難解語の文脈訳（0〜3個）")

class BatchTranslationResponse(BaseModel):
    items: List[SentenceTranslation]
```

### 4.2 バックエンド内部ブロックモデル
```python
class TextBlock(BaseModel):
    id: str
    type: str  # 'heading' | 'sentence' | 'table'
    original_text: str
    annotated_html: Optional[str] = None
    translated_text: Optional[str] = None
```

---

## 5. コア処理ロジック仕様

### 5.1 PDF解析 & 文分割 (`parse_pdf_to_blocks`)
1. `pymupdf4llm.to_markdown()` でPDFをMarkdownに変換。
2. 段落（`\n\n`）単位でループ処理：
   - `#` で始まる行 ➔ `type="heading"`
   - `|` で始まる行 ➔ `type="table"`（文分割スキップ）
   - それ以外 ➔ `nltk.sent_tokenize()` で1文ずつに分解し `type="sentence"`

### 5.2 安全な単語ホバーHTMLタグ注入 (`safe_word_annotation`)
文字列破壊およびHTML崩壊を防ぐため、以下の順序を厳守すること：

```python
import html
import re

def safe_word_annotation(original_text: str, words: List[WordTranslation]) -> str:
    # 1. 原文全体を先にエスケープ
    escaped_text = html.escape(original_text)
    
    # 2. 単語長が長い順（熟語優先）にソート
    sorted_words = sorted(words, key=lambda x: len(x.word), reverse=True)
    
    for item in sorted_words:
        target_word = html.escape(item.word)
        # 属性値用に引用符等を二重エスケープ
        meaning = html.escape(item.context_meaning, quote=True)
        
        # \b による単語境界マッチ（英数字以外の境界も考慮）
        pattern = re.compile(rf'\b({re.escape(target_word)})\b', re.IGNORECASE)
        replacement = f'<span class="word-hover" data-meaning="{meaning}">\\1</span>'
        escaped_text = pattern.sub(replacement, escaped_text)
        
    return escaped_text
```

### 5.3 表（Table）の処理
- Markdownの表構造（`| Col1 | Col2 |`）は維持したままLLMに送り、セル内言語のみ翻訳させる。
- HTML合成時、Markdownの表を `<table>` タグへ変換し、`<div class="table-container">` で囲んで `overflow-x: auto;` を適用する。

---

## 6. 非同期ジョブ & バッチ処理設計 (FastAPI)

巨大PDF（50〜200P）対応のため、同期レスポンスではなくジョブ管理方式をとる。

### エンドポイント仕様
1. **`POST /api/translate`**: PDFファイルを受け取り `job_id` (UUID) を即座に返す。
2. **`GET /api/status/{job_id}`**: 進捗率（`status`: `queued` | `parsing` | `translating` | `completed` | `failed`, `progress`: 0〜100）を返す。
3. **`GET /api/download/{job_id}`**: 完成した `.html` ファイルをレスポンス。

### バッチ & API実行パラメータ
- バッチサイズ: 25ブロック単位
- 同時実行制限: `asyncio.Semaphore(5)` で過剰リクエストを制限
- Anthropic SDK Client: `max_retries=5` を設定

---

## 7. 生成される単一HTML/CSS/JSの仕様

### 7.1 機能要件
- **3つの表示モード (CSS切り替え)**:
  - `mode-default`: 原文のみ表示（単語ホバーで `data-meaning` ツールチップ表示）
  - `mode-full-ja`: 日本語訳のみ表示
  - `mode-block`: 1文ごとに原文と日本語訳を交互並列表示
- **テーマ切り替え**: Dark / Light / System 追従 (`data-theme` 属性操作)
- **フォントサイズ調整**: CSS変数 `--base-font-size` を JS で変更
- **サイドバー**: 見出し（`heading`）から自動抽出した目次 (TOC) スムーススクロール

### 7.2 制約事項
- 外部CSS/JSライブラリ (Tailwind, Bootstrap, CDN等) への依存を**完全排除**。
- Pure CSS / Pure Vanilla JS のみを使用し、オフライン環境（ローカルChrome）で単体起動可能にすること。