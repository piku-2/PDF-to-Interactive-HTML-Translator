# PDF-to-Interactive-HTML Translator 仕様書

## 1. プロジェクト概要
アップロードされた英語PDF（論文・技術文書・レポート等）を解析し、高度なインタラクティブ機能（文脈単語ホバー、1文ホバー訳、3表示モード切り替え、マルチLLMフォールバック、目次サイドバー）が組み込まれた**1ファイル完結型の高品質日本語対訳HTML**を生成するFastAPIバックエンドWebアプリケーション。

---

## 2. 技術スタック & 主要ライブラリ
- **Language:** Python 3.10+
- **Framework:** FastAPI, Uvicorn
- **PDF Parsing (Non-AI):** `pymupdf4llm`
- **Text Splitting:** `nltk` (`nltk.sent_tokenize`)
- **LLM Engine & Priority Fallback:**
  - 優先度設定: `.env` の `LLM_PROVIDER_PRIORITY` (デフォルト: `anthropic,gemini,openai,grok,deepseek,kimi,glm,openrouter,nvidia_nim`)
  - 代表モデル: `claude-haiku-5-5` (Anthropic), `gemini-2.5-flash-lite` (Gemini), `gpt-4o-mini` (OpenAI), `grok-2-latest` (Grok), `deepseek-chat` (DeepSeek)
  - 全プロバイダー共通で JSON / Structured Outputs を使用
- **Template Engine:** Jinja2 (HTML/CSS/JS 合成)

---

## 3. 3つの表示モードとインタラクティブUIの厳密定義

生成されるHTMLは、コントロールバーの切り替えにより以下の3つのモードへ瞬時に変貌します。

| モード名 | CSSクラス | 画面表示の挙動 | カーソル（ホバー）動作 |
| :--- | :--- | :--- | :--- |
| **① 1単語カーソル翻訳** | `mode-word-hover` | **英語原文のみ**を表示。<br>日本語全文は非表示。 | 英文中の対象単語にカーソルを乗せると、**「その単語の文脈訳」**がツールチップで表示される。 |
| **② 1文カーソル翻訳** | `mode-sentence-hover` | **英語原文のみ**を表示。<br>日本語全文は非表示。 | 1文にカーソルを乗せると、**「その1文全体の日本語訳」**が文章上部にポップアップ/吹き出しで表示される。（単語ホバーも同時に動作） |
| **③ 全文翻訳** | `mode-full-ja` | **日本語訳のみ**を美しいドキュメント形式で全文表示。 | 原文は非表示。日本語テキストをスムーズに流し読みできる。 |

---

## 4. データ構造定義 (Pydantic / JSON Schema)

### 4.1 LLM 出力スキーマ
```python
from typing import List
from pydantic import BaseModel, Field


class WordTranslation(BaseModel):
    word: str = Field(
        description="本文中に登場しているそのままの単語・熟語（原形ではなく本文内の完全一致表記）"
    )
    context_meaning: str = Field(
        description="この文脈における具体的な日本語の意味・ニュアンス"
    )


class SentenceTranslation(BaseModel):
    id: str = Field(description="入力ブロックのID (例: 'b-12')")
    translated: str = Field(description="1文の自然で正確な日本語訳")
    words: List[WordTranslation] = Field(
        default_factory=list,
        description="読者がつまずきそうな重要単語・専門用語・熟語・動詞句（1文につき可能な限り網羅的に3〜8個以上抽出すること）",
    )


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

### 5.1 PDF解析 & レイアウト保持 (`parse_pdf_to_blocks`)

1. `pymupdf4llm.to_markdown()` でPDFをMarkdownに変換。
2. 段落（`\n\n`）単位でループ処理：
* `#`, `##`, `###` ➔ `type="heading"`, `level=1~3`
* `|` で始まる行 ➔ `type="table"`（文分割スキップ・Markdown表構造を保持）
* コードブロック (`````) / 引用 (`>`) ➔ 適切なタイプとしてマーキング
* それ以外の本文 ➔ `nltk.sent_tokenize()` で1文ずつに分解し `type="sentence"`



### 5.2 安全な単語ホバーHTMLタグ注入 (`safe_word_annotation`)

HTML構文破壊を防ぐため、以下の順序を厳守すること：

```python
import html
import re


def safe_word_annotation(original_text: str, words: List[WordTranslation]) -> str:
    # 1. 原文全体を先にエスケープ
    escaped_text = html.escape(original_text)

    # 2. 単語長が長い順（熟語・複合語優先）にソートして部分一致事故を防止
    sorted_words = sorted(words, key=lambda x: len(x.word), reverse=True)

    for item in sorted_words:
        target_word = html.escape(item.word)
        meaning = html.escape(item.context_meaning, quote=True)

        # \b による単語境界マッチ
        pattern = re.compile(rf"\b({re.escape(target_word)})\b", re.IGNORECASE)
        replacement = f'<span class="word-hover" data-meaning="{meaning}">\\1</span>'
        escaped_text = pattern.sub(replacement, escaped_text)

    return escaped_text
```

### 5.3 表（Table）の処理

* Markdownの表構造（`| Col1 | Col2 |`）は維持したままLLMに送り、セル内言語のみ翻訳させる。
* HTML合成時、Markdownの表を `<table>` タグへ変換し、`<div class="table-container">` で囲んで `overflow-x: auto;` を適用する。

---

## 6. デザイン＆レイアウト（元PDFの美しさを再現するCSS）

単なるプレーンテキストの羅列にならないよう、以下のデザインシステムを単一HTML内に埋め込みます。

* **ドキュメントコンテナ:** 中央寄せ（`max-width: 860px`）、論文・レポートらしい適度な行間（`line-height: 1.75`）とパディング。
* **タイポグラフィ:** 見出し（`h1`〜`h3`）のメリハリあるフォントサイズと下部ボーダー。
* **1文ホバーポップアップ (CSS/JS):** `mode-sentence-hover` 時、`.sentence-block:hover` に対処して、日本語訳を要素のすぐ上（またはアコーディオン風）にツールチップ/ポップアップ表示。
* **表（Table）の再現:** `markdown` パッケージで `<table>` に変換し、`overflow-x: auto;` のスクロールコンテナで囲み、美しい境界線とヘッダー背景色を付与。

---

## 7. 非同期ジョブ & 多言語LLMプロバイダー自動切り替え

1. **`POST /api/translate`**: PDFファイルを受け取り `job_id` (UUID) を即座に返す。
2. **`GET /api/status/{job_id}`**: 進捗率（`status`: `queued` | `parsing` | `translating` | `completed` | `failed`, `progress`: 0〜100）を返す。
3. **`GET /api/download/{job_id}`**: 完成した `.html` ファイルをレスポンス。
4. **自動フォールバック:** `.env` の有効なキー順に試行。429/500エラー発生時はログを出力して即座に次のプロバイダーへフォールバック。

---

## 8. 生成される単一HTMLの完全制約

* 外部CSS/JSライブラリ（Tailwind, Bootstrap等）への依存は**完全不可**。
* Pure CSS / Pure Vanilla JS のみを使用し、完全オフライン（ローカルChrome）で動作すること。
