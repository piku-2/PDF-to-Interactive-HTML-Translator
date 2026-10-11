"""Prompts shared by every LLM provider."""

from __future__ import annotations

import json

from app.models import TextBlock

SYSTEM_PROMPT = """あなたは英語の論文・技術文書・レポートを日本語へ翻訳する翻訳者です。
入力は JSON 配列です。各要素は id, type, text を持ちます。type は heading, sentence, table, code, quote のいずれかです。

必ずツール submit_translations を1回だけ呼び、入力ブロックを過不足なく items に入れて返してください。前置きや解説は書かないでください。

規則:
- id は入力の値をそのまま使う。入力に無い id を作らない。入力にある id を落とさない。
- translated は自然で正確な日本語にする。数値、記号、固有名詞、引用、否定、程度の意味を保つ。
- heading は見出しとして簡潔に訳す。番号が付いていれば残す。# は付けない。
- sentence は、その1文に対応する翻訳にする。隣の文と混ぜない。
- quote は引用文として訳す。引用符を新たに足しすぎない。
- table は Markdown の表のまま返す。行数、列数、区切り行（| --- |）を維持し、セルの中の語句だけを訳す。空セルは空のままにする。
- code はプログラムの構造、記号、識別子、インデントを崩さない。translated はコメントと人間向けの文字列だけを日本語にしたコードにする。コメントが無ければ原文と同じコードを返す。
- words は、そのブロックの text に実際に書かれている表記と一致する英語の単語・熟語・動詞句。原形に戻さない。text に無い語句は入れない。
- sentence と quote では、読者がつまずきそうな重要語・専門用語・熟語・動詞句を可能な限り網羅する。1文につき3個以上、目安は8個前後かそれより多く。原文が短く3個に満たないときだけ、取れる範囲でよい。冠詞や be 動詞などの易しい機能語は後回しにする。
- heading、table、code も同じ基準で、原文にある語だけを入れる。
- context_meaning は、その出現箇所での具体的な日本語の意味にする。単語帳のような一般訳だけにしない。
"""

# OpenAI-compatible APIs require the lowercase word "json" somewhere in the prompt
# when response_format is json_object.
JSON_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """
応答は json オブジェクト1個だけです。説明、前置き、コードフェンスは禁止です。
json の形:
{"items":[{"id":"b-1","translated":"日本語","words":[{"word":"term","context_meaning":"この文脈の意味"}]}]}
words が無いブロックは "words": [] にしてください。
"""
)

ANTHROPIC_TOOL = {
    "name": "submit_translations",
    "description": "各ブロックの日本語訳と、原文にある重要語・専門用語・熟語の文脈訳を返す。words は1文につき3個以上を目標にする。",
    "strict": True,
    "cache_control": {"type": "ephemeral"},
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "translated": {"type": "string"},
                        "words": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "word": {"type": "string"},
                                    "context_meaning": {"type": "string"},
                                },
                                "required": ["word", "context_meaning"],
                            },
                        },
                    },
                    "required": ["id", "translated", "words"],
                },
            }
        },
        "required": ["items"],
    },
}


def block_payload(blocks: list[TextBlock]) -> str:
    payload = [
        {"id": block.id, "type": block.type, "text": block.original_text}
        for block in blocks
    ]
    return json.dumps(payload, ensure_ascii=False)


def anthropic_user_text(blocks: list[TextBlock]) -> str:
    return (
        "次のブロックを仕様どおりに翻訳し、submit_translations で返してください。\n"
        + block_payload(blocks)
    )


def json_user_text(blocks: list[TextBlock]) -> str:
    return (
        "次のブロックを仕様どおりに翻訳し、json だけを返してください。\n"
        + block_payload(blocks)
    )
