"""Pydantic models from SPEC.md section 4."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class WordTranslation(BaseModel):
    word: str = Field(description="本文中に登場しているそのままの単語・熟語（原形ではなく本文内の完全一致表記）")
    context_meaning: str = Field(description="この文脈における具体的な日本語の意味・ニュアンス")


class SentenceTranslation(BaseModel):
    id: str = Field(description="入力ブロックのID (例: 'b-12')")
    translated: str = Field(description="1文の自然な日本語訳")
    words: List[WordTranslation] = Field(default_factory=list, description="専門用語や難解語の文脈訳（0〜3個）")


class BatchTranslationResponse(BaseModel):
    items: List[SentenceTranslation]


class TextBlock(BaseModel):
    id: str
    type: Literal["heading", "sentence", "table"]
    original_text: str
    annotated_html: Optional[str] = None
    translated_text: Optional[str] = None
    # Heading depth (1-6). Sentences and tables keep 0.
    level: int = 0


class TranslateAccepted(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "parsing", "translating", "completed", "failed"]
    progress: int = Field(ge=0, le=100)
    error: Optional[str] = None
    filename: Optional[str] = None
