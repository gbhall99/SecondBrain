"""Pydantic models for LLM knowledge extraction (→ JSON schema for Ollama).

Refs (``*_ref``) are indices into the ``entities`` list of the SAME response, or
-1 for the conversation owner ("I"/"me"). Every item cites the transcript
``source_segment_ids`` it came from so provenance is preserved.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ENTITY_TYPES = ("person", "project", "organization", "topic", "place")
OWNER_REF = -1


class ExEntity(BaseModel):
    type: Literal["person", "project", "organization", "topic", "place"]
    name: str
    aliases: list[str] = Field(default_factory=list)
    source_segment_ids: list[int] = Field(default_factory=list)
    confidence: float = 0.5


class ExFact(BaseModel):
    subject_ref: int = OWNER_REF
    predicate: str
    object_text: str = ""
    object_ref: int | None = None
    source_segment_ids: list[int] = Field(default_factory=list)
    confidence: float = 0.5


class ExActionItem(BaseModel):
    owed_by_ref: int | None = None
    owed_to_ref: int | None = None
    description: str
    due_date: str | None = None
    source_segment_ids: list[int] = Field(default_factory=list)
    confidence: float = 0.5


class ExStatement(BaseModel):
    summary: str
    participant_refs: list[int] = Field(default_factory=list)
    source_segment_ids: list[int] = Field(default_factory=list)
    confidence: float = 0.5


class ExtractionResult(BaseModel):
    entities: list[ExEntity] = Field(default_factory=list)
    facts: list[ExFact] = Field(default_factory=list)
    action_items: list[ExActionItem] = Field(default_factory=list)
    decisions: list[ExStatement] = Field(default_factory=list)
    ideas: list[ExStatement] = Field(default_factory=list)


def extraction_json_schema() -> dict:
    return ExtractionResult.model_json_schema()
