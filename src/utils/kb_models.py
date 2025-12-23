from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class KBManifestRow(BaseModel):
    doc_id: str
    title: str
    category: str = "general"
    sub_category: str = "basics"
    source_name: str = "mixed"
    source_url: str = ""
    language: str = "en"
    license_or_usage_notes: str = ""
    created_at: str = ""
    updated_at: str = ""
    local_path: str
    summary: str = ""
    tags: str = ""


class GlossaryTerm(BaseModel):
    term: str
    definition: str
    category: str = "general"
    examples: str = ""


class ValidationIssue(BaseModel):
    level: str  # ERROR | WARN | INFO
    message: str
    location: Optional[str] = None


class ValidationReport(BaseModel):
    ok: bool
    errors: List[ValidationIssue] = Field(default_factory=list)
    warnings: List[ValidationIssue] = Field(default_factory=list)
    infos: List[ValidationIssue] = Field(default_factory=list)

    def add_error(self, msg: str, location: Optional[str] = None) -> None:
        self.errors.append(ValidationIssue(level="ERROR", message=msg, location=location))

    def add_warning(self, msg: str, location: Optional[str] = None) -> None:
        self.warnings.append(ValidationIssue(level="WARN", message=msg, location=location))

    def add_info(self, msg: str, location: Optional[str] = None) -> None:
        self.infos.append(ValidationIssue(level="INFO", message=msg, location=location))

    def finalize(self) -> "ValidationReport":
        self.ok = len(self.errors) == 0
        return self
