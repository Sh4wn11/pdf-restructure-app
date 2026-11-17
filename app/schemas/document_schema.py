from pydantic import BaseModel, ConfigDict, Field


class ContentSection(BaseModel):
    """A single logical section within the restructured output document."""

    model_config = ConfigDict(str_strip_whitespace=True)

    heading: str = Field(
        ...,
        description="Clear, professional section heading.",
    )
    body_paragraphs: list[str] = Field(
        default_factory=list,
        description="Sanitized body paragraphs with OCR artifacts and filler removed.",
    )
    highlight_callout: str | None = Field(
        default=None,
        description="Important metric, quote, or key takeaway box if applicable; otherwise null.",
    )
    key_bullets: list[str] = Field(
        default_factory=list,
        description="Concise action items or bulleted highlights for this section.",
    )


class RestructuredDocument(BaseModel):
    """Root schema for the sanitized and reorganized PDF document."""

    model_config = ConfigDict(str_strip_whitespace=True)

    document_title: str = Field(
        ...,
        description="Formal, descriptive title for the restructured document.",
    )
    subtitle: str = Field(
        default="",
        description="Brief clarifying subtitle or document category; empty string if not applicable.",
    )
    executive_summary: str = Field(
        ...,
        description="Concise 2 to 4 sentence executive overview of the core content.",
    )
    metadata_date: str = Field(
        default="N/A",
        description="Normalized document date in YYYY-MM-DD format, or 'N/A' if not found.",
    )
    author_or_source: str = Field(
        default="N/A",
        description="Identified author, department, or originating organization, or 'N/A'.",
    )
    sections: list[ContentSection] = Field(
        ...,
        min_length=1,
        description="Ordered list of restructured content sections.",
    )