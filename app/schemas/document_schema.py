from pydantic import BaseModel, ConfigDict, Field


class ContentSection(BaseModel):
    """A single structured section within the Orchard job profile PDF."""

    model_config = ConfigDict(str_strip_whitespace=True)

    heading: str = Field(
        ...,
        description=(
            "Section heading. Must be one of: 'Assignment Overview', "
            "'Schedule & Coverage', 'Clinical Details', 'Requirements', or 'Skills'."
        ),
    )
    body_paragraphs: list[str] = Field(
        default_factory=list,
        description=(
            "Sanitized paragraphs strictly from source text. For 'Assignment Overview', "
            "include a 'Location & Facility Details:' paragraph or lead-in."
        ),
    )
    highlight_callout: str | None = Field(
        default=None,
        description="Optional callout (e.g., Perm compensation/benefits summary or key shift note); null if none.",
    )
    key_bullets: list[str] = Field(
        default_factory=list,
        description="Explicitly stated facts formatted as concise bullet points.",
    )


class RestructuredDocument(BaseModel):
    """Root schema for sanitized Orchard agency job postings."""

    model_config = ConfigDict(str_strip_whitespace=True)

    job_id: str = Field(
        default="Job-00000",
        description="Extracted VMS Job ID formatted as 'Job-<ID>' (e.g., 'Job-20159').",
    )
    role_type: str = Field(
        default="Locums",
        description="Classified role type: 'Locums', 'Locums to Perm', or 'Perm'.",
    )
    document_title: str = Field(
        ...,
        description="Sanitized specialty/role title (e.g., 'Locums — Hospitalist Physician'). Never include facility or city names.",
    )
    subtitle: str = Field(
        default="",
        description="Brief role classification and state subtitle (e.g., 'Locums to Perm Opportunity | OH').",
    )
    executive_summary: str = Field(
        ...,
        description="Strictly factual 2-3 sentence summary of the position with facility and city names redacted.",
    )
    metadata_date: str = Field(
        default="N/A",
        description="Posting or start/target date in YYYY-MM-DD format, or 'N/A' if not stated.",
    )
    nearest_airport: str = Field(
        default="N/A",
        description="Formatted strictly as: '[State] (approximately [Time] away from [Airport Code] Airport)'. Example: 'OH (approximately 20 minutes away from CMH Airport)'.",
    )
    author_or_source: str = Field(
        default="Orchard",
        description="Agency brand identifier.",
    )
    sections: list[ContentSection] = Field(
        ...,
        min_length=5,
        max_length=5,
        description=(
            "Exactly 5 sections in this strict order: "
            "1. Assignment Overview, 2. Schedule & Coverage, 3. Clinical Details, 4. Requirements, 5. Skills."
        ),
    )