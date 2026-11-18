import asyncio
import re
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, status
from fastapi.responses import HTMLResponse

from app.config import settings
from app.schemas.document_schema import ContentSection, RestructuredDocument
from app.services.ai_sanitizer import sanitize_and_structure_async
from app.services.ingestion import ingest_user_input
from app.services.pdf_renderer import compile_html_to_pdf
from app.services.template_engine import render_document_html

router = APIRouter()


def _sanitize_pdf_filename(job_id_or_title: str) -> str:
    """
    Format the PDF filename strictly as 'Orchard_(JOB ID).pdf' (e.g., 'Orchard_Job-20159.pdf')
    while stripping any unsafe header/path characters.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", job_id_or_title.strip()).strip("_")
    if not cleaned:
        cleaned = "Job-Unlisted"
    if cleaned.lower().startswith("orchard_"):
        safe_stem = cleaned[:60]
    else:
        safe_stem = f"Orchard_{cleaned}"[:60]
    return f"{safe_stem}.pdf"


def _build_sample_document() -> RestructuredDocument:
    """Return a sample sanitized Orchard VMS job posting for instant HTML/CSS design previews."""
    return RestructuredDocument(
        job_id="Job-20159",
        role_type="Locums",
        document_title="Locums — Inpatient Hospitalist Physician",
        subtitle="Locums Opportunity | OH",
        executive_summary=(
            "A Facility in OH is seeking a Board-Certified Internal Medicine or Family Medicine "
            "Hospitalist for ongoing inpatient locums coverage. Day and night shifts are available."
        ),
        metadata_date="2026-11-18",
        nearest_airport="OH (approximately 20 minutes away from CMH Airport)",
        sections=[
            ContentSection(
                heading="Assignment Overview",
                body_paragraphs=[
                    "Location & Facility Details: 280-bed acute care Facility located in OH utilizing Epic EMR. "
                    "Open ICU model with full subspecialty backup."
                ],
                highlight_callout=None,
                key_bullets=[
                    "Role Type: Locums coverage (minimum 7 shifts per month).",
                    "Setting: Inpatient acute care Facility in OH.",
                    "EMR System: Epic.",
                ],
            ),
            ContentSection(
                heading="Schedule & Coverage",
                body_paragraphs=[
                    "7-on / 7-off block scheduling with 12-hour shifts (7:00 AM – 7:00 PM or 7:00 PM – 7:00 AM)."
                ],
                highlight_callout=None,
                key_bullets=[
                    "No night call required for daytime rounding physicians.",
                    " Holiday and weekend coverage included in block rotation.",
                ],
            ),
            ContentSection(
                heading="Clinical Details",
                body_paragraphs=[
                    "Average daily census of 15–18 patients per day rounder, including 2–3 admissions per shift."
                ],
                highlight_callout=None,
                key_bullets=[
                    "Dedicated nocturnist handles overnight admissions.",
                    "Intensivist and pulmonology support available on-site.",
                ],
            ),
            ContentSection(
                heading="Skills",
                body_paragraphs=[],
                highlight_callout=None,
                key_bullets=[],
                required_skills=[
                    "Inpatient acute care management and hospitalist rounding",
                    "Level I/Screening (viability) ultrasound and FHR interpretation",
                    "Routine and complicated delivery (Vaginal, C-section, VBAC)",
                ],
                preferred_skills=[
                    "High-risk clinical case management",
                    "OB ultrasound performance and interpretation",
                ],
            ),
            ContentSection(
                heading="Requirements",
                body_paragraphs=[
                    "Candidates must hold an active OH medical license and current DEA registration."
                ],
                highlight_callout=None,
                key_bullets=[
                    "Board Certified in Internal Medicine (ABIM) or Family Medicine (ABFM).",
                    "Active ACLS and BLS certifications.",
                    "Clean malpractice history for emergency temp privileges (30–45 days).",
                ],
            ),
        ],
    )


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_web_ui() -> HTMLResponse:
    """Serve the single-page web application interface."""
    index_path = settings.TEMPLATES_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Frontend template index.html is missing.",
        )
    html_content = await asyncio.to_thread(index_path.read_text, encoding="utf-8")
    return HTMLResponse(content=html_content)


@router.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Return 204 No Content so browsers do not log a 404 error on /favicon.ico."""
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/api/v1/transform")
async def transform_to_pdf(
    raw_text: str | None = Form(default=None),
    pdf_file: UploadFile | None = File(default=None),
    custom_instructions: str = Form(default=""),
) -> Response:
    """Ingest text or PDF, sanitize and structure with Gemini, and compile to an Orchard PDF."""
    extracted_text = await ingest_user_input(raw_text=raw_text, pdf_file=pdf_file)
    structured_doc = await sanitize_and_structure_async(
        raw_text=extracted_text,
        custom_instructions=custom_instructions,
    )
    rendered_html = await asyncio.to_thread(render_document_html, structured_doc)
    pdf_bytes = await compile_html_to_pdf(rendered_html)

    safe_filename = _sanitize_pdf_filename(structured_doc.job_id)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{safe_filename}"',
            "X-Document-Filename": safe_filename,
        },
    )


@router.post("/api/v1/preview-html", response_class=HTMLResponse)
async def preview_structured_html(
    raw_text: str | None = Form(default=None),
    pdf_file: UploadFile | None = File(default=None),
    custom_instructions: str = Form(default=""),
) -> HTMLResponse:
    """Run the ingestion and AI structuring pipeline and return raw HTML/CSS for live previewing."""
    extracted_text = await ingest_user_input(raw_text=raw_text, pdf_file=pdf_file)
    structured_doc = await sanitize_and_structure_async(
        raw_text=extracted_text,
        custom_instructions=custom_instructions,
    )
    rendered_html = await asyncio.to_thread(render_document_html, structured_doc)
    return HTMLResponse(content=rendered_html)


@router.post("/api/v1/design-preview")
async def preview_design_without_ai(
    output_format: str = Form(default="pdf"),
) -> Response:
    """Compile the sample Orchard job profile into PDF or HTML without calling the Gemini API."""
    sample_doc = _build_sample_document()
    rendered_html = await asyncio.to_thread(render_document_html, sample_doc)

    if output_format.lower() == "html":
        return HTMLResponse(content=rendered_html)

    pdf_bytes = await compile_html_to_pdf(rendered_html)
    safe_filename = _sanitize_pdf_filename(sample_doc.job_id)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{safe_filename}"',
            "X-Document-Filename": safe_filename,
        },
    )