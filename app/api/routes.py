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


def _sanitize_pdf_filename(title: str) -> str:
    """Sanitize document title into a safe ASCII filename to prevent header injection."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", title.strip()).strip("_")
    safe_stem = cleaned[:60] if cleaned else "restructured_document"
    return f"{safe_stem}.pdf"


def _build_sample_document() -> RestructuredDocument:
    """Return a deterministic sample document for instant CSS/HTML layout testing."""
    return RestructuredDocument(
        document_title="Q3 Commercial Fleet & Operations Report",
        subtitle="Executive Performance Summary & Infrastructure Plan",
        executive_summary=(
            "During October 2026, commercial fleet operations generated USD 1.4M in total "
            "revenue across truck and motorcycle divisions. Service bay expansion is scheduled "
            "for Q4 to support growing maintenance volume."
        ),
        metadata_date="2026-10-01",
        author_or_source="Fleet Operations Division",
        sections=[
            ContentSection(
                heading="Unit Delivery & Revenue Breakdown",
                body_paragraphs=[
                    "Demand for heavy commercial trucks remained strong throughout the quarter, "
                    "accounting for the majority of gross dealership billings.",
                    "Motorcycle fleet orders increased among regional logistics partners seeking "
                    "last-mile delivery efficiency.",
                ],
                highlight_callout="Key Milestone: USD 1.4M total revenue with 60 total vehicles delivered.",
                key_bullets=[
                    "Delivered 42 commercial trucks to enterprise accounts.",
                    "Delivered 18 fleet motorcycles for urban courier routes.",
                    "Maintained a 98.4% on-time vehicle handover rate.",
                ],
            ),
            ContentSection(
                heading="Q4 Infrastructure & Service Action Plan",
                body_paragraphs=[
                    "To prevent maintenance bottlenecks as active fleet contracts scale, "
                    "workshop capacity will be upgraded before year-end."
                ],
                highlight_callout=None,
                key_bullets=[
                    "Expand primary service bay capacity by Q4.",
                    "Standardize preventive diagnostic checklists across all bays.",
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


@router.post("/api/v1/transform")
async def transform_to_pdf(
    raw_text: str | None = Form(default=None),
    pdf_file: UploadFile | None = File(default=None),
    custom_instructions: str = Form(default=""),
) -> Response:
    """Ingest text or PDF, sanitize and structure with Gemini, and compile to a branded PDF."""
    extracted_text = await ingest_user_input(raw_text=raw_text, pdf_file=pdf_file)
    structured_doc = await sanitize_and_structure_async(
        raw_text=extracted_text,
        custom_instructions=custom_instructions,
    )
    rendered_html = await asyncio.to_thread(render_document_html, structured_doc)
    pdf_bytes = await compile_html_to_pdf(rendered_html)

    safe_filename = _sanitize_pdf_filename(structured_doc.document_title)
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
    """
    Compile the sample document into PDF or HTML without calling the Gemini API.
    Ideal for rapid custom HTML/CSS design iteration.
    """
    sample_doc = _build_sample_document()
    rendered_html = await asyncio.to_thread(render_document_html, sample_doc)

    if output_format.lower() == "html":
        return HTMLResponse(content=rendered_html)

    pdf_bytes = await compile_html_to_pdf(rendered_html)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'inline; filename="design_preview.pdf"',
            "X-Document-Filename": "design_preview.pdf",
        },
    )