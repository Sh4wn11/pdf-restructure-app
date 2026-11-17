import asyncio
import re
import pymupdf as fitz  # Modern PyMuPDF import (silences fitz FutureWarning)
from fastapi import HTTPException, UploadFile, status
from app.config import settings

PDF_MAGIC_HEADER: bytes = b"%PDF-"


def _normalize_extracted_text(text: str) -> str:
    """Collapse excessive blank lines and strip trailing whitespace per line."""
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Synchronously validate and extract text page-by-page from raw PDF bytes in memory.
    Raises FastAPI HTTPException on invalid headers, encrypted PDFs, or empty text.
    """
    if not pdf_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded PDF file is empty.",
        )

    if len(pdf_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"PDF exceeds the maximum allowed size of {settings.MAX_UPLOAD_MB} MB.",
        )

    # Pure-Python magic-byte verification (avoids Windows libmagic DLL issues)
    if not pdf_bytes[:5].startswith(PDF_MAGIC_HEADER):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Invalid file format. File header does not match a valid PDF (%PDF-).",
        )

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            if doc.is_encrypted:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Password-protected or encrypted PDFs are not supported.",
                )

            if doc.page_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="The uploaded PDF contains zero pages.",
                )

            extracted_pages: list[str] = []
            for page_index, page in enumerate(doc, start=1):
                page_text = page.get_text("text").strip()
                if page_text:
                    extracted_pages.append(f"--- [Page {page_index}] ---\n{page_text}")

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse PDF file structure: {exc}",
        ) from exc

    if not extracted_pages:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No extractable text was found in the PDF. "
                "The file may contain only scanned images without an OCR text layer."
            ),
        )

    return _normalize_extracted_text("\n\n".join(extracted_pages))


async def ingest_user_input(
    raw_text: str | None = None,
    pdf_file: UploadFile | None = None,
) -> str:
    """
    Primary async ingestion entry point accepting either raw text or an uploaded PDF.
    If both are provided, the uploaded PDF takes precedence.
    """
    has_pdf = pdf_file is not None and bool(pdf_file.filename)
    has_text = raw_text is not None and bool(raw_text.strip())

    if not has_pdf and not has_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide either raw text input or a valid PDF file.",
        )

    if has_pdf and pdf_file is not None:
        try:
            # Read at most max_upload_bytes + 1 byte to prevent memory exhaustion
            max_bytes = settings.max_upload_bytes
            pdf_bytes = await pdf_file.read(max_bytes + 1)
        finally:
            await pdf_file.close()

        if len(pdf_bytes) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"PDF exceeds the maximum allowed size of {settings.MAX_UPLOAD_MB} MB.",
            )

        # Offload CPU-bound PyMuPDF parsing to a worker thread
        return await asyncio.to_thread(extract_text_from_pdf_bytes, pdf_bytes)

    cleaned_text = _normalize_extracted_text(raw_text or "")
    if len(cleaned_text.encode("utf-8")) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Text input exceeds the maximum allowed size of {settings.MAX_UPLOAD_MB} MB.",
        )

    return cleaned_text