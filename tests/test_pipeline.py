import pymupdf as fitz
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.routes import _sanitize_pdf_filename
from app.main import app
from app.services.ingestion import extract_text_from_pdf_bytes

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_rejects_spoofed_non_pdf_bytes() -> None:
    spoofed_bytes = b"PK\x03\x04not_a_real_pdf_file"
    with pytest.raises(HTTPException) as exc_info:
        extract_text_from_pdf_bytes(spoofed_bytes)
    assert exc_info.value.status_code == 415


def test_extracts_valid_in_memory_pdf() -> None:
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "Unit Test Fleet Report 2026")
        pdf_bytes = doc.tobytes()

    extracted = extract_text_from_pdf_bytes(pdf_bytes)
    assert "Unit Test Fleet Report 2026" in extracted


def test_filename_sanitization_blocks_header_injection() -> None:
    malicious_title = 'Report\r\nContent-Type: text/html; ../../evil'
    safe_name = _sanitize_pdf_filename(malicious_title)
    assert "\r" not in safe_name
    assert "\n" not in safe_name
    assert ".." not in safe_name
    assert safe_name.endswith(".pdf")


def test_zero_quota_design_preview_compiles_pdf() -> None:
    response = client.post("/api/v1/design-preview", data={"output_format": "pdf"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-")