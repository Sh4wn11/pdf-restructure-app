import asyncio
import re
from dataclasses import dataclass, field
import pymupdf as fitz  # Modern PyMuPDF import
from fastapi import HTTPException, UploadFile, status
from app.config import settings

PDF_MAGIC_HEADER: bytes = b"%PDF-"
CHECKMARK_TOKENS: set[str] = {"X", "x", "✓", "✔"}

# Section headers that signal the end of a multi-page Specialty Skills table
SKILL_TABLE_END_MARKERS: tuple[str, ...] = (
    "credentialing:",
    "provider approval:",
    "timesheet approval",
    "other details",
    "description/other details",
    "contacts",
    "call details",
    "practice details",
    "coverage details",
)

# Repeating VMS page footer phrases to ignore while tracking multi-page tables
FOOTER_IGNORE_MARKERS: tuple[str, ...] = (
    "locumsmart assignment request as of",
    "locumsmart | po box",
    "page 1 of",
    "page 2 of",
    "page 3 of",
    "page 4 of",
)


@dataclass
class _VisualLine:
    """Represents a single horizontal line of words reconstructed from (x, y) coordinates."""

    y_center: float
    words: list[tuple[float, float, float, float, str]] = field(default_factory=list)


@dataclass
class _SkillRow:
    """Represents a logical row inside a Required / Preferred / Not Required skills grid."""

    text: str
    statuses: list[str] = field(default_factory=list)


def _normalize_extracted_text(text: str) -> str:
    """Collapse excessive blank lines and strip trailing whitespace per line."""
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def _group_words_into_lines(
    raw_words: list[tuple],
    y_tolerance: float = 4.5,
) -> list[_VisualLine]:
    """Group positioned words (x0, y0, x1, y1, text, ...) into sorted horizontal visual lines."""
    if not raw_words:
        return []

    sorted_words = sorted(raw_words, key=lambda w: ((w[1] + w[3]) / 2.0, w[0]))
    lines: list[_VisualLine] = []

    for w in sorted_words:
        x0, y0, x1, y1, text = float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4]).strip()
        if not text:
            continue
        y_mid = (y0 + y1) / 2.0

        if lines and abs(y_mid - lines[-1].y_center) <= y_tolerance:
            lines[-1].words.append((x0, y0, x1, y1, text))
            # Update running average Y-center
            lines[-1].y_center = sum((item[1] + item[3]) / 2.0 for item in lines[-1].words) / len(
                lines[-1].words
            )
        else:
            lines.append(_VisualLine(y_center=y_mid, words=[(x0, y0, x1, y1, text)]))

    for line in lines:
        line.words.sort(key=lambda item: item[0])

    return lines


def _format_standard_line(words: list[tuple[float, float, float, float, str]], col_gap: float = 28.0) -> str:
    """Join words on a regular line, inserting ' | ' when horizontal gaps indicate separate columns."""
    if not words:
        return ""
    parts: list[str] = [words[0][4]]
    prev_x1 = words[0][2]
    for x0, _, x1, _, text in words[1:]:
        if (x0 - prev_x1) > col_gap:
            parts.append(f"| {text}")
        else:
            parts.append(text)
        prev_x1 = max(prev_x1, x1)
    return " ".join(parts).strip()


def _detect_skill_column_anchors(
    lines: list[_VisualLine],
    line_idx: int,
) -> tuple[float, float, float] | None:
    """
    Detect a 'Required | Preferred | Not Required' table header row and return
    the horizontal X-center coordinates (req_x, pref_x, not_req_x).
    """
    target_line = lines[line_idx]
    pref_x: float | None = None
    for x0, _, x1, _, text in target_line.words:
        cleaned = text.strip("|,:;()").lower()
        if cleaned == "preferred" and x0 > 180.0:
            pref_x = (x0 + x1) / 2.0
            break

    if pref_x is None:
        return None

    # Gather words within +/- 18pt vertically to catch two-line 'Not \n Required' headers
    band_words: list[tuple[float, float, float, float, str]] = []
    for candidate_line in lines:
        if abs(candidate_line.y_center - target_line.y_center) <= 18.0:
            band_words.extend(candidate_line.words)

    req_candidates: list[float] = []
    not_req_candidates: list[float] = []

    for x0, _, x1, _, text in band_words:
        cleaned = text.strip("|,:;()").lower()
        x_mid = (x0 + x1) / 2.0
        if cleaned == "required" and 150.0 < x_mid < (pref_x - 20.0):
            req_candidates.append(x_mid)
        elif cleaned in {"not", "required"} and x_mid > (pref_x + 20.0):
            not_req_candidates.append(x_mid)

    if not req_candidates:
        return None

    req_x = sum(req_candidates) / len(req_candidates)
    if not_req_candidates:
        not_req_x = sum(not_req_candidates) / len(not_req_candidates)
    else:
        not_req_x = pref_x + (pref_x - req_x)

    return (req_x, pref_x, not_req_x)


def _classify_checkmark_x(
    x_mid: float,
    anchors: tuple[float, float, float],
    max_distance: float = 45.0,
) -> str | None:
    """Map an 'X' checkmark's horizontal center to REQUIRED, PREFERRED, or NOT_REQUIRED."""
    req_x, pref_x, not_req_x = anchors
    labeled_distances = [
        (abs(x_mid - req_x), "REQUIRED"),
        (abs(x_mid - pref_x), "PREFERRED"),
        (abs(x_mid - not_req_x), "NOT_REQUIRED — EXCLUDE"),
    ]
    best_dist, best_label = min(labeled_distances, key=lambda item: item[0])
    if best_dist <= max_distance:
        return best_label
    return None


def _is_wrapped_continuation(prev_text: str, curr_text: str) -> bool:
    """Return True if curr_text is a wrapped second line of prev_text inside a skill cell."""
    if not prev_text or not curr_text:
        return False
    prev_stripped = prev_text.rstrip()
    curr_stripped = curr_text.lstrip()
    if prev_stripped.endswith(",") or prev_stripped.endswith("("):
        return True
    if curr_stripped and curr_stripped[0].islower():
        return True
    return False


def _flush_skill_rows(skill_rows: list[_SkillRow]) -> list[str]:
    """Format accumulated skill rows with explicit [SKILL_STATUS: ...] tags."""
    formatted: list[str] = []
    for row in skill_rows:
        if not row.text:
            continue
        if row.statuses:
            unique_statuses = list(dict.fromkeys(row.statuses))
            status_tag = ", ".join(f"[SKILL_STATUS: {s}]" for s in unique_statuses)
            formatted.append(f"{row.text} -> {status_tag}")
        else:
            formatted.append(row.text)
    skill_rows.clear()
    return formatted


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Validate and extract text page-by-page from raw PDF bytes in memory.
    Preserves X-coordinate column alignment for 'Required | Preferred | Not Required'
    skill tables across multi-page boundaries.
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
            # Persist column X-anchors across pages so Page 3 tables without headers stay aligned
            active_skill_anchors: tuple[float, float, float] | None = None

            for page_index, page in enumerate(doc, start=1):
                raw_words = page.get_text("words")
                visual_lines = _group_words_into_lines(raw_words)
                if not visual_lines:
                    continue

                page_output_lines: list[str] = [f"--- [Page {page_index}] ---"]
                pending_skill_rows: list[_SkillRow] = []

                for idx, vline in enumerate(visual_lines):
                    joined_Preview = _format_standard_line(vline.words)
                    lower_preview = joined_Preview.lower()

                    # Ignore repeating VMS page footers without resetting active_skill_anchors
                    if any(marker in lower_preview for marker in FOOTER_IGNORE_MARKERS):
                        if active_skill_anchors is None:
                            page_output_lines.append(joined_Preview)
                        continue

                    # Check if this line is a 'Required | Preferred | Not Required' header
                    detected_anchors = _detect_skill_column_anchors(visual_lines, idx)
                    if detected_anchors is not None:
                        page_output_lines.extend(_flush_skill_rows(pending_skill_rows))
                        active_skill_anchors = detected_anchors
                        page_output_lines.append(
                            "Specialty Skills Table (Columns: REQUIRED | PREFERRED | NOT_REQUIRED):"
                        )
                        continue

                    if active_skill_anchors is not None:
                        # Skip standalone 'Required' word wrapped from 'Not \n Required' header
                        if lower_preview.strip("| ") in {"required", "not required"}:
                            continue

                        # Check if the skills table has ended and a new section began
                        if any(lower_preview.startswith(m) for m in SKILL_TABLE_END_MARKERS):
                            page_output_lines.extend(_flush_skill_rows(pending_skill_rows))
                            active_skill_anchors = None
                            page_output_lines.append(joined_Preview)
                            continue

                        req_x = active_skill_anchors[0]
                        checkbox_boundary_x = req_x - 35.0

                        left_words: list[tuple[float, float, float, float, str]] = []
                        row_statuses: list[str] = []

                        for x0, y0, x1, y1, text in vline.words:
                            token = text.strip("| ")
                            if not token:
                                continue
                            x_mid = (x0 + x1) / 2.0
                            if x_mid >= checkbox_boundary_x and token in CHECKMARK_TOKENS:
                                status_label = _classify_checkmark_x(x_mid, active_skill_anchors)
                                if status_label:
                                    row_statuses.append(status_label)
                            elif x_mid < checkbox_boundary_x:
                                left_words.append((x0, y0, x1, y1, token))

                        left_text = " ".join(w[4] for w in left_words).strip()

                        # Case 1: Vertically centered 'X' on a line with no left text
                        if not left_text and row_statuses:
                            if pending_skill_rows:
                                pending_skill_rows[-1].statuses.extend(row_statuses)
                            continue

                        # Case 2: Multi-line wrapped skill text (e.g. 'Colposcopy,\ncryosurgery, LEEP, etc.')
                        if (
                            left_text
                            and pending_skill_rows
                            and (not row_statuses or not pending_skill_rows[-1].statuses)
                            and _is_wrapped_continuation(pending_skill_rows[-1].text, left_text)
                        ):
                            pending_skill_rows[-1].text = f"{pending_skill_rows[-1].text} {left_text}"
                            pending_skill_rows[-1].statuses.extend(row_statuses)
                            continue

                        if left_text:
                            pending_skill_rows.append(
                                _SkillRow(text=left_text, statuses=row_statuses)
                            )
                        continue

                    page_output_lines.append(joined_Preview)

                page_output_lines.extend(_flush_skill_rows(pending_skill_rows))
                if len(page_output_lines) > 1:
                    extracted_pages.append("\n".join(page_output_lines))

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
            max_bytes = settings.max_upload_bytes
            pdf_bytes = await pdf_file.read(max_bytes + 1)
        finally:
            await pdf_file.close()

        if len(pdf_bytes) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"PDF exceeds the maximum allowed size of {settings.MAX_UPLOAD_MB} MB.",
            )

        return await asyncio.to_thread(extract_text_from_pdf_bytes, pdf_bytes)

    cleaned_text = _normalize_extracted_text(raw_text or "")
    if len(cleaned_text.encode("utf-8")) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Text input exceeds the maximum allowed size of {settings.MAX_UPLOAD_MB} MB.",
        )

    return cleaned_text