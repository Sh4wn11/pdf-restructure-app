import asyncio
import logging
from fastapi import HTTPException, status
from playwright.sync_api import Route, sync_playwright

logger = logging.getLogger(__name__)


def _ssrf_guard_route(route: Route) -> None:
    """
    SSRF Security Guard: Only permit inline 'data:' URIs (Base64 images/fonts)
    and 'about:blank'. Block all external HTTP/HTTPS or local file:// fetches.
    """
    url = route.request.url
    if url.startswith("data:") or url.startswith("about:"):
        route.continue_()
    else:
        logger.warning("Blocked unauthorized external resource fetch during PDF render: %s", url)
        route.abort("accessdenied")


def _compile_html_to_pdf_sync(rendered_html: str) -> bytes:
    """
    Synchronously compile rendered HTML/CSS into A4 PDF bytes in memory via headless Chromium.
    Running inside a worker thread guarantees Windows ProactorEventLoop compatibility under Uvicorn.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-dev-shm-usage", "--no-sandbox"],
        )
        try:
            context = browser.new_context(java_script_enabled=False)
            page = context.new_page()

            page.route("**/*", _ssrf_guard_route)
            page.set_content(rendered_html, wait_until="load")

            pdf_bytes: bytes = page.pdf(
                format="A4",
                print_background=True,
                prefer_css_page_size=True,
            )
            return pdf_bytes
        finally:
            browser.close()


async def compile_html_to_pdf(rendered_html: str) -> bytes:
    """
    Async entry point that validates the HTML string and offloads PDF compilation
    to a background worker thread.
    """
    if not rendered_html or not rendered_html.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot compile an empty HTML document to PDF.",
        )

    try:
        return await asyncio.to_thread(_compile_html_to_pdf_sync, rendered_html)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to compile HTML to PDF via Playwright")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF rendering engine failed: {exc}",
        ) from exc