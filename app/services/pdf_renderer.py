import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from fastapi import HTTPException, status
from playwright.sync_api import Browser, Playwright, Route, sync_playwright

logger = logging.getLogger(__name__)

# Single-thread executor keeps sync_playwright bound to one consistent OS thread
# and prevents concurrent Chromium spikes on Render's 512MB RAM tier.
_PDF_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chromium_pdf_worker")
_playwright_instance: Playwright | None = None
_browser_instance: Browser | None = None

CHROMIUM_ARGS: list[str] = [
    "--disable-dev-shm-usage",
    "--no-sandbox",
    "--disable-gpu",
    "--disable-extensions",
    "--disable-sync",
    "--disable-translate",
    "--no-first-run",
    "--no-zygote",
]


def _get_or_create_browser() -> Browser:
    """Return a warm, persistent headless Chromium instance inside the PDF worker thread."""
    global _playwright_instance, _browser_instance
    if _browser_instance is None or not _browser_instance.is_connected():
        if _playwright_instance is None:
            _playwright_instance = sync_playwright().start()
        _browser_instance = _playwright_instance.chromium.launch(
            headless=True,
            args=CHROMIUM_ARGS,
        )
    return _browser_instance


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
    """Compile HTML/CSS into A4 PDF bytes using the warm Chromium browser."""
    browser = _get_or_create_browser()
    context = browser.new_context(java_script_enabled=False)
    try:
        page = context.new_page()
        page.route("**/*", _ssrf_guard_route)
        page.set_content(rendered_html, wait_until="domcontentloaded")

        pdf_bytes: bytes = page.pdf(
            format="A4",
            print_background=True,
            prefer_css_page_size=True,
        )
        return pdf_bytes
    finally:
        # Always close the tab/context immediately to keep RAM usage flat
        context.close()


async def compile_html_to_pdf(rendered_html: str) -> bytes:
    """
    Async entry point that validates the HTML string and dispatches PDF compilation
    to the persistent Chromium worker thread.
    """
    if not rendered_html or not rendered_html.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot compile an empty HTML document to PDF.",
        )

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            _PDF_EXECUTOR,
            _compile_html_to_pdf_sync,
            rendered_html,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to compile HTML to PDF via Playwright")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF rendering engine failed: {exc}",
        ) from exc