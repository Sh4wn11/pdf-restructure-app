import base64
import mimetypes
from functools import lru_cache
from pathlib import Path
from jinja2 import FileSystemLoader, select_autoescape
from jinja2.sandbox import SandboxedEnvironment

from app.config import settings
from app.schemas.document_schema import RestructuredDocument

DEFAULT_LOGO_SVG: str = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="140" height="40" viewBox="0 0 140 40">'
    '<rect width="40" height="40" rx="8" fill="#1e3a8a"/>'
    '<path d="M12 20L18 26L28 14" stroke="#ffffff" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>'
    '<text x="50" y="25" font-family="Segoe UI, Arial, sans-serif" font-size="16" font-weight="700" fill="#0f172a">DOCFLOW</text>'
    "</svg>"
)
DEFAULT_LOGO_DATA_URI: str = (
    f"data:image/svg+xml;base64,{base64.b64encode(DEFAULT_LOGO_SVG.encode('utf-8')).decode('ascii')}"
)


@lru_cache(maxsize=1)
def get_jinja_env() -> SandboxedEnvironment:
    """Create a cached, XSS-safe SandboxedEnvironment for HTML template rendering."""
    return SandboxedEnvironment(
        loader=FileSystemLoader(str(settings.TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def load_static_image_base64(filename: str) -> str | None:
    """
    Safely read an image from app/static/images and return a Base64 Data URI.
    Blocks directory traversal outside app/static/images.
    """
    images_dir: Path = (settings.STATIC_DIR / "images").resolve()
    target_path: Path = (images_dir / filename).resolve()

    if not target_path.is_relative_to(images_dir) or not target_path.is_file():
        return None

    mime_type, _ = mimetypes.guess_type(target_path.name)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/png"

    encoded = base64.b64encode(target_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def load_print_stylesheet() -> str:
    """Read app/templates/styles/print_theme.css so it can be inlined into the document."""
    css_path: Path = settings.TEMPLATES_DIR / "styles" / "print_theme.css"
    if css_path.is_file():
        return css_path.read_text(encoding="utf-8")
    return ""


def render_document_html(
    doc: RestructuredDocument,
    logo_filename: str = "logo.png",
) -> str:
    """
    Bind the validated RestructuredDocument schema, CSS stylesheet, and Base64 images
    into app/templates/base_report.html and return the complete HTML string.
    """
    env = get_jinja_env()
    template = env.get_template("base_report.html")

    logo_data_uri = load_static_image_base64(logo_filename) or DEFAULT_LOGO_DATA_URI
    css_styles = load_print_stylesheet()

    return template.render(
        doc=doc.model_dump(),
        css_styles=css_styles,
        logo_data_uri=logo_data_uri,
    )