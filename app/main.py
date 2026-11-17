from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.config import settings


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Ensure required static and template directories exist before serving requests."""
    (settings.STATIC_DIR / "images").mkdir(parents=True, exist_ok=True)
    (settings.STATIC_DIR / "fonts").mkdir(parents=True, exist_ok=True)
    (settings.TEMPLATES_DIR / "styles").mkdir(parents=True, exist_ok=True)
    yield


def create_app() -> FastAPI:
    """Instantiate and configure the FastAPI web application."""
    application = FastAPI(
        title="AI Document Restructure & PDF Utility",
        description="Sanitizes raw text or PDFs with Gemini and compiles branded PDFs.",
        version="1.0.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    application.mount(
        "/static",
        StaticFiles(directory=str(settings.STATIC_DIR)),
        name="static",
    )

    application.include_router(api_router)

    @application.get("/health", tags=["System"])
    async def health_check() -> dict[str, str]:
        return {
            "status": "ok",
            "primary_model": settings.GEMINI_MODEL,
        }

    return application


app: FastAPI = create_app()