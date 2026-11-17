from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve directory paths relative to this file (app/config.py)
BASE_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = BASE_DIR.parent
TEMPLATES_DIR: Path = BASE_DIR / "templates"
STATIC_DIR: Path = BASE_DIR / "static"


class Settings(BaseSettings):
    """Application configuration loaded and validated from .env or environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    GEMINI_API_KEY: str = Field(
        ...,
        min_length=1,
        description="Google Gemini API key required for document structuring.",
    )
    GEMINI_MODEL: str = Field(
        default="gemini-3.8-flash",
        min_length=1,
        description="Target Gemini model identifier.",
    )
    GEMINI_FALLBACK_MODELS: list[str] = Field(
        default=["gemini-3.8-flash-lite", "gemini-3.8-pro"],
        description="Ordered fallback models if the primary model returns 503/429/404.",
    )
    MAX_UPLOAD_MB: int = Field(
        default=20,
        gt=0,
        le=100,
        description="Maximum allowed PDF upload size in megabytes.",
    )

    # Path attributes exposed on the settings instance for convenience
    BASE_DIR: Path = BASE_DIR
    TEMPLATES_DIR: Path = TEMPLATES_DIR
    STATIC_DIR: Path = STATIC_DIR

    @property
    def max_upload_bytes(self) -> int:
        """Convert MAX_UPLOAD_MB to exact byte ceiling."""
        return self.MAX_UPLOAD_MB * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()


settings: Settings = get_settings()