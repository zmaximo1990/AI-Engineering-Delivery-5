"""Runtime settings loaded from the environment and the project ``.env`` file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parent

_PROVIDERS = {"openai", "anthropic", "gemini"}


class Settings(BaseSettings):
    """Process settings. Empty environment values fall back to the field default."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    default_max_tokens: int = 1024
    default_temperature: float = 0.0
    gemini_max_tokens: int = 1024
    default_provider: str = Field(default="openai", validation_alias="LLM_PROVIDER")

    data_dir: Path = PROJECT_DIR / "data"
    fixtures_path: Path = PROJECT_DIR / "data" / "fixtures.csv"
    core_db_path: Path = PROJECT_DIR / "data" / "catalog.sqlite"
    checkpoint_db_path: Path = PROJECT_DIR / "data" / "checkpoints.sqlite"
    trace_log_path: Path = PROJECT_DIR / "trace.log"

    transcript_excerpt_chars: int = 400
    recursion_limit: int = 10

    @model_validator(mode="before")
    @classmethod
    def _blank_as_unset(cls, data: object) -> object:
        """Treat blank strings as missing so the field default is used."""
        if not isinstance(data, dict):
            return data
        cleaned: dict[object, object] = {}
        for key, value in data.items():
            if isinstance(value, str):
                value = value.strip()
                if value == "":
                    continue
            cleaned[key] = value
        return cleaned

    @field_validator("default_provider", mode="before")
    @classmethod
    def _provider(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        lowered = value.strip().lower()
        if lowered not in _PROVIDERS:
            allowed = ", ".join(sorted(_PROVIDERS))
            raise ValueError(f"LLM_PROVIDER must be one of: {allowed}")
        return lowered

    @field_validator("transcript_excerpt_chars")
    @classmethod
    def _excerpt_floor(cls, value: int) -> int:
        return max(80, value)

    @field_validator("recursion_limit")
    @classmethod
    def _recursion_floor(cls, value: int) -> int:
        return max(2, value)

    @field_validator(
        "data_dir",
        "fixtures_path",
        "core_db_path",
        "checkpoint_db_path",
        "trace_log_path",
        mode="before",
    )
    @classmethod
    def _resolve_project_path(cls, value: object) -> Path:
        path = value if isinstance(value, Path) else Path(str(value).strip())
        if not path.is_absolute():
            path = PROJECT_DIR / path
        return path


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once."""
    return Settings()


settings = get_settings()
