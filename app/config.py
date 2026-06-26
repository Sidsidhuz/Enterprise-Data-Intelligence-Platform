"""
app/config.py
==============

Centralized application configuration.

Every other module that needs a path, a limit, or a tunable value should
import `settings` from here rather than hardcoding values or reading
environment variables directly. This keeps configuration in one place and
makes the rest of the codebase easy to test (you can construct a `Settings`
object with different values in a test without touching environment
variables at all).

All filesystem paths are resolved relative to the project root, so the
application behaves the same regardless of the current working directory
it's launched from.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The project root is two levels up from this file: app/config.py -> app/ -> root/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Application settings, loaded from environment variables / a `.env` file,
    falling back to sensible local-development defaults.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Storage ---
    data_dir: str = "data"
    database_url: str = "sqlite:///./autoinsight.db"

    # --- Upload constraints ---
    max_upload_size_mb: int = 50
    allowed_extensions: str = "csv,xlsx,xls"

    # --- AutoML ---
    automl_tuning_time_budget_seconds: int = 60

    # --- Logging ---
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Derived properties (computed, not read directly from environment)
    # ------------------------------------------------------------------

    @property
    def data_path(self) -> Path:
        """Absolute path to the root data directory."""
        path = (PROJECT_ROOT / self.data_dir).resolve()
        return path

    @property
    def raw_dir(self) -> Path:
        return self.data_path / "raw"

    @property
    def cleaned_dir(self) -> Path:
        return self.data_path / "cleaned"

    @property
    def models_dir(self) -> Path:
        return self.data_path / "models"

    @property
    def reports_dir(self) -> Path:
        return self.data_path / "reports"

    @property
    def plots_dir(self) -> Path:
        return self.data_path / "plots"

    @property
    def allowed_extensions_set(self) -> set[str]:
        """Allowed upload extensions as a lowercase set, e.g. {'csv', 'xlsx'}."""
        return {ext.strip().lower().lstrip(".") for ext in self.allowed_extensions.split(",") if ext.strip()}

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def ensure_data_dirs_exist(self) -> None:
        """Create every data subdirectory if it doesn't already exist."""
        for path in (self.raw_dir, self.cleaned_dir, self.models_dir, self.reports_dir, self.plots_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.

    Using a function (rather than a module-level singleton) makes it easy to
    override settings in tests via dependency injection / monkeypatching the
    cache, while still avoiding re-parsing the environment on every call in
    normal operation.
    """
    return Settings()


# A ready-to-use settings instance for convenient importing:
#   from app.config import settings
settings = get_settings()
