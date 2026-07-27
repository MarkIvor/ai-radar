"""Pydantic-Settings конфигурация AI Radar.

Бутстрап-настройки читаются из .env (или переменных окружения).
Runtime-настройки (мастер-пароль, порог VETO, LLM-провайдеры, API-ключи)
живут в SQLite и переопределяют бутстрап после первого запуска.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
"""Абсолютный путь к корню проекта."""


class Settings(BaseSettings):
    """Бутстрап-конфиг приложения.

    Все значения можно переопределить через .env или переменные окружения.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Безопасность ---
    master_password: str = Field(default="airadar")
    jwt_secret: str = Field(default="change-me")
    access_token_ttl_min: int = Field(default=30, ge=1)
    refresh_token_ttl_days: int = Field(default=7, ge=1)

    # --- Хранилище ---
    database_path: str = Field(default="storage/airadar.db")

    # --- AI Judge (Сигнал 2) ---
    judge_mode: str = Field(default="ensemble")  # "ensemble" | "single"
    judge_http_timeout_sec: int = Field(default=60, ge=5)
    veto_threshold: int = Field(default=75, ge=50, le=95)
    judge_max_retries: int = Field(default=2, ge=0, le=5)

    # --- UI ---
    docs_enabled: bool = Field(default=True)

    # --- Пресет LLM по умолчанию (необязательно) ---
    default_llm_base_url: str | None = None
    default_llm_model: str | None = None
    default_llm_api_key: str | None = None

    @field_validator("judge_mode")
    @classmethod
    def _validate_judge_mode(cls, v: str) -> str:
        v = (v or "ensemble").strip().lower()
        if v not in {"ensemble", "single"}:
            raise ValueError(f"judge_mode must be 'ensemble' or 'single', got: {v}")
        return v

    @property
    def database_uri(self) -> str:
        """Абсолютный путь к файлу SQLite для aiosqlite."""
        p = Path(self.database_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Синглтон-инстанс настроек."""
    return Settings()


def reset_settings_cache() -> None:
    """Сбросить кэш (для тестов)."""
    get_settings.cache_clear()
