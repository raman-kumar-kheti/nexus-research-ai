"""Centralized configuration for NexusResearch AI."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    PROJECT_NAME: str = "NexusResearch AI"
    VERSION: str = "4.0.0"

    # LLM Provider: "groq" | "openai_compatible" | "mock"
    LLM_PROVIDER: str = "groq"
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    OPENAI_API_BASE: str = ""
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Execution Guardrails
    MAX_ITERATIONS: int = 3
    MAX_TOKEN_BUDGET: int = 16000
    MAX_COST_USD: float = 0.25
    AGENT_TIMEOUT_SECONDS: float = 30.0
    MAX_RETRIES: int = 2
    REQUIRE_HUMAN_APPROVAL: bool = True

    # Auth & Storage
    JWT_SECRET_KEY: str = ""
    DATABASE_URL: str = ""
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent
    DB_PATH: Path = ROOT_DIR / "memory" / "research.db"
    CHECKPOINT_DB_PATH: Path = ROOT_DIR / "memory" / "checkpoints.db"
    OUTPUTS_DIR: Path = ROOT_DIR / "outputs"

    def model_post_init(self, __context: Any) -> None:
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        if not self.JWT_SECRET_KEY:
            self.JWT_SECRET_KEY = secrets.token_urlsafe(32)
        if not self.GROQ_API_KEY and self.LLM_PROVIDER == "groq":
            # Gracefully fall back to deterministic mock mode when no API key is configured.
            self.LLM_PROVIDER = "mock"


settings = Settings()

# Backward-compatible module constants
GROQ_API_KEY = settings.GROQ_API_KEY
GROQ_MODEL = settings.GROQ_MODEL
DB_PATH = str(settings.DB_PATH)
OUTPUTS_DIR = str(settings.OUTPUTS_DIR)
