"""Centralized runtime configuration.

All env-driven config flows through `Settings`. Loaded once via `get_settings()`.

The module also calls `load_dotenv()` at import time so any legacy
`os.environ.get(...)` callsites (still present in agents/tools) see the same
values during the migration.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── LLM (LiteLLM proxy) ──────────────────────────────────────────────
    default_model_name: str = Field(default="lm-studio-qwen3.5")
    default_model_timeout: int = Field(default=300)
    litellm_api_base: Optional[str] = Field(default=None)
    litellm_api_key: Optional[str] = Field(default=None)

    # ── Observability (Langfuse) ─────────────────────────────────────────
    use_langfuse: bool = Field(default=False)
    langfuse_host: Optional[str] = Field(default=None)
    langfuse_public_key: Optional[str] = Field(default=None)
    langfuse_secret_key: Optional[str] = Field(default=None)

    # ── GitLab fs auth ───────────────────────────────────────────────────
    gitlab_private_token: Optional[str] = Field(default=None)
    gitlab_oauth_token: Optional[str] = Field(default=None)
    ci_job_token: Optional[str] = Field(default=None)

    # ── Storage paths (override in k8s) ──────────────────────────────────
    # Where FileArtifactService writes. None → CLI default (./artifacts).
    artifacts_dir: Optional[Path] = Field(default=None, alias="CONTRACTOR_ARTIFACTS_DIR")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def _build_default_model() -> LiteLlm:
    s = get_settings()
    return LiteLlm(model=s.default_model_name, timeout=s.default_model_timeout)


# Many modules import this directly; keep the symbol working.
DEFAULT_MODEL: LiteLlm = _build_default_model()
