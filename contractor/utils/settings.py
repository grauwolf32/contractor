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
    default_model_name: str = Field(default="lm-studio-qwen3.6")
    default_model_timeout: int = Field(default=300)
    litellm_api_base: Optional[str] = Field(default=None)
    litellm_api_key: Optional[str] = Field(default=None)

    # ── LLM sampling (None → backend / model default) ────────────────────
    # Applied to every LiteLlm built via `build_model`. Lower temperature
    # tightens structured-output adherence; leave None to keep the model's
    # own defaults.
    model_temperature: Optional[float] = Field(default=None)
    model_top_p: Optional[float] = Field(default=None)

    # ── Tool defaults (global baseline; agent code may override) ─────────
    # These are the fallbacks used when a tool/agent factory is called
    # without an explicit value. Keep them equal to the historical
    # hardcoded constants so behaviour is unchanged unless tuned via env.
    http_timeout: float = Field(default=30.0)
    http_body_preview_chars: int = Field(default=2048)
    http_history_size: int = Field(default=20)
    http_retry_attempts: int = Field(default=3)
    http_retry_base_delay: float = Field(default=0.5)
    http_retry_max_delay: float = Field(default=8.0)
    fs_max_items: int = Field(default=100)
    fs_max_output: int = Field(default=80_000)
    code_max_walk_depth: int = Field(default=50)
    code_max_files_per_walk: int = Field(default=100_000)
    graph_max_results: int = Field(default=200)
    graph_max_paths: int = Field(default=25)
    graph_max_path_depth: int = Field(default=30)
    likec4_validate_timeout: float = Field(default=120.0)

    # ── Observability (Langfuse) ─────────────────────────────────────────
    use_langfuse: bool = Field(default=False)
    langfuse_host: Optional[str] = Field(default=None)
    langfuse_public_key: Optional[str] = Field(default=None)
    langfuse_secret_key: Optional[str] = Field(default=None)

    # ── Caido proxy ─────────────────────────────────────────────────────
    caido_url: Optional[str] = Field(default=None)
    caido_auth_token: Optional[str] = Field(default=None)

    # ── GitLab fs auth ───────────────────────────────────────────────────
    gitlab_private_token: Optional[str] = Field(default=None)
    gitlab_oauth_token: Optional[str] = Field(default=None)
    ci_job_token: Optional[str] = Field(default=None)

    # ── Storage paths (override in k8s) ──────────────────────────────────
    # Base dir for artifact stores. None → CLI default (./artifacts). The CLI
    # namespaces a per-project subdir under this base, so multiple projects
    # don't share one store.
    artifacts_dir: Optional[Path] = Field(default=None, alias="CONTRACTOR_ARTIFACTS_DIR")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def build_model(
    model_name: Optional[str] = None,
    timeout: Optional[int] = None,
) -> LiteLlm:
    """Construct a ``LiteLlm`` applying the configured sampling defaults.

    ``model_name`` / ``timeout`` fall back to ``Settings`` when omitted.
    ``model_temperature`` / ``model_top_p`` are forwarded to litellm only
    when set, so leaving them unset preserves the model's own defaults.
    """
    s = get_settings()
    kwargs: dict = {
        "model": model_name if model_name is not None else s.default_model_name,
        "timeout": timeout if timeout is not None else s.default_model_timeout,
    }
    if s.model_temperature is not None:
        kwargs["temperature"] = s.model_temperature
    if s.model_top_p is not None:
        kwargs["top_p"] = s.model_top_p
    return LiteLlm(**kwargs)


def _build_default_model() -> LiteLlm:
    return build_model()


# Many modules import this directly; keep the symbol working.
DEFAULT_MODEL: LiteLlm = _build_default_model()
