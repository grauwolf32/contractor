from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pytest
import yaml
from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAYGROUND_ROOT = REPO_ROOT / "tests" / "playground"
FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures"

# Mirror the CLI: load the same `.env` files the CLI does so eval runs see
# the LiteLLM proxy / Langfuse / GitLab config the user has already wired
# up. `cli/.env` is the canonical source (where `USE_LITELLM_PROXY`,
# `LITELLM_PROXY_API_BASE`, etc. live); `<repo>/.env` is also tried as a
# convenience fallback. Existing process env always wins.
for _env_path in (REPO_ROOT / "cli" / ".env", REPO_ROOT / ".env"):
    if _env_path.exists():
        load_dotenv(_env_path, override=False)


@dataclass(frozen=True)
class EvalFixture:
    """A self-contained eval fixture: source tree + ground-truth artifacts."""

    slug: str
    source_root: Path
    expected_oas: dict[str, Any]
    expected_vulnerabilities: list[dict[str, Any]]
    swe_cases: list[dict[str, Any]]
    trace_cases: list[dict[str, Any]]


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_load_yaml(path: Path, default: Any) -> Any:
    return _load_yaml(path) if path.is_file() else default


def _maybe_load_json(path: Path, default: Any) -> Any:
    return _load_json(path) if path.is_file() else default


def _load_fixture(slug: str) -> EvalFixture:
    fixture_dir = FIXTURES_ROOT / slug
    meta = _load_yaml(fixture_dir / "meta.yaml")
    source_root = (REPO_ROOT / meta["source_root"]).resolve()
    if not source_root.is_dir():
        raise RuntimeError(
            f"Fixture {slug} source_root not found: {source_root}. "
            "Did you initialise the playground submodule?"
        )
    return EvalFixture(
        slug=slug,
        source_root=source_root,
        expected_oas=_maybe_load_yaml(fixture_dir / "oas.expected.yaml", {}) or {},
        expected_vulnerabilities=_maybe_load_json(
            fixture_dir / "vulnerabilities.expected.json", []
        ),
        swe_cases=_maybe_load_json(fixture_dir / "swe-cases.json", []),
        trace_cases=_maybe_load_json(fixture_dir / "trace-cases.json", []),
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip eval tests unless explicitly opted in."""
    if config.getoption("-m"):
        return
    if os.environ.get("CONTRACTOR_RUN_EVAL"):
        return
    skip_eval = pytest.mark.skip(
        reason="eval tests are slow + LLM-bound; run with `pytest -m eval` "
        "or set CONTRACTOR_RUN_EVAL=1"
    )
    for item in items:
        if "eval" in item.keywords:
            item.add_marker(skip_eval)


@pytest.fixture(scope="session", autouse=True)
def _eval_observability() -> None:
    """Initialise Langfuse instrumentation once per eval session.

    No-op when Langfuse is disabled in settings — the harness still tags
    runs with ``prompt_version`` etc., the calls just don't emit anything.
    """
    from contractor.utils import observability

    observability.init()


@pytest.fixture(scope="session")
def eval_model() -> LiteLlm:
    """LiteLlm instance used by eval agents.

    Override the model via CONTRACTOR_EVAL_MODEL. Defaults to the project's
    DEFAULT_MODEL configuration so evals exercise the same provider used
    in production runs.
    """
    override = os.environ.get("CONTRACTOR_EVAL_MODEL")
    if override:
        return LiteLlm(model=override, timeout=600)
    from contractor.utils.settings import DEFAULT_MODEL

    return DEFAULT_MODEL


@pytest.fixture(
    scope="session",
    params=["spring", "fastapi", "vulnyapi", "vaultpay"],
    ids=lambda s: f"fixture[{s}]",
)
def fixture(request: pytest.FixtureRequest) -> EvalFixture:
    return _load_fixture(request.param)


@pytest.fixture(scope="session")
def fixture_fs(fixture: EvalFixture):
    """A sandboxed read-only filesystem rooted at the fixture's source tree."""
    from cli.fs import RootedLocalFileSystem

    return RootedLocalFileSystem(str(fixture.source_root))


def select_fixture(slug: str) -> Optional[EvalFixture]:
    """Helper for tests that need a specific fixture (not parametrized)."""
    return _load_fixture(slug)
