from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
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


@dataclass
class EvalFixture:
    """A self-contained eval fixture: source tree + lazily-loaded ground-truth.

    Case files (``*-cases.json``) and expected-output files are loaded on
    first access and cached for the lifetime of the fixture.  This avoids
    reading planner cases when only running trace evals, and means adding a
    new case type never requires touching this dataclass.
    """

    slug: str
    source_root: Path
    benchmark: Optional[str]
    _fixture_dir: Path = field(repr=False)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def load_cases(self, case_type: str) -> list[dict[str, Any]]:
        """Load ``{case_type}-cases.json`` from the fixture directory."""
        key = f"cases:{case_type}"
        if key not in self._cache:
            self._cache[key] = _maybe_load_json(
                self._fixture_dir / f"{case_type}-cases.json", [],
            )
        return self._cache[key]

    def _load_expected_yaml(self, name: str) -> dict[str, Any]:
        key = f"yaml:{name}"
        if key not in self._cache:
            self._cache[key] = _maybe_load_yaml(
                self._fixture_dir / f"{name}.yaml", {},
            ) or {}
        return self._cache[key]

    def _load_expected_json(self, name: str) -> list[dict[str, Any]]:
        key = f"json:{name}"
        if key not in self._cache:
            self._cache[key] = _maybe_load_json(
                self._fixture_dir / f"{name}.json", [],
            )
        return self._cache[key]

    # -- ground-truth accessors ------------------------------------------------

    @property
    def expected_oas(self) -> dict[str, Any]:
        return self._load_expected_yaml("oas.expected")

    @property
    def expected_vulnerabilities(self) -> list[dict[str, Any]]:
        return self._load_expected_json("vulnerabilities.expected")

    # -- convenience case-list properties --------------------------------------

    @property
    def swe_cases(self) -> list[dict[str, Any]]:
        return self.load_cases("swe")

    @property
    def trace_cases(self) -> list[dict[str, Any]]:
        return self.load_cases("trace")

    @property
    def task_cases(self) -> list[dict[str, Any]]:
        return self.load_cases("task")

    @property
    def planner_cases(self) -> list[dict[str, Any]]:
        return self.load_cases("planner")

    @property
    def vuln_cases(self) -> list[dict[str, Any]]:
        return self.load_cases("vuln")


# ---------------------------------------------------------------------------
# Fixture discovery & caching
# ---------------------------------------------------------------------------

def _discover_fixture_slugs() -> list[str]:
    """Auto-discover all fixture slugs that have ``meta.yaml``."""
    if not FIXTURES_ROOT.is_dir():
        return []
    return sorted(
        d.name
        for d in FIXTURES_ROOT.iterdir()
        if d.is_dir() and (d / "meta.yaml").is_file()
    )


def _discover_slugs_with_cases(case_type: str) -> list[str]:
    """Auto-discover fixture slugs containing ``{case_type}-cases.json``."""
    return [
        slug
        for slug in _discover_fixture_slugs()
        if (FIXTURES_ROOT / slug / f"{case_type}-cases.json").is_file()
    ]


_fixture_cache: dict[str, EvalFixture] = {}


def _load_fixture(slug: str) -> EvalFixture:
    if slug in _fixture_cache:
        return _fixture_cache[slug]
    fixture_dir = FIXTURES_ROOT / slug
    meta = _load_yaml(fixture_dir / "meta.yaml")
    source_root = (REPO_ROOT / meta["source_root"]).resolve()
    if not source_root.is_dir():
        raise RuntimeError(
            f"Fixture {slug} source_root not found: {source_root}. "
            "Did you initialise the playground submodule?"
        )
    fix = EvalFixture(
        slug=slug,
        source_root=source_root,
        benchmark=meta.get("benchmark"),
        _fixture_dir=fixture_dir,
    )
    _fixture_cache[slug] = fix
    return fix


def _load_case_params(case_type: str) -> list[tuple[str, str]]:
    """Return ``(slug, case_id)`` pairs for parametrization.

    Only reads the JSON files — fixture validation (source_root exists)
    is deferred to the per-case fixture that calls ``_load_fixture``.
    """
    params: list[tuple[str, str]] = []
    for slug in _discover_fixture_slugs():
        cases = _maybe_load_json(
            FIXTURES_ROOT / slug / f"{case_type}-cases.json", [],
        )
        for case in cases:
            params.append((slug, case["id"]))
    return params


# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------

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


_CASE_PARAM_MAP = {
    "trace_case": "trace",
    "swe_case": "swe",
    "planner_case": "planner",
    "exploitability_case": "exploitability",
}


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Auto-parametrize per-case fixtures (``trace_case``, ``swe_case``, …).

    Each ``(slug, case_id)`` pair becomes its own pytest item, giving
    independent timeouts, xdist parallelism, and per-case CI visibility.
    """
    for fixture_name, case_type in _CASE_PARAM_MAP.items():
        if fixture_name in metafunc.fixturenames:
            params = _load_case_params(case_type)
            metafunc.parametrize(
                fixture_name,
                params,
                ids=[f"{slug}/{cid}" for slug, cid in params],
                indirect=True,
            )


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _eval_observability() -> None:
    """Initialise Langfuse instrumentation once per eval session."""
    from contractor.utils import observability

    observability.init()


@pytest.fixture(scope="session")
def eval_sink():
    """Session-wide collector that writes one ``eval/v1`` envelope per
    ``(scenario, unit)`` at the end of the run.

    Per-fixture eval tests call ``eval_sink.record(...)`` once with their scored
    :class:`~tests.eval.results.CaseResult`; the aggregated envelopes land in
    ``eval_runs/<unit>/eval_results.json`` for analytics-ui.
    """
    from tests.eval.results import EvalSink

    sink = EvalSink()
    yield sink
    for path in sink.flush():
        print(f"[eval_sink] results -> {path}", flush=True)


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
    params=_discover_fixture_slugs(),
    ids=lambda s: f"fixture[{s}]",
)
def fixture(request: pytest.FixtureRequest) -> EvalFixture:
    return _load_fixture(request.param)


@pytest.fixture(scope="session")
def fixture_fs(fixture: EvalFixture):
    """A sandboxed read-only filesystem rooted at the fixture's source tree."""
    from cli.fs import RootedLocalFileSystem

    return RootedLocalFileSystem(str(fixture.source_root))


@pytest.fixture(
    scope="session",
    params=_discover_slugs_with_cases("vuln"),
    ids=lambda s: f"vuln[{s}]",
)
def vuln_fixture(request: pytest.FixtureRequest) -> EvalFixture:
    """Session-scoped fixture parametrized over all vuln benchmark fixtures."""
    return _load_fixture(request.param)


# ---------------------------------------------------------------------------
# Per-case fixtures (resolved via pytest_generate_tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def trace_case(request: pytest.FixtureRequest) -> tuple[EvalFixture, dict]:
    slug, case_id = request.param
    fix = _load_fixture(slug)
    case = next(c for c in fix.trace_cases if c["id"] == case_id)
    return fix, case


@pytest.fixture(scope="session")
def swe_case(request: pytest.FixtureRequest) -> tuple[EvalFixture, dict]:
    slug, case_id = request.param
    fix = _load_fixture(slug)
    case = next(c for c in fix.swe_cases if c["id"] == case_id)
    return fix, case


@pytest.fixture(scope="session")
def planner_case(request: pytest.FixtureRequest) -> tuple[EvalFixture, dict]:
    slug, case_id = request.param
    fix = _load_fixture(slug)
    case = next(c for c in fix.planner_cases if c["id"] == case_id)
    return fix, case


@pytest.fixture(scope="session")
def exploitability_case(request: pytest.FixtureRequest) -> tuple[EvalFixture, dict]:
    slug, case_id = request.param
    fix = _load_fixture(slug)
    case = next(c for c in fix.load_cases("exploitability") if c["id"] == case_id)
    return fix, case


# ---------------------------------------------------------------------------
# Public helpers for scripts
# ---------------------------------------------------------------------------

def select_fixture(slug: str) -> Optional[EvalFixture]:
    """Helper for tests/scripts that need a specific fixture (not parametrized)."""
    return _load_fixture(slug)
