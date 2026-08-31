"""Test-suite warning policy for pinned third-party compatibility notices.

The hardening gate promotes every other warning to an error. These exact
messages originate in the pinned ADK, A2A, and Starlette dependencies rather
than Contractor code and have upstream migration work outside this slice.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from starlette.exceptions import StarletteDeprecationWarning

if TYPE_CHECKING:
    from contractor_runtime.capabilities import CapabilitySnapshot


def _install_dependency_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"BaseAgentConfig is deprecated and will be removed.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Using `httpx` with `starlette\.testclient` is deprecated.*",
        category=StarletteDeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"label\(\) is deprecated\. Use is_required\(\) or is_repeated\(\) instead\.",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"\[EXPERIMENTAL\] feature FeatureName\.JSON_SCHEMA_FOR_FUNC_DECL is enabled\.",
        category=UserWarning,
    )


def pytest_configure() -> None:
    _install_dependency_warning_filters()


with warnings.catch_warnings():
    _install_dependency_warning_filters()
    # Warm lazy imports whose pinned releases warn at class/import time. Test
    # modules can then import them normally under the global -W error policy.
    from google.adk.agents import LlmAgent as _LlmAgent  # noqa: F401
    from starlette.testclient import TestClient as _TestClient  # noqa: F401


@pytest.fixture(autouse=True)
def ignore_pinned_dependency_runtime_warnings() -> Iterator[None]:
    with warnings.catch_warnings():
        _install_dependency_warning_filters()
        yield


@pytest.fixture
def runtime_capabilities(tmp_path: Path) -> CapabilitySnapshot:
    """Full declared inventory for tests that are not exercising probes."""

    from contractor_runtime.capabilities import CapabilitySnapshot
    from contractor_runtime.factories import built_in_factories

    factories = built_in_factories(tmp_path)
    return CapabilitySnapshot.create(
        runtimes=factories.worker_runtimes,
        toolsets={ref: factory.exported_tools for ref, factory in factories.toolsets.items()},
        sandbox_profiles=factories.sandbox_profiles,
    )
