from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from google.adk import features

from contractor_runtime.worker import adk_features
from contractor_runtime.worker import factory as worker_factory

UNSUPPORTED = (
    features.FeatureName.FUNCTION_TOOL_ARG_VALIDATION,
    features.FeatureName.IN_MEMORY_SESSION_SERVICE_LIGHT_COPY,
    features.FeatureName.SKILL_LIFECYCLE,
)


def test_supported_adk_feature_defaults_are_accepted() -> None:
    assert all(not features.is_feature_enabled(feature) for feature in UNSUPPORTED)
    adk_features.require_supported_adk_features()


def test_supported_features_allow_the_actual_native_capability_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def native_probe() -> bool:
        calls.append("native probe")
        return True

    monkeypatch.setattr(adk_features.features, "is_feature_enabled", lambda _: False)
    monkeypatch.setattr(worker_factory, "probe_native_agent_skills", native_probe)
    assert asyncio.run(worker_factory.AdkWorkerRuntimeFactory().probe())
    assert calls == ["native probe"]


@pytest.mark.parametrize("feature", UNSUPPORTED)
@pytest.mark.parametrize("boundary", ["probe", "create"])
def test_unsupported_feature_stops_factory_before_any_resources(
    monkeypatch: pytest.MonkeyPatch, feature: features.FeatureName, boundary: str
) -> None:
    touched: list[str] = []

    async def native_probe() -> bool:
        touched.append("native probe")
        raise AssertionError("feature rejection must precede native probes")

    async def prepare(*args: object, **kwargs: object) -> None:
        touched.append("skills")
        raise AssertionError("feature rejection must precede Skill preparation")

    def model(*args: object) -> None:
        touched.append("model")
        raise AssertionError("feature rejection must precede model creation")

    def artifact(*args: object) -> None:
        touched.append("artifact client")
        raise AssertionError("feature rejection must precede client creation")

    monkeypatch.setattr(adk_features.features, "is_feature_enabled", lambda item: item == feature)
    monkeypatch.setattr(worker_factory, "probe_native_agent_skills", native_probe)
    monkeypatch.setattr(worker_factory, "prepare_agent_skills", prepare)
    factory = worker_factory.AdkWorkerRuntimeFactory(model, artifact)

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match=f"Unsupported ADK feature enabled: {feature.value}"):
            if boundary == "probe":
                await factory.probe()
            else:
                # No context member may be accessed before this guard. A missing
                # guard would fail on the sentinel rather than the expected error.
                await factory.create(object())  # type: ignore[arg-type]
        assert touched == []

    asyncio.run(scenario())


def test_warning_strict_feature_detection_rejects_instead_of_running_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def warning(*args: object) -> bool:
        raise UserWarning("feature activation warning")

    monkeypatch.setattr(adk_features.features, "is_feature_enabled", warning)
    with pytest.raises(RuntimeError, match="FUNCTION_TOOL_ARG_VALIDATION"):
        adk_features.require_supported_adk_features()


def test_public_feature_api_env_and_programmatic_activation_are_rejected() -> None:
    # The SDK's programmatic overrides are process-wide. Isolate the real public
    # API checks rather than leave overrides or warning-once state in this suite.
    script = """
import os
from google.adk import features
from contractor_runtime.worker.adk_features import require_supported_adk_features

require_supported_adk_features()
for name in (
    "FUNCTION_TOOL_ARG_VALIDATION",
    "IN_MEMORY_SESSION_SERVICE_LIGHT_COPY",
    "SKILL_LIFECYCLE",
):
    feature = features.FeatureName[name]
    for source in ("environment", "override"):
        if source == "environment":
            os.environ["ADK_ENABLE_" + name] = "1"
        else:
            features.override_feature_enabled(feature, True)
        try:
            require_supported_adk_features()
        except RuntimeError as error:
            if str(error) != "Unsupported ADK feature enabled: " + name:
                raise AssertionError("feature rejection lost its closed name") from error
        else:
            raise AssertionError("unsupported feature was accepted: " + name)
        finally:
            os.environ.pop("ADK_ENABLE_" + name, None)
            if source == "override":
                features.override_feature_enabled(feature, False)
require_supported_adk_features()
"""
    environment = dict(os.environ)
    for feature in UNSUPPORTED:
        environment.pop("ADK_ENABLE_" + feature.value, None)
        environment.pop("ADK_DISABLE_" + feature.value, None)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
