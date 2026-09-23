from __future__ import annotations

import json
import logging
import math

import pytest
from pydantic import SecretStr

from contractor_runtime.capabilities import _log_probe
from contractor_runtime.log import MAX_LOG_EXTRA_TEXT, JsonFormatter, configure_logging


def test_dependency_info_logs_cannot_publish_private_urls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    dependency_names = ("httpx", "httpcore", "uvicorn", "uvicorn.error")
    original_dependency_levels = {name: logging.getLogger(name).level for name in dependency_names}
    try:
        configure_logging("info")
        logging.getLogger("httpx").info("HTTP Request: POST https://private-runtime.example/secret")
        logging.getLogger("contractor_runtime.test").info("safe lifecycle event")

        rendered = capsys.readouterr().err
        assert "private-runtime.example" not in rendered
        assert "safe lifecycle event" in rendered
        assert all(
            logging.getLogger(name).getEffectiveLevel() >= logging.WARNING
            for name in dependency_names
        )
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)
        for name, level in original_dependency_levels.items():
            logging.getLogger(name).setLevel(level)


def _format(**extra: object) -> dict[str, object]:
    record = logging.getLogger("contractor_runtime.test").makeRecord(
        "contractor_runtime.test",
        logging.INFO,
        __file__,
        1,
        "structured event %s",
        ("ok",),
        None,
        extra=extra,
    )
    return json.loads(JsonFormatter().format(record))


def test_json_logs_keep_reviewed_structured_extra_fields() -> None:
    event = _format(
        capabilityRef="adk@1",
        capabilityKind="runtime",
        probeOutcome="unavailable",
        durationMs=12,
        instanceId="runtime-1",
        podmanImageDigest="sha256:" + "a" * 64,
        podmanCPUs=1.5,
        podmanBindDiskQuotaEnforced=False,
        podmanNetwork=None,
    )
    del event["timestamp"]
    assert event == {
        "level": "info",
        "logger": "contractor_runtime.test",
        "message": "structured event ok",
        "capabilityRef": "adk@1",
        "capabilityKind": "runtime",
        "probeOutcome": "unavailable",
        "durationMs": 12,
        "instanceId": "runtime-1",
        "podmanImageDigest": "sha256:" + "a" * 64,
        "podmanCPUs": 1.5,
        "podmanBindDiskQuotaEnforced": False,
        "podmanNetwork": None,
    }


def test_json_logs_drop_unreviewed_and_non_scalar_extra_fields() -> None:
    secret = "recognizable-log-extra-secret"
    event = _format(
        authorization=secret,
        gatewayToken=SecretStr(secret),
        capabilityRef={"nested": secret},
        instanceId=SecretStr(secret),
        durationMs=math.inf,
        probeOutcome="x" * (MAX_LOG_EXTRA_TEXT + 1),
        capabilityKind=True,
    )
    assert secret not in json.dumps(event)
    assert set(event) == {"timestamp", "level", "logger", "message", "capabilityKind"}
    assert event["capabilityKind"] is True


def test_capability_probe_logs_render_their_structured_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    try:
        configure_logging("info")
        _log_probe("adk@1", "runtime", "unavailable", 7)
        rendered = capsys.readouterr().err.strip().splitlines()
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)
    event = json.loads(rendered[-1])
    assert event["message"] == "Runtime capability probe unavailable"
    assert event["capabilityRef"] == "adk@1"
    assert event["capabilityKind"] == "runtime"
    assert event["probeOutcome"] == "unavailable"
    assert event["durationMs"] == 7
