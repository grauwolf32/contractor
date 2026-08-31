from __future__ import annotations

import logging

import pytest

from contractor_runtime.log import configure_logging


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
