"""Contractor explorer UI — a dependency-free local web app for browsing
agent prompts, task templates, workflow pipelines, and skills.

Launch with the ``analytics-ui`` console script or ``python -m analytics_ui``.
"""
from __future__ import annotations

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    from analytics_ui.server import main as _main

    return _main(argv)
