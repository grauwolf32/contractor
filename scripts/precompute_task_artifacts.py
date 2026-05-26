#!/usr/bin/env python3
"""Precompute dependency_information + project_information artifacts for eval fixtures.

Saves result text to ``tests/eval/fixtures/<slug>/artifacts/`` so that
downstream evals (e.g. likec4_build) can skip the expensive SWE-agent
steps and iterate on a single task in isolation.

Usage::

    poetry run python scripts/precompute_task_artifacts.py vulnyapi crapi-workshop
    poetry run python scripts/precompute_task_artifacts.py --all
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.eval.conftest import FIXTURES_ROOT, _load_fixture  # noqa: E402
from tests.eval.task_harness import run_task_pipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("precompute")

ALL_SLUGS = [
    "vulnyapi",
    "crapi-workshop",
    "crapi-identity",
    "vaultpay",
    "fastapi",
    "spring",
]

TASKS_TO_PRECOMPUTE = [
    "dependency_information",
    "project_information",
]


async def precompute_fixture(slug: str, model) -> None:
    fixture = _load_fixture(slug)
    logger.info("Starting precompute for %s (source: %s)", slug, fixture.source_root)

    from cli.fs import RootedLocalFileSystem

    from contractor.agents.swe_agent.agent import build_swe_agent

    fs = RootedLocalFileSystem(str(fixture.source_root))

    def queue(runner) -> None:
        swe_builder = partial(
            build_swe_agent,
            name="swe_agent",
            fs=fs,
            model=model,
            max_tokens=100_000,
        )
        runner.add_variable(name="project_path", value=str(fixture.source_root))
        runner.add_task(
            name="dependency_information",
            worker_builder=swe_builder,
            iterations=1,
            max_attempts=4,
            max_steps=20,
            namespace="dependency_information",
            model=model,
        )
        runner.add_task(
            name="project_information",
            worker_builder=swe_builder,
            iterations=1,
            max_attempts=4,
            max_steps=20,
            artifacts=["dependency_information/result"],
            namespace="project_information",
            model=model,
        )

    artifact_keys = [
        "dependency_information/result",
        "project_information/result",
    ]

    run = await run_task_pipeline(
        queue_fn=queue,
        artifact_keys=artifact_keys,
        namespace=f"precompute-{slug}",
        timeout_s=1800.0,
        runner_name=f"precompute-{slug}",
    )

    out_dir = FIXTURES_ROOT / slug / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in artifact_keys:
        text = run.artifacts.get(key, "")
        if not text:
            logger.warning("No artifact for %s / %s — skipping", slug, key)
            continue
        safe_name = key.replace("/", "_") + ".txt"
        dest = out_dir / safe_name
        dest.write_text(text, encoding="utf-8")
        logger.info("Saved %s (%d chars)", dest.relative_to(REPO_ROOT), len(text))

    logger.info("Done: %s", slug)


async def main(slugs: list[str]) -> None:
    import os

    from dotenv import load_dotenv

    for env_path in (REPO_ROOT / "cli" / ".env", REPO_ROOT / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)

    from contractor.utils import observability

    observability.init()

    override = os.environ.get("CONTRACTOR_EVAL_MODEL")
    if override:
        from google.adk.models.lite_llm import LiteLlm

        model = LiteLlm(model=override, timeout=600)
    else:
        from contractor.utils.settings import DEFAULT_MODEL

        model = DEFAULT_MODEL

    logger.info("Using model: %s", model)
    logger.info("Fixtures: %s", slugs)

    for slug in slugs:
        try:
            await precompute_fixture(slug, model)
        except Exception:
            logger.exception("Failed to precompute %s", slug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "slugs",
        nargs="*",
        help="Fixture slugs to precompute (default: all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Precompute all fixtures",
    )
    args = parser.parse_args()

    slugs = ALL_SLUGS if args.all or not args.slugs else args.slugs
    asyncio.run(main(slugs))
