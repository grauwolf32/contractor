#!/usr/bin/env python3
"""A/B the observation arm on the oas_build task eval (default vs lean_no_errors).

Runs the planner-driven ``oas_update`` task (the OpenAPI build) through the shared
run_task_pipeline harness — which now resolves the observation arm from
CONTRACTOR_EVAL_OBSERVATIONS — for arm ``off`` (baseline) then ``lean`` (enabled,
no tool error counts), scoring the produced OpenAPI schema against ground truth
(endpoint precision/recall/F1 + component-schema recall). Streams a comparison.

Uses fixtures that ship precomputed dependency/project artifacts so only the
oas_update task runs (faster, isolates the planner task).

Usage::

    AB_FIXTURE=vulnyapi poetry run python scripts/ab_oas_build_task.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ARMS = {
    "off": '{"enabled": false}',
    "lean": '{"enabled": true, "include_tool_errors": false}',
}


def _load_precomputed(fixtures_root: Path, slug: str) -> dict[str, str] | None:
    art = fixtures_root / slug / "artifacts"
    out: dict[str, str] = {}
    for key in ("dependency_information", "project_information"):
        p = art / f"{key}_result.txt"
        if not p.is_file():
            return None
        out[f"{key}/result"] = p.read_text(encoding="utf-8")
    return out


async def run_arm(slug: str, arm: str, overlay: str, out_dir: Path) -> dict:
    os.environ["CONTRACTOR_EVAL_OBSERVATIONS"] = overlay

    import yaml

    from cli.fs import RootedLocalFileSystem
    from contractor.agents.oas_builder_agent.agent import build_oas_builder_agent
    from contractor.agents.swe_agent.agent import build_swe_agent
    from contractor.tools.observations import ObservationConfig
    from contractor.utils.settings import DEFAULT_MODEL, build_model
    from tests.eval.conftest import FIXTURES_ROOT, _load_fixture
    from tests.eval.results import metrics_from_task
    from tests.eval.scorers import score_oas_schema
    from tests.eval.task_harness import run_task_pipeline

    arm_tag = ObservationConfig.resolve(None, os.environ).as_tag()
    print(f"\n###### oas_build {slug} / arm={arm}  {overlay} ######", flush=True)
    print(f"  observations arm: {json.dumps(arm_tag)}", flush=True)

    model_alias = os.environ.get("CONTRACTOR_EVAL_MODEL") or DEFAULT_MODEL.model
    llm = build_model(model_alias, 600)
    fixture = _load_fixture(slug)
    fs = RootedLocalFileSystem(str(fixture.source_root))
    precomputed = _load_precomputed(FIXTURES_ROOT, slug)

    def queue(runner) -> None:
        swe_builder = partial(build_swe_agent, name="swe_agent", fs=fs, model=llm, max_tokens=100_000)
        oas_builder = partial(build_oas_builder_agent, name="oas_builder", fs=fs, model=llm, max_tokens=100_000)
        runner.add_variable(name="project_path", value=str(fixture.source_root))
        if precomputed is None:
            runner.add_task(name="dependency_information", worker_builder=swe_builder,
                            iterations=1, max_attempts=2, max_steps=20,
                            namespace="dependency_information", model=llm)
            runner.add_task(name="project_information", worker_builder=swe_builder,
                            iterations=1, max_attempts=2, max_steps=20,
                            artifacts=["dependency_information/result"],
                            namespace="project_information", model=llm)
        runner.add_task(name="oas_update", worker_builder=oas_builder,
                        iterations=2, max_attempts=4, max_steps=20,
                        artifacts=["dependency_information/result", "project_information/result"],
                        namespace="openapi-building", model=llm)

    oas_key = "user:oas-openapi-building"
    try:
        run = await run_task_pipeline(
            queue_fn=queue,
            artifact_keys=["oas_update/result", oas_key],
            namespace=f"ab-oas-{slug}-{arm}",
            timeout_s=float(os.environ.get("AB_TIMEOUT", "3000")),
            runner_name=f"oas-build-{slug}-{arm}",
            preloaded_artifacts=precomputed,
            output_dir=out_dir / arm,
        )
    except Exception as exc:
        print(f"  [{slug}/{arm}] ERROR: {exc!r}", flush=True)
        return {"arm": arm}

    text = run.artifacts.get(oas_key, "") or run.result_text("oas_update")
    schema = yaml.safe_load(text) if text else {}
    if not isinstance(schema, dict):
        schema = {}
    result = score_oas_schema(schema, fixture.expected_oas,
                              min_endpoint_precision=0.5, min_endpoint_recall=0.6,
                              min_schema_recall=0.3)
    ep = result.meta["endpoint_score"]
    sc = result.meta["schemas_score"]
    m = metrics_from_task(run.metrics)
    row = {
        "arm": arm,
        "endpoint_f1": round(getattr(ep, "f1", 0.0), 3),
        "endpoint_p": round(getattr(ep, "precision", 0.0), 3),
        "endpoint_r": round(getattr(ep, "recall", 0.0), 3),
        "schema_recall": round(getattr(sc, "recall", 0.0), 3),
        "passed": result.passed,
        "tools": m.get("total_tool_calls", 0),
        "tokens": m.get("total_tokens", 0),
        "llm": m.get("llm_calls", 0),
        "tool_errors": m.get("tool_errors", 0),
        "observations": arm_tag,
    }
    print(f"ROW {arm:5} | endpoint F1={row['endpoint_f1']:.3f} "
          f"(P={row['endpoint_p']:.3f} R={row['endpoint_r']:.3f}) "
          f"schema_recall={row['schema_recall']:.3f} passed={row['passed']} "
          f"| tools={row['tools']} tok={row['tokens']} llm={row['llm']}", flush=True)
    return row


async def main() -> None:
    slug = os.environ.get("AB_FIXTURE", "vulnyapi")
    out = REPO / "eval_runs" / "ab_oas_build" / slug
    out.mkdir(parents=True, exist_ok=True)
    print(f"A/B oas_build observations  fixture={slug}  arms={list(ARMS)}", flush=True)

    rows = {}
    for arm, overlay in ARMS.items():
        rows[arm] = await run_arm(slug, arm, overlay, out)
        (out / "ab_summary.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    off, lean = rows.get("off", {}), rows.get("lean", {})
    print("\n===== oas_build A/B SUMMARY (off vs lean) =====", flush=True)
    for arm in ("off", "lean"):
        r = rows.get(arm) or {}
        if "endpoint_f1" in r:
            print(f"  {arm:5} endpointF1={r['endpoint_f1']:.3f} schemaR={r['schema_recall']:.3f} "
                  f"tools={r['tools']} tok={r['tokens']}", flush=True)
    if off.get("endpoint_f1") is not None and lean.get("endpoint_f1") is not None:
        print(f"  DELTA dEndpointF1={lean['endpoint_f1'] - off['endpoint_f1']:+.3f} "
              f"dSchemaR={lean['schema_recall'] - off['schema_recall']:+.3f} "
              f"dTok={lean['tokens'] - off['tokens']:+d}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
