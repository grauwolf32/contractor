"""Resume Config B's tail (consolidation + wave-2) on the FINISHED B artifacts.

The full A/B left Config B's recon + wave-1 build complete but wave-2 timed out.
Rather than re-run those expensive stages, this copies the finished B artifact
store and runs only:

    knowledge_consolidation  (fresh; new output_format -> result IS the knowledge)
    oas_update  (wave-2)     (fed ONLY knowledge_consolidation/result)

Wave-2's oas_builder loads the wave-1 cumulative spec (present in the copied
store) and refines it from the consolidated knowledge alone. The old
consolidation artifacts are cleared first so the new consolidation is clean.

Usage:
    poetry run python scripts/resume_oas_ab_b_tail.py \
        --src eval_runs/oas-ab-pydio-idm-2/config-b/artifacts \
        --project ~/src/pydio-cells --folder idm \
        --expected ~/src/pydio-cells/common/proto/rest/cellsapi-rest.swagger.json \
        --path-prefixes "/acl,/role,/user,/policy,/workspace,/share,/graph,/meta,/search" \
        --model lm-studio-qwen3.6 --timeout 7200 --out eval_runs/oas-ab-b-tail
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from cli.fs import RootedLocalFileSystem
from contractor.agents.librarian_agent import build_librarian_agent
from contractor.agents.oas_builder_agent.agent import build_oas_builder_agent
from contractor.tools.artifact_pool import KeywordPoolBackend
from contractor.utils.settings import build_model
from tests.eval.scorers import diff_detail, score_oas_schema
from tests.eval.task_harness import run_task_pipeline

USER_ID = "eval-user"
OAS_KEY = "user:oas-openapi-building"

OAS_CONSOLIDATION_FOCUS = (
    "Consolidate everything known about the HTTP REST API surface into a clean, "
    "per-endpoint knowledge base the OpenAPI builder can consume. For each "
    "endpoint cluster the method + path, the request params/body schema, the "
    "response schema(s), and the auth requirement, citing the source files. "
    "Flag endpoints that are mentioned but missing from the current draft spec, "
    "and any conflicting signatures. The goal is to close gaps and fix errors in "
    "the OpenAPI draft on the next build pass."
)


def _load_spec_from_dir(artifact_dir: Path) -> dict[str, Any]:
    base = (Path(artifact_dir) / "users" / USER_ID / "artifacts"
            / "oas-openapi-building" / "versions")
    if not base.is_dir():
        return {}
    for v in sorted((int(d.name) for d in base.iterdir() if d.name.isdigit()),
                    reverse=True):
        f = base / str(v) / "oas-openapi-building"
        if f.is_file():
            spec = yaml.safe_load(f.read_text(encoding="utf-8"))
            if isinstance(spec, dict) and spec:
                return spec
    return {}


def _score(actual: dict, expected_full: dict, prefixes: tuple[str, ...]) -> dict:
    def sc(exp):
        r = score_oas_schema(actual, exp, min_endpoint_precision=0,
                             min_endpoint_recall=0, min_schema_recall=0)
        return diff_detail(r)
    scoped = dict(expected_full)
    scoped["paths"] = {p: v for p, v in expected_full.get("paths", {}).items()
                       if not prefixes or p.startswith(prefixes)}
    return {"vs_full": sc(expected_full), "vs_scoped": sc(scoped),
            "draft_paths": len(actual.get("paths", {}) or {})}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="finished config-b artifacts dir")
    ap.add_argument("--project", required=True)
    ap.add_argument("--folder", default=".")
    ap.add_argument("--expected", required=True)
    ap.add_argument("--path-prefixes", default="")
    ap.add_argument("--model", default=None)
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--out", default="eval_runs/oas-ab-b-tail")
    args = ap.parse_args()

    out = Path(args.out)
    work = out / "work"
    src = Path(args.src).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(src, work)
    # Clear the prior consolidation so the re-run is clean.
    art_base = work / "users" / USER_ID / "artifacts"
    for stale in ("knowledge_consolidation", "memory/knowledge:oas"):
        shutil.rmtree(art_base / stale, ignore_errors=True)

    pre_spec = _load_spec_from_dir(work)
    print(f"[resume] copied store; wave-1 spec has "
          f"{len(pre_spec.get('paths', {}))} paths to refine")

    fs = RootedLocalFileSystem(root_path=str(Path(args.project).expanduser()))
    llm = build_model(args.model, int(args.timeout))

    raw = Path(args.expected).expanduser().read_text(encoding="utf-8")
    expected = yaml.safe_load(raw) or {}
    if "definitions" in expected and "components" not in expected:
        expected["components"] = {"schemas": expected.get("definitions", {})}
    prefixes = tuple(p for p in (args.path_prefixes or "").split(",") if p)

    def queue(runner) -> None:
        runner.add_variable(name="project_path", value=args.folder)
        librarian = partial(
            build_librarian_agent, name="librarian_agent", fs=fs,
            artifact_service=runner.artifact_service, app_name=runner.name,
            user_id=USER_ID, pool_backend=KeywordPoolBackend(), model=llm,
        )
        # Pin the spec artifact so wave-2 refines the existing cumulative spec
        # from a FRESH memory namespace (no inherited recon/notes).
        oas = partial(build_oas_builder_agent, name="oas_builder", fs=fs,
                      model=llm, max_tokens=100_000,
                      oas_artifact_name="openapi-building")
        runner.add_task(
            # One consolidation pass is a complete synthesis; requiring 2
            # successful passes (the old 2/4) can't finish now that the result
            # inlines the full knowledge base. iterations=1, retried up to 3x.
            name="knowledge_consolidation", ref="consolidate",
            worker_builder=librarian, iterations=1, max_attempts=3, max_steps=24,
            params={"focus": OAS_CONSOLIDATION_FOCUS},
            artifacts=["dependency_information/result",
                       "project_information_short/result", "oas_update/result"],
            namespace="knowledge:oas", model=llm,
        )
        runner.add_task(
            name="oas_update", ref="oas_update_wave2",
            worker_builder=oas, iterations=2, max_attempts=4, max_steps=20,
            artifacts=["knowledge_consolidation/result"],
            namespace="openapi-rebuild", model=llm,
        )

    status = "ok"
    text = ""
    try:
        run = await run_task_pipeline(
            queue_fn=queue, artifact_keys=["oas_update/result", OAS_KEY],
            namespace="oas-ab-b-tail", timeout_s=args.timeout,
            runner_name="oas-ab-config-b", output_dir=out, artifact_dir=work,
        )
        text = run.artifacts.get(OAS_KEY, "") or run.result_text("oas_update")
    except Exception as exc:  # noqa: BLE001 - score the partial build on timeout
        status = f"{type(exc).__name__}: {exc}"[:120]
        print(f"[resume] did not finish ({status}) — scoring partial from disk")

    actual = yaml.safe_load(text) if text else {}
    if not isinstance(actual, dict) or not actual.get("paths"):
        actual = _load_spec_from_dir(work)
    if not isinstance(actual, dict):
        actual = {}

    scores = _score(actual, expected, prefixes)
    result = {"config": "config-b-retail", "status": status,
              "wave1_paths": len(pre_spec.get("paths", {})), **scores}
    (out / "comparison.json").write_text(json.dumps(result, indent=2),
                                         encoding="utf-8")
    print("\n===== CONFIG B (consolidation + wave-2 rerun) =====")
    print(f"  status={status}")
    print(f"  wave-1 start paths : {result['wave1_paths']}")
    print(f"  final draft paths  : {scores['draft_paths']}")
    print(f"  vs FULL  GT        : {scores['vs_full']}")
    print(f"  vs scoped GT       : {scores['vs_scoped']}")
    print(f"\nwrote {out / 'comparison.json'}")


if __name__ == "__main__":
    asyncio.run(main())
