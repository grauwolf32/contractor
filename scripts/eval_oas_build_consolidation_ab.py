"""A/B eval: single deep oas_build vs split build + consolidation + rebuild.

Tests research direction B3 (multi-pass) / the min_iter–max_iter grounding:
min_iterations = a floor of (re-)exploration opportunities WITH cumulative
memory; max_iterations = a cap on failed variations. Config B inserts a
``knowledge_consolidation`` step between two shorter build waves so the second
wave's cumulative memory is clean.

  Config A (baseline):  dep_info -> proj_info -> oas_update(iter=4, att=8)
  Config B (split):     dep_info -> proj_info -> oas_update(iter=2, att=4)
                                  -> knowledge_consolidation
                                  -> oas_update(iter=2, att=4)

Both produce the cumulative ``user:oas-openapi-building`` spec, scored against a
ground-truth OpenAPI/Swagger spec via the existing endpoint/schema scorer.

Usage:
    poetry run python scripts/eval_oas_build_consolidation_ab.py \
        --project ~/src/pydio-cells --folder . \
        --expected /path/to/cellsapi-rest.swagger.json \
        --config both --model lm-studio-qwen3.6 --out eval_runs/oas-ab-pydio
"""

from __future__ import annotations

import argparse
import asyncio
import json
from functools import partial
from pathlib import Path
from typing import Any

import litellm
import yaml

from cli.fs import RootedLocalFileSystem
from contractor.agents.librarian_agent import build_librarian_agent
from contractor.agents.oas_builder_agent.agent import build_oas_builder_agent
from contractor.agents.swe_agent.agent import build_swe_agent
from contractor.tools.artifact_pool import KeywordPoolBackend
from contractor.utils.settings import build_model
from tests.eval.scorers import diff_detail, score_oas_schema
from tests.eval.task_harness import run_task_pipeline

# litellm's async logging worker can raise CancelledError/TimeoutError at
# shutdown and crash the run; we use none of its callbacks here, so disable them.
litellm.success_callback = []
litellm.failure_callback = []
litellm.callbacks = []

USER_ID = "eval-user"
OAS_KEY = "user:oas-openapi-building"

# Consolidation focus steers the librarian toward the API surface so wave 2
# builds a more complete/correct spec (passed as a workflow param — the task
# itself stays domain-agnostic).
OAS_CONSOLIDATION_FOCUS = (
    "Consolidate everything known about the HTTP REST API surface into a clean, "
    "per-endpoint knowledge base the OpenAPI builder can consume. For each "
    "endpoint cluster the method + path, the request params/body schema, the "
    "response schema(s), and the auth requirement, citing the source files. "
    "Flag endpoints that are mentioned but missing from the current draft spec, "
    "and any conflicting signatures. The goal is to close gaps and fix errors in "
    "the OpenAPI draft on the next build pass."
)


def _normalize_expected(spec: dict[str, Any]) -> dict[str, Any]:
    """Make a Swagger-2.0 ground truth comparable to an OAS-3 build.

    Endpoint scoring reads ``paths`` (same in both); schema-recall reads
    ``components.schemas`` (OAS 3) — Swagger 2.0 keeps these under
    ``definitions``, so lift them so the schema-name set lines up.
    """
    if "definitions" in spec and "components" not in spec:
        spec = dict(spec)
        spec["components"] = {"schemas": spec.get("definitions", {})}
    return spec


# (iterations, max_attempts, max_steps) per stage. ``--smoke`` shrinks every
# stage to a single quick pass so the full wiring runs end-to-end in minutes.
def _budgets(smoke: bool) -> dict[str, tuple[int, int, int]]:
    if smoke:
        return {"recon": (1, 1, 6), "build_a": (1, 1, 6),
                "build_b": (1, 1, 6), "consolidate": (1, 1, 6)}
    # consolidate: one pass is a complete synthesis, but emitting the full
    # inlined knowledge base + finishing is heavy for a 35b — give it more
    # planner steps (24) and retries (4) so it reliably completes.
    return {"recon": (1, 2, 20), "build_a": (4, 8, 16),
            "build_b": (2, 4, 16), "consolidate": (1, 4, 24)}


def _add_recon(runner, swe, llm, b, folder: str) -> None:
    """Queue the shared recon tasks (dependency + short project inventory)."""
    runner.add_variable(name="project_path", value=folder)
    ri, ra, rs = b["recon"]
    runner.add_task(
        name="dependency_information", ref="dependency_information",
        worker_builder=swe, iterations=ri, max_attempts=ra, max_steps=rs,
        namespace="dependency_information", model=llm,
    )
    runner.add_task(
        name="project_information_short", ref="project_information_short",
        worker_builder=swe, iterations=ri, max_attempts=ra, max_steps=rs,
        artifacts=["dependency_information/result"],
        namespace="project_information_short", model=llm,
    )


def _build_queue_a(fs, folder: str, llm, smoke: bool = False, skip_recon: bool = False):
    b = _budgets(smoke)
    swe = partial(build_swe_agent, name="swe_agent", fs=fs, model=llm, max_tokens=80_000)
    oas = partial(
        build_oas_builder_agent, name="oas_builder", fs=fs, model=llm, max_tokens=80_000
    )

    def queue(runner) -> None:
        runner.add_variable(name="project_path", value=folder)
        if not skip_recon:
            _add_recon(runner, swe, llm, b, folder)
        bi, ba, bs = b["build_a"]
        runner.add_task(
            name="oas_update", ref="oas_update",
            worker_builder=oas, iterations=bi, max_attempts=ba, max_steps=bs,
            artifacts=["dependency_information/result", "project_information_short/result"],
            namespace="openapi-building", model=llm,
        )

    return queue


def _build_queue_b(fs, folder: str, llm, smoke: bool = False, skip_recon: bool = False):
    b = _budgets(smoke)
    swe = partial(build_swe_agent, name="swe_agent", fs=fs, model=llm, max_tokens=80_000)
    # Pin the spec artifact to one key for BOTH waves, so wave-2 can run in a
    # fresh MEMORY namespace (no inherited recon/notes) yet keep refining the
    # same cumulative spec instead of starting an empty one.
    oas = partial(
        build_oas_builder_agent, name="oas_builder", fs=fs, model=llm,
        max_tokens=80_000, oas_artifact_name="openapi-building",
    )

    def queue(runner) -> None:
        runner.add_variable(name="project_path", value=folder)
        # Consolidation worker is bound to THIS run's artifact store/app so its
        # cross-namespace pool tools see wave-1's output.
        librarian = partial(
            build_librarian_agent, name="librarian_agent", fs=fs,
            artifact_service=runner.artifact_service, app_name=runner.name,
            user_id=USER_ID, pool_backend=KeywordPoolBackend(), model=llm,
        )
        if not skip_recon:
            _add_recon(runner, swe, llm, b, folder)
        # Wave 1 — shorter build.
        wi, wa, ws = b["build_b"]
        runner.add_task(
            name="oas_update", ref="oas_update_wave1",
            worker_builder=oas, iterations=wi, max_attempts=wa, max_steps=ws,
            artifacts=["dependency_information/result", "project_information_short/result"],
            namespace="openapi-building", model=llm,
        )
        # Consolidate the scattered API-surface knowledge.
        ci, ca, cs = b["consolidate"]
        runner.add_task(
            name="knowledge_consolidation", ref="consolidate",
            worker_builder=librarian, iterations=ci, max_attempts=ca, max_steps=cs,
            params={"focus": OAS_CONSOLIDATION_FOCUS},
            artifacts=[
                "dependency_information/result",
                "project_information_short/result",
                "oas_update/result",
            ],
            namespace="knowledge:oas", model=llm,
        )
        # Wave 2 — refine the SAME cumulative spec (oas_artifact_name pins it),
        # but in a FRESH memory namespace so it does NOT inherit wave-1's
        # injected recon or working notes. Its only injected knowledge is the
        # consolidation result, so the build genuinely relies on the
        # consolidated memory — the point of the split.
        runner.add_task(
            name="oas_update", ref="oas_update_wave2",
            worker_builder=oas, iterations=wi, max_attempts=wa, max_steps=ws,
            artifacts=["knowledge_consolidation/result"],
            namespace="openapi-rebuild", model=llm,
        )

    return queue


RECON_KEYS = ["dependency_information/result", "project_information_short/result"]


def _load_spec_from_dir(
    artifact_dir: Path, *, app_name: str
) -> dict[str, Any]:
    """Recover the latest built OpenAPI spec from a persisted ADK artifact tree.

    Lets us score the partial spec when an arm times out or crashes mid-build
    instead of losing the whole run — the cumulative spec is saved every step.
    """
    root = Path(artifact_dir)
    base = (
        root
        / "apps"
        / app_name
        / "users"
        / USER_ID
        / "artifacts"
        / "oas-openapi-building"
        / "versions"
    )
    if not base.is_dir():
        base = (
            root
            / "users"
            / USER_ID
            / "artifacts"
            / "oas-openapi-building"
            / "versions"
        )
    if not base.is_dir():
        return {}
    versions = sorted(int(d.name) for d in base.iterdir() if d.name.isdigit())
    for v in reversed(versions):  # newest first; fall back if the tail is empty
        f = base / str(v) / "oas-openapi-building"
        if f.is_file():
            spec = yaml.safe_load(f.read_text(encoding="utf-8"))
            if isinstance(spec, dict) and spec:
                return spec
    return {}


async def _precompute_recon(fs, folder, llm, *, out: Path, timeout: float) -> dict[str, str]:
    """Run recon ONCE; return its artifacts to inject into both A and B so they
    start from identical context (fair A/B + no double recon cost)."""
    swe = partial(build_swe_agent, name="swe_agent", fs=fs, model=llm, max_tokens=80_000)

    def queue(runner) -> None:
        _add_recon(runner, swe, llm, _budgets(False), folder)

    run = await run_task_pipeline(
        queue_fn=queue, artifact_keys=RECON_KEYS, namespace="oas-ab-recon",
        timeout_s=timeout, runner_name="oas-ab-recon",
        output_dir=out / "recon", artifact_dir=out / "recon" / "artifacts",
    )
    preloaded = {k: run.artifacts.get(k, "") for k in RECON_KEYS}
    print(f"[recon] dep={len(preloaded[RECON_KEYS[0]])}c "
          f"proj={len(preloaded[RECON_KEYS[1]])}c")
    return preloaded


async def _run_config(
    name: str, queue, expected: dict[str, Any], *, out: Path, timeout: float,
    preloaded: dict[str, str] | None = None,
) -> dict[str, Any]:
    art_dir = out / name / "artifacts"
    text = ""
    status = "ok"
    try:
        run = await run_task_pipeline(
            queue_fn=queue,
            artifact_keys=["oas_update/result", OAS_KEY],
            namespace=f"oas-ab-{name}",
            timeout_s=timeout,
            runner_name=f"oas-ab-{name}",
            preloaded_artifacts=preloaded or None,
            output_dir=out / name,
            artifact_dir=art_dir,
        )
        text = run.artifacts.get(OAS_KEY, "") or run.result_text("oas_update")
    except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 - isolate one arm
        # CancelledError (BaseException) included: litellm's background logging
        # worker can raise it at shutdown and would otherwise kill the whole run.
        status = f"{type(exc).__name__}: {exc}"[:120]
        print(f"\n[{name}] run did not finish ({status}) — "
              f"scoring the partial spec from disk")

    actual = yaml.safe_load(text) if text else {}
    if not isinstance(actual, dict) or not actual.get("paths"):
        # Timed out / crashed (or empty result) -> recover the partial build.
        recovered = _load_spec_from_dir(art_dir, app_name=f"oas-ab-{name}")
        if recovered:
            actual = recovered
            if status == "ok":
                status = "partial(from-disk)"
    if not isinstance(actual, dict):
        actual = {}
    result = score_oas_schema(
        actual, expected,
        min_endpoint_precision=0.0, min_endpoint_recall=0.0, min_schema_recall=0.0,
    )
    detail = diff_detail(result)
    n_paths = len((actual or {}).get("paths", {}) or {})
    print(f"\n[{name}] status={status} endpoints F1={detail.get('f1')} "
          f"P={detail.get('precision')} R={detail.get('recall')} "
          f"| draft paths={n_paths}")
    return {"config": name, "status": status, "detail": detail,
            "draft_paths": n_paths, "produced_spec": bool(actual.get("paths"))}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--folder", default=".")
    ap.add_argument("--expected", required=True, help="ground-truth OpenAPI/Swagger json|yaml")
    ap.add_argument("--config", choices=["a", "b", "both"], default="both")
    ap.add_argument("--model", default=None)
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--out", default="eval_runs/oas-build-ab")
    ap.add_argument("--smoke", action="store_true",
                    help="single quick pass per stage to validate wiring")
    ap.add_argument("--no-precompute", dest="precompute", action="store_false",
                    help="re-run recon inside each arm instead of sharing it")
    ap.set_defaults(precompute=True)
    ap.add_argument("--path-prefixes", default="",
                    help="comma-separated path prefixes to scope the ground "
                    "truth to (e.g. '/acl,/role,/user' when --folder is one "
                    "service). Empty = score against the whole spec.")
    ap.add_argument("--recon-cache", default="",
                    help="reuse recon from a prior run's recon dir "
                    "(<dir>/dependency_information_result + "
                    "project_information_short_result) instead of recomputing")
    args = ap.parse_args()

    project = str(Path(args.project).expanduser())
    fs = RootedLocalFileSystem(root_path=project)
    llm = build_model(args.model, int(args.timeout))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = Path(args.expected).expanduser().read_text(encoding="utf-8")
    expected = _normalize_expected(yaml.safe_load(raw) or {})
    prefixes = tuple(p for p in (args.path_prefixes or "").split(",") if p)
    if prefixes:
        expected = dict(expected)
        expected["paths"] = {
            p: v for p, v in (expected.get("paths") or {}).items()
            if p.startswith(prefixes)
        }
    print(f"ground truth: {len(expected.get('paths', {}))} paths, "
          f"{len(expected.get('components', {}).get('schemas', {}))} schemas"
          + (f" (scoped to {prefixes})" if prefixes else ""))

    # Precompute recon once and inject into both arms (skip in smoke). This
    # makes A and B start from identical context, so the A/B measures the build
    # strategy, not recon variance.
    preloaded: dict[str, str] | None = None
    if args.recon_cache:
        cache = Path(args.recon_cache).expanduser()
        preloaded = {
            k: (cache / k.replace("/", "_")).read_text(encoding="utf-8")
            for k in RECON_KEYS
        }
        print(f"[recon] reused from {cache}: "
              f"dep={len(preloaded[RECON_KEYS[0]])}c "
              f"proj={len(preloaded[RECON_KEYS[1]])}c")
    elif args.precompute and not args.smoke:
        preloaded = await _precompute_recon(
            fs, args.folder, llm, out=out, timeout=args.timeout)
    skip = preloaded is not None

    results = []
    if args.config in ("a", "both"):
        results.append(await _run_config(
            "config-a", _build_queue_a(fs, args.folder, llm, args.smoke, skip),
            expected, out=out, timeout=args.timeout, preloaded=preloaded))
    if args.config in ("b", "both"):
        results.append(await _run_config(
            "config-b", _build_queue_b(fs, args.folder, llm, args.smoke, skip),
            expected, out=out, timeout=args.timeout, preloaded=preloaded))

    (out / "comparison.json").write_text(
        json.dumps({"ground_truth_paths": len(expected.get("paths", {})),
                    "results": results}, indent=2), encoding="utf-8")
    print("\n===== A/B COMPARISON =====")
    for r in results:
        d = r["detail"]
        print(f"  {r['config']:9} F1={d.get('f1')}  P={d.get('precision')}  "
              f"R={d.get('recall')}  draft_paths={r['draft_paths']}")
    print(f"\nwrote {out/'comparison.json'}")


if __name__ == "__main__":
    asyncio.run(main())
