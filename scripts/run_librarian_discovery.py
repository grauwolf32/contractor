"""Ad-hoc harness: run the knowledge_discovery task with the librarian agent
against an EXISTING artifact store + project, using the keyword pool backend.

Not part of the CLI workflow registry — it points the runner straight at a
pre-existing FileArtifactService store so we can test discovery on real prior
runs. Usage:

    poetry run python scripts/run_librarian_discovery.py \
        --store /tmp/lib-test-store \
        --project tests/playground/python/vulnyapi \
        --namespace knowledge:vulnyapi
"""

from __future__ import annotations

import argparse
import asyncio
from functools import partial

from google.adk.artifacts import FileArtifactService

from cli.fs import RootedLocalFileSystem
from contractor.agents.librarian_agent import build_librarian_agent
from contractor.runners.task_runner import TaskRunner
from contractor.tools.artifact_pool import ArtifactPool, KeywordPoolBackend
from contractor.utils.settings import build_model

APP_NAME = "contractor"
USER_ID = "cli-user"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="FileArtifactService root_dir")
    ap.add_argument("--project", required=True, help="project source root")
    ap.add_argument("--namespace", default="knowledge:discovery")
    ap.add_argument(
        "--task",
        default="knowledge_discovery",
        help="task template: knowledge_discovery | knowledge_consolidation",
    )
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--focus",
        default=None,
        help="workflow-injected focus/direction (task 'focus' param); "
        "omitted -> the task's default",
    )
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--max-steps", type=int, default=15)
    # Lower max-tokens => the worker hits the summarization limit sooner and
    # keeps a smaller context window (mitigates long-context OOM crashes on
    # memory-hungry local models). max-attempts retries a crashed/failed task.
    ap.add_argument("--max-tokens", type=int, default=80000)
    ap.add_argument("--max-attempts", type=int, default=1)
    args = ap.parse_args()

    artifact_service = FileArtifactService(root_dir=args.store)
    fs = RootedLocalFileSystem(root_path=args.project)
    llm = build_model(args.model, args.timeout)

    # Show the pool the librarian will read before the run.
    pool = ArtifactPool(
        artifact_service=artifact_service, app_name=APP_NAME, user_id=USER_ID
    )
    docs = await pool.documents()
    print(f"[pool] {len(docs)} documents across "
          f"{len({d.namespace for d in docs})} namespaces (reserved excluded)")

    # Resolve a focus preset shortcut (mirrors how a workflow would pass a
    # domain context constant via params) or use the raw string.
    focus = args.focus
    if focus in ("security", "security-consolidation"):
        from contractor.agents.librarian_agent.focuses import (
            SECURITY_CONSOLIDATION_FOCUS,
            SECURITY_DISCOVERY_FOCUS,
        )

        focus = (
            SECURITY_CONSOLIDATION_FOCUS
            if "consolidation" in args.task or focus == "security-consolidation"
            else SECURITY_DISCOVERY_FOCUS
        )

    worker_builder = partial(
        build_librarian_agent,
        name="librarian_agent",
        fs=fs,
        artifact_service=artifact_service,
        app_name=APP_NAME,
        user_id=USER_ID,
        pool_backend=KeywordPoolBackend(),
        max_tokens=args.max_tokens,
        model=llm,
    )

    runner = TaskRunner(name=APP_NAME, artifact_service=artifact_service)
    runner.add_variable(name="project_path", value=".")
    runner.add_task(
        name=args.task,
        ref=f"{args.task}:run",
        worker_builder=worker_builder,
        namespace=args.namespace,
        params={"focus": focus} if focus else {},
        artifact_key=f"{args.task}/{args.namespace.replace(':', '_')}",
        artifacts=[],
        iterations=1,
        max_attempts=args.max_attempts,
        max_steps=args.max_steps,
        timeout_s=float(args.timeout),
        model=llm,
    )

    async def on_event(ev) -> None:
        print(f"[event] {ev.type} task={ev.task_name}")

    results = await runner.run(user_id=USER_ID, on_event=on_event)

    print("\n===== RESULT =====")
    for r in results:
        print(f"status={r.status} summary={r.summary!r}")
        print((r.result or "")[:2000])

    print("\n===== NOTES WRITTEN TO", args.namespace, "=====")
    notes = await pool.load_notes(args.namespace)
    for name, note in notes.items():
        print(f"- {name}  tags={note.get('tags')}")
        print(f"    {str(note.get('memory') or '')[:200]}")


if __name__ == "__main__":
    asyncio.run(main())
