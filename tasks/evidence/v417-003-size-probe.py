"""Test-only serialized-size and transient-copy measurements for V417-003.

Run from runtime with its locked environment:
uv run python ../tasks/evidence/v417-003-size-probe.py /path/to/report.json
No model or external service is called. Inputs are deterministic test data.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import tracemalloc
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime/tests"))

from test_worker_compaction import call_event, new_coordinator, reply_event  # noqa: E402


async def measure(name: str, count: int, payload: object) -> dict[str, object]:
    coordinator = new_coordinator()
    events = []
    for ordinal in range(1, count + 1):
        value = json.loads(json.dumps(payload)) if ordinal < count else {"status": "recent"}
        for event in (
            call_event(f"call-{ordinal}", f"c{ordinal}"),
            reply_event(f"reply-{ordinal}", f"c{ordinal}", value=value),
        ):
            events.append(event)
            await coordinator.observe(event)
    index = coordinator._archive
    if index is None:
        raise RuntimeError("Missing archive")
    groups = index.groups
    raw_bytes = sum(len(event.model_dump_json(exclude_none=True).encode()) for event in events)
    projection_bytes = sum(len(group.projection.encode()) for group in groups)
    projection_heap_bytes = sum(sys.getsizeof(group.projection) for group in groups)
    tracemalloc.start()
    plan = await coordinator.plan()
    plan_current, plan_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    document_bytes = len(plan.document.encode())
    summary = coordinator.validate_summary(plan, json.dumps({"summary": "Earlier steps checked."}))
    marker = coordinator.marker(plan, sdk_invocation_id="probe-sdk")
    coordinator.commit(plan, summary, marker)
    tracemalloc.start()
    page = coordinator.history_page("inv-1", 1, 0)
    page_current, page_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    page_bytes = len(json.dumps(page, ensure_ascii=False, separators=(",", ":")).encode())
    result = {
        "case": name,
        "groups": len(groups),
        "events": len(events),
        "raw_session_events_utf8_bytes": raw_bytes,
        "retained_projection_utf8_bytes": projection_bytes,
        "retained_projection_with_registered_overhead_bytes": index._bytes,
        "retained_projection_python_string_heap_bytes": projection_heap_bytes,
        "selected_groups": plan.last_group - plan.first_group + 1,
        "compactor_document_utf8_bytes": document_bytes,
        "plan_traced_live_bytes": plan_current,
        "plan_traced_peak_bytes": plan_peak,
        "history_page_utf8_bytes": page_bytes,
        "history_page_traced_live_bytes": page_current,
        "history_page_traced_peak_bytes": page_peak,
        "summary_utf8_bytes": len(summary.encode()),
        "marker_utf8_bytes": len(marker.model_dump_json(exclude_none=True).encode()),
    }
    coordinator.end()
    if coordinator._archive is not None or coordinator.active:
        raise RuntimeError("Invocation archive survived cleanup")
    return result


async def main() -> None:
    cases = [
        ("ordinary-tool-results", 40, {"status": "checked", "detail": "source line\n" * 80}),
        (
            "http-results",
            128,
            {"status": 200, "headers": {"content-type": "text/html"}, "body": "response\n" * 1024},
        ),
        ("multibyte-source-results", 64, {"source": "проверка доступа\n" * 1500}),
        ("large-multipage-result", 2, {str(i): "x" * 65536 for i in range(64)}),
        ("4096-group-ceiling", 4096, {"status": "checked"}),
        ("near-32-MiB-history", 500, {"body": "x" * 65536}),
    ]
    report = {
        "scope": (
            "Deterministic production archive/coordinator, synthetic observations; "
            "not live-model quality or an RSS ceiling"
        ),
        "base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "runtime_patch_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "--", "runtime/src", "runtime/tests"], cwd=ROOT)
        ).hexdigest(),
        "runtime_lock_sha256": hashlib.sha256((ROOT / "runtime/uv.lock").read_bytes()).hexdigest(),
        "google_adk": version("google-adk"),
        "python": sys.version.split()[0],
        "measurement": (
            "UTF-8 JSON sizes; tracemalloc starts after archive admission and measures each "
            "operation separately. Heap peaks are observations, not timing or pass thresholds. "
            "Raw events are serialized individually and remain session-owned references."
        ),
        "cases": [await measure(*case) for case in cases],
    }
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Recorded {len(cases)} deterministic size/copy measurements")


if __name__ == "__main__":
    asyncio.run(main())
