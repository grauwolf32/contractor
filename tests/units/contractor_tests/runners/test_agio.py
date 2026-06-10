"""Guards for the Agio event taxonomy.

``cli/metrics.MetricsSink`` filters on ``ALL_AGIO_EVENT_TYPES`` — any event
type emitted by a runner but not mirrored in ``AgioEventType`` silently
vanishes from ``metrics.jsonl``. These tests make the mirroring explicit so
a new ``EventType`` member fails loudly here instead.
"""

from __future__ import annotations

from contractor.runners.agio import ALL_AGIO_EVENT_TYPES, AgioEventType
from contractor.runners.models import EventType


class TestTaxonomyMirroring:
    def test_every_task_runner_event_type_is_persisted(self):
        # models.EventType (what TaskRunner emits) must be a subset of the
        # Agio taxonomy (what MetricsSink persists). A member added to
        # EventType without an AgioEventType mirror would be dropped from
        # metrics.jsonl without any error.
        missing = {e.value for e in EventType} - ALL_AGIO_EVENT_TYPES
        assert not missing, (
            f"EventType member(s) {sorted(missing)} are not mirrored in "
            f"AgioEventType — events of these types would silently vanish "
            f"from metrics.jsonl"
        )

    def test_agent_runner_string_events_are_persisted(self):
        # AgentRunner emits these as string literals (no enum), and
        # Workflow.emit_task_skipped emits "task_skipped" — all must stay
        # in the taxonomy or RouterWorkflow / skip events disappear from
        # metrics.jsonl.
        assert {
            "agent_run_started",
            "agent_run_finished",
            "final_text",
            "task_skipped",
        } <= ALL_AGIO_EVENT_TYPES

    def test_all_agio_event_types_matches_enum(self):
        assert frozenset(t.value for t in AgioEventType) == ALL_AGIO_EVENT_TYPES
