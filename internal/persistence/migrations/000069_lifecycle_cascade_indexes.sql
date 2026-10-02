-- The planner-event cascade probes this key once per deleted Run event.
-- Historical planner events without Run identity do not need entries.
CREATE INDEX planner_events_run_event_idx
    ON planner_events (run_id, run_event_sequence)
    WHERE run_id IS NOT NULL;

-- Run and planner-session deletion also probes these referencing tables.
CREATE INDEX artifact_scopes_run_idx
    ON artifact_scopes (run_id) WHERE run_id IS NOT NULL;
CREATE INDEX gateway_allocation_routes_run_idx
    ON gateway_allocation_routes (run_id);
CREATE INDEX gateway_recovery_waits_run_idx
    ON gateway_recovery_waits (run_id);
CREATE INDEX gateway_recovery_failures_run_idx
    ON gateway_recovery_failures (run_id);
CREATE INDEX run_stage_resumptions_run_idx
    ON run_stage_resumptions (run_id);
CREATE INDEX planner_execution_reports_session_idx
    ON planner_execution_reports (session_id);
