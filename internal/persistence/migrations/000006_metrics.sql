CREATE TABLE allocation_execution_reports (
    report_id text PRIMARY KEY CHECK (btrim(report_id) <> ''),
    stage_execution_id text NOT NULL,
    allocation_id text NOT NULL,
    logical_agent_name text NOT NULL,
    report_schema_version text NOT NULL CHECK (btrim(report_schema_version) <> ''),
    report jsonb NOT NULL CHECK (
        jsonb_typeof(report) = 'object'
        AND report->>'reportId' = report_id
        AND report->>'allocationId' = allocation_id
        AND jsonb_typeof(report->'worker') = 'object'
        AND jsonb_typeof(report->'runtime') = 'object'
    ),
    received_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    expires_at timestamptz NOT NULL DEFAULT (clock_timestamp() + interval '30 days'),
    FOREIGN KEY (stage_execution_id, allocation_id, logical_agent_name)
        REFERENCES stage_allocations(stage_execution_id, allocation_id, logical_agent_name)
        ON DELETE CASCADE
);

CREATE INDEX allocation_execution_reports_stage_idx
    ON allocation_execution_reports (stage_execution_id, logical_agent_name);
CREATE INDEX allocation_execution_reports_allocation_idx
    ON allocation_execution_reports (allocation_id, received_at DESC);
CREATE UNIQUE INDEX allocation_execution_reports_complete_identity
    ON allocation_execution_reports (stage_execution_id, allocation_id, logical_agent_name)
    WHERE (report->'worker'->>'complete')::boolean
      AND (report->'runtime'->>'complete')::boolean;
CREATE INDEX allocation_execution_reports_expiry_idx
    ON allocation_execution_reports (expires_at, stage_execution_id);

CREATE TABLE planner_execution_reports (
    report_id text PRIMARY KEY CHECK (btrim(report_id) <> ''),
    stage_execution_id text NOT NULL REFERENCES stage_executions(stage_execution_id)
        ON DELETE CASCADE,
    session_id text NOT NULL REFERENCES planner_sessions(session_id) ON DELETE CASCADE,
    invocation_id text NOT NULL CHECK (btrim(invocation_id) <> ''),
    started_at timestamptz NOT NULL,
    finished_at timestamptz NOT NULL CHECK (finished_at >= started_at),
    report_schema_version text NOT NULL CHECK (btrim(report_schema_version) <> ''),
    report jsonb NOT NULL CHECK (
        jsonb_typeof(report) = 'object'
        AND report->>'reportId' = report_id
    ),
    received_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    expires_at timestamptz NOT NULL DEFAULT (clock_timestamp() + interval '30 days')
);

CREATE INDEX planner_execution_reports_expiry_idx
    ON planner_execution_reports (expires_at, stage_execution_id);
CREATE INDEX planner_execution_reports_stage_idx
    ON planner_execution_reports (stage_execution_id, received_at DESC);
CREATE UNIQUE INDEX planner_execution_reports_complete_identity
    ON planner_execution_reports (stage_execution_id, session_id)
    WHERE (report->>'complete')::boolean;

CREATE TABLE stage_metrics (
    stage_execution_id text PRIMARY KEY REFERENCES stage_executions(stage_execution_id)
        ON DELETE CASCADE,
    metrics_schema_version text NOT NULL CHECK (btrim(metrics_schema_version) <> ''),
    metrics jsonb NOT NULL CHECK (
        jsonb_typeof(metrics) = 'object'
        AND jsonb_typeof(metrics->'workers') = 'object'
        AND jsonb_typeof(metrics->'runtime') = 'object'
    ),
    summary jsonb NOT NULL CHECK (jsonb_typeof(summary) = 'object'),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    expires_at timestamptz NOT NULL DEFAULT (clock_timestamp() + interval '30 days')
);

CREATE INDEX stage_metrics_expiry_idx ON stage_metrics (expires_at, stage_execution_id);

CREATE OR REPLACE FUNCTION contractor_protect_telemetry_report_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'telemetry report cannot be changed' USING ERRCODE = '23514';
END;
$$;

CREATE TRIGGER allocation_execution_reports_immutable
BEFORE UPDATE ON allocation_execution_reports
FOR EACH ROW EXECUTE FUNCTION contractor_protect_telemetry_report_immutable();

CREATE TRIGGER planner_execution_reports_immutable
BEFORE UPDATE ON planner_execution_reports
FOR EACH ROW EXECUTE FUNCTION contractor_protect_telemetry_report_immutable();
