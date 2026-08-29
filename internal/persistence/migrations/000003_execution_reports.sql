ALTER TABLE stage_allocations
    ADD CONSTRAINT stage_allocation_report_identity
    UNIQUE (stage_execution_id, allocation_id, logical_agent_name);

CREATE TABLE stage_execution_reports (
    stage_execution_id text NOT NULL,
    allocation_id text NOT NULL,
    logical_agent_name text NOT NULL,
    report_schema_version text NOT NULL CHECK (btrim(report_schema_version) <> ''),
    report jsonb NOT NULL CHECK (
        jsonb_typeof(report) = 'object'
        AND jsonb_typeof(report->'allocationId') = 'string'
        AND report->>'allocationId' = allocation_id
        AND jsonb_typeof(report->'startedAt') = 'string'
        AND jsonb_typeof(report->'finishedAt') = 'string'
        AND jsonb_typeof(report->'complete') = 'boolean'
        AND jsonb_typeof(report->'counters') = 'object'
        AND jsonb_typeof(report->'errors') = 'array'
        AND jsonb_typeof(report->'truncated') = 'boolean'
    ),
    received_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (stage_execution_id, logical_agent_name),
    UNIQUE (allocation_id),
    FOREIGN KEY (stage_execution_id, allocation_id, logical_agent_name)
        REFERENCES stage_allocations(stage_execution_id, allocation_id, logical_agent_name)
        ON DELETE CASCADE
);

CREATE INDEX stage_execution_reports_stage_idx
    ON stage_execution_reports (stage_execution_id, logical_agent_name);

CREATE OR REPLACE FUNCTION contractor_protect_execution_report_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'execution report cannot be changed' USING ERRCODE = '23514';
END;
$$;

CREATE TRIGGER stage_execution_reports_immutable
BEFORE UPDATE OR DELETE ON stage_execution_reports
FOR EACH ROW EXECUTE FUNCTION contractor_protect_execution_report_immutable();
