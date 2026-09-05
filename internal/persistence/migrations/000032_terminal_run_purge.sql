ALTER TABLE artifact_pins
    ADD COLUMN run_id text;

-- The historical rows predate explicit Run ownership. Temporarily remove the
-- immutable guard while the migration derives that ownership from the exact,
-- collision-safe pin identifiers written by the application.
DROP TRIGGER artifact_pins_immutable ON artifact_pins;

WITH candidates AS (
    SELECT pin.ctid AS pin_row, run.run_id, octet_length(run.run_id) AS specificity
    FROM artifact_pins AS pin
    JOIN workflow_runs AS run
      ON pin.pin_kind IN ('run_input', 'run_output')
     AND left(pin.pin_id, length(run.run_id) + 1) = run.run_id || ':'
    UNION ALL
    SELECT pin.ctid AS pin_row, execution.run_id,
           octet_length(execution.stage_execution_id) AS specificity
    FROM artifact_pins AS pin
    JOIN stage_executions AS execution
      ON pin.pin_kind IN ('stage_context', 'stage_result')
     AND left(pin.pin_id, length(execution.stage_execution_id) + 1)
         = execution.stage_execution_id || ':'
), resolved AS (
    SELECT DISTINCT ON (pin_row) pin_row, run_id
    FROM candidates
    ORDER BY pin_row, specificity DESC, run_id
)
UPDATE artifact_pins AS pin
SET run_id = resolved.run_id
FROM resolved
WHERE pin.ctid = resolved.pin_row;

ALTER TABLE artifact_pins
    ALTER COLUMN run_id SET NOT NULL,
    ADD CONSTRAINT artifact_pins_run_id_fkey
        FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id) ON DELETE CASCADE;

CREATE INDEX artifact_pins_run_idx ON artifact_pins (run_id, pin_kind, pin_id);

CREATE TRIGGER artifact_pins_immutable
BEFORE UPDATE OR DELETE ON artifact_pins
FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_immutable();

CREATE OR REPLACE FUNCTION contractor_lifecycle_purge_enabled()
RETURNS boolean
LANGUAGE sql
STABLE
AS $$
    SELECT COALESCE(current_setting('contractor.lifecycle_purge', true), '')
        IN ('run', 'project')
$$;

CREATE OR REPLACE FUNCTION contractor_protect_artifact_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'DELETE' AND contractor_lifecycle_purge_enabled() THEN
        RETURN OLD;
    END IF;
    RAISE EXCEPTION 'immutable ArtifactStore row cannot be changed' USING ERRCODE = '23514';
END;
$$;

CREATE OR REPLACE FUNCTION contractor_protect_artifact_blob_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        IF contractor_lifecycle_purge_enabled() THEN
            RETURN OLD;
        END IF;
        RAISE EXCEPTION 'immutable Artifact blob cannot be changed' USING ERRCODE = '23514';
    END IF;
    IF ROW(NEW.sha256, NEW.payload, NEW.size_bytes, NEW.created_at)
       IS DISTINCT FROM
       ROW(OLD.sha256, OLD.payload, OLD.size_bytes, OLD.created_at)
    THEN
        RAISE EXCEPTION 'immutable Artifact blob cannot be changed' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION contractor_protect_execution_report_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'DELETE' AND contractor_lifecycle_purge_enabled() THEN
        RETURN OLD;
    END IF;
    RAISE EXCEPTION 'execution report cannot be changed' USING ERRCODE = '23514';
END;
$$;
