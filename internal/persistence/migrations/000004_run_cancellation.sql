ALTER TABLE workflow_runs
    ADD COLUMN cancellation_schema_version text,
    ADD COLUMN run_cancellation jsonb;

-- Versions before this migration could represent cancelling through the
-- generic state transition API. Preserve such rows with an explicit legacy
-- cancellation before enforcing the new shape.
UPDATE workflow_runs
SET cancellation_schema_version = 'contractor/v1alpha1',
    run_cancellation = jsonb_build_object(
        'code', 'user_cancelled',
        'requestedAt', updated_at
    )
WHERE state IN ('cancelling', 'cancelled');

CREATE FUNCTION contractor_valid_run_cancellation(payload jsonb)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT COALESCE(
        jsonb_typeof(payload) = 'object'
        AND payload->>'code' = 'user_cancelled'
        AND jsonb_typeof(payload->'requestedAt') = 'string'
        AND btrim(payload->>'requestedAt') <> ''
        AND (
            NOT (payload ? 'requestedBy')
            OR jsonb_typeof(payload->'requestedBy') = 'string'
                AND btrim(payload->>'requestedBy') <> ''
        )
        AND (
            NOT (payload ? 'reason')
            OR jsonb_typeof(payload->'reason') = 'string'
                AND btrim(payload->>'reason') <> ''
        ),
        false
    )
$$;

ALTER TABLE workflow_runs
    ADD CONSTRAINT workflow_run_cancellation_shape CHECK (
        (state IN ('cancelling', 'cancelled')) = (
            cancellation_schema_version IS NOT NULL
            AND btrim(cancellation_schema_version) <> ''
            AND run_cancellation IS NOT NULL
            AND contractor_valid_run_cancellation(run_cancellation)
        )
        AND (
            state IN ('cancelling', 'cancelled')
            OR cancellation_schema_version IS NULL AND run_cancellation IS NULL
        )
    );

DROP INDEX workflow_runs_scheduler_idx;
CREATE INDEX workflow_runs_scheduler_idx
    ON workflow_runs (state, scheduler_claim_expires_at, created_at)
    WHERE state IN ('running', 'cancelling');

CREATE OR REPLACE FUNCTION contractor_protect_workflow_run_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.run_id IS DISTINCT FROM OLD.run_id
        OR NEW.owner_id IS DISTINCT FROM OLD.owner_id
        OR NEW.workflow_name IS DISTINCT FROM OLD.workflow_name
        OR NEW.workflow_version IS DISTINCT FROM OLD.workflow_version
        OR NEW.workflow_schema_version IS DISTINCT FROM OLD.workflow_schema_version
        OR NEW.workflow_snapshot IS DISTINCT FROM OLD.workflow_snapshot
        OR NEW.parameters IS DISTINCT FROM OLD.parameters
        OR NEW.created_at IS DISTINCT FROM OLD.created_at
    THEN
        RAISE EXCEPTION 'immutable WorkflowRun fields cannot be changed' USING ERRCODE = '23514';
    END IF;
    IF OLD.run_cancellation IS NOT NULL
        AND (
            NEW.cancellation_schema_version IS DISTINCT FROM OLD.cancellation_schema_version
            OR NEW.run_cancellation IS DISTINCT FROM OLD.run_cancellation
        )
    THEN
        RAISE EXCEPTION 'WorkflowRun cancellation cannot be changed' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;
