ALTER TABLE audits
    ADD COLUMN deletion_requested_at timestamptz;

UPDATE audits
SET deletion_requested_at = COALESCE(updated_at, clock_timestamp())
WHERE state = 'deleting';

ALTER TABLE audits
    ADD CONSTRAINT audits_deletion_intent_shape CHECK (
        (deletion_requested_at IS NULL AND state <> 'deleting')
        OR
        (deletion_requested_at IS NOT NULL AND state IN ('cancelling', 'deleting'))
    );

ALTER TABLE audit_items
    ADD COLUMN origin jsonb;

-- Historical task packages retain the exact source internally, but extracting
-- their archive member in SQL would be unsafe. Mark that bounded legacy gap
-- explicitly; every post-migration materialization writes complete origin.
UPDATE audit_items
SET origin = jsonb_build_object(
    'schema', 'contractor.audit.item-origin.v1',
    'entryKey', item_key,
    'provenanceIncomplete', true
);

ALTER TABLE audit_items
    ALTER COLUMN origin SET NOT NULL,
    ADD CONSTRAINT audit_items_origin_shape CHECK (
        jsonb_typeof(origin) = 'object'
        AND origin->>'schema' = 'contractor.audit.item-origin.v1'
        AND origin->>'entryKey' = item_key
        AND octet_length(origin::text) <= 16384
        AND (
            origin->'provenanceIncomplete' = 'true'::jsonb
            OR (
                jsonb_typeof(origin->'sourceRef') = 'object'
                AND origin->>'sourceContentDigest' ~ '^sha256:[0-9a-f]{64}$'
                AND origin->>'canonicalInventoryDigest' ~ '^sha256:[0-9a-f]{64}$'
                AND btrim(origin->>'sourceMediaType') <> ''
                AND octet_length(origin->>'sourceMediaType') <= 256
            )
        )
    );

ALTER TABLE audit_execution_items
    DROP CONSTRAINT audit_execution_items_collection_disposition_check,
    ADD CONSTRAINT audit_execution_items_collection_disposition_check CHECK (
        collection_disposition IS NULL OR collection_disposition IN (
            'accepted-result', 'missing-output', 'invalid-result',
            'execution-failed', 'execution-cancelled', 'collection-contract-invalid'
        )
    );

ALTER TABLE audit_collection_receipts
    DROP CONSTRAINT audit_collection_receipts_disposition_check,
    ADD CONSTRAINT audit_collection_receipts_disposition_check CHECK (disposition IN (
        'accepted-result', 'missing-output', 'invalid-result',
        'execution-failed', 'execution-cancelled', 'collection-contract-invalid'
    ));

ALTER TABLE audit_executions
    ADD COLUMN run_provenance jsonb,
    ADD COLUMN run_deleted_at timestamptz;

UPDATE audit_executions AS execution
SET run_provenance = jsonb_build_object(
        'schema', 'contractor.audit.run-provenance.v1',
        'runId', run.run_id,
        'workflow', jsonb_build_object(
            'name', run.workflow_name,
            'version', run.workflow_version,
            'schemaVersion', run.workflow_schema_version,
            'configurationRef', jsonb_build_object(
                'name', run.workflow_name,
                'version', run.workflow_version
            ),
            'closureDigest', 'sha256:' || encode(
                pg_catalog.sha256(convert_to(run.workflow_snapshot::text, 'UTF8')),
                'hex'
            )
        )
    )
FROM workflow_runs AS run
WHERE execution.run_id = run.run_id;

-- A pre-feature database may contain a collected execution whose Run was
-- already removed. Preserve its known identity and mark the historical gap;
-- all executions bound after this migration receive complete provenance in
-- BindRun before the association commits.
UPDATE audit_executions AS execution
SET run_provenance = jsonb_build_object(
        'schema', 'contractor.audit.run-provenance.v1',
        'runId', execution.run_id,
        'provenanceIncomplete', true
    ),
    run_deleted_at = CASE WHEN execution.state = 'collected'
        THEN COALESCE(execution.terminal_observed_at, execution.updated_at)
        ELSE NULL
    END
WHERE execution.run_id IS NOT NULL
  AND execution.run_provenance IS NULL
  AND execution.state = 'collected'
  AND NOT EXISTS (
      SELECT 1 FROM workflow_runs AS run WHERE run.run_id = execution.run_id
  );

ALTER TABLE audit_executions
    ADD CONSTRAINT audit_executions_run_provenance_shape CHECK (
        (run_id IS NULL AND run_provenance IS NULL AND run_deleted_at IS NULL)
        OR
        (run_id IS NOT NULL AND run_provenance IS NOT NULL
            AND jsonb_typeof(run_provenance) = 'object'
            AND run_provenance->>'schema' = 'contractor.audit.run-provenance.v1'
            AND run_provenance->>'runId' = run_id
            AND octet_length(run_provenance::text) <= 16384)
    ),
    ADD CONSTRAINT audit_executions_run_deletion_shape CHECK (
        run_deleted_at IS NULL OR (run_id IS NOT NULL AND state = 'collected')
    );

CREATE INDEX audit_executions_audit_run_cleanup_idx
    ON audit_executions (audit_id, run_deleted_at, created_at, execution_id)
    WHERE run_id IS NOT NULL;

-- The deferred association constraint distinguishes a live relation from its
-- durable Audit-owned tombstone. Run deletion and run_deleted_at are committed
-- together, so neither intermediate statement is authoritative on its own.
CREATE OR REPLACE FUNCTION contractor_require_matching_audit_run()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.run_id IS NOT NULL AND (
        (NEW.run_deleted_at IS NULL AND NOT EXISTS (
            SELECT 1
              FROM workflow_runs AS run
             WHERE run.run_id = NEW.run_id
               AND run.audit_execution_id = NEW.execution_id
               AND run.audit_submission_key = NEW.submission_key
               AND run.publication_mode = 'audit-managed'
        ))
        OR
        (NEW.run_deleted_at IS NOT NULL AND EXISTS (
            SELECT 1 FROM workflow_runs AS run WHERE run.run_id = NEW.run_id
        ))
    ) THEN
        RAISE EXCEPTION 'AuditExecution Run association is not authoritative'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE INDEX audits_project_deletion_intent_idx
    ON audits (project_id, deletion_requested_at, created_at, audit_id);

DROP INDEX IF EXISTS audits_reconcile_idx;
CREATE INDEX audits_reconcile_idx
    ON audits (updated_at, audit_id)
    WHERE state IN ('active', 'waiting_review', 'paused', 'finalizing', 'cancelling', 'deleting');

-- Project deletion fences ordinary writes immediately, but trusted Audit
-- cancellation may still need to retain an already-produced exact result.
-- The public Artifact API independently reserves every audit-* namespace.
DROP TRIGGER artifact_scopes_project_admission ON artifact_scopes;
DROP TRIGGER artifact_bindings_project_admission ON artifact_bindings;

CREATE OR REPLACE FUNCTION contractor_guard_project_artifact_scope_admission()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    current_lifecycle text;
BEGIN
    IF NEW.scope_kind <> 'project' THEN
        RETURN NEW;
    END IF;
    SELECT lifecycle_state INTO current_lifecycle
      FROM projects WHERE project_id = NEW.scope_id
      FOR SHARE;
    IF current_lifecycle = 'deleting' AND NOT EXISTS (
        SELECT 1 FROM audits
         WHERE project_id = NEW.scope_id
           AND deletion_requested_at IS NOT NULL
           AND state IN ('cancelling', 'deleting')
    ) THEN
        RAISE EXCEPTION 'Project is deleting' USING ERRCODE = '55000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION contractor_guard_project_artifact_binding_admission()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    current_lifecycle text;
BEGIN
    IF NEW.scope_kind <> 'project' THEN
        RETURN NEW;
    END IF;
    SELECT lifecycle_state INTO current_lifecycle
      FROM projects WHERE project_id = NEW.scope_id
      FOR SHARE;
    IF current_lifecycle = 'deleting' AND NOT (
        EXISTS (
            SELECT 1 FROM audits
             WHERE project_id = NEW.scope_id
               AND deletion_requested_at IS NOT NULL
               AND state IN ('cancelling', 'deleting')
               AND NEW.namespace = 'audit-' || encode(
                   pg_catalog.sha256(
                       convert_to('contractor.audit.identity.v1', 'UTF8') || decode('00', 'hex') ||
                       convert_to('audit', 'UTF8') || decode('00', 'hex') ||
                       convert_to(audit_id, 'UTF8')
                   ),
                   'hex'
               )
        )
    ) THEN
        RAISE EXCEPTION 'Project is deleting' USING ERRCODE = '55000';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER artifact_scopes_project_admission
BEFORE INSERT ON artifact_scopes
FOR EACH ROW EXECUTE FUNCTION contractor_guard_project_artifact_scope_admission();

CREATE TRIGGER artifact_bindings_project_admission
BEFORE INSERT OR UPDATE ON artifact_bindings
FOR EACH ROW EXECUTE FUNCTION contractor_guard_project_artifact_binding_admission();

CREATE OR REPLACE FUNCTION contractor_lifecycle_purge_enabled()
RETURNS boolean
LANGUAGE sql
STABLE
AS $$
    SELECT COALESCE(current_setting('contractor.lifecycle_purge', true), '')
        IN ('run', 'project', 'audit')
$$;
