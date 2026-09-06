ALTER TABLE artifact_pins
    DROP CONSTRAINT artifact_pins_pin_kind_check,
    ADD CONSTRAINT artifact_pins_pin_kind_check CHECK (pin_kind IN (
        'run_input', 'stage_context', 'stage_result', 'run_output',
        'finding_proposal', 'finding_evidence'
    ));

CREATE TABLE finding_proposal_receipts (
    receipt_id text PRIMARY KEY CHECK (
        receipt_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'
    ),
    proposal_id text NOT NULL UNIQUE CHECK (
        proposal_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'
    ),
    allocation_id text NOT NULL CHECK (btrim(allocation_id) <> '' AND octet_length(allocation_id) <= 256),
    runtime_agent_id text NOT NULL CHECK (btrim(runtime_agent_id) <> '' AND octet_length(runtime_agent_id) <= 256),
    runtime_instance_id text NOT NULL CHECK (btrim(runtime_instance_id) <> '' AND octet_length(runtime_instance_id) <= 256),
    stage_execution_id text NOT NULL CHECK (btrim(stage_execution_id) <> '' AND octet_length(stage_execution_id) <= 256),
    logical_agent_name text NOT NULL CHECK (btrim(logical_agent_name) <> '' AND octet_length(logical_agent_name) <= 128),
    invocation_id text NOT NULL CHECK (btrim(invocation_id) <> '' AND octet_length(invocation_id) <= 128),
    submission_id text NOT NULL CHECK (btrim(submission_id) <> '' AND octet_length(submission_id) <= 128),
    client_key text NOT NULL CHECK (client_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$'),
    request_digest text NOT NULL CHECK (request_digest ~ '^sha256:[0-9a-f]{64}$'),
    run_id text NOT NULL CHECK (btrim(run_id) <> '' AND octet_length(run_id) <= 256),
    owner_id text NOT NULL CHECK (btrim(owner_id) <> '' AND octet_length(owner_id) <= 256),
    project_id text,
    audit_execution_id text,
    audit_id text,
    audit_role text CHECK (audit_role IS NULL OR audit_role IN ('discovery', 'check', 'assessment')),
    workflow_name text NOT NULL CHECK (btrim(workflow_name) <> '' AND octet_length(workflow_name) <= 128),
    workflow_version text NOT NULL CHECK (btrim(workflow_version) <> '' AND octet_length(workflow_version) <= 128),
    workflow_schema_version text NOT NULL CHECK (btrim(workflow_schema_version) <> '' AND octet_length(workflow_schema_version) <= 128),
    workflow_configuration_ref jsonb NOT NULL CHECK (
        jsonb_typeof(workflow_configuration_ref) = 'object'
        AND octet_length(workflow_configuration_ref::text) <= 4096
    ),
    workflow_closure_digest text NOT NULL CHECK (workflow_closure_digest ~ '^sha256:[0-9a-f]{64}$'),
    proposal_ref jsonb NOT NULL CHECK (jsonb_typeof(proposal_ref) = 'object' AND octet_length(proposal_ref::text) <= 4096),
    proposal_digest text NOT NULL CHECK (proposal_digest ~ '^sha256:[0-9a-f]{64}$'),
    proposal_media_type text NOT NULL CHECK (proposal_media_type = 'application/json'),
    proposal_size_bytes bigint NOT NULL CHECK (proposal_size_bytes >= 0 AND proposal_size_bytes <= 8388608),
    evidence jsonb NOT NULL CHECK (jsonb_typeof(evidence) = 'array' AND octet_length(evidence::text) <= 1048576),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (allocation_id, invocation_id, submission_id),
    CONSTRAINT finding_proposal_receipts_audit_shape CHECK (
        (audit_execution_id IS NULL AND audit_id IS NULL AND audit_role IS NULL)
        OR
        (audit_execution_id IS NOT NULL AND audit_id IS NOT NULL AND audit_role IS NOT NULL)
    )
);

CREATE INDEX finding_proposal_receipts_run_idx
    ON finding_proposal_receipts (run_id, created_at, receipt_id);
CREATE INDEX finding_proposal_receipts_audit_idx
    ON finding_proposal_receipts (audit_id, created_at, receipt_id)
    WHERE audit_id IS NOT NULL;

CREATE OR REPLACE FUNCTION contractor_protect_finding_proposal_receipt()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'finding proposal receipts are immutable' USING ERRCODE = '23514';
END;
$$;

CREATE TRIGGER finding_proposal_receipts_protect_immutable
BEFORE UPDATE OR DELETE ON finding_proposal_receipts
FOR EACH ROW EXECUTE FUNCTION contractor_protect_finding_proposal_receipt();

CREATE TABLE finding_proposal_retention (
    receipt_id text PRIMARY KEY REFERENCES finding_proposal_receipts(receipt_id) ON DELETE CASCADE,
    state text NOT NULL DEFAULT 'source-held' CHECK (state IN ('source-held', 'audit-held', 'discarded')),
    source_run_deleted_at timestamptz,
    discarded_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT finding_proposal_retention_shape CHECK (
        (state = 'source-held' AND source_run_deleted_at IS NULL AND discarded_at IS NULL)
        OR (state = 'audit-held' AND discarded_at IS NULL)
        OR (state = 'discarded' AND source_run_deleted_at IS NOT NULL AND discarded_at IS NOT NULL)
    )
);

CREATE TABLE finding_proposal_audit_holds (
    receipt_id text NOT NULL REFERENCES finding_proposal_receipts(receipt_id) ON DELETE CASCADE,
    audit_id text NOT NULL REFERENCES audits(audit_id) ON DELETE CASCADE,
    project_id text NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    proposal_ref jsonb NOT NULL CHECK (jsonb_typeof(proposal_ref) = 'object' AND octet_length(proposal_ref::text) <= 4096),
    evidence jsonb NOT NULL CHECK (jsonb_typeof(evidence) = 'array' AND octet_length(evidence::text) <= 1048576),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (receipt_id, audit_id)
);

CREATE INDEX finding_proposal_audit_holds_audit_idx
    ON finding_proposal_audit_holds (audit_id, created_at, receipt_id);

CREATE TRIGGER finding_proposal_audit_holds_protect_update
BEFORE UPDATE ON finding_proposal_audit_holds
FOR EACH ROW EXECUTE FUNCTION contractor_protect_finding_proposal_receipt();

CREATE OR REPLACE FUNCTION contractor_reconcile_finding_retention()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    target_receipt text;
BEGIN
    IF TG_OP = 'DELETE' THEN
        target_receipt := OLD.receipt_id;
    ELSE
        target_receipt := NEW.receipt_id;
    END IF;
    IF EXISTS (
        SELECT 1 FROM finding_proposal_audit_holds WHERE receipt_id = target_receipt
    ) THEN
        UPDATE finding_proposal_retention
           SET state = 'audit-held', discarded_at = NULL, updated_at = clock_timestamp()
         WHERE receipt_id = target_receipt;
    ELSE
        UPDATE finding_proposal_retention
           SET state = CASE WHEN source_run_deleted_at IS NULL THEN 'source-held' ELSE 'discarded' END,
               discarded_at = CASE WHEN source_run_deleted_at IS NULL THEN NULL ELSE COALESCE(discarded_at, clock_timestamp()) END,
               updated_at = clock_timestamp()
         WHERE receipt_id = target_receipt;
    END IF;
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER finding_proposal_audit_holds_reconcile
AFTER INSERT OR DELETE ON finding_proposal_audit_holds
FOR EACH ROW EXECUTE FUNCTION contractor_reconcile_finding_retention();
