-- Audit is a durable controller-owned state machine layered over ordinary
-- Projects and WorkflowRuns. The schema keeps one-to-many execution members
-- even while the first Server capability admits batch_size = 1 only.

CREATE TABLE audits (
    audit_id text PRIMARY KEY CHECK (
        audit_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'
    ),
    owner_id text NOT NULL CHECK (
        btrim(owner_id) <> '' AND octet_length(owner_id) <= 256
    ),
    project_id text NOT NULL,
    profile_name text NOT NULL CHECK (btrim(profile_name) <> '' AND octet_length(profile_name) <= 128),
    profile_version text NOT NULL CHECK (btrim(profile_version) <> '' AND octet_length(profile_version) <= 128),
    profile_digest text NOT NULL CHECK (profile_digest ~ '^sha256:[0-9a-f]{64}$'),
    profile_snapshot jsonb NOT NULL CHECK (
        jsonb_typeof(profile_snapshot) = 'object' AND octet_length(profile_snapshot::text) <= 8388608
    ),
    input_selection jsonb NOT NULL CHECK (
        jsonb_typeof(input_selection) = 'object' AND octet_length(input_selection::text) <= 8388608
    ),
    baseline_snapshot jsonb CHECK (
        baseline_snapshot IS NULL OR (
            jsonb_typeof(baseline_snapshot) = 'object' AND octet_length(baseline_snapshot::text) <= 16777216
        )
    ),
    state text NOT NULL DEFAULT 'draft' CHECK (state IN (
        'draft', 'active', 'waiting_review', 'paused', 'finalizing',
        'cancelling', 'completed', 'cancelled', 'failed', 'deleting'
    )),
    revision bigint NOT NULL DEFAULT 1 CHECK (revision > 0),
    current_round_id text,
    dispatch_state text NOT NULL DEFAULT 'open' CHECK (dispatch_state IN ('open', 'closed')),
    hold_state text NOT NULL DEFAULT 'pending' CHECK (hold_state IN ('pending', 'held', 'released')),
    deadline_at timestamptz,
    max_rounds integer NOT NULL CHECK (max_rounds > 0 AND max_rounds <= 32),
    batch_size integer NOT NULL CHECK (batch_size > 0 AND batch_size <= 64),
    max_items_per_round integer NOT NULL CHECK (max_items_per_round > 0 AND max_items_per_round <= 10000),
    max_items_total integer NOT NULL CHECK (max_items_total > 0 AND max_items_total <= 100000),
    max_submitted_runs integer NOT NULL CHECK (max_submitted_runs > 0 AND max_submitted_runs <= 1000000),
    max_item_run_attempts integer NOT NULL CHECK (max_item_run_attempts > 0 AND max_item_run_attempts <= 10),
    max_evidence_bytes bigint NOT NULL CHECK (max_evidence_bytes > 0 AND max_evidence_bytes <= 1073741824),
    reserved_run_count integer NOT NULL DEFAULT 0 CHECK (reserved_run_count >= 0),
    submitted_run_count integer NOT NULL DEFAULT 0 CHECK (submitted_run_count >= 0),
    outstanding_run_count integer NOT NULL DEFAULT 0 CHECK (outstanding_run_count >= 0),
    retained_evidence_bytes bigint NOT NULL DEFAULT 0 CHECK (retained_evidence_bytes >= 0),
    next_event_sequence bigint NOT NULL DEFAULT 1 CHECK (next_event_sequence > 0),
    stop_reason_code text CHECK (
        stop_reason_code IS NULL OR (btrim(stop_reason_code) <> '' AND octet_length(stop_reason_code) <= 128)
    ),
    stop_reason_message text CHECK (
        stop_reason_message IS NULL OR octet_length(stop_reason_message) <= 4096
    ),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    started_at timestamptz,
    finished_at timestamptz,
    CONSTRAINT audits_project_owner_fkey FOREIGN KEY (project_id, owner_id)
        REFERENCES projects(project_id, owner_id) ON DELETE RESTRICT,
    CONSTRAINT audits_project_owner_identity UNIQUE (audit_id, project_id, owner_id),
    CONSTRAINT audits_stop_reason_shape CHECK (
        (stop_reason_code IS NULL) = (stop_reason_message IS NULL)
    ),
    CONSTRAINT audits_counter_bounds CHECK (
        reserved_run_count <= max_submitted_runs
        AND submitted_run_count <= reserved_run_count
        AND outstanding_run_count <= reserved_run_count
        AND retained_evidence_bytes <= max_evidence_bytes
    ),
    CONSTRAINT audits_time_shape CHECK (
        (state = 'draft' AND started_at IS NULL AND finished_at IS NULL)
        OR (state IN ('active', 'waiting_review', 'paused', 'finalizing', 'cancelling')
            AND started_at IS NOT NULL AND deadline_at IS NOT NULL AND finished_at IS NULL)
        OR (state IN ('completed', 'cancelled', 'failed')
            AND started_at IS NOT NULL AND deadline_at IS NOT NULL AND finished_at IS NOT NULL)
        OR state = 'deleting'
    ),
    CONSTRAINT audits_baseline_shape CHECK (
        state IN ('draft', 'deleting') OR baseline_snapshot IS NOT NULL
    )
);

CREATE INDEX audits_owner_created_idx
    ON audits (owner_id, created_at DESC, audit_id DESC);
CREATE INDEX audits_project_state_idx
    ON audits (project_id, state, created_at, audit_id);
CREATE INDEX audits_reconcile_idx
    ON audits (updated_at, audit_id)
    WHERE state IN ('active', 'waiting_review', 'paused', 'finalizing', 'cancelling');

CREATE TABLE audit_rounds (
    round_id text PRIMARY KEY CHECK (round_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'),
    audit_id text NOT NULL REFERENCES audits(audit_id) ON DELETE CASCADE,
    ordinal integer NOT NULL CHECK (ordinal > 0),
    manifest_ref jsonb NOT NULL CHECK (
        jsonb_typeof(manifest_ref) = 'object' AND octet_length(manifest_ref::text) <= 4096
    ),
    manifest_digest text NOT NULL CHECK (manifest_digest ~ '^sha256:[0-9a-f]{64}$'),
    state text NOT NULL CHECK (state IN ('proposed', 'accepted', 'executing', 'assessing', 'closed')),
    expected_item_count integer NOT NULL CHECK (expected_item_count >= 0 AND expected_item_count <= 10000),
    revision bigint NOT NULL DEFAULT 1 CHECK (revision > 0),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (audit_id, ordinal),
    UNIQUE (round_id, audit_id)
);

ALTER TABLE audits
    ADD CONSTRAINT audits_current_round_fkey
    FOREIGN KEY (current_round_id, audit_id)
    REFERENCES audit_rounds(round_id, audit_id)
    ON DELETE SET NULL (current_round_id)
    DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE audit_items (
    item_id text PRIMARY KEY CHECK (item_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'),
    audit_id text NOT NULL,
    round_id text NOT NULL,
    item_key text NOT NULL CHECK (btrim(item_key) <> '' AND octet_length(item_key) <= 256),
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    kind text NOT NULL CHECK (btrim(kind) <> '' AND octet_length(kind) <= 128),
    subject_key text NOT NULL CHECK (btrim(subject_key) <> '' AND octet_length(subject_key) <= 512),
    task_ref jsonb NOT NULL CHECK (
        jsonb_typeof(task_ref) = 'object' AND octet_length(task_ref::text) <= 4096
    ),
    task_digest text NOT NULL CHECK (task_digest ~ '^sha256:[0-9a-f]{64}$'),
    workflow_role text NOT NULL CHECK (btrim(workflow_role) <> '' AND octet_length(workflow_role) <= 128),
    state text NOT NULL CHECK (state IN ('pending', 'awaiting_review', 'ready', 'submitted', 'collecting', 'settled')),
    final_disposition text CHECK (final_disposition IS NULL OR final_disposition IN (
        'accepted-result', 'missing-output', 'invalid-result',
        'execution-failed', 'execution-cancelled', 'excluded', 'not-applicable'
    )),
    accepted_result_ref jsonb CHECK (
        accepted_result_ref IS NULL OR (
            jsonb_typeof(accepted_result_ref) = 'object' AND octet_length(accepted_result_ref::text) <= 4096
        )
    ),
    accepted_result_digest text CHECK (
        accepted_result_digest IS NULL OR accepted_result_digest ~ '^sha256:[0-9a-f]{64}$'
    ),
    last_execution_item_id text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT audit_items_round_fkey FOREIGN KEY (round_id, audit_id)
        REFERENCES audit_rounds(round_id, audit_id) ON DELETE CASCADE,
    CONSTRAINT audit_items_audit_identity UNIQUE (item_id, audit_id, round_id),
    CONSTRAINT audit_items_coverage_identity UNIQUE (
        item_id, audit_id, round_id, item_key, subject_key
    ),
    CONSTRAINT audit_items_round_key UNIQUE (round_id, item_key),
    CONSTRAINT audit_items_round_ordinal UNIQUE (round_id, ordinal),
    CONSTRAINT audit_items_settlement_shape CHECK (
        (state = 'settled') = (final_disposition IS NOT NULL)
    ),
    CONSTRAINT audit_items_result_shape CHECK (
        (accepted_result_ref IS NULL) = (accepted_result_digest IS NULL)
        AND (accepted_result_ref IS NULL OR final_disposition = 'accepted-result')
    )
);

CREATE INDEX audit_items_dispatch_idx
    ON audit_items (audit_id, state, round_id, ordinal, item_id)
    WHERE state <> 'settled';

CREATE TABLE audit_executions (
    execution_id text PRIMARY KEY CHECK (execution_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'),
    audit_id text NOT NULL REFERENCES audits(audit_id) ON DELETE CASCADE,
    round_id text,
    role text NOT NULL CHECK (role IN ('discovery', 'check', 'assessment')),
    role_attempt integer CHECK (role_attempt IS NULL OR role_attempt > 0),
    manifest_ref jsonb NOT NULL CHECK (
        jsonb_typeof(manifest_ref) = 'object' AND octet_length(manifest_ref::text) <= 4096
    ),
    manifest_digest text NOT NULL CHECK (manifest_digest ~ '^sha256:[0-9a-f]{64}$'),
    submission_key text NOT NULL CHECK (
        submission_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
    ),
    request_digest text NOT NULL CHECK (request_digest ~ '^sha256:[0-9a-f]{64}$'),
    -- Kept as durable tombstone provenance after ordinary Run deletion. The
    -- trusted binding method verifies the Run row and V25-005 supplies the
    -- deletion gate; no cascading foreign key may erase this identity.
    run_id text,
    state text NOT NULL DEFAULT 'intent' CHECK (state IN ('intent', 'submitted', 'collecting', 'collected')),
    terminal_outcome text CHECK (
        terminal_outcome IS NULL OR terminal_outcome IN ('succeeded', 'failed', 'cancelled', 'submission-failed')
    ),
    terminal_run_generation text CHECK (
        terminal_run_generation IS NULL OR (btrim(terminal_run_generation) <> '' AND octet_length(terminal_run_generation) <= 256)
    ),
    terminal_run_sequence bigint CHECK (terminal_run_sequence IS NULL OR terminal_run_sequence >= 0),
    terminal_observed_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT audit_executions_round_fkey FOREIGN KEY (round_id, audit_id)
        REFERENCES audit_rounds(round_id, audit_id) ON DELETE CASCADE,
    CONSTRAINT audit_executions_identity UNIQUE (execution_id, audit_id, round_id),
    CONSTRAINT audit_executions_audit_identity UNIQUE (execution_id, audit_id),
    CONSTRAINT audit_executions_submission_key UNIQUE (submission_key),
    CONSTRAINT audit_executions_role_shape CHECK (
        (role = 'check' AND round_id IS NOT NULL AND role_attempt IS NULL)
        OR (role IN ('discovery', 'assessment') AND role_attempt IS NOT NULL)
    ),
    CONSTRAINT audit_executions_run_shape CHECK (
        (state = 'intent' AND run_id IS NULL)
        OR (state = 'submitted' AND run_id IS NOT NULL)
        OR (state IN ('collecting', 'collected') AND (
            run_id IS NOT NULL OR terminal_outcome = 'submission-failed'
        ))
    ),
    CONSTRAINT audit_executions_terminal_shape CHECK (
        (state IN ('intent', 'submitted')
            AND terminal_outcome IS NULL AND terminal_run_generation IS NULL
            AND terminal_run_sequence IS NULL AND terminal_observed_at IS NULL)
        OR (state IN ('collecting', 'collected')
            AND terminal_outcome IS NOT NULL AND terminal_observed_at IS NOT NULL
            AND (
                (run_id IS NOT NULL AND terminal_run_generation IS NOT NULL AND terminal_run_sequence IS NOT NULL)
                OR (run_id IS NULL AND terminal_outcome = 'submission-failed'
                    AND terminal_run_generation IS NULL AND terminal_run_sequence IS NULL)
            ))
    )
);

CREATE UNIQUE INDEX audit_executions_run_unique
    ON audit_executions (run_id) WHERE run_id IS NOT NULL;
CREATE UNIQUE INDEX audit_executions_role_attempt_unique
    ON audit_executions (
        audit_id, COALESCE(round_id, ''), role, COALESCE(role_attempt, 0)
    ) WHERE role IN ('discovery', 'assessment');
CREATE INDEX audit_executions_reconcile_idx
    ON audit_executions (audit_id, state, created_at, execution_id)
    WHERE state IN ('intent', 'submitted', 'collecting');

CREATE TABLE audit_execution_items (
    execution_item_id text PRIMARY KEY CHECK (
        execution_item_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'
    ),
    execution_id text NOT NULL,
    audit_id text NOT NULL,
    round_id text NOT NULL,
    item_id text NOT NULL,
    batch_ordinal integer NOT NULL CHECK (batch_ordinal >= 0),
    item_attempt integer NOT NULL CHECK (item_attempt > 0),
    task_ref jsonb NOT NULL CHECK (
        jsonb_typeof(task_ref) = 'object' AND octet_length(task_ref::text) <= 4096
    ),
    task_digest text NOT NULL CHECK (task_digest ~ '^sha256:[0-9a-f]{64}$'),
    input_refs jsonb NOT NULL CHECK (
        jsonb_typeof(input_refs) = 'array' AND octet_length(input_refs::text) <= 1048576
    ),
    state text NOT NULL DEFAULT 'submitted' CHECK (state IN ('submitted', 'collecting', 'settled')),
    collection_disposition text CHECK (collection_disposition IS NULL OR collection_disposition IN (
        'accepted-result', 'missing-output', 'invalid-result',
        'execution-failed', 'execution-cancelled'
    )),
    result_ref jsonb CHECK (
        result_ref IS NULL OR (jsonb_typeof(result_ref) = 'object' AND octet_length(result_ref::text) <= 4096)
    ),
    result_digest text CHECK (
        result_digest IS NULL OR result_digest ~ '^sha256:[0-9a-f]{64}$'
    ),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    collected_at timestamptz,
    CONSTRAINT audit_execution_items_execution_fkey
        FOREIGN KEY (execution_id, audit_id, round_id)
        REFERENCES audit_executions(execution_id, audit_id, round_id) ON DELETE CASCADE,
    CONSTRAINT audit_execution_items_item_fkey
        FOREIGN KEY (item_id, audit_id, round_id)
        REFERENCES audit_items(item_id, audit_id, round_id) ON DELETE CASCADE,
    CONSTRAINT audit_execution_items_execution_ordinal UNIQUE (execution_id, batch_ordinal),
    CONSTRAINT audit_execution_items_execution_member UNIQUE (execution_id, item_id),
    CONSTRAINT audit_execution_items_item_identity UNIQUE (execution_item_id, item_id),
    CONSTRAINT audit_execution_items_item_attempt UNIQUE (item_id, item_attempt),
    CONSTRAINT audit_execution_items_collection_shape CHECK (
        (state IN ('submitted', 'collecting')
            AND collection_disposition IS NULL AND result_ref IS NULL
            AND result_digest IS NULL AND collected_at IS NULL)
        OR (state = 'settled'
            AND collection_disposition IS NOT NULL AND collected_at IS NOT NULL
            AND (result_ref IS NULL) = (result_digest IS NULL)
            AND (result_ref IS NULL OR collection_disposition = 'accepted-result'))
    )
);

ALTER TABLE audit_items
    ADD CONSTRAINT audit_items_last_execution_item_fkey
    FOREIGN KEY (last_execution_item_id, item_id)
    REFERENCES audit_execution_items(execution_item_id, item_id)
    DEFERRABLE INITIALLY DEFERRED;

CREATE INDEX audit_execution_items_audit_state_idx
    ON audit_execution_items (audit_id, state, execution_id, batch_ordinal);

CREATE TABLE audit_collection_receipts (
    receipt_id text PRIMARY KEY CHECK (receipt_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'),
    audit_id text NOT NULL REFERENCES audits(audit_id) ON DELETE CASCADE,
    execution_id text NOT NULL,
    run_id text,
    terminal_outcome text NOT NULL CHECK (terminal_outcome IN ('succeeded', 'failed', 'cancelled', 'submission-failed')),
    terminal_run_generation text,
    terminal_run_sequence bigint,
    disposition text NOT NULL CHECK (disposition IN (
        'accepted-result', 'missing-output', 'invalid-result',
        'execution-failed', 'execution-cancelled'
    )),
    source_output_ref jsonb CHECK (
        source_output_ref IS NULL OR (
            jsonb_typeof(source_output_ref) = 'object' AND octet_length(source_output_ref::text) <= 4096
        )
    ),
    source_output_digest text CHECK (
        source_output_digest IS NULL OR source_output_digest ~ '^sha256:[0-9a-f]{64}$'
    ),
    retained_refs jsonb NOT NULL DEFAULT '[]'::jsonb CHECK (
        jsonb_typeof(retained_refs) = 'array' AND octet_length(retained_refs::text) <= 8388608
    ),
    error_code text CHECK (
        error_code IS NULL OR (btrim(error_code) <> '' AND octet_length(error_code) <= 128)
    ),
    request_digest text NOT NULL CHECK (request_digest ~ '^sha256:[0-9a-f]{64}$'),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (execution_id),
    CONSTRAINT audit_collection_receipts_execution_fkey
        FOREIGN KEY (execution_id, audit_id)
        REFERENCES audit_executions(execution_id, audit_id) ON DELETE CASCADE,
    CONSTRAINT audit_collection_receipts_observation_shape CHECK (
        (run_id IS NOT NULL AND terminal_run_generation IS NOT NULL
            AND btrim(terminal_run_generation) <> '' AND terminal_run_sequence >= 0)
        OR (run_id IS NULL AND terminal_outcome = 'submission-failed'
            AND terminal_run_generation IS NULL AND terminal_run_sequence IS NULL)
    ),
    CONSTRAINT audit_collection_receipts_source_shape CHECK (
        (source_output_ref IS NULL) = (source_output_digest IS NULL)
        AND ((disposition IN ('accepted-result', 'invalid-result')) = (source_output_ref IS NOT NULL))
    )
);

CREATE TABLE audit_artifact_links (
    audit_id text NOT NULL REFERENCES audits(audit_id) ON DELETE CASCADE,
    logical_key text NOT NULL CHECK (btrim(logical_key) <> '' AND octet_length(logical_key) <= 512),
    artifact_ref jsonb NOT NULL CHECK (
        jsonb_typeof(artifact_ref) = 'object' AND octet_length(artifact_ref::text) <= 4096
    ),
    artifact_digest text NOT NULL CHECK (artifact_digest ~ '^sha256:[0-9a-f]{64}$'),
    media_type text NOT NULL CHECK (btrim(media_type) <> '' AND octet_length(media_type) <= 256),
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    source_provenance jsonb NOT NULL CHECK (
        jsonb_typeof(source_provenance) = 'object' AND octet_length(source_provenance::text) <= 1048576
    ),
    display_ref text NOT NULL DEFAULT '' CHECK (octet_length(display_ref) <= 1024),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (audit_id, logical_key)
);

CREATE TABLE audit_coverage_rows (
    audit_id text NOT NULL,
    round_id text NOT NULL,
    item_id text NOT NULL,
    item_key text NOT NULL CHECK (btrim(item_key) <> ''),
    subject_key text NOT NULL CHECK (btrim(subject_key) <> ''),
    status text NOT NULL CHECK (status IN (
        'not-tested', 'inconclusive', 'satisfied', 'violated',
        'not-applicable', 'blocked', 'excluded'
    )),
    requested jsonb NOT NULL CHECK (jsonb_typeof(requested) = 'array' AND octet_length(requested::text) <= 1048576),
    completed jsonb NOT NULL CHECK (jsonb_typeof(completed) = 'array' AND octet_length(completed::text) <= 1048576),
    gaps jsonb NOT NULL CHECK (jsonb_typeof(gaps) = 'array' AND octet_length(gaps::text) <= 1048576),
    rationale text NOT NULL DEFAULT '' CHECK (octet_length(rationale) <= 4096),
    result_ref jsonb,
    result_digest text,
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (item_id),
    UNIQUE (round_id, item_key),
    CONSTRAINT audit_coverage_item_fkey FOREIGN KEY (
        item_id, audit_id, round_id, item_key, subject_key
    ) REFERENCES audit_items(
        item_id, audit_id, round_id, item_key, subject_key
    ) ON DELETE CASCADE,
    CONSTRAINT audit_coverage_result_shape CHECK (
        (result_ref IS NULL) = (result_digest IS NULL)
        AND (result_ref IS NULL OR (
            jsonb_typeof(result_ref) = 'object'
            AND octet_length(result_ref::text) <= 4096
            AND result_digest ~ '^sha256:[0-9a-f]{64}$'
        ))
    )
);

CREATE TABLE audit_events (
    audit_id text NOT NULL REFERENCES audits(audit_id) ON DELETE CASCADE,
    sequence_number bigint NOT NULL CHECK (sequence_number > 0),
    kind text NOT NULL CHECK (
        kind ~ '^[a-z][a-z0-9_.-]{0,127}$'
    ),
    entity_id text NOT NULL CHECK (btrim(entity_id) <> '' AND octet_length(entity_id) <= 256),
    entity_revision bigint CHECK (entity_revision IS NULL OR entity_revision > 0),
    summary jsonb NOT NULL CHECK (
        jsonb_typeof(summary) = 'object' AND octet_length(summary::text) <= 65536
    ),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (audit_id, sequence_number)
);

CREATE TABLE audit_idempotency (
    owner_id text NOT NULL CHECK (btrim(owner_id) <> '' AND octet_length(owner_id) <= 256),
    operation text NOT NULL CHECK (operation ~ '^[a-z][a-z0-9_.-]{0,127}$'),
    idempotency_key text NOT NULL CHECK (
        idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
    ),
    request_digest text NOT NULL CHECK (request_digest ~ '^sha256:[0-9a-f]{64}$'),
    audit_id text NOT NULL REFERENCES audits(audit_id) ON DELETE CASCADE,
    resource_id text NOT NULL CHECK (btrim(resource_id) <> '' AND octet_length(resource_id) <= 256),
    response_snapshot jsonb NOT NULL CHECK (
        jsonb_typeof(response_snapshot) = 'object' AND octet_length(response_snapshot::text) <= 1048576
    ),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (owner_id, operation, idempotency_key)
);

CREATE TABLE audit_controller_claims (
    audit_id text PRIMARY KEY REFERENCES audits(audit_id) ON DELETE CASCADE,
    epoch bigint NOT NULL DEFAULT 0 CHECK (epoch >= 0),
    holder_id text CHECK (
        holder_id IS NULL OR (btrim(holder_id) <> '' AND octet_length(holder_id) <= 256)
    ),
    claimed_at timestamptz,
    expires_at timestamptz,
    CONSTRAINT audit_controller_claim_shape CHECK (
        (holder_id IS NULL AND claimed_at IS NULL AND expires_at IS NULL)
        OR (holder_id IS NOT NULL AND claimed_at IS NOT NULL AND expires_at > claimed_at)
    )
);

CREATE INDEX audit_controller_claim_expiry_idx
    ON audit_controller_claims (expires_at, audit_id);

-- Audit admission takes the same Project row lock as Run and ProjectScope
-- admission and additionally rejects Evaluation containers.
CREATE OR REPLACE FUNCTION contractor_require_active_audit_project(
    checked_project_id text,
    checked_owner_id text
)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    current_kind text;
    current_lifecycle text;
BEGIN
    SELECT kind, lifecycle_state
      INTO current_kind, current_lifecycle
      FROM projects
     WHERE project_id = checked_project_id AND owner_id = checked_owner_id
     FOR SHARE;

    IF NOT FOUND OR current_kind <> 'project' THEN
        RAISE EXCEPTION 'Audit Project is missing' USING ERRCODE = '23503';
    END IF;
    IF current_lifecycle = 'deleting' THEN
        RAISE EXCEPTION 'Project is deleting' USING ERRCODE = '55000';
    END IF;
END;
$$;

CREATE OR REPLACE FUNCTION contractor_guard_audit_admission()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    PERFORM contractor_require_active_audit_project(NEW.project_id, NEW.owner_id);
    RETURN NEW;
END;
$$;

CREATE TRIGGER audits_project_admission
BEFORE INSERT ON audits
FOR EACH ROW EXECUTE FUNCTION contractor_guard_audit_admission();

CREATE OR REPLACE FUNCTION contractor_guard_audit_child_admission()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    checked_project_id text;
    checked_owner_id text;
BEGIN
    SELECT project_id, owner_id INTO checked_project_id, checked_owner_id
      FROM audits WHERE audit_id = NEW.audit_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Audit is missing' USING ERRCODE = '23503';
    END IF;
    PERFORM contractor_require_active_audit_project(checked_project_id, checked_owner_id);
    RETURN NEW;
END;
$$;

CREATE TRIGGER audit_rounds_project_admission
BEFORE INSERT ON audit_rounds
FOR EACH ROW EXECUTE FUNCTION contractor_guard_audit_child_admission();

CREATE TRIGGER audit_executions_project_admission
BEFORE INSERT ON audit_executions
FOR EACH ROW EXECUTE FUNCTION contractor_guard_audit_child_admission();

CREATE OR REPLACE FUNCTION contractor_protect_audit_event()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'Audit events are immutable' USING ERRCODE = '23514';
END;
$$;

CREATE TRIGGER audit_events_protect_immutable
BEFORE UPDATE ON audit_events
FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_event();

CREATE OR REPLACE FUNCTION contractor_protect_audit_receipt()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.run_id IS NULL AND OLD.run_id IS NOT NULL
       AND ROW(NEW.receipt_id, NEW.audit_id, NEW.execution_id,
               NEW.terminal_outcome, NEW.terminal_run_generation,
               NEW.terminal_run_sequence, NEW.disposition,
               NEW.source_output_ref, NEW.source_output_digest,
               NEW.retained_refs, NEW.error_code, NEW.request_digest,
               NEW.created_at)
           IS NOT DISTINCT FROM
           ROW(OLD.receipt_id, OLD.audit_id, OLD.execution_id,
               OLD.terminal_outcome, OLD.terminal_run_generation,
               OLD.terminal_run_sequence, OLD.disposition,
               OLD.source_output_ref, OLD.source_output_digest,
               OLD.retained_refs, OLD.error_code, OLD.request_digest,
               OLD.created_at)
    THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'Audit collection receipts are immutable' USING ERRCODE = '23514';
END;
$$;

CREATE TRIGGER audit_collection_receipts_protect_immutable
BEFORE UPDATE ON audit_collection_receipts
FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_receipt();
