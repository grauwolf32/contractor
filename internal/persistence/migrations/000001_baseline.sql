-- Contractor schema baseline.
--
-- This file creates the whole schema of a new database. It replaces
-- migrations 000001-000102, which were folded on 2026-10-11 while the
-- project had no deployments to upgrade (V367-001). A database migrated by
-- those files has a different ledger and must be recreated: the migrator
-- reports drift instead of upgrading it.
--
-- Tables are grouped by context, each followed by its keys, indexes and
-- triggers. Functions come first because CHECK constraints and triggers
-- call them, foreign keys come last, and the rows every database starts
-- with close the file. Later schema changes go into new numbered files.

-- Function bodies reference functions and tables created later in this file.
SET LOCAL check_function_bodies = false;

-- Functions

CREATE FUNCTION artifact_blob_backend_guard() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    -- Reads of already selected settings need no global serialization.
    IF NOT EXISTS (SELECT 1 FROM artifact_blob_settings WHERE backend = NEW.backend) THEN
        UPDATE artifact_blob_settings SET backend = NEW.backend
        WHERE singleton AND backend IS NULL;
        IF NOT EXISTS (SELECT 1 FROM artifact_blob_settings WHERE backend = NEW.backend) THEN
            RAISE EXCEPTION 'artifact blob backend mismatch' USING ERRCODE = '23514';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_admit_audit_finding() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    admitted_finding_id text;
    event_sequence bigint;
    finding_revision bigint;
BEGIN
    -- Findings belong to an Audit; the same immutable receipt may be imported
    -- into several compatible Audits. Preserve all previously admitted IDs.
    admitted_finding_id := 'finding-' || encode(sha256(convert_to(
        jsonb_build_array(NEW.audit_id, NEW.receipt_id)::text, 'UTF8'
    )), 'hex');
    INSERT INTO audit_findings (
        finding_id, audit_id, first_receipt_id, first_proposal_ref
    ) VALUES (
        admitted_finding_id, NEW.audit_id, NEW.receipt_id, NEW.proposal_ref
    )
    ON CONFLICT (audit_id, first_receipt_id) DO NOTHING
    RETURNING revision INTO finding_revision;

    IF finding_revision IS NULL THEN
        RETURN NEW;
    END IF;

    INSERT INTO audit_finding_contributions (
        finding_id, audit_id, receipt_id, relation, proposal_ref
    ) VALUES (
        admitted_finding_id, NEW.audit_id, NEW.receipt_id, 'first', NEW.proposal_ref
    );

    UPDATE audits
       SET revision = revision + 1,
           next_event_sequence = next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
     WHERE audit_id = NEW.audit_id
    RETURNING next_event_sequence - 1 INTO event_sequence;

    INSERT INTO audit_events (
        audit_id, sequence_number, kind, entity_id, entity_revision, summary
    ) VALUES (
        NEW.audit_id, event_sequence, 'finding.proposed', admitted_finding_id,
        finding_revision, jsonb_build_object('state', 'proposed')
    );
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_append_workflow_run_lifecycle_event(target_run_id text, resource_kind text, target_stage_execution_id text, resource_state text) RETURNS bigint
    LANGUAGE plpgsql
    AS $$
DECLARE
    allocated_sequence bigint;
    lifecycle_data jsonb;
BEGIN
    IF resource_kind NOT IN ('run', 'stageExecution')
        OR btrim(resource_state) = ''
        OR (resource_kind = 'run' AND target_stage_execution_id IS NOT NULL)
        OR (
            resource_kind = 'stageExecution'
            AND (target_stage_execution_id IS NULL OR btrim(target_stage_execution_id) = '')
        )
    THEN
        RAISE EXCEPTION 'invalid WorkflowRun lifecycle event' USING ERRCODE = '23514';
    END IF;

    UPDATE workflow_runs
    SET next_run_event_sequence = next_run_event_sequence + 1
    WHERE run_id = target_run_id
    RETURNING next_run_event_sequence - 1 INTO allocated_sequence;

    IF allocated_sequence IS NULL THEN
        RAISE EXCEPTION 'WorkflowRun lifecycle owner is missing' USING ERRCODE = '23503';
    END IF;

    lifecycle_data := jsonb_build_object(
        'runId', target_run_id,
        'resource', resource_kind,
        'state', resource_state
    );
    IF target_stage_execution_id IS NOT NULL THEN
        lifecycle_data := lifecycle_data || jsonb_build_object(
            'stageExecutionId', target_stage_execution_id
        );
    END IF;

    INSERT INTO workflow_run_events (
        run_id, sequence_number, event_id, event_schema_version, kind, data
    ) VALUES (
        target_run_id,
        allocated_sequence,
        'lifecycle-' || md5(
            random()::text || clock_timestamp()::text || target_run_id || allocated_sequence::text
        ),
        'contractor/v1alpha1',
        'lifecycle.changed',
        lifecycle_data
    );
    RETURN allocated_sequence;
END;
$$;

CREATE FUNCTION contractor_capture_stage_execution_lifecycle_event() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        PERFORM contractor_append_workflow_run_lifecycle_event(
            NEW.run_id, 'stageExecution', NEW.stage_execution_id, NEW.state
        );
        RETURN NEW;
    END IF;

    IF OLD.state IS DISTINCT FROM NEW.state THEN
        -- StartPlanner allocates this lifecycle event and planner.started in one
        -- explicit two-sequence CTE to avoid updating the owning Run twice in
        -- the same SQL statement.
        IF OLD.state = 'preparing'
            AND NEW.state = 'running'
            AND OLD.planner_session_id IS NULL
            AND NEW.planner_session_id IS NOT NULL
        THEN
            RETURN NEW;
        END IF;
        PERFORM contractor_append_workflow_run_lifecycle_event(
            NEW.run_id, 'stageExecution', NEW.stage_execution_id, NEW.state
        );
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_capture_workflow_run_lifecycle_event() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'INSERT' OR OLD.state IS DISTINCT FROM NEW.state THEN
        PERFORM contractor_append_workflow_run_lifecycle_event(
            NEW.run_id, 'run', NULL, NEW.state
        );
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_check_llm_credential_exclusive_state() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    PERFORM 1 FROM credential_operations
    WHERE credential_id = NEW.credential_id AND operation_kind = 'create'
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'credential ID is not reserved' USING ERRCODE = '23503';
    END IF;
    IF EXISTS (
        SELECT 1 FROM credential_operations
        WHERE credential_id = NEW.credential_id AND operation_kind = 'delete' AND phase = 'completed'
    ) THEN
        RAISE EXCEPTION 'credential ID has a tombstone' USING ERRCODE = '23505';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_eval_audit_child_changed() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    r record;
BEGIN
    FOR r IN
        SELECT s.experiment_id, s.member_id
        FROM eval_submissions s
        WHERE s.execution_kind = 'audit' AND s.execution_id = COALESCE(NEW.audit_id, OLD.audit_id)
    LOOP
        PERFORM contractor_eval_dirty(r.experiment_id, r.member_id);
    END LOOP;
    RETURN NULL;
END;
$$;

CREATE FUNCTION contractor_eval_collection_changed() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE leaving boolean := TG_OP = 'DELETE';
BEGIN
    IF TG_OP = 'UPDATE' THEN
        leaving := ROW(OLD.owner_id, OLD.project_id) IS DISTINCT FROM ROW(NEW.owner_id, NEW.project_id);
    END IF;
    -- Deleted and moved rows leave their old collections; moves also enter new ones.
    IF leaving THEN
        INSERT INTO eval_collections(owner_id, project_id) VALUES (OLD.owner_id, ''), (OLD.owner_id, OLD.project_id)
        ON CONFLICT (owner_id, project_id) DO UPDATE SET revision = eval_collections.revision + 1;
    END IF;
    IF TG_OP <> 'DELETE' THEN
        INSERT INTO eval_collections(owner_id, project_id) VALUES (NEW.owner_id, ''), (NEW.owner_id, NEW.project_id)
        ON CONFLICT (owner_id, project_id) DO UPDATE SET revision = eval_collections.revision + 1;
    END IF;
    RETURN NULL;
END; $$;

CREATE FUNCTION contractor_eval_collections_changed() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    INSERT INTO eval_collections(owner_id, project_id)
    SELECT DISTINCT changed.owner_id, collection.project_id
    FROM changed_experiments changed
    CROSS JOIN LATERAL (VALUES (''), (changed.project_id)) AS collection(project_id)
    ORDER BY 1, 2
    ON CONFLICT (owner_id, project_id) DO UPDATE SET revision = eval_collections.revision + 1;
    RETURN NULL;
END; $$;

CREATE FUNCTION contractor_eval_collections_updated() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    INSERT INTO eval_collections(owner_id, project_id)
    WITH changed AS (
        SELECT prior.owner_id AS old_owner, prior.project_id AS old_project,
               next_row.owner_id AS new_owner, next_row.project_id AS new_project
          FROM old_experiments prior JOIN new_experiments next_row USING (experiment_id)
         WHERE ROW(prior.owner_id, prior.project_id, prior.name, prior.control_mode,
                   prior.state, prior.expected_count, prior.draft, prior.dataset_id)
               IS DISTINCT FROM
               ROW(next_row.owner_id, next_row.project_id, next_row.name, next_row.control_mode,
                   next_row.state, next_row.expected_count, next_row.draft, next_row.dataset_id)
    )
    SELECT DISTINCT collection.owner_id, collection.project_id
      FROM changed
      CROSS JOIN LATERAL (VALUES
          (old_owner, ''), (old_owner, old_project),
          (new_owner, ''), (new_owner, new_project)
      ) AS collection(owner_id, project_id)
     ORDER BY 1, 2
    ON CONFLICT (owner_id, project_id) DO UPDATE SET revision = eval_collections.revision + 1;
    RETURN NULL;
END; $$;

CREATE FUNCTION contractor_eval_complete_matrix() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE expected integer; actual integer;
BEGIN
    SELECT expected_count INTO expected FROM eval_experiments WHERE experiment_id=NEW.experiment_id;
    SELECT count(*) INTO actual FROM eval_members WHERE experiment_id=NEW.experiment_id;
    IF expected IS NULL OR expected <> actual OR actual < 1 THEN
        RAISE EXCEPTION 'Evaluation expected matrix is incomplete' USING ERRCODE = '23514';
    END IF;
    RETURN NULL;
END; $$;

CREATE FUNCTION contractor_eval_dirty(selected_experiment text, selected_member text) RETURNS void
    LANGUAGE plpgsql
    AS $$
BEGIN
    INSERT INTO eval_member_projections(experiment_id, member_id)
    VALUES (selected_experiment, selected_member)
    ON CONFLICT (experiment_id, member_id)
    DO UPDATE SET revision = eval_member_projections.revision + 1;

    INSERT INTO eval_projection_queue(experiment_id) VALUES (selected_experiment)
    ON CONFLICT (experiment_id)
    DO UPDATE SET revision = eval_projection_queue.revision + 1;
END;
$$;

CREATE FUNCTION contractor_eval_evidence_deleted() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    r record;
BEGIN
    FOR r IN
        SELECT DISTINCT experiment_id, member_id FROM eval_evidence_refs
        WHERE scope_kind = OLD.scope_kind AND scope_id = OLD.scope_id
            AND namespace = OLD.namespace AND name = OLD.name AND revision = OLD.revision
    LOOP
        PERFORM contractor_eval_dirty(r.experiment_id, r.member_id);
    END LOOP;
    RETURN OLD;
END;
$$;

CREATE FUNCTION contractor_eval_execution_changed() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    selected_kind text;
    execution_ref text;
    r record;
BEGIN
    IF TG_TABLE_NAME = 'workflow_runs' THEN
        selected_kind := 'run';
        execution_ref := COALESCE(NEW.run_id, OLD.run_id);
    ELSE
        selected_kind := 'audit';
        execution_ref := COALESCE(NEW.audit_id, OLD.audit_id);
    END IF;

    FOR r IN
        SELECT s.experiment_id, s.member_id
        FROM eval_submissions s
        WHERE s.execution_kind = selected_kind AND s.execution_id = execution_ref
        UNION
        SELECT s.experiment_id, s.member_id
        FROM audit_executions x
        JOIN eval_submissions s ON s.execution_kind = 'audit' AND s.execution_id = x.audit_id
        WHERE selected_kind = 'run' AND x.run_id = execution_ref
    LOOP
        PERFORM contractor_eval_dirty(r.experiment_id, r.member_id);
    END LOOP;
    RETURN NULL;
END;
$$;

CREATE FUNCTION contractor_eval_fence_project() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.lifecycle_state = 'deleting' AND OLD.lifecycle_state = 'active' THEN
        UPDATE eval_experiments
           SET deletion_requested_at = CASE
                   WHEN project_id = NEW.project_id THEN COALESCE(deletion_requested_at, clock_timestamp())
                   ELSE deletion_requested_at
               END,
               state = 'cancelling',
               revision = revision + 1,
               updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
         WHERE project_id = NEW.project_id
            OR (state NOT IN ('finished', 'cancelled') AND experiment_id IN (
                SELECT experiment_id
                  FROM eval_submissions
                 WHERE execution_project_id = NEW.project_id
            ));
    END IF;
    RETURN NEW;
END; $$;

CREATE FUNCTION contractor_eval_immutable() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'DELETE' AND current_setting('contractor.eval_purge', true) = 'on' THEN RETURN OLD; END IF;
    RAISE EXCEPTION 'Evaluation record is immutable' USING ERRCODE = '23514';
END; $$;

CREATE FUNCTION contractor_eval_member_bound() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NOT EXISTS(SELECT 1 FROM eval_experiments WHERE experiment_id=NEW.experiment_id AND NEW.ordinal<expected_count) THEN
        RAISE EXCEPTION 'Evaluation member is outside frozen matrix' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END; $$;

CREATE FUNCTION contractor_eval_member_changed() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    PERFORM contractor_eval_dirty(NEW.experiment_id, NEW.member_id);
    RETURN NULL;
END;
$$;

CREATE FUNCTION contractor_eval_members_inserted() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    INSERT INTO eval_member_projections(experiment_id, member_id)
    SELECT experiment_id, member_id FROM inserted_members;

    INSERT INTO eval_projection_queue(experiment_id)
    SELECT DISTINCT experiment_id FROM inserted_members
    ON CONFLICT (experiment_id)
    DO UPDATE SET revision = eval_projection_queue.revision + 1;
    RETURN NULL;
END;
$$;

CREATE FUNCTION contractor_eval_metrics_changed() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    r record;
    selected_run text;
BEGIN
    IF TG_TABLE_NAME = 'stage_executions' THEN
        selected_run := COALESCE(NEW.run_id, OLD.run_id);
    ELSE
        SELECT run_id INTO selected_run FROM stage_executions
        WHERE stage_execution_id = COALESCE(NEW.stage_execution_id, OLD.stage_execution_id);
    END IF;

    FOR r IN
        SELECT s.experiment_id, s.member_id
        FROM eval_submissions s
        WHERE s.execution_kind = 'run' AND s.execution_id = selected_run
        UNION
        SELECT s.experiment_id, s.member_id
        FROM audit_executions x
        JOIN eval_submissions s ON s.execution_kind = 'audit' AND s.execution_id = x.audit_id
        WHERE x.run_id = selected_run
    LOOP
        PERFORM contractor_eval_dirty(r.experiment_id, r.member_id);
    END LOOP;
    RETURN NULL;
END;
$$;

CREATE FUNCTION contractor_eval_output_changed() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    r record;
    selected_scope text;
    selected_id text;
    selected_namespace text;
BEGIN
    selected_scope := COALESCE(NEW.scope_kind, OLD.scope_kind);
    selected_id := COALESCE(NEW.scope_id, OLD.scope_id);
    selected_namespace := COALESCE(NEW.namespace, OLD.namespace);
    IF selected_scope = 'run' AND selected_namespace = 'outputs' THEN
        FOR r IN
            SELECT s.experiment_id, s.member_id
            FROM eval_submissions s
            WHERE s.execution_kind = 'run' AND s.execution_id = selected_id
            UNION
            SELECT s.experiment_id, s.member_id
            FROM audit_executions x
            JOIN eval_submissions s ON s.execution_kind = 'audit' AND s.execution_id = x.audit_id
            WHERE x.run_id = selected_id
        LOOP
            PERFORM contractor_eval_dirty(r.experiment_id, r.member_id);
        END LOOP;
    END IF;
    RETURN NULL;
END;
$$;

CREATE FUNCTION contractor_eval_protect_experiment() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF ROW(NEW.experiment_id,NEW.owner_id,NEW.project_id,NEW.portable_id,NEW.control_mode,NEW.created_at)
       IS DISTINCT FROM ROW(OLD.experiment_id,OLD.owner_id,OLD.project_id,OLD.portable_id,OLD.control_mode,OLD.created_at)
       OR NEW.revision NOT IN (OLD.revision, OLD.revision + 1)
       OR (NEW.revision = OLD.revision
           AND ROW(NEW.name,NEW.draft,NEW.dataset_id,NEW.dataset_revision,NEW.max_in_flight,
                   NEW.wall_ms,NEW.token_limit,NEW.deletion_requested_at)
               IS DISTINCT FROM
               ROW(OLD.name,OLD.draft,OLD.dataset_id,OLD.dataset_revision,OLD.max_in_flight,
                   OLD.wall_ms,OLD.token_limit,OLD.deletion_requested_at))
       OR NEW.updated_at <= OLD.updated_at
       OR NEW.observed_tokens < OLD.observed_tokens
       OR (OLD.started_at IS NOT NULL AND ROW(NEW.started_at,NEW.deadline_at) IS DISTINCT FROM ROW(OLD.started_at,OLD.deadline_at))
       OR (OLD.deletion_requested_at IS NOT NULL AND NEW.deletion_requested_at IS DISTINCT FROM OLD.deletion_requested_at)
       OR (EXISTS (SELECT 1 FROM eval_frozen_plans WHERE experiment_id=OLD.experiment_id)
           AND ROW(NEW.draft,NEW.dataset_id,NEW.dataset_revision,NEW.expected_count,NEW.max_in_flight,NEW.wall_ms,NEW.token_limit)
               IS DISTINCT FROM ROW(OLD.draft,OLD.dataset_id,OLD.dataset_revision,OLD.expected_count,OLD.max_in_flight,OLD.wall_ms,OLD.token_limit))
    THEN RAISE EXCEPTION 'Evaluation revision or immutable fields changed' USING ERRCODE = '23514'; END IF;
    RETURN NEW;
END; $$;

CREATE FUNCTION contractor_eval_protect_submission() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF ROW(NEW.experiment_id,NEW.member_id,NEW.actor_id,NEW.created_at,NEW.execution_kind)
       IS DISTINCT FROM ROW(OLD.experiment_id,OLD.member_id,OLD.actor_id,OLD.created_at,OLD.execution_kind)
       OR (OLD.execution_id IS NOT NULL AND NEW.execution_id IS DISTINCT FROM OLD.execution_id)
       OR (OLD.execution_project_id IS NOT NULL AND NEW.execution_project_id IS DISTINCT FROM OLD.execution_project_id)
       OR (OLD.state IN ('terminal','rejected') AND ROW(NEW.state,NEW.terminal_state) IS DISTINCT FROM ROW(OLD.state,OLD.terminal_state))
    THEN RAISE EXCEPTION 'Evaluation submission is immutable' USING ERRCODE = '23514'; END IF;
    RETURN NEW;
END; $$;

CREATE FUNCTION contractor_eval_protect_suboperation() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF ROW(NEW.experiment_id,NEW.member_id,NEW.kind,NEW.operation_key,NEW.request)
       IS DISTINCT FROM ROW(OLD.experiment_id,OLD.member_id,OLD.kind,OLD.operation_key,OLD.request)
       OR (OLD.state <> 'intent' AND ROW(NEW.state,NEW.response) IS DISTINCT FROM ROW(OLD.state,OLD.response))
    THEN RAISE EXCEPTION 'Evaluation suboperation is immutable' USING ERRCODE = '23514'; END IF;
    RETURN NEW;
END; $$;

CREATE FUNCTION contractor_eval_retain_deleted_execution() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE bound_experiment text; bound_member text; kind text; execution text; outcome text; never_started boolean;
BEGIN
    IF TG_TABLE_NAME='workflow_runs' THEN
        SELECT op.experiment_id,op.member_id INTO bound_experiment,bound_member
        FROM eval_suboperations op JOIN eval_experiments e USING(experiment_id)
        WHERE op.kind='run-create' AND op.operation_key=OLD.request_idempotency_key
          AND e.owner_id=OLD.owner_id AND e.project_id=OLD.project_id AND op.request_sha256=OLD.request_digest;
        kind:='run';execution:=OLD.run_id;outcome:=OLD.state;never_started:=false;
    ELSE
        SELECT op.experiment_id,op.member_id INTO bound_experiment,bound_member
        FROM audit_idempotency i JOIN eval_suboperations op ON op.operation_key=i.idempotency_key
        JOIN eval_experiments e ON e.experiment_id=op.experiment_id
        JOIN eval_submissions s ON s.experiment_id=op.experiment_id AND s.member_id=op.member_id
        WHERE i.audit_id=OLD.audit_id AND i.operation='audit.create' AND op.kind='audit-create'
          AND e.owner_id=OLD.owner_id AND i.owner_id=OLD.owner_id AND s.execution_project_id=OLD.project_id AND op.request_sha256=i.request_digest;
        kind:='audit';execution:=OLD.audit_id;never_started:=(OLD.started_at IS NULL);
        -- Audit deletion can replace its old terminal state with deleting.
        -- Drain is confirmed by ordinary purge, but quality remains unknown.
        outcome:=CASE WHEN OLD.state IN ('completed','cancelled','failed') THEN OLD.state ELSE 'unknown' END;
    END IF;
    IF bound_experiment IS NOT NULL THEN
        INSERT INTO eval_execution_tombstones(experiment_id,member_id,execution_kind,execution_id,terminal_state,never_started)
        VALUES(bound_experiment,bound_member,kind,execution,outcome,never_started);
        UPDATE eval_experiments SET view_generation=view_generation+1,
            updated_at=GREATEST(clock_timestamp(),updated_at+interval '1 microsecond')
        WHERE experiment_id=bound_experiment;
    END IF;
    RETURN OLD;
END; $$;

CREATE FUNCTION contractor_eval_view_immutable() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        IF current_setting('contractor.eval_purge', true) = 'on' THEN RETURN OLD; END IF;
        IF NOT EXISTS (
            SELECT 1 FROM eval_projection_queue q
            WHERE q.experiment_id = OLD.experiment_id AND q.generation = OLD.generation
        ) THEN RETURN OLD; END IF;
    END IF;
    RAISE EXCEPTION 'Evaluation record is immutable' USING ERRCODE = '23514';
END;
$$;

CREATE FUNCTION contractor_guard_audit_admission() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    PERFORM contractor_require_active_audit_project(NEW.project_id, NEW.owner_id);
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_guard_audit_child_admission() RETURNS trigger
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

CREATE FUNCTION contractor_guard_project_artifact_admission() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.scope_kind = 'project' THEN
        PERFORM contractor_require_active_project(NEW.scope_id);
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_guard_project_artifact_binding_admission() RETURNS trigger
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

CREATE FUNCTION contractor_guard_project_artifact_scope_admission() RETURNS trigger
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

CREATE FUNCTION contractor_guard_workflow_run_project_admission() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.project_id IS NOT NULL THEN
        PERFORM contractor_require_active_project(NEW.project_id, NEW.owner_id);
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_lifecycle_purge_enabled() RETURNS boolean
    LANGUAGE sql STABLE
    AS $$
    SELECT COALESCE(current_setting('contractor.lifecycle_purge', true), '')
        IN ('run', 'project', 'audit')
$$;

CREATE FUNCTION contractor_notify_workflow_run_event() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    PERFORM pg_notify('contractor_run_events_v1', NEW.run_id);
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_protect_artifact_binding() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.scope_kind IS DISTINCT FROM OLD.scope_kind
        OR NEW.scope_id IS DISTINCT FROM OLD.scope_id
        OR NEW.namespace IS DISTINCT FROM OLD.namespace
        OR NEW.name IS DISTINCT FROM OLD.name
        OR NEW.created_at IS DISTINCT FROM OLD.created_at
    THEN
        RAISE EXCEPTION 'Artifact binding identity cannot be changed' USING ERRCODE = '23514';
    END IF;
    IF OLD.frozen AND (
        NEW.current_revision IS DISTINCT FROM OLD.current_revision OR NOT NEW.frozen
    ) THEN
        -- Only the atomic continuation transaction may thaw an output binding;
        -- it must retain the exact current revision. Other frozen bindings and
        -- all immutable revisions retain their original protection.
        IF NOT (
            OLD.scope_kind = 'run' AND OLD.namespace = 'outputs'
            AND NOT NEW.frozen AND NEW.current_revision = OLD.current_revision
            AND EXISTS (
                SELECT 1 FROM run_stage_resumptions AS receipt
                JOIN workflow_runs AS run ON run.run_id = receipt.run_id
                JOIN stage_executions AS target ON target.stage_execution_id = receipt.target_execution_id
                WHERE receipt.run_id = OLD.scope_id
                  AND receipt.transaction_id = pg_current_xact_id()
                  AND run.state = 'pending' AND run.publication_mode = 'ordinary'
                  AND target.run_id = run.run_id AND target.state = 'preparing'
            )
        ) THEN
            RAISE EXCEPTION 'frozen Artifact binding cannot be changed' USING ERRCODE = '23514';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_protect_artifact_blob_immutable() RETURNS trigger
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

CREATE FUNCTION contractor_protect_artifact_immutable() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'DELETE' AND contractor_lifecycle_purge_enabled() THEN
        RETURN OLD;
    END IF;
    RAISE EXCEPTION 'immutable ArtifactStore row cannot be changed' USING ERRCODE = '23514';
END;
$$;

CREATE FUNCTION contractor_protect_artifact_version() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'DELETE' AND contractor_lifecycle_purge_enabled() THEN
        RETURN OLD;
    END IF;
    IF TG_OP = 'UPDATE' AND OLD.git_resolved_commit IS NULL AND NEW.git_resolved_commit IS NOT NULL
        AND ROW(NEW.version_id, NEW.blob_sha256, NEW.media_type, NEW.created_at)
            IS NOT DISTINCT FROM ROW(OLD.version_id, OLD.blob_sha256, OLD.media_type, OLD.created_at)
    THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'immutable ArtifactStore row cannot be changed' USING ERRCODE = '23514';
END;
$$;

CREATE FUNCTION contractor_protect_audit_completion() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
 IF NEW.audit_completion IS DISTINCT FROM OLD.audit_completion AND
    (OLD.audit_completion IS NOT NULL OR OLD.state <> 'initializing') THEN
  RAISE EXCEPTION 'Audit completion is immutable' USING ERRCODE = '23514';
 END IF;
 RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_protect_audit_event() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    RAISE EXCEPTION 'Audit events are immutable' USING ERRCODE = '23514';
END;
$$;

CREATE FUNCTION contractor_protect_audit_execution_collection() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF OLD.collection_receipt_id IS NOT NULL
        AND ROW(NEW.execution_id, NEW.audit_id, NEW.run_id, NEW.terminal_outcome,
                NEW.terminal_run_generation, NEW.terminal_run_sequence,
                NEW.collection_receipt_id, NEW.collection_disposition,
                NEW.collection_source_output_ref, NEW.collection_source_output_digest,
                NEW.collection_retained_refs, NEW.collection_error_code,
                NEW.collection_request_digest, NEW.collected_at)
            IS DISTINCT FROM
            ROW(OLD.execution_id, OLD.audit_id, OLD.run_id, OLD.terminal_outcome,
                OLD.terminal_run_generation, OLD.terminal_run_sequence,
                OLD.collection_receipt_id, OLD.collection_disposition,
                OLD.collection_source_output_ref, OLD.collection_source_output_digest,
                OLD.collection_retained_refs, OLD.collection_error_code,
                OLD.collection_request_digest, OLD.collected_at)
    THEN
        RAISE EXCEPTION 'Audit collection receipts are immutable' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_protect_audit_finding_history() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    RAISE EXCEPTION 'Audit finding history is immutable' USING ERRCODE = '23514';
END;
$$;

CREATE FUNCTION contractor_protect_audit_item_source() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF OLD.proposal_receipt_id IS NOT NULL
        AND ROW(NEW.proposal_receipt_id, NEW.proposal_check_ordinal, NEW.proposal_ref, NEW.proposal_digest)
        IS DISTINCT FROM
        ROW(OLD.proposal_receipt_id, OLD.proposal_check_ordinal, OLD.proposal_ref, OLD.proposal_digest)
    THEN
        RAISE EXCEPTION 'Audit finding history is immutable' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_protect_audit_preparation_baseline() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF OLD.phase IN ('preparing', 'inventory') OR EXISTS (
        SELECT 1 FROM audit_executions WHERE audit_id = OLD.audit_id AND role = 'prepare'
    ) THEN
        IF ROW(NEW.baseline_snapshot, NEW.profile_snapshot, NEW.input_selection,
               NEW.profile_name, NEW.profile_version, NEW.profile_digest, NEW.owner_id, NEW.project_id)
           IS DISTINCT FROM ROW(OLD.baseline_snapshot, OLD.profile_snapshot, OLD.input_selection,
               OLD.profile_name, OLD.profile_version, OLD.profile_digest, OLD.owner_id, OLD.project_id)
        THEN RAISE EXCEPTION 'Audit preparation baseline is immutable' USING ERRCODE = '23514'; END IF;
        IF NEW.phase = 'not-started' THEN
            RAISE EXCEPTION 'Audit preparation cannot return to not-started' USING ERRCODE = '23514';
        END IF;
    END IF;
    IF NEW.phase IN ('preparing', 'inventory') THEN
        IF NEW.baseline_snapshot->>'schema' IS DISTINCT FROM 'contractor.audit.baseline.v1'
           OR jsonb_typeof(NEW.baseline_snapshot->'inputs') IS DISTINCT FROM 'object'
           OR NEW.baseline_snapshot ? 'inventory'
           OR NOT EXISTS (SELECT 1 FROM jsonb_each(NEW.profile_snapshot->'workflows') AS role WHERE role.value->>'kind' = 'prepare')
        THEN RAISE EXCEPTION 'Audit preparation baseline or profile is invalid' USING ERRCODE = '23514'; END IF;
    END IF;
    IF NEW.phase = 'inventory' AND OLD.phase IS DISTINCT FROM 'inventory' AND EXISTS (
        SELECT 1 FROM jsonb_each(NEW.profile_snapshot->'workflows') AS role
         WHERE role.value->>'kind' = 'prepare' AND NOT EXISTS (
             SELECT 1 FROM audit_executions AS execution
              WHERE execution.audit_id = NEW.audit_id AND execution.role = 'prepare'
                AND execution.workflow_role = role.key AND execution.collection_disposition = 'accepted-result'
         )
    ) THEN RAISE EXCEPTION 'Audit preparation is incomplete' USING ERRCODE = '23514'; END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_protect_credential_operation() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.operation_id IS DISTINCT FROM OLD.operation_id
        OR NEW.idempotency_key IS DISTINCT FROM OLD.idempotency_key
        OR NEW.request_hash IS DISTINCT FROM OLD.request_hash
        OR NEW.credential_id IS DISTINCT FROM OLD.credential_id
        OR NEW.operation_kind IS DISTINCT FROM OLD.operation_kind
        OR NEW.request_schema_version IS DISTINCT FROM OLD.request_schema_version
        OR NEW.request IS DISTINCT FROM OLD.request
        OR NEW.created_at IS DISTINCT FROM OLD.created_at
        OR OLD.phase <> 'prepared'
        OR NEW.phase NOT IN ('completed', 'abandoned')
        OR (NEW.phase = 'abandoned' AND OLD.operation_kind <> 'create')
        OR NEW.updated_at < OLD.updated_at
    THEN
        RAISE EXCEPTION 'credential operation mutation is invalid' USING ERRCODE = '23514';
    END IF;
    IF NEW.operation_kind = 'delete' AND NEW.phase = 'completed' THEN
        PERFORM 1 FROM credential_operations
        WHERE credential_id = NEW.credential_id AND operation_kind = 'create'
        FOR UPDATE;
        IF EXISTS (SELECT 1 FROM llm_credentials WHERE credential_id = NEW.credential_id) THEN
            RAISE EXCEPTION 'credential ID is still active' USING ERRCODE = '23505';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_protect_finding_proposal_receipt() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    RAISE EXCEPTION 'finding proposal receipts are immutable' USING ERRCODE = '23514';
END;
$$;

CREATE FUNCTION contractor_protect_finding_proposal_receipt_retention() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'UPDATE'
        AND to_jsonb(NEW) - ARRAY['retention_state', 'source_run_deleted_at', 'discarded_at', 'retention_updated_at']
            = to_jsonb(OLD) - ARRAY['retention_state', 'source_run_deleted_at', 'discarded_at', 'retention_updated_at']
    THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'finding proposal receipts are immutable' USING ERRCODE = '23514';
END;
$$;

CREATE FUNCTION contractor_protect_immutable_credential_row() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    RAISE EXCEPTION 'credential identity, active record, and tombstone rows are immutable'
        USING ERRCODE = '23514';
END;
$$;

CREATE FUNCTION contractor_protect_owner_queue_control() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.owner_id IS DISTINCT FROM OLD.owner_id THEN
        RAISE EXCEPTION 'Owner Queue control identity cannot be changed' USING ERRCODE = '23514';
    END IF;
    IF NEW.paused IS NOT DISTINCT FROM OLD.paused
        OR NEW.revision <> OLD.revision + 1
        OR NEW.updated_at <= OLD.updated_at
    THEN
        RAISE EXCEPTION 'Owner Queue control update must change state, revision and time' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_protect_project() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    public_change boolean;
BEGIN
    IF NEW.project_id IS DISTINCT FROM OLD.project_id
        OR NEW.owner_id IS DISTINCT FROM OLD.owner_id
        OR NEW.kind IS DISTINCT FROM OLD.kind
        OR NEW.request_idempotency_key IS DISTINCT FROM OLD.request_idempotency_key
        OR NEW.request_digest IS DISTINCT FROM OLD.request_digest
        OR NEW.created_at IS DISTINCT FROM OLD.created_at
    THEN
        RAISE EXCEPTION 'Project immutable identity cannot be changed' USING ERRCODE = '23514';
    END IF;

    IF OLD.lifecycle_state = 'deleting'
        AND (
            NEW.name IS DISTINCT FROM OLD.name
            OR NEW.description IS DISTINCT FROM OLD.description
            OR NEW.http_target_url IS DISTINCT FROM OLD.http_target_url
            OR NEW.http_target_credential_id IS DISTINCT FROM OLD.http_target_credential_id
            OR NEW.http_target_credential_kind IS DISTINCT FROM OLD.http_target_credential_kind
            OR NEW.lifecycle_state IS DISTINCT FROM OLD.lifecycle_state
            OR NEW.deletion_requested_at IS DISTINCT FROM OLD.deletion_requested_at
        )
    THEN
        RAISE EXCEPTION 'Project is deleting' USING ERRCODE = '55000';
    END IF;

    IF OLD.lifecycle_state = 'active'
        AND NEW.lifecycle_state = 'deleting'
        AND (
            NEW.deletion_phase IS DISTINCT FROM 'cancelling'
            OR NEW.deletion_requested_at IS NULL
            OR NEW.deletion_claim_id IS NOT NULL
            OR NEW.deletion_claimed_at IS NOT NULL
            OR NEW.deletion_claim_expires_at IS NOT NULL
        )
    THEN
        RAISE EXCEPTION 'Project deletion must begin in cancelling' USING ERRCODE = '23514';
    END IF;

    IF OLD.lifecycle_state = 'deleting'
        AND NEW.deletion_phase IS DISTINCT FROM OLD.deletion_phase
        AND (
            OLD.deletion_claim_id IS NULL
            OR NEW.deletion_claim_id IS DISTINCT FROM OLD.deletion_claim_id
            OR NEW.deletion_claimed_at IS DISTINCT FROM OLD.deletion_claimed_at
            OR NEW.deletion_claim_expires_at IS DISTINCT FROM OLD.deletion_claim_expires_at
            OR NOT (
                (OLD.deletion_phase = 'cancelling' AND NEW.deletion_phase = 'draining')
                OR (OLD.deletion_phase = 'draining' AND NEW.deletion_phase = 'purging_runs')
                OR (OLD.deletion_phase = 'purging_runs' AND NEW.deletion_phase = 'purging_artifacts')
            )
        )
    THEN
        RAISE EXCEPTION 'Project deletion phase transition is invalid' USING ERRCODE = '23514';
    END IF;

    public_change := ROW(
        NEW.name, NEW.description,
        NEW.http_target_url, NEW.http_target_credential_id, NEW.http_target_credential_kind,
        NEW.lifecycle_state, NEW.deletion_phase, NEW.deletion_requested_at
    ) IS DISTINCT FROM ROW(
        OLD.name, OLD.description,
        OLD.http_target_url, OLD.http_target_credential_id, OLD.http_target_credential_kind,
        OLD.lifecycle_state, OLD.deletion_phase, OLD.deletion_requested_at
    );

    IF public_change THEN
        IF NEW.revision <> OLD.revision + 1 OR NEW.updated_at <= OLD.updated_at THEN
            RAISE EXCEPTION 'Project update must advance revision and time' USING ERRCODE = '23514';
        END IF;
    ELSIF NEW.revision IS DISTINCT FROM OLD.revision
        OR NEW.updated_at IS DISTINCT FROM OLD.updated_at
    THEN
        RAISE EXCEPTION 'Project lease update cannot change public revision' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_protect_run_stage_resumption() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    RAISE EXCEPTION 'Run continuation receipts are immutable' USING ERRCODE = '23514';
END;
$$;

CREATE FUNCTION contractor_protect_runtime_agent_principal() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.runtime_agent_id IS DISTINCT FROM OLD.runtime_agent_id
       OR NEW.created_by IS DISTINCT FROM OLD.created_by
       OR NEW.created_at IS DISTINCT FROM OLD.created_at
       OR NEW.label_revision IS DISTINCT FROM OLD.label_revision + 1
       OR NEW.labels IS NOT DISTINCT FROM OLD.labels
    THEN
        RAISE EXCEPTION 'Runtime Agent principal update must be one semantic label revision'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_protect_runtime_config_immutable() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    RAISE EXCEPTION 'immutable RuntimeConfig row cannot be changed'
        USING ERRCODE = '23514';
END;
$$;

CREATE FUNCTION contractor_protect_runtime_credential_immutable() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'UPDATE' AND OLD.deleted_at IS NULL AND NEW.deleted_at IS NOT NULL
        AND ROW(NEW.credential_id, NEW.credential_kind, NEW.encryption_schema_version, NEW.key_id,
                NEW.nonce, NEW.ciphertext, NEW.created_by, NEW.created_at,
                NEW.idempotency_key_digest, NEW.request_mac)
            IS NOT DISTINCT FROM
            ROW(OLD.credential_id, OLD.credential_kind, OLD.encryption_schema_version, OLD.key_id,
                OLD.nonce, OLD.ciphertext, OLD.created_by, OLD.created_at,
                OLD.idempotency_key_digest, OLD.request_mac)
    THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'Runtime credential rows are immutable'
        USING ERRCODE = '23514';
END;
$$;

CREATE FUNCTION contractor_protect_runtime_label_binding() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        IF OLD.label = 'default' THEN
            RAISE EXCEPTION 'default RuntimeConfig binding cannot be deleted'
                USING ERRCODE = '23514';
        END IF;
        RETURN OLD;
    END IF;

    IF NEW.label IS DISTINCT FROM OLD.label
       OR NEW.created_by IS DISTINCT FROM OLD.created_by
       OR NEW.created_at IS DISTINCT FROM OLD.created_at
       OR NEW.revision IS DISTINCT FROM OLD.revision + 1
       OR ROW(NEW.config_name, NEW.config_version, NEW.config_digest)
          IS NOT DISTINCT FROM
          ROW(OLD.config_name, OLD.config_version, OLD.config_digest)
    THEN
        RAISE EXCEPTION 'RuntimeConfig binding update must be one semantic revision'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_protect_scheduler_settings() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'Scheduler settings singleton cannot be deleted'
            USING ERRCODE = '23514';
    END IF;
    IF NEW.singleton IS DISTINCT FROM OLD.singleton
        OR NEW.max_concurrent_runs IS NOT DISTINCT FROM OLD.max_concurrent_runs
        OR NEW.revision IS DISTINCT FROM OLD.revision + 1
        OR NEW.updated_at <= OLD.updated_at
    THEN
        RAISE EXCEPTION 'Scheduler settings update must be one semantic revision'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_protect_stage_allocation_identity() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.allocation_id IS DISTINCT FROM OLD.allocation_id
       OR NEW.stage_execution_id IS DISTINCT FROM OLD.stage_execution_id
       OR NEW.logical_agent_name IS DISTINCT FROM OLD.logical_agent_name
       OR NEW.namespace IS DISTINCT FROM OLD.namespace
       OR NEW.agent_template_ref IS DISTINCT FROM OLD.agent_template_ref
       OR NEW.worker_runtime_ref IS DISTINCT FROM OLD.worker_runtime_ref
       OR NEW.runtime_agent_instance_id IS DISTINCT FROM OLD.runtime_agent_instance_id
       OR NEW.runtime_agent_id IS DISTINCT FROM OLD.runtime_agent_id
       OR NEW.runtime_agent_label_revision IS DISTINCT FROM OLD.runtime_agent_label_revision
       OR NEW.runtime_configuration_schema_version IS DISTINCT FROM OLD.runtime_configuration_schema_version
       OR NEW.runtime_configuration IS DISTINCT FROM OLD.runtime_configuration
       OR NEW.created_at IS DISTINCT FROM OLD.created_at
    THEN
        RAISE EXCEPTION 'Stage allocation identity and Runtime configuration are immutable'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_protect_stage_allocation_performance_policy() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.performance_collection_policy IS DISTINCT FROM OLD.performance_collection_policy THEN
        RAISE EXCEPTION 'Stage allocation performance collection policy is immutable'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_protect_stage_execution_immutable() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.stage_execution_id IS DISTINCT FROM OLD.stage_execution_id
        OR NEW.run_id IS DISTINCT FROM OLD.run_id
        OR NEW.stage_name IS DISTINCT FROM OLD.stage_name
        OR NEW.attempt IS DISTINCT FROM OLD.attempt
        OR NEW.previous_execution_id IS DISTINCT FROM OLD.previous_execution_id
        OR NEW.execution_config_variant IS DISTINCT FROM OLD.execution_config_variant
        OR NEW.escalation_ordinal IS DISTINCT FROM OLD.escalation_ordinal
        OR NEW.stage_spec_schema_version IS DISTINCT FROM OLD.stage_spec_schema_version
        OR NEW.stage_spec_snapshot IS DISTINCT FROM OLD.stage_spec_snapshot
        OR NEW.stage_context_schema_version IS DISTINCT FROM OLD.stage_context_schema_version
        OR NEW.stage_context_snapshot IS DISTINCT FROM OLD.stage_context_snapshot
        OR NEW.created_at IS DISTINCT FROM OLD.created_at
    THEN
        RAISE EXCEPTION 'immutable StageExecution fields cannot be changed' USING ERRCODE = '23514';
    END IF;
    IF OLD.planner_session_id IS NOT NULL
        AND (NEW.planner_session_id IS DISTINCT FROM OLD.planner_session_id
            OR NEW.planner_invocation_id IS DISTINCT FROM OLD.planner_invocation_id)
    THEN
        RAISE EXCEPTION 'Planner identity cannot be changed' USING ERRCODE = '23514';
    END IF;
    IF OLD.candidate_stage_result IS NOT NULL
        AND (NEW.candidate_stage_result IS DISTINCT FROM OLD.candidate_stage_result
            OR NEW.candidate_result_schema_version IS DISTINCT FROM OLD.candidate_result_schema_version)
    THEN
        RAISE EXCEPTION 'candidate StageResult cannot be changed' USING ERRCODE = '23514';
    END IF;
    IF OLD.accepted_stage_result IS NOT NULL
        AND (NEW.accepted_stage_result IS DISTINCT FROM OLD.accepted_stage_result
            OR NEW.accepted_result_schema_version IS DISTINCT FROM OLD.accepted_result_schema_version)
    THEN
        RAISE EXCEPTION 'accepted StageResult cannot be changed' USING ERRCODE = '23514';
    END IF;
    IF OLD.stage_termination IS NOT NULL
        AND (NEW.stage_termination IS DISTINCT FROM OLD.stage_termination
            OR NEW.termination_schema_version IS DISTINCT FROM OLD.termination_schema_version)
    THEN
        RAISE EXCEPTION 'StageTermination cannot be changed' USING ERRCODE = '23514';
    END IF;
    IF OLD.finalization_id IS NOT NULL
        AND (NEW.finalization_id IS DISTINCT FROM OLD.finalization_id
            OR NEW.finalization_deadline IS DISTINCT FROM OLD.finalization_deadline)
    THEN
        RAISE EXCEPTION 'finalization identity cannot be changed' USING ERRCODE = '23514';
    END IF;
    IF OLD.abort_id IS NOT NULL
        AND (NEW.abort_id IS DISTINCT FROM OLD.abort_id
            OR NEW.abort_deadline IS DISTINCT FROM OLD.abort_deadline)
    THEN
        RAISE EXCEPTION 'abort identity cannot be changed' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_protect_stage_transition_decision_immutable() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    RAISE EXCEPTION 'Stage transition decisions are immutable' USING ERRCODE = '23514';
END;
$$;

CREATE FUNCTION contractor_protect_telemetry_report_immutable() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    RAISE EXCEPTION 'telemetry report cannot be changed' USING ERRCODE = '23514';
END;
$$;

CREATE FUNCTION contractor_protect_workflow_run_event_immutable() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    RAISE EXCEPTION 'WorkflowRun events are immutable' USING ERRCODE = '23514';
END;
$$;

CREATE FUNCTION contractor_protect_workflow_run_immutable() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.run_id IS DISTINCT FROM OLD.run_id
        OR NEW.owner_id IS DISTINCT FROM OLD.owner_id
        OR NEW.project_id IS DISTINCT FROM OLD.project_id
        OR NEW.workflow_name IS DISTINCT FROM OLD.workflow_name
        OR NEW.workflow_version IS DISTINCT FROM OLD.workflow_version
        OR NEW.workflow_schema_version IS DISTINCT FROM OLD.workflow_schema_version
        OR NEW.workflow_snapshot IS DISTINCT FROM OLD.workflow_snapshot
        OR NEW.parameters IS DISTINCT FROM OLD.parameters
        OR NEW.metadata_labels IS DISTINCT FROM OLD.metadata_labels
        OR NEW.request_idempotency_key IS DISTINCT FROM OLD.request_idempotency_key
        OR NEW.request_digest IS DISTINCT FROM OLD.request_digest
        OR NEW.run_event_generation IS DISTINCT FROM OLD.run_event_generation
        OR NEW.runtime_labels IS DISTINCT FROM OLD.runtime_labels
        OR NEW.runtime_config_snapshot IS DISTINCT FROM OLD.runtime_config_snapshot
        OR NEW.project_http_target_snapshot IS DISTINCT FROM OLD.project_http_target_snapshot
        OR NEW.publication_mode IS DISTINCT FROM OLD.publication_mode
        OR NEW.audit_execution_id IS DISTINCT FROM OLD.audit_execution_id
        OR NEW.audit_submission_key IS DISTINCT FROM OLD.audit_submission_key
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

CREATE FUNCTION contractor_protect_workflow_run_skill_snapshot() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.skill_snapshot IS DISTINCT FROM OLD.skill_snapshot
        AND (OLD.state <> 'initializing' OR NEW.state <> 'initializing')
    THEN
        RAISE EXCEPTION 'terminal WorkflowRun Skill snapshot cannot be changed' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_reconcile_finding_retention() RETURNS trigger
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
        UPDATE finding_proposal_receipts
           SET retention_state = 'audit-held', discarded_at = NULL, retention_updated_at = clock_timestamp()
         WHERE receipt_id = target_receipt;
    ELSE
        UPDATE finding_proposal_receipts
           SET retention_state = CASE WHEN source_run_deleted_at IS NULL THEN 'source-held' ELSE 'discarded' END,
               discarded_at = CASE WHEN source_run_deleted_at IS NULL THEN NULL ELSE COALESCE(discarded_at, clock_timestamp()) END,
               retention_updated_at = clock_timestamp()
         WHERE receipt_id = target_receipt;
    END IF;
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_require_active_audit_project(checked_project_id text, checked_owner_id text) RETURNS void
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

CREATE FUNCTION contractor_require_active_project(checked_project_id text, checked_owner_id text DEFAULT NULL::text) RETURNS void
    LANGUAGE plpgsql
    AS $$
DECLARE
    current_lifecycle text;
BEGIN
    SELECT lifecycle_state
    INTO current_lifecycle
    FROM projects
    WHERE project_id = checked_project_id
      AND (checked_owner_id IS NULL OR owner_id = checked_owner_id)
    FOR SHARE;

    IF current_lifecycle = 'deleting' THEN
        RAISE EXCEPTION 'Project is deleting' USING ERRCODE = '55000';
    END IF;
END;
$$;

CREATE FUNCTION contractor_require_allocation_completion() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE pinned jsonb; stage_name text;
BEGIN
 IF TG_OP = 'UPDATE' AND NEW.completion_contract IS DISTINCT FROM OLD.completion_contract THEN
  RAISE EXCEPTION 'Allocation completion is immutable' USING ERRCODE = '23514';
 END IF;
 SELECT r.audit_completion, s.stage_name INTO pinned, stage_name
 FROM stage_executions s JOIN workflow_runs r USING (run_id)
 WHERE s.stage_execution_id = NEW.stage_execution_id;
 IF pinned ->> 'stage' = stage_name AND pinned ->> 'agent' = NEW.logical_agent_name THEN
  IF NEW.completion_contract IS DISTINCT FROM pinned -> 'contract' THEN
   RAISE EXCEPTION 'Allocation completion differs from Run authority' USING ERRCODE = '23514';
  END IF;
 ELSIF NEW.completion_contract IS NOT NULL THEN
  RAISE EXCEPTION 'Allocation has no completion authority' USING ERRCODE = '23514';
 END IF;
 RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_require_audit_completion() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE stored workflow_runs%ROWTYPE; binding jsonb;
BEGIN
 IF TG_OP = 'UPDATE' AND OLD.audit_completion IS NULL AND NEW.audit_completion IS NULL THEN RETURN NULL; END IF;
 SELECT * INTO stored FROM workflow_runs WHERE run_id = NEW.run_id;
 IF NOT FOUND OR stored.publication_mode <> 'audit-managed' THEN RETURN NULL; END IF;
 SELECT a.profile_snapshot #> ARRAY['workflows', e.workflow_role] INTO binding
 FROM audit_executions e JOIN audits a USING (audit_id)
 WHERE e.execution_id = stored.audit_execution_id AND e.role = 'check';
 IF (binding -> 'workerCompletion') IS NOT NULL AND binding -> 'workerCompletion' <> 'null'::jsonb THEN
  IF stored.audit_completion IS NULL OR
     stored.audit_completion ->> 'stage' IS DISTINCT FROM binding #>> '{workerCompletion,stage}' OR
     stored.audit_completion ->> 'agent' IS DISTINCT FROM binding #>> '{workerCompletion,agent}' OR
     stored.audit_completion #>> '{contract,kind}' IS DISTINCT FROM binding #>> '{workerCompletion,kind}' THEN
   RAISE EXCEPTION 'Audit completion does not match owned profile' USING ERRCODE = '23514';
  END IF;
 ELSIF stored.audit_completion IS NOT NULL THEN
  RAISE EXCEPTION 'Audit completion has no owned check binding' USING ERRCODE = '23514';
 END IF;
 RETURN NULL;
END;
$$;

CREATE FUNCTION contractor_require_complete_audit_run_binding() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.publication_mode = 'audit-managed' AND NOT EXISTS (
        SELECT 1
        FROM audit_executions AS execution
        WHERE execution.execution_id = NEW.audit_execution_id
          AND execution.submission_key = NEW.audit_submission_key
          AND execution.run_id = NEW.run_id
    ) THEN
        RAISE EXCEPTION 'Audit-managed WorkflowRun association is incomplete'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_require_matching_audit_run() RETURNS trigger
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

CREATE FUNCTION contractor_valid_allocation_runtime_configuration(candidate jsonb) RETURNS boolean
    LANGUAGE plpgsql IMMUTABLE STRICT
    AS $_$
DECLARE
    model_policy jsonb := candidate->'modelPolicy';
    origins jsonb := candidate->'origins';
    provenance jsonb := candidate->'provenance';
    has_model boolean := candidate ? 'modelPolicy';
BEGIN
    IF jsonb_typeof(candidate) IS DISTINCT FROM 'object'
       OR NOT (candidate ?& ARRAY['origins', 'provenance']::text[])
       OR EXISTS (
           SELECT 1 FROM jsonb_object_keys(candidate) AS key
           WHERE key <> ALL (ARRAY['modelPolicy', 'origins', 'provenance']::text[])
       )
       OR jsonb_typeof(origins) IS DISTINCT FROM 'object'
       OR jsonb_typeof(provenance) IS DISTINCT FROM 'object'
       OR NOT (provenance ?& ARRAY[
           'agentLabels', 'default', 'runLabels', 'runtimeAdapters', 'runtimeCredentialRefs'
       ]::text[])
       OR EXISTS (
           SELECT 1 FROM jsonb_object_keys(provenance) AS key
           WHERE key <> ALL (ARRAY[
               'agentLabels', 'default', 'llmCredential', 'llmGatewayConfig',
               'runLabels', 'runtimeAdapters', 'runtimeCredentialRefs'
           ]::text[])
       )
       OR jsonb_typeof(provenance->'default') IS DISTINCT FROM 'object'
       OR jsonb_typeof(provenance->'runLabels') IS DISTINCT FROM 'array'
       OR jsonb_typeof(provenance->'agentLabels') IS DISTINCT FROM 'array'
       OR jsonb_typeof(provenance->'runtimeAdapters') IS DISTINCT FROM 'array'
       OR jsonb_typeof(provenance->'runtimeCredentialRefs') IS DISTINCT FROM 'array'
    THEN
        RETURN false;
    END IF;

    IF has_model THEN
        IF jsonb_typeof(model_policy) IS DISTINCT FROM 'object'
           OR (SELECT array_agg(key ORDER BY key) FROM jsonb_object_keys(model_policy) AS key)
              IS DISTINCT FROM ARRAY['digest', 'policyId', 'version']::text[]
           OR jsonb_typeof(model_policy->'policyId') IS DISTINCT FROM 'string'
           OR jsonb_typeof(model_policy->'version') IS DISTINCT FROM 'string'
           OR jsonb_typeof(model_policy->'digest') IS DISTINCT FROM 'string'
           OR model_policy->>'policyId' !~ '^[a-z][a-z0-9_-]*$'
           OR model_policy->>'version' !~ '^[A-Za-z0-9][A-Za-z0-9._+-]*$'
           OR model_policy->>'digest' !~ '^sha256:[0-9a-f]{64}$'
           OR jsonb_typeof(provenance->'llmGatewayConfig') IS DISTINCT FROM 'object'
        THEN
            RETURN false;
        END IF;
    ELSIF provenance ?| ARRAY['llmGatewayConfig', 'llmCredential']::text[]
       OR origins ?| ARRAY['llmGateway', 'llmCredential']::text[]
    THEN
        RETURN false;
    END IF;
    RETURN true;
EXCEPTION WHEN OTHERS THEN
    RETURN false;
END;
$_$;

CREATE FUNCTION contractor_valid_pinned_runtime_config(candidate jsonb, expected_label text, explicit boolean) RETURNS boolean
    LANGUAGE sql IMMUTABLE STRICT
    AS $_$
    SELECT jsonb_typeof(candidate) = 'object'
       AND (SELECT array_agg(key ORDER BY key) FROM jsonb_object_keys(candidate) AS key)
           = ARRAY['bindingRevision', 'config', 'explicit', 'label']::text[]
       AND candidate->>'label' = expected_label
       AND candidate->'explicit' = to_jsonb(explicit)
       AND jsonb_typeof(candidate->'bindingRevision') = 'number'
       AND (candidate->>'bindingRevision')::numeric >= 1
       AND (candidate->>'bindingRevision')::numeric <= 18446744073709551615
       AND scale((candidate->>'bindingRevision')::numeric) = 0
       AND jsonb_typeof(candidate->'config') = 'object'
       AND (SELECT array_agg(key ORDER BY key) FROM jsonb_object_keys(candidate->'config') AS key)
           = ARRAY['digest', 'name', 'version']::text[]
       AND candidate->'config'->>'name' ~ '^[a-z][a-z0-9_-]*$'
       AND length(candidate->'config'->>'name') <= 63
       AND candidate->'config'->>'version' ~ '^[A-Za-z0-9][A-Za-z0-9._-]*$'
       AND length(candidate->'config'->>'version') <= 128
       AND candidate->'config'->>'digest' ~ '^sha256:[0-9a-f]{64}$'
$_$;

CREATE FUNCTION contractor_valid_run_cancellation(payload jsonb) RETURNS boolean
    LANGUAGE sql IMMUTABLE
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

CREATE FUNCTION contractor_valid_run_metadata_labels(labels jsonb) RETURNS boolean
    LANGUAGE sql IMMUTABLE
    AS $_$
    SELECT jsonb_typeof(labels) = 'object'
       AND (SELECT count(*) FROM jsonb_object_keys(labels)) <= 32
       AND NOT EXISTS (
           SELECT 1 FROM jsonb_each(labels) AS entry
           WHERE jsonb_typeof(entry.value) <> 'string'
              OR octet_length(entry.key) NOT BETWEEN 1 AND 63
              OR entry.key !~ '^[a-z][a-z0-9]*([._-][a-z0-9]+)*$'
              OR entry.key ~ '^contractor\.'
              OR octet_length(entry.value #>> '{}') > 256
       )
$_$;

CREATE FUNCTION contractor_valid_run_runtime_snapshot(candidate jsonb, labels text[]) RETURNS boolean
    LANGUAGE plpgsql IMMUTABLE STRICT
    AS $_$
DECLARE
    item jsonb;
    index_value integer := 1;
    previous_value text := '';
    credential_value text;
BEGIN
    IF jsonb_typeof(candidate) <> 'object'
       OR (SELECT array_agg(key ORDER BY key) FROM jsonb_object_keys(candidate) AS key)
          <> ARRAY['default', 'labels', 'llmCredentialIds', 'runtimeCredentialIds']::text[]
       OR contractor_valid_pinned_runtime_config(candidate->'default', 'default', false) IS DISTINCT FROM true
       OR jsonb_typeof(candidate->'labels') <> 'array'
       OR jsonb_array_length(candidate->'labels') <> cardinality(labels)
       OR jsonb_typeof(candidate->'llmCredentialIds') <> 'array'
       OR jsonb_typeof(candidate->'runtimeCredentialIds') <> 'array'
    THEN
        RETURN false;
    END IF;

    FOR item IN SELECT value FROM jsonb_array_elements(candidate->'labels')
    LOOP
        IF index_value > cardinality(labels)
           OR contractor_valid_pinned_runtime_config(item, labels[index_value], true) IS DISTINCT FROM true
        THEN
            RETURN false;
        END IF;
        index_value := index_value + 1;
    END LOOP;

    FOR credential_value IN
        SELECT value #>> '{}' FROM jsonb_array_elements(candidate->'llmCredentialIds') AS value
    LOOP
        IF credential_value IS NULL
           OR credential_value !~ '^[a-z][a-z0-9_-]*$'
           OR length(credential_value) > 128
           OR credential_value <= previous_value
        THEN
            RETURN false;
        END IF;
        previous_value := credential_value;
    END LOOP;

    previous_value := '';
    FOR credential_value IN
        SELECT value #>> '{}' FROM jsonb_array_elements(candidate->'runtimeCredentialIds') AS value
    LOOP
        IF credential_value IS NULL
           OR credential_value !~ '^[a-z][a-z0-9_-]*$'
           OR length(credential_value) > 128
           OR credential_value <= previous_value
        THEN
            RETURN false;
        END IF;
        previous_value := credential_value;
    END LOOP;
    RETURN true;
EXCEPTION WHEN OTHERS THEN
    RETURN false;
END;
$_$;

CREATE FUNCTION contractor_valid_runtime_agent_labels(candidate text[]) RETURNS boolean
    LANGUAGE sql IMMUTABLE STRICT
    AS $_$
    SELECT cardinality(candidate) <= 32
       AND candidate = ARRAY(
           SELECT value
           FROM unnest(candidate) AS value
           ORDER BY value COLLATE "C"
       )
       AND NOT EXISTS (
           SELECT 1
           FROM unnest(candidate) AS value
           WHERE value !~ '^[a-z][a-z0-9_-]*$'
              OR length(value) > 63
              OR value = 'default'
       )
       AND cardinality(candidate) = (
           SELECT count(DISTINCT value) FROM unnest(candidate) AS value
       )
$_$;

CREATE FUNCTION contractor_valid_stage_result(payload jsonb) RETURNS boolean
    LANGUAGE sql IMMUTABLE
    AS $$
    SELECT COALESCE(
        jsonb_typeof(payload) = 'object'
        AND jsonb_typeof(payload->'apiVersion') = 'string'
        AND btrim(payload->>'apiVersion') <> ''
        AND payload->>'outcome' IN ('succeeded', 'failed')
        AND jsonb_typeof(payload->'summary') = 'string'
        AND btrim(payload->>'summary') <> ''
        AND jsonb_typeof(payload->'artifacts') = 'object'
        AND (
            (payload->>'outcome' = 'succeeded'
                AND (NOT (payload ? 'error') OR payload->'error' = 'null'::jsonb))
            OR
            (payload->>'outcome' = 'failed'
                AND jsonb_typeof(payload->'error') = 'object'
                AND jsonb_typeof(payload->'error'->'code') = 'string'
                AND btrim(payload->'error'->>'code') <> ''
                AND jsonb_typeof(payload->'error'->'message') = 'string'
                AND btrim(payload->'error'->>'message') <> ''
                AND jsonb_typeof(payload->'error'->'retryable') = 'boolean')
        ),
        false
    )
$$;

CREATE FUNCTION contractor_valid_stage_termination(payload jsonb) RETURNS boolean
    LANGUAGE sql IMMUTABLE
    AS $$
    SELECT COALESCE(
        jsonb_typeof(payload) = 'object'
        AND jsonb_typeof(payload->'outcome') = 'string'
        AND payload->>'outcome' IN ('cancelled', 'interrupted')
        AND jsonb_typeof(payload->'code') = 'string'
        AND btrim(payload->>'code') <> ''
        AND jsonb_typeof(payload->'message') = 'string'
        AND btrim(payload->>'message') <> ''
        AND jsonb_typeof(payload->'retryable') = 'boolean'
        AND jsonb_typeof(payload->'phase') = 'string'
        AND payload->>'phase' IN ('preparing', 'running')
        AND jsonb_typeof(payload->'occurredAt') = 'string'
        AND btrim(payload->>'occurredAt') <> '',
        false
    )
$$;

CREATE FUNCTION contractor_validate_audit_preparation_execution() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    audit audits%ROWTYPE;
    binding jsonb;
    mapping record;
    expected jsonb;
    output jsonb;
    namespace text;
BEGIN
    IF NEW.role <> 'prepare' AND (TG_OP = 'INSERT' OR OLD.role <> 'prepare') THEN RETURN NEW; END IF;
    IF TG_OP = 'UPDATE' THEN
        IF ROW(NEW.execution_id, NEW.audit_id, NEW.round_id, NEW.role, NEW.workflow_role,
               NEW.role_attempt, NEW.manifest_ref, NEW.manifest_digest, NEW.submission_key,
               NEW.request_digest, NEW.preparation_snapshot)
           IS DISTINCT FROM ROW(OLD.execution_id, OLD.audit_id, OLD.round_id, OLD.role, OLD.workflow_role,
               OLD.role_attempt, OLD.manifest_ref, OLD.manifest_digest, OLD.submission_key,
               OLD.request_digest, OLD.preparation_snapshot)
        THEN RAISE EXCEPTION 'Audit preparation execution identity is immutable' USING ERRCODE = '23514'; END IF;
        IF OLD.collection_receipt_id IS NOT NULL AND NEW.preparation_outputs IS DISTINCT FROM OLD.preparation_outputs THEN
            RAISE EXCEPTION 'Audit preparation outputs are immutable' USING ERRCODE = '23514';
        END IF;
        -- Retained data must remain readable after its source Run is deleted.
        IF OLD.collection_receipt_id IS NOT NULL OR NEW.collection_receipt_id IS NULL THEN RETURN NEW; END IF;
    END IF;

    SELECT * INTO STRICT audit FROM audits WHERE audit_id = NEW.audit_id FOR UPDATE;
    binding := audit.profile_snapshot #> ARRAY['workflows', NEW.workflow_role];
    IF binding->>'kind' IS DISTINCT FROM 'prepare' THEN
        RAISE EXCEPTION 'Audit preparation role is invalid' USING ERRCODE = '23514';
    END IF;

    IF TG_OP = 'INSERT' THEN
        IF audit.phase <> 'preparing' OR audit.current_round_id IS NOT NULL OR audit.state <> 'active'
           OR audit.dispatch_state <> 'open' OR NEW.role_attempt > (binding->>'maxRunAttempts')::integer
           OR NEW.role_attempt <> COALESCE((SELECT max(role_attempt) + 1 FROM audit_executions
               WHERE audit_id = NEW.audit_id AND role = 'prepare' AND workflow_role = NEW.workflow_role), 1)
           OR EXISTS (SELECT 1 FROM audit_executions WHERE audit_id = NEW.audit_id AND role = 'prepare'
               AND workflow_role = NEW.workflow_role AND (state <> 'collected' OR collection_disposition = 'accepted-result'))
        THEN RAISE EXCEPTION 'Audit preparation attempt is not admissible' USING ERRCODE = '23514'; END IF;
        IF EXISTS (SELECT 1 FROM jsonb_object_keys(NEW.preparation_snapshot->'inputs') AS name
                   WHERE NOT (binding->'inputs') ? name)
           OR EXISTS (SELECT 1 FROM jsonb_object_keys(NEW.preparation_snapshot->'parameters') AS name
                   WHERE NOT (binding->'parameters') ? name)
        THEN RAISE EXCEPTION 'Audit preparation submission has undeclared mappings' USING ERRCODE = '23514'; END IF;
        FOR mapping IN SELECT * FROM jsonb_each(binding->'inputs') LOOP
            expected := NULL;
            IF mapping.value->>'source' = 'audit-input' THEN
                expected := audit.baseline_snapshot #> ARRAY['inputs', mapping.value->>'name'];
            ELSIF mapping.value->>'source' = 'prepare-output' THEN
                SELECT accepted->'retained' INTO expected
                  FROM audit_executions AS producer CROSS JOIN LATERAL jsonb_array_elements(producer.preparation_outputs) AS accepted
                 WHERE producer.audit_id = NEW.audit_id AND producer.workflow_role = mapping.value->>'role'
                   AND producer.role = 'prepare' AND producer.collection_disposition = 'accepted-result'
                   AND accepted->>'logicalName' = mapping.value->>'name';
                IF expected IS NULL THEN RAISE EXCEPTION 'Audit preparation dependency is not accepted' USING ERRCODE = '23514'; END IF;
            ELSE RAISE EXCEPTION 'Audit preparation input source is invalid' USING ERRCODE = '23514'; END IF;
            IF NEW.preparation_snapshot #> ARRAY['inputs', mapping.key] IS DISTINCT FROM expected
               OR (expected IS NULL AND binding #>> ARRAY['workflow', 'inputs', mapping.key, 'required'] = 'true')
            THEN RAISE EXCEPTION 'Audit preparation input is not pinned exactly' USING ERRCODE = '23514'; END IF;
        END LOOP;
        FOR mapping IN SELECT * FROM jsonb_each(binding->'parameters') LOOP
            IF mapping.value->>'source' = 'literal' THEN expected := to_jsonb(COALESCE(mapping.value->>'value', ''));
            ELSIF mapping.value->>'source' = 'scope-field' THEN expected := audit.baseline_snapshot #> ARRAY['scope', mapping.value->>'name'];
            ELSE RAISE EXCEPTION 'Audit preparation parameter source is invalid' USING ERRCODE = '23514'; END IF;
            IF NEW.preparation_snapshot #> ARRAY['parameters', mapping.key] IS DISTINCT FROM expected THEN
                RAISE EXCEPTION 'Audit preparation parameter is not pinned exactly' USING ERRCODE = '23514';
            END IF;
        END LOOP;
        RETURN NEW;
    END IF;

    IF NEW.collection_disposition <> 'accepted-result' THEN RETURN NEW; END IF;
    IF jsonb_array_length(NEW.preparation_outputs) <> (SELECT count(*) FROM jsonb_each(binding->'outputs'))
       OR jsonb_array_length(NEW.preparation_outputs) <> (SELECT count(DISTINCT value->>'logicalName') FROM jsonb_array_elements(NEW.preparation_outputs))
    THEN RAISE EXCEPTION 'Audit preparation must accept all mapped outputs atomically' USING ERRCODE = '23514'; END IF;
    namespace := 'audit-' || encode(pg_catalog.sha256(
        convert_to('contractor.audit.identity.v1', 'UTF8') || decode('00', 'hex') || convert_to('audit', 'UTF8') || decode('00', 'hex') || convert_to(NEW.audit_id, 'UTF8')
    ), 'hex');
    FOR output IN SELECT * FROM jsonb_array_elements(NEW.preparation_outputs) LOOP
        IF binding #>> ARRAY['outputs', output->>'logicalName'] IS DISTINCT FROM output->>'workflowOutput'
           OR NOT COALESCE((binding #> ARRAY['workflow', 'outputs', output->>'workflowOutput', 'mediaTypes']) ?| ARRAY[output #>> '{source,mediaType}', '*/*'], false)
           OR output #>> '{source,ref,namespace}' IS DISTINCT FROM 'outputs'
           OR output #>> '{source,ref,name}' IS DISTINCT FROM output->>'workflowOutput'
           OR output #>> '{retained,ref,namespace}' IS DISTINCT FROM namespace
           OR output #>> '{source,digest}' IS DISTINCT FROM output #>> '{retained,digest}'
           OR NOT EXISTS (
               SELECT 1 FROM artifact_binding_revisions AS target
               JOIN artifact_bindings AS frozen USING (scope_kind, scope_id, namespace, name)
               JOIN artifact_versions AS version ON version.version_id = target.version_id
               JOIN artifact_blobs AS blob ON blob.sha256 = version.blob_sha256
               JOIN artifact_lineage AS lineage ON lineage.target_scope_kind = target.scope_kind
                 AND lineage.target_scope_id = target.scope_id AND lineage.target_namespace = target.namespace
                 AND lineage.target_name = target.name AND lineage.target_revision = target.revision
               JOIN artifact_binding_revisions AS source ON source.scope_kind = lineage.source_scope_kind
                 AND source.scope_id = lineage.source_scope_id AND source.namespace = lineage.source_namespace
                 AND source.name = lineage.source_name AND source.revision = lineage.source_revision
               JOIN artifact_bindings AS source_binding ON source_binding.scope_kind = source.scope_kind
                 AND source_binding.scope_id = source.scope_id AND source_binding.namespace = source.namespace AND source_binding.name = source.name
               WHERE target.scope_kind = 'project' AND target.scope_id = audit.project_id
                 AND target.namespace = output #>> '{retained,ref,namespace}' AND target.name = output #>> '{retained,ref,name}'
                 AND target.revision = output #>> '{retained,ref,revision}' AND frozen.frozen AND frozen.current_revision = target.revision
                 AND source.version_id = target.version_id AND source.scope_kind = 'run' AND source.scope_id = NEW.run_id
                 AND source.namespace = 'outputs' AND source.name = output->>'workflowOutput'
                 AND source.revision = output #>> '{source,ref,revision}' AND source_binding.frozen AND source_binding.current_revision = source.revision
                 AND lineage.lineage_kind = 'audit_import'
                 AND output #>> '{source,digest}' = 'sha256:' || encode(blob.sha256, 'hex')
                 AND version.media_type = output #>> '{source,mediaType}' AND version.media_type = output #>> '{retained,mediaType}'
                 AND blob.size_bytes = COALESCE((output #>> '{source,sizeBytes}')::bigint, 0)
                 AND blob.size_bytes = COALESCE((output #>> '{retained,sizeBytes}')::bigint, 0)
           ) OR NOT EXISTS (
               SELECT 1 FROM jsonb_array_elements(NEW.collection_retained_refs) AS link
                WHERE link->>'logicalKey' = 'prepare:' || NEW.execution_id || ':' || (output->>'logicalName')
                  AND link->'artifact' = output->'retained'
           )
        THEN RAISE EXCEPTION 'Audit preparation output has no exact frozen retention and lineage' USING ERRCODE = '23514'; END IF;
    END LOOP;
    IF NOT EXISTS (SELECT 1 FROM jsonb_array_elements(NEW.preparation_outputs) AS candidate
                    WHERE candidate #> '{source,ref}' = NEW.collection_source_output_ref
                      AND candidate #>> '{source,digest}' = NEW.collection_source_output_digest)
    THEN RAISE EXCEPTION 'Audit preparation receipt source is not a mapped output' USING ERRCODE = '23514'; END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_validate_manual_stage_execution() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        IF NEW.resume_source_execution_id IS DISTINCT FROM OLD.resume_source_execution_id THEN
            RAISE EXCEPTION 'manual continuation identity is immutable' USING ERRCODE = '23514';
        END IF;
        RETURN NEW;
    END IF;
    IF NEW.resume_source_execution_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM stage_executions AS source
        WHERE source.stage_execution_id = NEW.resume_source_execution_id
          AND source.run_id = NEW.run_id AND source.stage_name = NEW.stage_name
          AND source.attempt + 1 = NEW.attempt
          AND source.state IN ('failed', 'interrupted')
          AND source.execution_config_variant = NEW.execution_config_variant
          AND source.escalation_ordinal IS NOT DISTINCT FROM NEW.escalation_ordinal
          AND source.stage_spec_schema_version = NEW.stage_spec_schema_version
          AND source.stage_spec_snapshot = NEW.stage_spec_snapshot
          AND source.stage_context_schema_version = NEW.stage_context_schema_version
          AND source.stage_context_snapshot = NEW.stage_context_snapshot
    ) THEN
        RAISE EXCEPTION 'manual continuation must copy its terminal source' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE FUNCTION contractor_validate_run_stage_resumption() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM stage_executions AS target
        WHERE target.stage_execution_id = NEW.target_execution_id
          AND target.run_id = NEW.run_id
          AND target.resume_source_execution_id = NEW.source_execution_id
    ) THEN
        RAISE EXCEPTION 'manual continuation receipt must match its target' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;


-- Workflow Runs: Runs, Stages, scheduler claims, events, reports and metrics

CREATE TABLE allocation_execution_reports (
    report_id text NOT NULL,
    stage_execution_id text NOT NULL,
    allocation_id text NOT NULL,
    logical_agent_name text NOT NULL,
    report_schema_version text NOT NULL,
    report jsonb NOT NULL,
    received_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    expires_at timestamp with time zone DEFAULT (clock_timestamp() + '30 days'::interval) NOT NULL,
    CONSTRAINT allocation_execution_reports_check CHECK (((jsonb_typeof(report) = 'object'::text) AND ((report ->> 'reportId'::text) = report_id) AND ((report ->> 'allocationId'::text) = allocation_id) AND (jsonb_typeof((report -> 'worker'::text)) = 'object'::text) AND (jsonb_typeof((report -> 'runtime'::text)) = 'object'::text))),
    CONSTRAINT allocation_execution_reports_report_id_check CHECK ((btrim(report_id) <> ''::text)),
    CONSTRAINT allocation_execution_reports_report_schema_version_check CHECK ((btrim(report_schema_version) <> ''::text))
);

ALTER TABLE allocation_execution_reports
    ADD CONSTRAINT allocation_execution_reports_pkey PRIMARY KEY (report_id);

CREATE INDEX allocation_execution_reports_allocation_idx ON allocation_execution_reports USING btree (allocation_id, received_at DESC);

CREATE UNIQUE INDEX allocation_execution_reports_complete_identity ON allocation_execution_reports USING btree (stage_execution_id, allocation_id, logical_agent_name) WHERE ((((report -> 'worker'::text) ->> 'complete'::text))::boolean AND (((report -> 'runtime'::text) ->> 'complete'::text))::boolean);

CREATE INDEX allocation_execution_reports_expiry_idx ON allocation_execution_reports USING btree (expires_at, stage_execution_id);

CREATE INDEX allocation_execution_reports_stage_idx ON allocation_execution_reports USING btree (stage_execution_id, logical_agent_name);

CREATE TRIGGER allocation_execution_reports_immutable BEFORE UPDATE ON allocation_execution_reports FOR EACH ROW EXECUTE FUNCTION contractor_protect_telemetry_report_immutable();

CREATE TABLE performance_minutes (
    server_generation text NOT NULL,
    minute_start timestamp with time zone NOT NULL,
    schema_version smallint NOT NULL,
    coverage_seconds double precision NOT NULL,
    payload bytea NOT NULL,
    expires_at timestamp with time zone NOT NULL,
    CONSTRAINT performance_minutes_check CHECK ((expires_at = (minute_start + '168:00:00'::interval))),
    CONSTRAINT performance_minutes_coverage_seconds_check CHECK (((coverage_seconds >= (0)::double precision) AND (coverage_seconds <= (60)::double precision))),
    CONSTRAINT performance_minutes_minute_start_check CHECK ((minute_start = date_trunc('minute'::text, minute_start))),
    CONSTRAINT performance_minutes_payload_check CHECK (((octet_length(payload) >= 1) AND (octet_length(payload) <= 32768))),
    CONSTRAINT performance_minutes_schema_version_check CHECK ((schema_version = 1)),
    CONSTRAINT performance_minutes_server_generation_check CHECK (((length(server_generation) >= 1) AND (length(server_generation) <= 128)))
);

ALTER TABLE performance_minutes
    ADD CONSTRAINT performance_minutes_pkey PRIMARY KEY (server_generation, minute_start);

CREATE INDEX performance_minutes_expiry_idx ON performance_minutes USING btree (expires_at, server_generation, minute_start);

CREATE INDEX performance_minutes_range_idx ON performance_minutes USING btree (minute_start, server_generation);

CREATE TABLE planner_execution_reports (
    report_id text NOT NULL,
    stage_execution_id text NOT NULL,
    session_id text NOT NULL,
    invocation_id text NOT NULL,
    started_at timestamp with time zone NOT NULL,
    finished_at timestamp with time zone NOT NULL,
    report_schema_version text NOT NULL,
    report jsonb NOT NULL,
    received_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    expires_at timestamp with time zone DEFAULT (clock_timestamp() + '30 days'::interval) NOT NULL,
    CONSTRAINT planner_execution_reports_check CHECK ((finished_at >= started_at)),
    CONSTRAINT planner_execution_reports_check1 CHECK (((jsonb_typeof(report) = 'object'::text) AND ((report ->> 'reportId'::text) = report_id))),
    CONSTRAINT planner_execution_reports_invocation_id_check CHECK ((btrim(invocation_id) <> ''::text)),
    CONSTRAINT planner_execution_reports_report_id_check CHECK ((btrim(report_id) <> ''::text)),
    CONSTRAINT planner_execution_reports_report_schema_version_check CHECK ((btrim(report_schema_version) <> ''::text))
);

ALTER TABLE planner_execution_reports
    ADD CONSTRAINT planner_execution_reports_pkey PRIMARY KEY (report_id);

CREATE UNIQUE INDEX planner_execution_reports_complete_identity ON planner_execution_reports USING btree (stage_execution_id, session_id) WHERE ((report ->> 'complete'::text))::boolean;

CREATE INDEX planner_execution_reports_expiry_idx ON planner_execution_reports USING btree (expires_at, stage_execution_id);

CREATE INDEX planner_execution_reports_session_idx ON planner_execution_reports USING btree (session_id);

CREATE INDEX planner_execution_reports_stage_idx ON planner_execution_reports USING btree (stage_execution_id, received_at DESC);

CREATE TRIGGER planner_execution_reports_immutable BEFORE UPDATE ON planner_execution_reports FOR EACH ROW EXECUTE FUNCTION contractor_protect_telemetry_report_immutable();

CREATE TABLE planner_sessions (
    session_id text NOT NULL,
    stage_execution_id text NOT NULL,
    invocation_id text NOT NULL,
    state_schema_version text NOT NULL,
    state jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    next_event_sequence bigint DEFAULT 1 NOT NULL,
    CONSTRAINT planner_session_next_event_sequence_positive CHECK ((next_event_sequence > 0)),
    CONSTRAINT planner_session_state_size CHECK ((octet_length((state)::text) <= 2097152)),
    CONSTRAINT planner_sessions_invocation_id_check CHECK ((btrim(invocation_id) <> ''::text)),
    CONSTRAINT planner_sessions_session_id_check CHECK ((btrim(session_id) <> ''::text)),
    CONSTRAINT planner_sessions_state_check CHECK ((jsonb_typeof(state) = 'object'::text)),
    CONSTRAINT planner_sessions_state_schema_version_check CHECK ((btrim(state_schema_version) <> ''::text))
);

ALTER TABLE planner_sessions
    ADD CONSTRAINT planner_sessions_invocation_id_key UNIQUE (invocation_id);

ALTER TABLE planner_sessions
    ADD CONSTRAINT planner_sessions_pkey PRIMARY KEY (session_id);

ALTER TABLE planner_sessions
    ADD CONSTRAINT planner_sessions_stage_execution_id_key UNIQUE (stage_execution_id);

ALTER TABLE planner_sessions
    ADD CONSTRAINT planner_sessions_stage_execution_id_session_id_invocation_i_key UNIQUE (stage_execution_id, session_id, invocation_id);

CREATE TABLE run_stage_resumptions (
    source_execution_id text NOT NULL,
    run_id text NOT NULL,
    target_execution_id text NOT NULL,
    requested_by text NOT NULL,
    requested_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    transaction_id xid8 DEFAULT pg_current_xact_id() NOT NULL
);

ALTER TABLE run_stage_resumptions
    ADD CONSTRAINT run_stage_resumption_target_source_unique UNIQUE (target_execution_id, source_execution_id);

ALTER TABLE run_stage_resumptions
    ADD CONSTRAINT run_stage_resumptions_pkey PRIMARY KEY (source_execution_id);

ALTER TABLE run_stage_resumptions
    ADD CONSTRAINT run_stage_resumptions_target_execution_id_key UNIQUE (target_execution_id);

CREATE INDEX run_stage_resumptions_run_idx ON run_stage_resumptions USING btree (run_id);

CREATE TRIGGER run_stage_resumptions_immutable BEFORE UPDATE ON run_stage_resumptions FOR EACH ROW EXECUTE FUNCTION contractor_protect_run_stage_resumption();

CREATE TRIGGER run_stage_resumptions_validate BEFORE INSERT ON run_stage_resumptions FOR EACH ROW EXECUTE FUNCTION contractor_validate_run_stage_resumption();

CREATE TABLE stage_allocations (
    allocation_id text NOT NULL,
    stage_execution_id text NOT NULL,
    logical_agent_name text NOT NULL,
    namespace text NOT NULL,
    agent_template_ref jsonb NOT NULL,
    worker_runtime_ref jsonb NOT NULL,
    runtime_agent_instance_id text NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    release_attempted_at timestamp with time zone,
    release_completed_at timestamp with time zone,
    runtime_agent_id text,
    runtime_agent_label_revision numeric(20,0),
    runtime_configuration_schema_version text,
    runtime_configuration jsonb,
    performance_collection_policy text,
    completion_contract jsonb,
    CONSTRAINT stage_allocation_performance_collection_policy CHECK (((performance_collection_policy IS NULL) OR (performance_collection_policy = ANY (ARRAY['requested'::text, 'disabled'::text, 'unsupported'::text])))),
    CONSTRAINT stage_allocation_release_timestamps CHECK (((release_completed_at IS NULL) OR ((release_attempted_at IS NOT NULL) AND (release_completed_at >= release_attempted_at)))),
    CONSTRAINT stage_allocation_runtime_configuration_shape CHECK ((((runtime_agent_id IS NULL) AND (runtime_agent_label_revision IS NULL) AND (runtime_configuration_schema_version IS NULL) AND (runtime_configuration IS NULL)) OR ((runtime_agent_id ~ '^[0-9a-f]{64}$'::text) AND (runtime_agent_label_revision >= (1)::numeric) AND (runtime_agent_label_revision <= '18446744073709551615'::numeric) AND (runtime_configuration_schema_version = 'contractor.runtime-config-provenance/v2'::text) AND contractor_valid_allocation_runtime_configuration(runtime_configuration)))),
    CONSTRAINT stage_allocations_agent_template_ref_check CHECK ((jsonb_typeof(agent_template_ref) = 'object'::text)),
    CONSTRAINT stage_allocations_allocation_id_check CHECK ((btrim(allocation_id) <> ''::text)),
    CONSTRAINT stage_allocations_completion_contract_check CHECK (((completion_contract IS NULL) OR ((jsonb_typeof(completion_contract) = 'object'::text) AND ((completion_contract ->> 'kind'::text) = 'audit-check-results@1'::text)))),
    CONSTRAINT stage_allocations_logical_agent_name_check CHECK ((btrim(logical_agent_name) <> ''::text)),
    CONSTRAINT stage_allocations_namespace_check CHECK (((btrim(namespace) <> ''::text) AND (POSITION(('/'::text) IN (namespace)) = 0))),
    CONSTRAINT stage_allocations_runtime_agent_instance_id_check CHECK ((btrim(runtime_agent_instance_id) <> ''::text)),
    CONSTRAINT stage_allocations_worker_runtime_ref_check CHECK ((jsonb_typeof(worker_runtime_ref) = 'object'::text))
);

ALTER TABLE stage_allocations
    ADD CONSTRAINT stage_allocation_report_identity UNIQUE (stage_execution_id, allocation_id, logical_agent_name);

ALTER TABLE stage_allocations
    ADD CONSTRAINT stage_allocations_pkey PRIMARY KEY (allocation_id);

ALTER TABLE stage_allocations
    ADD CONSTRAINT stage_allocations_stage_execution_id_logical_agent_name_key UNIQUE (stage_execution_id, logical_agent_name);

CREATE INDEX stage_allocations_live_llm_credential_idx ON stage_allocations USING btree (((runtime_configuration #>> '{provenance,llmCredential,credentialId}'::text[]))) WHERE ((release_completed_at IS NULL) AND (runtime_configuration IS NOT NULL));

CREATE INDEX stage_allocations_live_runtime_agent_idx ON stage_allocations USING btree (runtime_agent_id, allocation_id) WHERE ((runtime_agent_id IS NOT NULL) AND (release_completed_at IS NULL));

CREATE INDEX stage_allocations_live_runtime_credentials_idx ON stage_allocations USING gin ((((runtime_configuration -> 'provenance'::text) -> 'runtimeCredentialRefs'::text))) WHERE ((release_completed_at IS NULL) AND (runtime_configuration IS NOT NULL));

CREATE INDEX stage_allocations_pending_release_idx ON stage_allocations USING btree (release_attempted_at, stage_execution_id, allocation_id) WHERE (release_completed_at IS NULL);

CREATE TRIGGER stage_allocations_protect_identity BEFORE UPDATE ON stage_allocations FOR EACH ROW EXECUTE FUNCTION contractor_protect_stage_allocation_identity();

CREATE TRIGGER stage_allocations_protect_performance_policy BEFORE UPDATE ON stage_allocations FOR EACH ROW EXECUTE FUNCTION contractor_protect_stage_allocation_performance_policy();

CREATE TRIGGER stage_allocations_require_completion BEFORE INSERT OR UPDATE ON stage_allocations FOR EACH ROW EXECUTE FUNCTION contractor_require_allocation_completion();

CREATE TABLE stage_executions (
    stage_execution_id text NOT NULL,
    run_id text NOT NULL,
    stage_name text NOT NULL,
    attempt integer NOT NULL,
    previous_execution_id text,
    stage_spec_schema_version text NOT NULL,
    stage_spec_snapshot jsonb NOT NULL,
    stage_context_schema_version text NOT NULL,
    stage_context_snapshot jsonb NOT NULL,
    state text NOT NULL,
    state_reason_code text NOT NULL,
    state_reason_message text DEFAULT ''::text NOT NULL,
    planner_session_id text,
    planner_invocation_id text,
    candidate_result_schema_version text,
    candidate_stage_result jsonb,
    accepted_result_schema_version text,
    accepted_stage_result jsonb,
    termination_schema_version text,
    stage_termination jsonb,
    finalization_id text,
    finalization_deadline timestamp with time zone,
    abort_id text,
    abort_deadline timestamp with time zone,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    planner_started_at timestamp with time zone,
    terminal_at timestamp with time zone,
    execution_config_variant text DEFAULT 'base'::text NOT NULL,
    escalation_ordinal integer,
    resume_source_execution_id text,
    admitted_at timestamp with time zone,
    CONSTRAINT stage_accepted_shape CHECK ((((accepted_result_schema_version IS NULL) = (accepted_stage_result IS NULL)) AND ((accepted_stage_result IS NULL) OR ((btrim(accepted_result_schema_version) <> ''::text) AND contractor_valid_stage_result(accepted_stage_result))))),
    CONSTRAINT stage_attempt_lineage_shape CHECK ((((attempt = 1) AND (previous_execution_id IS NULL)) OR ((attempt > 1) AND (previous_execution_id IS NOT NULL)))),
    CONSTRAINT stage_candidate_shape CHECK ((((candidate_result_schema_version IS NULL) = (candidate_stage_result IS NULL)) AND ((candidate_stage_result IS NULL) OR ((btrim(candidate_result_schema_version) <> ''::text) AND contractor_valid_stage_result(candidate_stage_result))))),
    CONSTRAINT stage_execution_config_variant_shape CHECK ((((execution_config_variant = 'base'::text) AND (escalation_ordinal IS NULL)) OR ((execution_config_variant = ANY (ARRAY['failed_escalation'::text, 'interrupted_escalation'::text])) AND (escalation_ordinal IS NOT NULL) AND (escalation_ordinal > 0)))),
    CONSTRAINT stage_execution_resume_previous CHECK (((resume_source_execution_id IS NULL) OR ((previous_execution_id IS NOT NULL) AND (resume_source_execution_id = previous_execution_id)))),
    CONSTRAINT stage_execution_state_shape CHECK ((((state = 'preparing'::text) AND (planner_session_id IS NULL) AND (candidate_stage_result IS NULL) AND (accepted_stage_result IS NULL) AND (stage_termination IS NULL) AND (finalization_id IS NULL) AND (finalization_deadline IS NULL) AND (abort_id IS NULL) AND (abort_deadline IS NULL) AND (terminal_at IS NULL)) OR ((state = 'running'::text) AND (planner_session_id IS NOT NULL) AND (candidate_stage_result IS NULL) AND (accepted_stage_result IS NULL) AND (stage_termination IS NULL) AND (finalization_id IS NULL) AND (finalization_deadline IS NULL) AND (abort_id IS NULL) AND (abort_deadline IS NULL) AND (planner_started_at IS NOT NULL) AND (terminal_at IS NULL)) OR ((state = 'finalizing'::text) AND (planner_session_id IS NOT NULL) AND (candidate_stage_result IS NOT NULL) AND (accepted_stage_result IS NULL) AND (stage_termination IS NULL) AND (finalization_id IS NOT NULL) AND (btrim(finalization_id) <> ''::text) AND (finalization_deadline IS NOT NULL) AND (abort_id IS NULL) AND (abort_deadline IS NULL) AND (planner_started_at IS NOT NULL) AND (terminal_at IS NULL)) OR ((state = 'aborting'::text) AND (candidate_stage_result IS NULL) AND (accepted_stage_result IS NULL) AND (stage_termination IS NOT NULL) AND (finalization_id IS NULL) AND (finalization_deadline IS NULL) AND (abort_id IS NOT NULL) AND (btrim(abort_id) <> ''::text) AND (abort_deadline IS NOT NULL) AND (terminal_at IS NULL) AND ((((stage_termination ->> 'phase'::text) = 'preparing'::text) AND (planner_session_id IS NULL)) OR (((stage_termination ->> 'phase'::text) = 'running'::text) AND (planner_session_id IS NOT NULL) AND (planner_started_at IS NOT NULL)))) OR ((state = ANY (ARRAY['succeeded'::text, 'failed'::text])) AND (planner_session_id IS NOT NULL) AND (candidate_stage_result IS NOT NULL) AND (accepted_stage_result = candidate_stage_result) AND (accepted_result_schema_version = candidate_result_schema_version) AND ((accepted_stage_result ->> 'outcome'::text) = state) AND (stage_termination IS NULL) AND (finalization_id IS NOT NULL) AND (btrim(finalization_id) <> ''::text) AND (finalization_deadline IS NOT NULL) AND (abort_id IS NULL) AND (abort_deadline IS NULL) AND (planner_started_at IS NOT NULL) AND (terminal_at IS NOT NULL)) OR ((state = ANY (ARRAY['interrupted'::text, 'cancelled'::text])) AND (candidate_stage_result IS NULL) AND (accepted_stage_result IS NULL) AND (stage_termination IS NOT NULL) AND ((stage_termination ->> 'outcome'::text) = state) AND (finalization_id IS NULL) AND (finalization_deadline IS NULL) AND (abort_id IS NOT NULL) AND (btrim(abort_id) <> ''::text) AND (abort_deadline IS NOT NULL) AND (terminal_at IS NOT NULL) AND ((((stage_termination ->> 'phase'::text) = 'preparing'::text) AND (planner_session_id IS NULL)) OR (((stage_termination ->> 'phase'::text) = 'running'::text) AND (planner_session_id IS NOT NULL) AND (planner_started_at IS NOT NULL)))))),
    CONSTRAINT stage_executions_attempt_check CHECK ((attempt > 0)),
    CONSTRAINT stage_executions_stage_context_schema_version_check CHECK ((btrim(stage_context_schema_version) <> ''::text)),
    CONSTRAINT stage_executions_stage_context_snapshot_check CHECK ((jsonb_typeof(stage_context_snapshot) = 'object'::text)),
    CONSTRAINT stage_executions_stage_execution_id_check CHECK ((btrim(stage_execution_id) <> ''::text)),
    CONSTRAINT stage_executions_stage_name_check CHECK ((btrim(stage_name) <> ''::text)),
    CONSTRAINT stage_executions_stage_spec_schema_version_check CHECK ((btrim(stage_spec_schema_version) <> ''::text)),
    CONSTRAINT stage_executions_stage_spec_snapshot_check CHECK ((jsonb_typeof(stage_spec_snapshot) = 'object'::text)),
    CONSTRAINT stage_executions_state_check CHECK ((state = ANY (ARRAY['preparing'::text, 'running'::text, 'finalizing'::text, 'aborting'::text, 'succeeded'::text, 'failed'::text, 'interrupted'::text, 'cancelled'::text]))),
    CONSTRAINT stage_executions_state_reason_code_check CHECK ((btrim(state_reason_code) <> ''::text)),
    CONSTRAINT stage_planner_ids_shape CHECK (((planner_session_id IS NULL) = (planner_invocation_id IS NULL))),
    CONSTRAINT stage_termination_payload_shape CHECK ((((termination_schema_version IS NULL) = (stage_termination IS NULL)) AND ((stage_termination IS NULL) OR ((btrim(termination_schema_version) <> ''::text) AND contractor_valid_stage_termination(stage_termination)))))
);

ALTER TABLE stage_executions
    ADD CONSTRAINT stage_attempt_identity UNIQUE (run_id, stage_name, attempt);

ALTER TABLE stage_executions
    ADD CONSTRAINT stage_execution_lineage_key UNIQUE (run_id, stage_name, stage_execution_id);

ALTER TABLE stage_executions
    ADD CONSTRAINT stage_execution_run_identity UNIQUE (run_id, stage_execution_id);

ALTER TABLE stage_executions
    ADD CONSTRAINT stage_executions_pkey PRIMARY KEY (stage_execution_id);

CREATE UNIQUE INDEX stage_execution_escalation_ordinal_unique ON stage_executions USING btree (run_id, stage_name, execution_config_variant, escalation_ordinal) WHERE ((execution_config_variant <> 'base'::text) AND (resume_source_execution_id IS NULL));

CREATE INDEX stage_executions_recovery_idx ON stage_executions USING btree (state, updated_at) WHERE (state = ANY (ARRAY['preparing'::text, 'running'::text, 'finalizing'::text, 'aborting'::text]));

CREATE INDEX stage_executions_run_created_idx ON stage_executions USING btree (run_id, created_at, stage_execution_id);

CREATE INDEX stage_executions_run_terminal_history_idx ON stage_executions USING btree (run_id, terminal_at DESC, stage_execution_id DESC) WHERE (terminal_at IS NOT NULL);

CREATE INDEX stage_executions_terminal_history_idx ON stage_executions USING btree (terminal_at DESC, stage_execution_id DESC) WHERE (terminal_at IS NOT NULL);

CREATE TRIGGER eval_stages_projection AFTER INSERT OR DELETE OR UPDATE OF state ON stage_executions FOR EACH ROW EXECUTE FUNCTION contractor_eval_metrics_changed();

CREATE TRIGGER stage_executions_capture_lifecycle_event AFTER INSERT OR UPDATE ON stage_executions FOR EACH ROW EXECUTE FUNCTION contractor_capture_stage_execution_lifecycle_event();

CREATE TRIGGER stage_executions_manual_continuation BEFORE INSERT OR UPDATE ON stage_executions FOR EACH ROW EXECUTE FUNCTION contractor_validate_manual_stage_execution();

CREATE TRIGGER stage_executions_protect_immutable BEFORE UPDATE ON stage_executions FOR EACH ROW EXECUTE FUNCTION contractor_protect_stage_execution_immutable();

CREATE TABLE stage_metrics (
    stage_execution_id text NOT NULL,
    metrics_schema_version text NOT NULL,
    metrics jsonb NOT NULL,
    summary jsonb NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    expires_at timestamp with time zone DEFAULT (clock_timestamp() + '30 days'::interval) NOT NULL,
    CONSTRAINT stage_metrics_metrics_check CHECK (((jsonb_typeof(metrics) = 'object'::text) AND (jsonb_typeof((metrics -> 'workers'::text)) = 'object'::text) AND (jsonb_typeof((metrics -> 'runtime'::text)) = 'object'::text))),
    CONSTRAINT stage_metrics_metrics_schema_version_check CHECK ((btrim(metrics_schema_version) <> ''::text)),
    CONSTRAINT stage_metrics_summary_check CHECK ((jsonb_typeof(summary) = 'object'::text))
);

ALTER TABLE stage_metrics
    ADD CONSTRAINT stage_metrics_pkey PRIMARY KEY (stage_execution_id);

CREATE INDEX stage_metrics_expiry_idx ON stage_metrics USING btree (expires_at, stage_execution_id);

CREATE TRIGGER eval_metrics_projection AFTER INSERT OR DELETE OR UPDATE ON stage_metrics FOR EACH ROW EXECUTE FUNCTION contractor_eval_metrics_changed();

CREATE TABLE stage_transition_decisions (
    source_execution_id text NOT NULL,
    run_id text NOT NULL,
    action text NOT NULL,
    target_stage_name text,
    target_execution_id text,
    decided_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    escalation_ordinal integer,
    escalation_exhausted boolean DEFAULT false NOT NULL,
    CONSTRAINT stage_transition_action CHECK ((action = ANY (ARRAY['next'::text, 'retry'::text, 'escalate'::text, 'succeed'::text, 'fail'::text]))),
    CONSTRAINT stage_transition_decision_shape CHECK ((((action = ANY (ARRAY['next'::text, 'retry'::text, 'escalate'::text])) AND (target_stage_name IS NOT NULL) AND (btrim(target_stage_name) <> ''::text) AND (target_execution_id IS NOT NULL) AND (btrim(target_execution_id) <> ''::text)) OR ((action = ANY (ARRAY['succeed'::text, 'fail'::text])) AND (target_stage_name IS NULL) AND (target_execution_id IS NULL)))),
    CONSTRAINT stage_transition_escalation_shape CHECK ((((action = 'escalate'::text) AND (escalation_ordinal IS NOT NULL) AND (escalation_ordinal > 0) AND (escalation_exhausted = false)) OR ((action = ANY (ARRAY['next'::text, 'fail'::text])) AND (escalation_ordinal IS NOT NULL) AND (escalation_ordinal > 0) AND (escalation_exhausted = true)) OR ((action <> 'escalate'::text) AND (escalation_ordinal IS NULL) AND (escalation_exhausted = false))))
);

ALTER TABLE stage_transition_decisions
    ADD CONSTRAINT stage_transition_decisions_pkey PRIMARY KEY (source_execution_id);

ALTER TABLE stage_transition_decisions
    ADD CONSTRAINT stage_transition_target_once UNIQUE (target_execution_id);

CREATE INDEX stage_transition_decisions_run_idx ON stage_transition_decisions USING btree (run_id, decided_at, source_execution_id);

CREATE TRIGGER stage_transition_decisions_protect_immutable BEFORE UPDATE ON stage_transition_decisions FOR EACH ROW EXECUTE FUNCTION contractor_protect_stage_transition_decision_immutable();

CREATE TABLE workflow_run_events (
    run_id text NOT NULL,
    sequence_number bigint NOT NULL,
    event_id text NOT NULL,
    event_schema_version text NOT NULL,
    kind text NOT NULL,
    data jsonb NOT NULL,
    occurred_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT workflow_run_events_data_check CHECK (((jsonb_typeof(data) = 'object'::text) AND (octet_length((data)::text) <= 2097152))),
    CONSTRAINT workflow_run_events_event_id_check CHECK ((btrim(event_id) <> ''::text)),
    CONSTRAINT workflow_run_events_event_schema_version_check CHECK ((btrim(event_schema_version) <> ''::text)),
    CONSTRAINT workflow_run_events_kind_check CHECK ((kind = ANY (ARRAY['planner.started'::text, 'planner.request_recorded'::text, 'planner.activity'::text, 'planner.plan_changed'::text, 'planner.current_changed'::text, 'planner.dispatch_selected'::text, 'planner.dispatch_started'::text, 'planner.dispatch_completed'::text, 'planner.finish_requested'::text, 'planner.completed'::text, 'planner.failed'::text, 'lifecycle.changed'::text]))),
    CONSTRAINT workflow_run_events_sequence_number_check CHECK ((sequence_number > 0))
);

ALTER TABLE workflow_run_events
    ADD CONSTRAINT workflow_run_events_event_id_key UNIQUE (event_id);

ALTER TABLE workflow_run_events
    ADD CONSTRAINT workflow_run_events_pkey PRIMARY KEY (run_id, sequence_number);

CREATE INDEX workflow_run_events_time_idx ON workflow_run_events USING btree (run_id, occurred_at, sequence_number);

CREATE TRIGGER workflow_run_events_notify_after_commit AFTER INSERT ON workflow_run_events FOR EACH ROW EXECUTE FUNCTION contractor_notify_workflow_run_event();

CREATE TRIGGER workflow_run_events_protect_immutable BEFORE UPDATE ON workflow_run_events FOR EACH ROW EXECUTE FUNCTION contractor_protect_workflow_run_event_immutable();

CREATE TABLE workflow_runs (
    run_id text NOT NULL,
    owner_id text NOT NULL,
    workflow_name text NOT NULL,
    workflow_version text NOT NULL,
    workflow_schema_version text NOT NULL,
    workflow_snapshot jsonb NOT NULL,
    parameters jsonb DEFAULT '{}'::jsonb NOT NULL,
    state text NOT NULL,
    state_reason_code text NOT NULL,
    state_reason_message text DEFAULT ''::text NOT NULL,
    scheduler_claim_id text,
    scheduler_claimed_at timestamp with time zone,
    scheduler_claim_expires_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    started_at timestamp with time zone,
    finished_at timestamp with time zone,
    cancellation_schema_version text,
    run_cancellation jsonb,
    request_idempotency_key text,
    request_digest text,
    run_event_generation text DEFAULT ('events-'::text || md5(((random())::text || (clock_timestamp())::text))) NOT NULL,
    next_run_event_sequence bigint DEFAULT 1 NOT NULL,
    runtime_labels text[] NOT NULL,
    runtime_config_snapshot jsonb NOT NULL,
    skill_snapshot jsonb DEFAULT '[]'::jsonb NOT NULL,
    project_id text,
    project_http_target_snapshot jsonb,
    publication_mode text DEFAULT 'ordinary'::text NOT NULL,
    audit_execution_id text,
    audit_submission_key text,
    audit_completion jsonb,
    scheduler_deferred boolean DEFAULT false NOT NULL,
    metadata_labels jsonb DEFAULT '{}'::jsonb NOT NULL,
    CONSTRAINT workflow_run_cancellation_shape CHECK ((((state = ANY (ARRAY['cancelling'::text, 'cancelled'::text])) = ((cancellation_schema_version IS NOT NULL) AND (btrim(cancellation_schema_version) <> ''::text) AND (run_cancellation IS NOT NULL) AND contractor_valid_run_cancellation(run_cancellation))) AND ((state = ANY (ARRAY['cancelling'::text, 'cancelled'::text])) OR ((cancellation_schema_version IS NULL) AND (run_cancellation IS NULL))))),
    CONSTRAINT workflow_run_claim_shape CHECK ((((scheduler_claim_id IS NULL) AND (scheduler_claimed_at IS NULL) AND (scheduler_claim_expires_at IS NULL)) OR ((scheduler_claim_id IS NOT NULL) AND (btrim(scheduler_claim_id) <> ''::text) AND (scheduler_claimed_at IS NOT NULL) AND (scheduler_claim_expires_at > scheduler_claimed_at)))),
    CONSTRAINT workflow_run_terminal_shape CHECK (((state = ANY (ARRAY['succeeded'::text, 'failed'::text, 'cancelled'::text])) = (finished_at IS NOT NULL))),
    CONSTRAINT workflow_run_terminal_unclaimed CHECK (((state <> ALL (ARRAY['succeeded'::text, 'failed'::text, 'cancelled'::text])) OR (scheduler_claim_id IS NULL))),
    CONSTRAINT workflow_runs_audit_authority_shape CHECK ((((publication_mode = 'ordinary'::text) AND (audit_execution_id IS NULL) AND (audit_submission_key IS NULL)) OR ((publication_mode = 'audit-managed'::text) AND (project_id IS NOT NULL) AND (audit_execution_id IS NOT NULL) AND (audit_submission_key IS NOT NULL)))),
    CONSTRAINT workflow_runs_check CHECK (((audit_completion IS NULL) OR ((publication_mode = 'audit-managed'::text) AND (jsonb_typeof(audit_completion) = 'object'::text) AND ((audit_completion #>> '{contract,kind}'::text[]) = 'audit-check-results@1'::text)))),
    CONSTRAINT workflow_runs_metadata_labels_check CHECK (contractor_valid_run_metadata_labels(metadata_labels)),
    CONSTRAINT workflow_runs_next_run_event_sequence_check CHECK ((next_run_event_sequence > 0)),
    CONSTRAINT workflow_runs_owner_id_check CHECK ((btrim(owner_id) <> ''::text)),
    CONSTRAINT workflow_runs_parameters_check CHECK ((jsonb_typeof(parameters) = 'object'::text)),
    CONSTRAINT workflow_runs_project_http_target_shape CHECK (((project_http_target_snapshot IS NULL) OR ((project_id IS NOT NULL) AND (jsonb_typeof(project_http_target_snapshot) = 'object'::text) AND (jsonb_typeof((project_http_target_snapshot -> 'url'::text)) = 'string'::text) AND (btrim((project_http_target_snapshot ->> 'url'::text)) = (project_http_target_snapshot ->> 'url'::text)) AND ((octet_length((project_http_target_snapshot ->> 'url'::text)) >= 1) AND (octet_length((project_http_target_snapshot ->> 'url'::text)) <= 2048)) AND ((NOT (project_http_target_snapshot ? 'credential'::text)) OR ((jsonb_typeof((project_http_target_snapshot -> 'credential'::text)) = 'object'::text) AND (jsonb_typeof((project_http_target_snapshot #> '{credential,credentialId}'::text[])) = 'string'::text) AND ((project_http_target_snapshot #>> '{credential,kind}'::text[]) = ANY (ARRAY['http-origin-basic@1'::text, 'http-origin-bearer@1'::text]))))))),
    CONSTRAINT workflow_runs_publication_mode_check CHECK ((publication_mode = ANY (ARRAY['ordinary'::text, 'audit-managed'::text]))),
    CONSTRAINT workflow_runs_request_idempotency_shape CHECK ((((request_idempotency_key IS NULL) AND (request_digest IS NULL)) OR ((request_idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'::text) AND (request_digest ~ '^sha256:[0-9a-f]{64}$'::text)))),
    CONSTRAINT workflow_runs_run_event_generation_check CHECK ((btrim(run_event_generation) <> ''::text)),
    CONSTRAINT workflow_runs_run_id_check CHECK ((btrim(run_id) <> ''::text)),
    CONSTRAINT workflow_runs_runtime_config_snapshot_shape CHECK (contractor_valid_run_runtime_snapshot(runtime_config_snapshot, runtime_labels)),
    CONSTRAINT workflow_runs_runtime_labels_check CHECK (contractor_valid_runtime_agent_labels(runtime_labels)),
    CONSTRAINT workflow_runs_skill_snapshot_array CHECK ((jsonb_typeof(skill_snapshot) = 'array'::text)),
    CONSTRAINT workflow_runs_state_check CHECK ((state = ANY (ARRAY['initializing'::text, 'pending'::text, 'running'::text, 'waiting'::text, 'cancelling'::text, 'succeeded'::text, 'failed'::text, 'cancelled'::text]))),
    CONSTRAINT workflow_runs_state_reason_code_check CHECK ((btrim(state_reason_code) <> ''::text)),
    CONSTRAINT workflow_runs_workflow_name_check CHECK ((btrim(workflow_name) <> ''::text)),
    CONSTRAINT workflow_runs_workflow_schema_version_check CHECK ((btrim(workflow_schema_version) <> ''::text)),
    CONSTRAINT workflow_runs_workflow_snapshot_check CHECK ((jsonb_typeof(workflow_snapshot) = 'object'::text)),
    CONSTRAINT workflow_runs_workflow_version_check CHECK ((btrim(workflow_version) <> ''::text))
);

ALTER TABLE workflow_runs
    ADD CONSTRAINT workflow_runs_audit_execution_unique UNIQUE (audit_execution_id);

ALTER TABLE workflow_runs
    ADD CONSTRAINT workflow_runs_audit_submission_unique UNIQUE (audit_submission_key);

ALTER TABLE workflow_runs
    ADD CONSTRAINT workflow_runs_pkey PRIMARY KEY (run_id);

ALTER TABLE workflow_runs
    ADD CONSTRAINT workflow_runs_run_project_key UNIQUE (run_id, project_id);

CREATE INDEX workflow_runs_admission_idx ON workflow_runs USING btree (state, scheduler_claim_expires_at, updated_at, created_at, run_id) WHERE (state = ANY (ARRAY['pending'::text, 'running'::text, 'waiting'::text, 'cancelling'::text]));

CREATE INDEX workflow_runs_metadata_labels_idx ON workflow_runs USING gin (metadata_labels jsonb_path_ops);

CREATE INDEX workflow_runs_owner_created_idx ON workflow_runs USING btree (owner_id, created_at, run_id);

CREATE UNIQUE INDEX workflow_runs_owner_idempotency_key ON workflow_runs USING btree (owner_id, request_idempotency_key) WHERE (request_idempotency_key IS NOT NULL);

CREATE INDEX workflow_runs_owner_nonterminal_queue_idx ON workflow_runs USING btree (owner_id, created_at, run_id) INCLUDE (project_id, workflow_name, workflow_version, state, updated_at, run_event_generation, next_run_event_sequence) WHERE (state = ANY (ARRAY['initializing'::text, 'running'::text, 'cancelling'::text]));

CREATE INDEX workflow_runs_owner_project_created_idx ON workflow_runs USING btree (owner_id, project_id, created_at DESC, run_id DESC) WHERE (project_id IS NOT NULL);

CREATE INDEX workflow_runs_project_http_target_credential_idx ON workflow_runs USING btree (((project_http_target_snapshot #>> '{credential,credentialId}'::text[])), created_at, run_id) WHERE ((state = ANY (ARRAY['initializing'::text, 'running'::text, 'cancelling'::text])) AND (project_http_target_snapshot ? 'credential'::text));

CREATE INDEX workflow_runs_publication_mode_idx ON workflow_runs USING btree (publication_mode, run_id) WHERE (publication_mode = 'audit-managed'::text);

CREATE INDEX workflow_runs_runtime_credentials_idx ON workflow_runs USING gin (((runtime_config_snapshot -> 'runtimeCredentialIds'::text))) WHERE (state = ANY (ARRAY['initializing'::text, 'running'::text, 'cancelling'::text]));

CREATE INDEX workflow_runs_runtime_labels_idx ON workflow_runs USING gin (runtime_labels);

CREATE INDEX workflow_runs_runtime_llm_credentials_idx ON workflow_runs USING gin (((runtime_config_snapshot -> 'llmCredentialIds'::text))) WHERE (state = ANY (ARRAY['initializing'::text, 'running'::text, 'cancelling'::text]));

CREATE INDEX workflow_runs_scheduler_idx ON workflow_runs USING btree (state, scheduler_claim_expires_at, created_at) WHERE (state = ANY (ARRAY['running'::text, 'cancelling'::text]));

CREATE INDEX workflow_runs_skill_initialization_scheduler_idx ON workflow_runs USING btree (scheduler_claim_expires_at, created_at, run_id) WHERE ((state = 'initializing'::text) AND (state_reason_code = 'skill_initialization_pending'::text));

CREATE TRIGGER eval_runs_projection AFTER DELETE OR UPDATE OF state, finished_at ON workflow_runs FOR EACH ROW EXECUTE FUNCTION contractor_eval_execution_changed();

CREATE TRIGGER workflow_runs_capture_lifecycle_event AFTER INSERT OR UPDATE ON workflow_runs FOR EACH ROW EXECUTE FUNCTION contractor_capture_workflow_run_lifecycle_event();

CREATE TRIGGER workflow_runs_eval_tombstone BEFORE DELETE ON workflow_runs FOR EACH ROW EXECUTE FUNCTION contractor_eval_retain_deleted_execution();

CREATE TRIGGER workflow_runs_project_admission BEFORE INSERT ON workflow_runs FOR EACH ROW EXECUTE FUNCTION contractor_guard_workflow_run_project_admission();

CREATE TRIGGER workflow_runs_protect_audit_completion BEFORE UPDATE ON workflow_runs FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_completion();

CREATE TRIGGER workflow_runs_protect_immutable BEFORE UPDATE ON workflow_runs FOR EACH ROW EXECUTE FUNCTION contractor_protect_workflow_run_immutable();

CREATE TRIGGER workflow_runs_protect_skill_snapshot BEFORE UPDATE ON workflow_runs FOR EACH ROW EXECUTE FUNCTION contractor_protect_workflow_run_skill_snapshot();

CREATE CONSTRAINT TRIGGER workflow_runs_require_audit_completion AFTER INSERT OR UPDATE ON workflow_runs DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION contractor_require_audit_completion();

CREATE CONSTRAINT TRIGGER workflow_runs_require_complete_audit_binding AFTER INSERT OR UPDATE ON workflow_runs DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION contractor_require_complete_audit_run_binding();


-- Artifacts: versioned bindings, names, provenance and output publication

CREATE TABLE artifact_binding_revisions (
    scope_kind text NOT NULL,
    scope_id text NOT NULL,
    namespace text NOT NULL,
    name text NOT NULL,
    revision text NOT NULL,
    version_id text NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT artifact_binding_revisions_name_check CHECK (((btrim(name) <> ''::text) AND (POSITION(('/'::text) IN (name)) = 0))),
    CONSTRAINT artifact_binding_revisions_namespace_check CHECK (((btrim(namespace) <> ''::text) AND (POSITION(('/'::text) IN (namespace)) = 0))),
    CONSTRAINT artifact_binding_revisions_revision_check CHECK ((btrim(revision) <> ''::text))
);

ALTER TABLE artifact_binding_revisions
    ADD CONSTRAINT artifact_binding_revisions_pkey PRIMARY KEY (scope_kind, scope_id, namespace, name, revision);

CREATE INDEX artifact_binding_revisions_history_idx ON artifact_binding_revisions USING btree (scope_kind, scope_id, namespace, name, created_at DESC, revision DESC);

CREATE INDEX artifact_binding_revisions_version_idx ON artifact_binding_revisions USING btree (version_id);

CREATE TRIGGER artifact_binding_revisions_immutable BEFORE DELETE OR UPDATE ON artifact_binding_revisions FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_immutable();

CREATE TRIGGER eval_artifact_projection BEFORE DELETE ON artifact_binding_revisions FOR EACH ROW EXECUTE FUNCTION contractor_eval_evidence_deleted();

CREATE TABLE artifact_bindings (
    scope_kind text NOT NULL,
    scope_id text NOT NULL,
    namespace text NOT NULL,
    name text NOT NULL,
    current_revision text NOT NULL,
    frozen boolean DEFAULT false NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT artifact_bindings_current_revision_check CHECK ((btrim(current_revision) <> ''::text)),
    CONSTRAINT artifact_bindings_name_check CHECK (((btrim(name) <> ''::text) AND (POSITION(('/'::text) IN (name)) = 0))),
    CONSTRAINT artifact_bindings_namespace_check CHECK (((btrim(namespace) <> ''::text) AND (POSITION(('/'::text) IN (namespace)) = 0)))
);

ALTER TABLE artifact_bindings
    ADD CONSTRAINT artifact_bindings_pkey PRIMARY KEY (scope_kind, scope_id, namespace, name);

CREATE TRIGGER artifact_bindings_project_admission BEFORE INSERT OR UPDATE ON artifact_bindings FOR EACH ROW EXECUTE FUNCTION contractor_guard_project_artifact_binding_admission();

CREATE TRIGGER artifact_bindings_protect_identity BEFORE UPDATE ON artifact_bindings FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_binding();

CREATE TRIGGER eval_outputs_projection AFTER INSERT OR DELETE OR UPDATE ON artifact_bindings FOR EACH ROW EXECUTE FUNCTION contractor_eval_output_changed();

CREATE TABLE artifact_blob_settings (
    singleton boolean DEFAULT true NOT NULL,
    backend text,
    CONSTRAINT artifact_blob_settings_backend_check CHECK ((backend = ANY (ARRAY['postgresql'::text, 'filesystem'::text]))),
    CONSTRAINT artifact_blob_settings_singleton_check CHECK (singleton)
);

ALTER TABLE artifact_blob_settings
    ADD CONSTRAINT artifact_blob_settings_pkey PRIMARY KEY (singleton);

CREATE TABLE artifact_blobs (
    sha256 bytea NOT NULL,
    payload bytea,
    size_bytes bigint NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    backend text DEFAULT 'postgresql'::text NOT NULL,
    object_key text,
    CONSTRAINT artifact_blobs_sha256_check CHECK ((octet_length(sha256) = 32)),
    CONSTRAINT artifact_blobs_size_check CHECK (((size_bytes >= 0) AND (size_bytes <= 67108864))),
    CONSTRAINT artifact_blobs_storage_check CHECK ((((backend = 'postgresql'::text) AND (object_key IS NULL) AND (payload IS NOT NULL) AND (size_bytes = octet_length(payload)) AND (sha256 = sha256(payload))) OR ((backend = 'filesystem'::text) AND (payload IS NULL) AND (object_key IS NOT NULL) AND (object_key ~ '^[0-9a-f]{2}/[0-9a-f]{32}$'::text))))
);

ALTER TABLE artifact_blobs
    ADD CONSTRAINT artifact_blobs_pkey PRIMARY KEY (sha256);

CREATE UNIQUE INDEX artifact_blobs_object_key_idx ON artifact_blobs USING btree (object_key) WHERE (object_key IS NOT NULL);

CREATE TRIGGER artifact_blob_backend_guard BEFORE INSERT OR UPDATE ON artifact_blobs FOR EACH ROW EXECUTE FUNCTION artifact_blob_backend_guard();

CREATE TRIGGER artifact_blobs_immutable BEFORE DELETE OR UPDATE ON artifact_blobs FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_blob_immutable();

CREATE TABLE artifact_lineage (
    target_scope_kind text NOT NULL,
    target_scope_id text NOT NULL,
    target_namespace text NOT NULL,
    target_name text NOT NULL,
    target_revision text NOT NULL,
    source_scope_kind text NOT NULL,
    source_scope_id text NOT NULL,
    source_namespace text NOT NULL,
    source_name text NOT NULL,
    source_revision text NOT NULL,
    lineage_kind text NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT artifact_lineage_lineage_kind_check CHECK ((lineage_kind = ANY (ARRAY['input_fork'::text, 'output_bind'::text, 'project_output_publish'::text, 'audit_import'::text])))
);

ALTER TABLE artifact_lineage
    ADD CONSTRAINT artifact_lineage_pkey PRIMARY KEY (target_scope_kind, target_scope_id, target_namespace, target_name, target_revision);

CREATE INDEX artifact_lineage_source_idx ON artifact_lineage USING btree (source_scope_kind, source_scope_id, source_namespace, source_name, source_revision);

CREATE TRIGGER artifact_lineage_immutable BEFORE DELETE OR UPDATE ON artifact_lineage FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_immutable();

CREATE TABLE artifact_scopes (
    scope_kind text NOT NULL,
    scope_id text NOT NULL,
    run_id text GENERATED ALWAYS AS (
CASE
    WHEN (scope_kind = 'run'::text) THEN scope_id
    ELSE NULL::text
END) STORED,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    project_id text GENERATED ALWAYS AS (
CASE
    WHEN (scope_kind = 'project'::text) THEN scope_id
    ELSE NULL::text
END) STORED,
    CONSTRAINT artifact_scopes_scope_id_check CHECK ((btrim(scope_id) <> ''::text)),
    CONSTRAINT artifact_scopes_scope_kind_check CHECK ((scope_kind = ANY (ARRAY['user'::text, 'project'::text, 'run'::text])))
);

ALTER TABLE artifact_scopes
    ADD CONSTRAINT artifact_scopes_pkey PRIMARY KEY (scope_kind, scope_id);

CREATE INDEX artifact_scopes_project_idx ON artifact_scopes USING btree (project_id) WHERE (project_id IS NOT NULL);

CREATE INDEX artifact_scopes_run_idx ON artifact_scopes USING btree (run_id) WHERE (run_id IS NOT NULL);

CREATE TRIGGER artifact_scopes_project_admission BEFORE INSERT ON artifact_scopes FOR EACH ROW EXECUTE FUNCTION contractor_guard_project_artifact_scope_admission();

CREATE TABLE artifact_versions (
    version_id text NOT NULL,
    blob_sha256 bytea NOT NULL,
    media_type text NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    git_repository_url text,
    git_requested_ref text,
    git_resolved_commit text,
    git_imported_at timestamp with time zone,
    CONSTRAINT artifact_versions_git_repository_url_check CHECK (((octet_length(git_repository_url) >= 1) AND (octet_length(git_repository_url) <= 4096))),
    CONSTRAINT artifact_versions_git_requested_ref_check CHECK (((octet_length(git_requested_ref) >= 1) AND (octet_length(git_requested_ref) <= 1024))),
    CONSTRAINT artifact_versions_git_resolved_commit_check CHECK ((git_resolved_commit ~ '^[0-9a-f]{40}$'::text)),
    CONSTRAINT artifact_versions_git_source CHECK ((((git_repository_url IS NULL) = (git_resolved_commit IS NULL)) AND ((git_repository_url IS NULL) = (git_imported_at IS NULL)) AND ((git_requested_ref IS NULL) OR (git_repository_url IS NOT NULL)))),
    CONSTRAINT artifact_versions_media_type_check CHECK (((media_type = lower(media_type)) AND (media_type ~ '^[a-z0-9][a-z0-9!#$%&''+.^_`|~-]*/[a-z0-9][a-z0-9!#$%&''+.^_`|~-]*$'::text))),
    CONSTRAINT artifact_versions_media_type_length CHECK (((length(media_type) >= 3) AND (length(media_type) <= 255))),
    CONSTRAINT artifact_versions_version_id_check CHECK ((btrim(version_id) <> ''::text))
);

ALTER TABLE artifact_versions
    ADD CONSTRAINT artifact_versions_pkey PRIMARY KEY (version_id);

CREATE INDEX artifact_versions_blob_idx ON artifact_versions USING btree (blob_sha256);

CREATE TRIGGER artifact_versions_immutable BEFORE DELETE OR UPDATE ON artifact_versions FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_version();

CREATE TABLE workflow_run_output_publications (
    run_id text NOT NULL,
    project_id text NOT NULL,
    output_name text NOT NULL,
    status text NOT NULL,
    source_scope_kind text GENERATED ALWAYS AS ('run'::text) STORED,
    source_namespace text NOT NULL,
    source_name text NOT NULL,
    source_revision text NOT NULL,
    target_scope_kind text GENERATED ALWAYS AS ('project'::text) STORED,
    target_namespace text NOT NULL,
    target_name text NOT NULL,
    target_revision text,
    error_code text,
    error_message text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT workflow_run_output_publications_check CHECK (((source_name = output_name) AND (target_name = output_name))),
    CONSTRAINT workflow_run_output_publications_check1 CHECK ((((status = 'published'::text) AND (target_revision IS NOT NULL) AND (error_code IS NULL) AND (error_message IS NULL)) OR ((status = 'already_present'::text) AND (target_revision IS NULL) AND (error_code IS NULL) AND (error_message IS NULL)) OR ((status = 'failed'::text) AND (target_revision IS NULL) AND (error_code IS NOT NULL) AND (btrim(error_code) <> ''::text) AND (octet_length(error_code) <= 128) AND (error_message IS NOT NULL) AND (btrim(error_message) <> ''::text) AND (octet_length(error_message) <= 2048)))),
    CONSTRAINT workflow_run_output_publications_output_name_check CHECK (((btrim(output_name) <> ''::text) AND (POSITION(('/'::text) IN (output_name)) = 0) AND (octet_length(output_name) <= 128))),
    CONSTRAINT workflow_run_output_publications_source_namespace_check CHECK ((source_namespace = 'outputs'::text)),
    CONSTRAINT workflow_run_output_publications_source_revision_check CHECK ((btrim(source_revision) <> ''::text)),
    CONSTRAINT workflow_run_output_publications_status_check CHECK ((status = ANY (ARRAY['published'::text, 'already_present'::text, 'failed'::text]))),
    CONSTRAINT workflow_run_output_publications_target_namespace_check CHECK ((target_namespace = 'outputs'::text))
);

ALTER TABLE workflow_run_output_publications
    ADD CONSTRAINT workflow_run_output_publications_pkey PRIMARY KEY (run_id, output_name);

CREATE INDEX workflow_run_output_publications_project_idx ON workflow_run_output_publications USING btree (project_id, created_at DESC, run_id, output_name);

CREATE TRIGGER workflow_run_output_publications_immutable BEFORE DELETE OR UPDATE ON workflow_run_output_publications FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_immutable();


-- Projects and the owner queue

CREATE TABLE owner_queue_controls (
    owner_id text NOT NULL,
    paused boolean DEFAULT false NOT NULL,
    revision bigint DEFAULT 1 NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT owner_queue_controls_owner_id_check CHECK (((btrim(owner_id) <> ''::text) AND (octet_length(owner_id) <= 256))),
    CONSTRAINT owner_queue_controls_revision_check CHECK ((revision > 0))
);

ALTER TABLE owner_queue_controls
    ADD CONSTRAINT owner_queue_controls_pkey PRIMARY KEY (owner_id);

CREATE TRIGGER owner_queue_controls_protect_mutation BEFORE UPDATE ON owner_queue_controls FOR EACH ROW EXECUTE FUNCTION contractor_protect_owner_queue_control();

CREATE TABLE projects (
    project_id text NOT NULL,
    owner_id text NOT NULL,
    kind text NOT NULL,
    name text NOT NULL,
    description text DEFAULT ''::text NOT NULL,
    revision bigint DEFAULT 1 NOT NULL,
    request_idempotency_key text NOT NULL,
    request_digest text NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    http_target_url text,
    http_target_credential_id text,
    http_target_credential_kind text,
    lifecycle_state text DEFAULT 'active'::text NOT NULL,
    deletion_phase text,
    deletion_requested_at timestamp with time zone,
    deletion_claim_id text,
    deletion_claimed_at timestamp with time zone,
    deletion_claim_expires_at timestamp with time zone,
    CONSTRAINT projects_deletion_phase_check CHECK ((deletion_phase = ANY (ARRAY['cancelling'::text, 'draining'::text, 'purging_runs'::text, 'purging_artifacts'::text]))),
    CONSTRAINT projects_deletion_shape CHECK ((((lifecycle_state = 'active'::text) AND (deletion_phase IS NULL) AND (deletion_requested_at IS NULL) AND (deletion_claim_id IS NULL) AND (deletion_claimed_at IS NULL) AND (deletion_claim_expires_at IS NULL)) OR ((lifecycle_state = 'deleting'::text) AND (deletion_phase IS NOT NULL) AND (deletion_requested_at IS NOT NULL) AND (((deletion_claim_id IS NULL) AND (deletion_claimed_at IS NULL) AND (deletion_claim_expires_at IS NULL)) OR ((deletion_claim_id IS NOT NULL) AND (deletion_claimed_at IS NOT NULL) AND (deletion_claim_expires_at > deletion_claimed_at)))))),
    CONSTRAINT projects_description_check CHECK ((octet_length(description) <= 4096)),
    CONSTRAINT projects_http_target_shape CHECK ((((http_target_url IS NULL) AND (http_target_credential_id IS NULL) AND (http_target_credential_kind IS NULL)) OR ((http_target_url IS NOT NULL) AND (btrim(http_target_url) = http_target_url) AND ((octet_length(http_target_url) >= 1) AND (octet_length(http_target_url) <= 2048)) AND (((http_target_credential_id IS NULL) AND (http_target_credential_kind IS NULL)) OR ((http_target_credential_id IS NOT NULL) AND (http_target_credential_kind = ANY (ARRAY['http-origin-basic@1'::text, 'http-origin-bearer@1'::text]))))))),
    CONSTRAINT projects_kind_check CHECK ((kind = ANY (ARRAY['project'::text, 'evaluation'::text]))),
    CONSTRAINT projects_lifecycle_state_check CHECK ((lifecycle_state = ANY (ARRAY['active'::text, 'deleting'::text]))),
    CONSTRAINT projects_name_check CHECK (((btrim(name) <> ''::text) AND (octet_length(name) <= 160))),
    CONSTRAINT projects_owner_id_check CHECK (((btrim(owner_id) <> ''::text) AND (octet_length(owner_id) <= 256))),
    CONSTRAINT projects_project_id_check CHECK ((project_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT projects_request_digest_check CHECK ((request_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT projects_request_idempotency_key_check CHECK ((request_idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'::text)),
    CONSTRAINT projects_revision_check CHECK ((revision > 0))
);

ALTER TABLE projects
    ADD CONSTRAINT projects_owner_id_request_idempotency_key_key UNIQUE (owner_id, request_idempotency_key);

ALTER TABLE projects
    ADD CONSTRAINT projects_owner_identity UNIQUE (owner_id, project_id);

ALTER TABLE projects
    ADD CONSTRAINT projects_pkey PRIMARY KEY (project_id);

ALTER TABLE projects
    ADD CONSTRAINT projects_project_owner_key UNIQUE (project_id, owner_id);

CREATE INDEX projects_deletion_claim_idx ON projects USING btree (deletion_claim_expires_at, deletion_requested_at, project_id) WHERE (lifecycle_state = 'deleting'::text);

CREATE INDEX projects_http_target_credential_idx ON projects USING btree (http_target_credential_id, project_id) WHERE (http_target_credential_id IS NOT NULL);

CREATE INDEX projects_owner_kind_created_idx ON projects USING btree (owner_id, kind, created_at DESC, project_id DESC);

CREATE TRIGGER projects_eval_fence AFTER UPDATE OF lifecycle_state ON projects FOR EACH ROW EXECUTE FUNCTION contractor_eval_fence_project();

CREATE TRIGGER projects_protect_identity BEFORE UPDATE ON projects FOR EACH ROW EXECUTE FUNCTION contractor_protect_project();

CREATE TABLE scheduler_settings (
    singleton boolean DEFAULT true NOT NULL,
    max_concurrent_runs integer NOT NULL,
    revision numeric(20,0) NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT scheduler_settings_max_concurrent_runs_check CHECK (((max_concurrent_runs >= 1) AND (max_concurrent_runs <= 32))),
    CONSTRAINT scheduler_settings_revision_check CHECK (((revision >= (1)::numeric) AND (revision <= '18446744073709551615'::numeric))),
    CONSTRAINT scheduler_settings_singleton_check CHECK (singleton)
);

ALTER TABLE scheduler_settings
    ADD CONSTRAINT scheduler_settings_pkey PRIMARY KEY (singleton);

CREATE TRIGGER scheduler_settings_protect_mutation BEFORE DELETE OR UPDATE ON scheduler_settings FOR EACH ROW EXECUTE FUNCTION contractor_protect_scheduler_settings();


-- Credentials: LLM Gateway keys, Runtime credentials and SSH keys

CREATE TABLE credential_operations (
    operation_id text NOT NULL,
    idempotency_key text NOT NULL,
    request_hash text NOT NULL,
    credential_id text NOT NULL,
    operation_kind text NOT NULL,
    phase text NOT NULL,
    request_schema_version text NOT NULL,
    request jsonb NOT NULL,
    created_at timestamp with time zone NOT NULL,
    updated_at timestamp with time zone NOT NULL,
    CONSTRAINT credential_operations_check CHECK ((updated_at >= created_at)),
    CONSTRAINT credential_operations_idempotency_key_check CHECK ((idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'::text)),
    CONSTRAINT credential_operations_operation_id_check CHECK (((operation_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]*$'::text) AND (octet_length(operation_id) <= 128))),
    CONSTRAINT credential_operations_operation_kind_check CHECK ((operation_kind = ANY (ARRAY['create'::text, 'delete'::text]))),
    CONSTRAINT credential_operations_phase_check CHECK ((phase = ANY (ARRAY['prepared'::text, 'completed'::text, 'abandoned'::text]))),
    CONSTRAINT credential_operations_request_check CHECK (((jsonb_typeof(request) = 'object'::text) AND (octet_length((request)::text) <= 65536))),
    CONSTRAINT credential_operations_request_hash_check CHECK ((request_hash ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT credential_operations_request_schema_version_check CHECK ((request_schema_version = 'contractor.credentials/v1'::text))
);

ALTER TABLE credential_operations
    ADD CONSTRAINT credential_operations_operation_kind_idempotency_key_key UNIQUE (operation_kind, idempotency_key);

ALTER TABLE credential_operations
    ADD CONSTRAINT credential_operations_pkey PRIMARY KEY (operation_id);

CREATE UNIQUE INDEX credential_operations_create_identity ON credential_operations USING btree (credential_id) WHERE (operation_kind = 'create'::text);

CREATE INDEX credential_operations_prepared_idx ON credential_operations USING btree (created_at, operation_id) WHERE (phase = 'prepared'::text);

CREATE TRIGGER credential_operations_protect_delete BEFORE DELETE ON credential_operations FOR EACH ROW EXECUTE FUNCTION contractor_protect_immutable_credential_row();

CREATE TRIGGER credential_operations_protect_transition BEFORE UPDATE ON credential_operations FOR EACH ROW EXECUTE FUNCTION contractor_protect_credential_operation();

CREATE TABLE git_ssh_keys (
    owner_id text NOT NULL,
    generation text NOT NULL,
    encryption_schema_version text NOT NULL,
    key_id text NOT NULL,
    nonce bytea NOT NULL,
    ciphertext bytea NOT NULL,
    fingerprint text NOT NULL,
    key_type text NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT git_ssh_keys_ciphertext_check CHECK (((octet_length(ciphertext) >= 17) AND (octet_length(ciphertext) <= 32784))),
    CONSTRAINT git_ssh_keys_encryption_schema_version_check CHECK ((encryption_schema_version = 'git-ssh-key@1'::text)),
    CONSTRAINT git_ssh_keys_generation_check CHECK ((generation ~ '^[0-9a-f]{32}$'::text)),
    CONSTRAINT git_ssh_keys_key_id_check CHECK ((key_id ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT git_ssh_keys_nonce_check CHECK ((octet_length(nonce) = 12)),
    CONSTRAINT git_ssh_keys_owner_id_check CHECK (((length(owner_id) >= 1) AND (length(owner_id) <= 256)))
);

ALTER TABLE git_ssh_keys
    ADD CONSTRAINT git_ssh_keys_pkey PRIMARY KEY (owner_id);

CREATE TABLE llm_credentials (
    credential_id text NOT NULL,
    llm_gateway_id text NOT NULL,
    llm_gateway_version text NOT NULL,
    llm_gateway_digest text NOT NULL,
    remote_key_id text NOT NULL,
    label text,
    gateway_policy jsonb NOT NULL,
    encryption_schema_version text NOT NULL,
    key_id text NOT NULL,
    nonce bytea NOT NULL,
    ciphertext bytea NOT NULL,
    created_at timestamp with time zone NOT NULL,
    CONSTRAINT llm_credentials_ciphertext_check CHECK (((octet_length(ciphertext) >= 17) AND (octet_length(ciphertext) <= 16400))),
    CONSTRAINT llm_credentials_encryption_schema_version_check CHECK ((encryption_schema_version = 'contractor.credentials/v1'::text)),
    CONSTRAINT llm_credentials_gateway_policy_check CHECK (((jsonb_typeof(gateway_policy) = 'object'::text) AND (octet_length((gateway_policy)::text) <= 65536))),
    CONSTRAINT llm_credentials_key_id_check CHECK ((key_id ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT llm_credentials_label_check CHECK (((label IS NULL) OR ((btrim(label) <> ''::text) AND (octet_length(label) <= 1024)))),
    CONSTRAINT llm_credentials_llm_gateway_digest_check CHECK ((llm_gateway_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT llm_credentials_llm_gateway_id_check CHECK (((llm_gateway_id ~ '^[a-z][a-z0-9_-]*$'::text) AND (octet_length(llm_gateway_id) <= 128))),
    CONSTRAINT llm_credentials_llm_gateway_version_check CHECK (((llm_gateway_version ~ '^[A-Za-z0-9][A-Za-z0-9._+-]*$'::text) AND (octet_length(llm_gateway_version) <= 64))),
    CONSTRAINT llm_credentials_nonce_check CHECK ((octet_length(nonce) = 12)),
    CONSTRAINT llm_credentials_remote_key_id_check CHECK (((btrim(remote_key_id) <> ''::text) AND (octet_length(remote_key_id) <= 512)))
);

ALTER TABLE llm_credentials
    ADD CONSTRAINT llm_credentials_llm_gateway_digest_remote_key_id_key UNIQUE (llm_gateway_digest, remote_key_id);

ALTER TABLE llm_credentials
    ADD CONSTRAINT llm_credentials_pkey PRIMARY KEY (credential_id);

CREATE INDEX llm_credentials_created_idx ON llm_credentials USING btree (created_at, credential_id);

CREATE TRIGGER llm_credentials_exclusive_state BEFORE INSERT ON llm_credentials FOR EACH ROW EXECUTE FUNCTION contractor_check_llm_credential_exclusive_state();

CREATE TRIGGER llm_credentials_protect_update BEFORE UPDATE ON llm_credentials FOR EACH ROW EXECUTE FUNCTION contractor_protect_immutable_credential_row();

CREATE TABLE runtime_credentials (
    credential_id text NOT NULL,
    credential_kind text NOT NULL,
    encryption_schema_version text NOT NULL,
    key_id text NOT NULL,
    nonce bytea NOT NULL,
    ciphertext bytea NOT NULL,
    created_by text NOT NULL,
    created_at timestamp with time zone NOT NULL,
    idempotency_key_digest text,
    request_mac bytea,
    deleted_by text,
    deleted_at timestamp with time zone,
    CONSTRAINT runtime_credentials_ciphertext_check CHECK (((octet_length(ciphertext) >= 17) AND (octet_length(ciphertext) <= 32784))),
    CONSTRAINT runtime_credentials_created_by_check CHECK (((btrim(created_by) <> ''::text) AND (octet_length(created_by) <= 256))),
    CONSTRAINT runtime_credentials_creation_receipt CHECK (((idempotency_key_digest IS NULL) = (request_mac IS NULL))),
    CONSTRAINT runtime_credentials_credential_id_check CHECK (((credential_id ~ '^[a-z][a-z0-9_-]*$'::text) AND (octet_length(credential_id) <= 128))),
    CONSTRAINT runtime_credentials_credential_kind_check CHECK ((credential_kind = ANY (ARRAY['otlp-headers@1'::text, 'http-proxy-basic@1'::text, 'http-proxy-bearer@1'::text, 'caido-bearer@1'::text, 'http-origin-basic@1'::text, 'http-origin-bearer@1'::text]))),
    CONSTRAINT runtime_credentials_deleted_by_check CHECK (((btrim(deleted_by) <> ''::text) AND (octet_length(deleted_by) <= 256))),
    CONSTRAINT runtime_credentials_deletion_marker CHECK (((deleted_by IS NULL) = (deleted_at IS NULL))),
    CONSTRAINT runtime_credentials_encryption_schema_version_check CHECK ((encryption_schema_version = 'contractor.runtime-credentials/v1'::text)),
    CONSTRAINT runtime_credentials_idempotency_key_digest_check CHECK ((idempotency_key_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT runtime_credentials_key_id_check CHECK ((key_id ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT runtime_credentials_nonce_check CHECK ((octet_length(nonce) = 12)),
    CONSTRAINT runtime_credentials_request_mac_check CHECK ((octet_length(request_mac) = 32))
);

ALTER TABLE runtime_credentials
    ADD CONSTRAINT runtime_credentials_credential_id_credential_kind_key UNIQUE (credential_id, credential_kind);

ALTER TABLE runtime_credentials
    ADD CONSTRAINT runtime_credentials_idempotency_key_digest_key UNIQUE (idempotency_key_digest);

ALTER TABLE runtime_credentials
    ADD CONSTRAINT runtime_credentials_pkey PRIMARY KEY (credential_id);

CREATE INDEX runtime_credentials_created_idx ON runtime_credentials USING btree (created_at, credential_id);

CREATE TRIGGER runtime_credentials_protect_mutation BEFORE DELETE OR UPDATE ON runtime_credentials FOR EACH ROW EXECUTE FUNCTION contractor_protect_runtime_credential_immutable();


-- Runtime configuration, Runtime Agents and model gateway recovery

CREATE TABLE gateway_allocation_routes (
    allocation_id text NOT NULL,
    model text NOT NULL,
    run_id text NOT NULL,
    route_key text NOT NULL
);

ALTER TABLE gateway_allocation_routes
    ADD CONSTRAINT gateway_allocation_routes_pkey PRIMARY KEY (allocation_id, model);

CREATE INDEX gateway_allocation_routes_run_idx ON gateway_allocation_routes USING btree (run_id);

CREATE TABLE gateway_recovery_failures (
    route_key text NOT NULL,
    request_id text NOT NULL,
    run_id text NOT NULL
);

ALTER TABLE gateway_recovery_failures
    ADD CONSTRAINT gateway_recovery_failures_pkey PRIMARY KEY (route_key, request_id);

CREATE INDEX gateway_recovery_failures_run_idx ON gateway_recovery_failures USING btree (run_id);

CREATE TABLE gateway_recovery_routes (
    route_key text NOT NULL,
    owner_id text NOT NULL,
    blocked boolean DEFAULT false NOT NULL,
    failure_code text DEFAULT ''::text NOT NULL,
    failure_count bigint DEFAULT 0 NOT NULL,
    blocked_at timestamp with time zone,
    next_probe_at timestamp with time zone,
    automatic_until timestamp with time zone,
    probe_id text,
    probe_run_id text,
    probe_until timestamp with time zone
);

ALTER TABLE gateway_recovery_routes
    ADD CONSTRAINT gateway_recovery_routes_pkey PRIMARY KEY (route_key);

CREATE TABLE gateway_recovery_waits (
    participant_id text NOT NULL,
    model text NOT NULL,
    run_id text NOT NULL,
    route_key text NOT NULL
);

ALTER TABLE gateway_recovery_waits
    ADD CONSTRAINT gateway_recovery_waits_pkey PRIMARY KEY (participant_id, model);

CREATE INDEX gateway_recovery_waits_route_idx ON gateway_recovery_waits USING btree (route_key);

CREATE INDEX gateway_recovery_waits_run_idx ON gateway_recovery_waits USING btree (run_id);

CREATE TABLE gateway_run_routes (
    run_id text NOT NULL,
    route_key text NOT NULL
);

ALTER TABLE gateway_run_routes
    ADD CONSTRAINT gateway_run_routes_pkey PRIMARY KEY (run_id, route_key);

CREATE TABLE runtime_agent_principals (
    runtime_agent_id text NOT NULL,
    labels text[] NOT NULL,
    label_revision numeric(20,0) NOT NULL,
    created_by text NOT NULL,
    created_at timestamp with time zone NOT NULL,
    updated_by text NOT NULL,
    updated_at timestamp with time zone NOT NULL,
    CONSTRAINT runtime_agent_principals_created_by_check CHECK (((btrim(created_by) <> ''::text) AND (length(created_by) <= 256))),
    CONSTRAINT runtime_agent_principals_label_revision_check CHECK (((label_revision >= (1)::numeric) AND (label_revision <= '18446744073709551615'::numeric))),
    CONSTRAINT runtime_agent_principals_labels_check CHECK (contractor_valid_runtime_agent_labels(labels)),
    CONSTRAINT runtime_agent_principals_runtime_agent_id_check CHECK ((runtime_agent_id ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT runtime_agent_principals_updated_by_check CHECK (((btrim(updated_by) <> ''::text) AND (length(updated_by) <= 256)))
);

ALTER TABLE runtime_agent_principals
    ADD CONSTRAINT runtime_agent_principals_pkey PRIMARY KEY (runtime_agent_id);

CREATE INDEX runtime_agent_principals_labels_idx ON runtime_agent_principals USING gin (labels);

CREATE INDEX runtime_agent_principals_updated_idx ON runtime_agent_principals USING btree (updated_at, runtime_agent_id);

CREATE TRIGGER runtime_agent_principals_protect_mutation BEFORE UPDATE ON runtime_agent_principals FOR EACH ROW EXECUTE FUNCTION contractor_protect_runtime_agent_principal();

CREATE TABLE runtime_config_versions (
    name text NOT NULL,
    version text NOT NULL,
    digest text NOT NULL,
    canonical_document text NOT NULL,
    built_in boolean DEFAULT false NOT NULL,
    actor_id text NOT NULL,
    created_at timestamp with time zone NOT NULL,
    idempotency_key_digest text,
    request_digest text,
    CONSTRAINT runtime_config_versions_actor_id_check CHECK (((btrim(actor_id) <> ''::text) AND (length(actor_id) <= 256))),
    CONSTRAINT runtime_config_versions_canonical_document_check CHECK (((octet_length(canonical_document) <= 131072) AND (jsonb_typeof((canonical_document)::jsonb) = 'object'::text))),
    CONSTRAINT runtime_config_versions_check CHECK ((((name = 'contractor-empty'::text) AND (version = '1'::text) AND built_in AND (digest = 'sha256:80a1754c01f8443c29fdc8f650a2254b2461694819918b204a55a7ad3425dc5f'::text) AND (canonical_document = '{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"contractor-empty","version":"1"},"spec":{}}'::text)) OR ((NOT built_in) AND (NOT ((name = 'contractor-empty'::text) AND (version = '1'::text)))))),
    CONSTRAINT runtime_config_versions_digest_check CHECK ((digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT runtime_config_versions_idempotency_key_digest_check CHECK ((idempotency_key_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT runtime_config_versions_name_check CHECK (((name ~ '^[a-z][a-z0-9_-]*$'::text) AND (length(name) <= 63))),
    CONSTRAINT runtime_config_versions_publication_receipt CHECK (((idempotency_key_digest IS NULL) = (request_digest IS NULL))),
    CONSTRAINT runtime_config_versions_request_digest_check CHECK ((request_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT runtime_config_versions_version_check CHECK (((version ~ '^[A-Za-z0-9][A-Za-z0-9._+-]*$'::text) AND (length(version) <= 128)))
);

ALTER TABLE runtime_config_versions
    ADD CONSTRAINT runtime_config_versions_idempotency_key_digest_key UNIQUE (idempotency_key_digest);

ALTER TABLE runtime_config_versions
    ADD CONSTRAINT runtime_config_versions_name_version_digest_key UNIQUE (name, version, digest);

ALTER TABLE runtime_config_versions
    ADD CONSTRAINT runtime_config_versions_pkey PRIMARY KEY (name, version);

CREATE INDEX runtime_config_versions_created_at_idx ON runtime_config_versions USING btree (created_at, name, version);

CREATE TRIGGER runtime_config_versions_protect_immutable BEFORE DELETE OR UPDATE ON runtime_config_versions FOR EACH ROW EXECUTE FUNCTION contractor_protect_runtime_config_immutable();

CREATE TABLE runtime_label_bindings (
    label text NOT NULL,
    config_name text NOT NULL,
    config_version text NOT NULL,
    config_digest text NOT NULL,
    revision numeric(20,0) NOT NULL,
    created_by text NOT NULL,
    created_at timestamp with time zone NOT NULL,
    updated_by text NOT NULL,
    updated_at timestamp with time zone NOT NULL,
    CONSTRAINT runtime_label_bindings_created_by_check CHECK (((btrim(created_by) <> ''::text) AND (length(created_by) <= 256))),
    CONSTRAINT runtime_label_bindings_label_check CHECK (((label ~ '^[a-z][a-z0-9_-]*$'::text) AND (length(label) <= 63))),
    CONSTRAINT runtime_label_bindings_revision_check CHECK (((revision >= (1)::numeric) AND (revision <= '18446744073709551615'::numeric))),
    CONSTRAINT runtime_label_bindings_updated_by_check CHECK (((btrim(updated_by) <> ''::text) AND (length(updated_by) <= 256)))
);

ALTER TABLE runtime_label_bindings
    ADD CONSTRAINT runtime_label_bindings_pkey PRIMARY KEY (label);

CREATE INDEX runtime_label_bindings_config_idx ON runtime_label_bindings USING btree (config_name, config_version, config_digest, label);

CREATE TRIGGER runtime_label_bindings_protect_mutation BEFORE DELETE OR UPDATE ON runtime_label_bindings FOR EACH ROW EXECUTE FUNCTION contractor_protect_runtime_label_binding();

CREATE TABLE runtime_management_operations (
    idempotency_key_digest text NOT NULL,
    request_digest text NOT NULL,
    operation_kind text NOT NULL,
    resource_id text NOT NULL,
    result jsonb NOT NULL,
    actor_id text NOT NULL,
    performed_at timestamp with time zone NOT NULL,
    CONSTRAINT runtime_management_operations_actor_id_check CHECK (((btrim(actor_id) <> ''::text) AND (length(actor_id) <= 256))),
    CONSTRAINT runtime_management_operations_idempotency_key_digest_check CHECK ((idempotency_key_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT runtime_management_operations_operation_kind_check CHECK ((operation_kind ~ '^[a-z][a-z0-9.-]{0,63}$'::text)),
    CONSTRAINT runtime_management_operations_request_digest_check CHECK ((request_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT runtime_management_operations_resource_id_check CHECK ((resource_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT runtime_management_operations_result_check CHECK (((jsonb_typeof(result) = 'object'::text) AND (octet_length((result)::text) <= 16384)))
);

ALTER TABLE runtime_management_operations
    ADD CONSTRAINT runtime_management_operations_pkey PRIMARY KEY (idempotency_key_digest);

CREATE INDEX runtime_management_operations_performed_at_idx ON runtime_management_operations USING btree (performed_at, idempotency_key_digest);

CREATE TRIGGER runtime_management_operations_protect_immutable BEFORE DELETE OR UPDATE ON runtime_management_operations FOR EACH ROW EXECUTE FUNCTION contractor_protect_runtime_config_immutable();


-- Audits: rounds, items, Audit-managed Runs and coverage

CREATE TABLE audit_artifact_links (
    audit_id text NOT NULL,
    logical_key text NOT NULL,
    artifact_ref jsonb NOT NULL,
    artifact_digest text NOT NULL,
    media_type text NOT NULL,
    size_bytes bigint NOT NULL,
    source_provenance jsonb NOT NULL,
    display_ref text DEFAULT ''::text NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT audit_artifact_links_artifact_digest_check CHECK ((artifact_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audit_artifact_links_artifact_ref_check CHECK (((jsonb_typeof(artifact_ref) = 'object'::text) AND (octet_length((artifact_ref)::text) <= 4096))),
    CONSTRAINT audit_artifact_links_display_ref_check CHECK ((octet_length(display_ref) <= 1024)),
    CONSTRAINT audit_artifact_links_logical_key_check CHECK (((btrim(logical_key) <> ''::text) AND (octet_length(logical_key) <= 512))),
    CONSTRAINT audit_artifact_links_media_type_check CHECK (((btrim(media_type) <> ''::text) AND (octet_length(media_type) <= 256))),
    CONSTRAINT audit_artifact_links_size_bytes_check CHECK ((size_bytes >= 0)),
    CONSTRAINT audit_artifact_links_source_provenance_check CHECK (((jsonb_typeof(source_provenance) = 'object'::text) AND (octet_length((source_provenance)::text) <= 1048576)))
);

ALTER TABLE audit_artifact_links
    ADD CONSTRAINT audit_artifact_links_pkey PRIMARY KEY (audit_id, logical_key);

CREATE TRIGGER eval_audit_outputs_projection AFTER INSERT OR DELETE OR UPDATE ON audit_artifact_links FOR EACH ROW EXECUTE FUNCTION contractor_eval_audit_child_changed();

CREATE TABLE audit_controller_claims (
    audit_id text NOT NULL,
    epoch bigint DEFAULT 0 NOT NULL,
    holder_id text,
    claimed_at timestamp with time zone,
    expires_at timestamp with time zone,
    CONSTRAINT audit_controller_claim_shape CHECK ((((holder_id IS NULL) AND (claimed_at IS NULL) AND (expires_at IS NULL)) OR ((holder_id IS NOT NULL) AND (claimed_at IS NOT NULL) AND (expires_at > claimed_at)))),
    CONSTRAINT audit_controller_claims_epoch_check CHECK ((epoch >= 0)),
    CONSTRAINT audit_controller_claims_holder_id_check CHECK (((holder_id IS NULL) OR ((btrim(holder_id) <> ''::text) AND (octet_length(holder_id) <= 256))))
);

ALTER TABLE audit_controller_claims
    ADD CONSTRAINT audit_controller_claims_pkey PRIMARY KEY (audit_id);

CREATE INDEX audit_controller_claim_expiry_idx ON audit_controller_claims USING btree (expires_at, audit_id);

CREATE TABLE audit_events (
    audit_id text NOT NULL,
    sequence_number bigint NOT NULL,
    kind text NOT NULL,
    entity_id text NOT NULL,
    entity_revision bigint,
    summary jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT audit_events_entity_id_check CHECK (((btrim(entity_id) <> ''::text) AND (octet_length(entity_id) <= 256))),
    CONSTRAINT audit_events_entity_revision_check CHECK (((entity_revision IS NULL) OR (entity_revision > 0))),
    CONSTRAINT audit_events_kind_check CHECK ((kind ~ '^[a-z][a-z0-9_.-]{0,127}$'::text)),
    CONSTRAINT audit_events_sequence_number_check CHECK ((sequence_number > 0)),
    CONSTRAINT audit_events_summary_check CHECK (((jsonb_typeof(summary) = 'object'::text) AND (octet_length((summary)::text) <= 65536)))
);

ALTER TABLE audit_events
    ADD CONSTRAINT audit_events_pkey PRIMARY KEY (audit_id, sequence_number);

CREATE INDEX audit_events_finding_proposal_rejected_idx ON audit_events USING btree (audit_id, entity_id) WHERE (kind = 'finding.proposal_rejected'::text);

CREATE TRIGGER audit_events_protect_immutable BEFORE UPDATE ON audit_events FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_event();

CREATE TABLE audit_execution_items (
    execution_item_id text NOT NULL,
    execution_id text NOT NULL,
    audit_id text NOT NULL,
    round_id text NOT NULL,
    item_id text NOT NULL,
    batch_ordinal integer NOT NULL,
    item_attempt integer NOT NULL,
    task_ref jsonb NOT NULL,
    task_digest text NOT NULL,
    input_refs jsonb NOT NULL,
    state text DEFAULT 'submitted'::text NOT NULL,
    collection_disposition text,
    result_ref jsonb,
    result_digest text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    collected_at timestamp with time zone,
    CONSTRAINT audit_execution_items_batch_ordinal_check CHECK ((batch_ordinal >= 0)),
    CONSTRAINT audit_execution_items_collection_disposition_check CHECK (((collection_disposition IS NULL) OR (collection_disposition = ANY (ARRAY['accepted-result'::text, 'missing-output'::text, 'invalid-result'::text, 'execution-failed'::text, 'execution-cancelled'::text, 'collection-contract-invalid'::text])))),
    CONSTRAINT audit_execution_items_collection_shape CHECK ((((state = ANY (ARRAY['submitted'::text, 'collecting'::text])) AND (collection_disposition IS NULL) AND (result_ref IS NULL) AND (result_digest IS NULL) AND (collected_at IS NULL)) OR ((state = 'settled'::text) AND (collection_disposition IS NOT NULL) AND (collected_at IS NOT NULL) AND ((result_ref IS NULL) = (result_digest IS NULL)) AND ((result_ref IS NULL) OR (collection_disposition = 'accepted-result'::text))))),
    CONSTRAINT audit_execution_items_execution_item_id_check CHECK ((execution_item_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT audit_execution_items_input_refs_check CHECK (((jsonb_typeof(input_refs) = 'array'::text) AND (octet_length((input_refs)::text) <= 1048576))),
    CONSTRAINT audit_execution_items_item_attempt_check CHECK ((item_attempt > 0)),
    CONSTRAINT audit_execution_items_result_digest_check CHECK (((result_digest IS NULL) OR (result_digest ~ '^sha256:[0-9a-f]{64}$'::text))),
    CONSTRAINT audit_execution_items_result_ref_check CHECK (((result_ref IS NULL) OR ((jsonb_typeof(result_ref) = 'object'::text) AND (octet_length((result_ref)::text) <= 4096)))),
    CONSTRAINT audit_execution_items_state_check CHECK ((state = ANY (ARRAY['submitted'::text, 'collecting'::text, 'settled'::text]))),
    CONSTRAINT audit_execution_items_task_digest_check CHECK ((task_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audit_execution_items_task_ref_check CHECK (((jsonb_typeof(task_ref) = 'object'::text) AND (octet_length((task_ref)::text) <= 4096)))
);

ALTER TABLE audit_execution_items
    ADD CONSTRAINT audit_execution_items_execution_member UNIQUE (execution_id, item_id);

ALTER TABLE audit_execution_items
    ADD CONSTRAINT audit_execution_items_execution_ordinal UNIQUE (execution_id, batch_ordinal);

ALTER TABLE audit_execution_items
    ADD CONSTRAINT audit_execution_items_item_attempt UNIQUE (item_id, item_attempt);

ALTER TABLE audit_execution_items
    ADD CONSTRAINT audit_execution_items_item_identity UNIQUE (execution_item_id, item_id);

ALTER TABLE audit_execution_items
    ADD CONSTRAINT audit_execution_items_pkey PRIMARY KEY (execution_item_id);

CREATE INDEX audit_execution_items_audit_state_idx ON audit_execution_items USING btree (audit_id, state, execution_id, batch_ordinal);

CREATE TABLE audit_executions (
    execution_id text NOT NULL,
    audit_id text NOT NULL,
    round_id text,
    role text NOT NULL,
    role_attempt integer,
    manifest_ref jsonb NOT NULL,
    manifest_digest text NOT NULL,
    submission_key text NOT NULL,
    request_digest text NOT NULL,
    run_id text,
    state text DEFAULT 'intent'::text NOT NULL,
    terminal_outcome text,
    terminal_run_generation text,
    terminal_run_sequence bigint,
    terminal_observed_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    run_provenance jsonb,
    run_deleted_at timestamp with time zone,
    workflow_role text NOT NULL,
    collection_receipt_id text,
    collection_disposition text,
    collection_source_output_ref jsonb,
    collection_source_output_digest text,
    collection_retained_refs jsonb,
    collection_error_code text,
    collection_request_digest text,
    collected_at timestamp with time zone,
    preparation_snapshot jsonb,
    preparation_outputs jsonb DEFAULT '[]'::jsonb NOT NULL,
    CONSTRAINT audit_executions_collection_disposition_check CHECK ((collection_disposition = ANY (ARRAY['accepted-result'::text, 'missing-output'::text, 'invalid-result'::text, 'execution-failed'::text, 'execution-cancelled'::text, 'collection-contract-invalid'::text]))),
    CONSTRAINT audit_executions_collection_error_code_check CHECK (((btrim(collection_error_code) <> ''::text) AND (octet_length(collection_error_code) <= 128))),
    CONSTRAINT audit_executions_collection_observation_shape CHECK (((collection_receipt_id IS NULL) OR ((run_id IS NOT NULL) AND (terminal_outcome IS NOT NULL) AND (terminal_run_generation IS NOT NULL) AND (btrim(terminal_run_generation) <> ''::text) AND (terminal_run_sequence >= 0)) OR ((run_id IS NULL) AND (terminal_outcome = 'submission-failed'::text) AND (terminal_run_generation IS NULL) AND (terminal_run_sequence IS NULL)))),
    CONSTRAINT audit_executions_collection_receipt_id_check CHECK ((collection_receipt_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT audit_executions_collection_request_digest_check CHECK ((collection_request_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audit_executions_collection_retained_refs_check CHECK (((jsonb_typeof(collection_retained_refs) = 'array'::text) AND (octet_length((collection_retained_refs)::text) <= 8388608))),
    CONSTRAINT audit_executions_collection_shape CHECK ((((collection_receipt_id IS NULL) = (collection_disposition IS NULL)) AND ((collection_receipt_id IS NULL) = (collection_retained_refs IS NULL)) AND ((collection_receipt_id IS NULL) = (collection_request_digest IS NULL)) AND ((collection_receipt_id IS NULL) = (collected_at IS NULL)) AND ((collection_receipt_id IS NOT NULL) OR ((collection_error_code IS NULL) AND (collection_source_output_ref IS NULL))))),
    CONSTRAINT audit_executions_collection_source_output_digest_check CHECK ((collection_source_output_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audit_executions_collection_source_output_ref_check CHECK (((jsonb_typeof(collection_source_output_ref) = 'object'::text) AND (octet_length((collection_source_output_ref)::text) <= 4096))),
    CONSTRAINT audit_executions_collection_source_shape CHECK ((((collection_source_output_ref IS NULL) = (collection_source_output_digest IS NULL)) AND ((collection_receipt_id IS NULL) OR ((collection_disposition = ANY (ARRAY['accepted-result'::text, 'invalid-result'::text])) = (collection_source_output_ref IS NOT NULL))))),
    CONSTRAINT audit_executions_execution_id_check CHECK ((execution_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT audit_executions_manifest_digest_check CHECK ((manifest_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audit_executions_manifest_ref_check CHECK (((jsonb_typeof(manifest_ref) = 'object'::text) AND (octet_length((manifest_ref)::text) <= 4096))),
    CONSTRAINT audit_executions_preparation_outputs_shape CHECK (((jsonb_typeof(preparation_outputs) = 'array'::text) AND (jsonb_array_length(preparation_outputs) <= 128) AND (octet_length((preparation_outputs)::text) <= 1048576) AND
CASE
    WHEN ((role = 'prepare'::text) AND (collection_disposition = 'accepted-result'::text)) THEN (jsonb_array_length(preparation_outputs) > 0)
    ELSE (preparation_outputs = '[]'::jsonb)
END)),
    CONSTRAINT audit_executions_preparation_shape CHECK ((((role = 'prepare'::text) AND (preparation_snapshot IS NOT NULL) AND (jsonb_typeof(preparation_snapshot) = 'object'::text) AND (jsonb_typeof((preparation_snapshot -> 'inputs'::text)) = 'object'::text) AND (jsonb_typeof((preparation_snapshot -> 'parameters'::text)) = 'object'::text) AND (preparation_snapshot ?& ARRAY['inputs'::text, 'parameters'::text]) AND (octet_length((preparation_snapshot)::text) <= 1048576)) OR ((role <> 'prepare'::text) AND (preparation_snapshot IS NULL)))),
    CONSTRAINT audit_executions_request_digest_check CHECK ((request_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audit_executions_role_attempt_check CHECK (((role_attempt IS NULL) OR (role_attempt > 0))),
    CONSTRAINT audit_executions_role_check CHECK ((role = ANY (ARRAY['prepare'::text, 'discovery'::text, 'check'::text, 'assessment'::text]))),
    CONSTRAINT audit_executions_role_shape CHECK ((((role = 'check'::text) AND (round_id IS NOT NULL) AND (role_attempt IS NULL)) OR ((role = ANY (ARRAY['discovery'::text, 'assessment'::text])) AND (role_attempt IS NOT NULL)) OR ((role = 'prepare'::text) AND (round_id IS NULL) AND ((role_attempt >= 1) AND (role_attempt <= 10))))),
    CONSTRAINT audit_executions_run_deletion_shape CHECK (((run_deleted_at IS NULL) OR ((run_id IS NOT NULL) AND (state = 'collected'::text)))),
    CONSTRAINT audit_executions_run_provenance_shape CHECK ((((run_id IS NULL) AND (run_provenance IS NULL) AND (run_deleted_at IS NULL)) OR ((run_id IS NOT NULL) AND (run_provenance IS NOT NULL) AND (jsonb_typeof(run_provenance) = 'object'::text) AND ((run_provenance ->> 'schema'::text) = 'contractor.audit.run-provenance.v1'::text) AND ((run_provenance ->> 'runId'::text) = run_id) AND (octet_length((run_provenance)::text) <= 16384)))),
    CONSTRAINT audit_executions_run_shape CHECK ((((state = 'intent'::text) AND (run_id IS NULL)) OR ((state = 'submitted'::text) AND (run_id IS NOT NULL)) OR ((state = ANY (ARRAY['collecting'::text, 'collected'::text])) AND ((run_id IS NOT NULL) OR (terminal_outcome = 'submission-failed'::text))))),
    CONSTRAINT audit_executions_state_check CHECK ((state = ANY (ARRAY['intent'::text, 'submitted'::text, 'collecting'::text, 'collected'::text]))),
    CONSTRAINT audit_executions_submission_key_check CHECK ((submission_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'::text)),
    CONSTRAINT audit_executions_terminal_outcome_check CHECK (((terminal_outcome IS NULL) OR (terminal_outcome = ANY (ARRAY['succeeded'::text, 'failed'::text, 'cancelled'::text, 'submission-failed'::text])))),
    CONSTRAINT audit_executions_terminal_run_generation_check CHECK (((terminal_run_generation IS NULL) OR ((btrim(terminal_run_generation) <> ''::text) AND (octet_length(terminal_run_generation) <= 256)))),
    CONSTRAINT audit_executions_terminal_run_sequence_check CHECK (((terminal_run_sequence IS NULL) OR (terminal_run_sequence >= 0))),
    CONSTRAINT audit_executions_terminal_shape CHECK ((((state = ANY (ARRAY['intent'::text, 'submitted'::text])) AND (terminal_outcome IS NULL) AND (terminal_run_generation IS NULL) AND (terminal_run_sequence IS NULL) AND (terminal_observed_at IS NULL)) OR ((state = ANY (ARRAY['collecting'::text, 'collected'::text])) AND (terminal_outcome IS NOT NULL) AND (terminal_observed_at IS NOT NULL) AND (((run_id IS NOT NULL) AND (terminal_run_generation IS NOT NULL) AND (terminal_run_sequence IS NOT NULL)) OR ((run_id IS NULL) AND (terminal_outcome = 'submission-failed'::text) AND (terminal_run_generation IS NULL) AND (terminal_run_sequence IS NULL)))))),
    CONSTRAINT audit_executions_workflow_role_shape CHECK (((btrim(workflow_role) <> ''::text) AND (octet_length(workflow_role) <= 128)))
);

ALTER TABLE audit_executions
    ADD CONSTRAINT audit_executions_audit_identity UNIQUE (execution_id, audit_id);

ALTER TABLE audit_executions
    ADD CONSTRAINT audit_executions_collection_receipt_id_key UNIQUE (collection_receipt_id);

ALTER TABLE audit_executions
    ADD CONSTRAINT audit_executions_identity UNIQUE (execution_id, audit_id, round_id);

ALTER TABLE audit_executions
    ADD CONSTRAINT audit_executions_pkey PRIMARY KEY (execution_id);

ALTER TABLE audit_executions
    ADD CONSTRAINT audit_executions_submission_identity UNIQUE (execution_id, submission_key);

ALTER TABLE audit_executions
    ADD CONSTRAINT audit_executions_submission_key UNIQUE (submission_key);

CREATE INDEX audit_executions_audit_run_cleanup_idx ON audit_executions USING btree (audit_id, run_deleted_at, created_at, execution_id) WHERE (run_id IS NOT NULL);

CREATE UNIQUE INDEX audit_executions_prepare_accepted_unique ON audit_executions USING btree (audit_id, workflow_role) WHERE ((role = 'prepare'::text) AND (collection_disposition = 'accepted-result'::text));

CREATE UNIQUE INDEX audit_executions_prepare_attempt_unique ON audit_executions USING btree (audit_id, workflow_role, role_attempt) WHERE (role = 'prepare'::text);

CREATE INDEX audit_executions_reconcile_idx ON audit_executions USING btree (audit_id, state, created_at, execution_id) WHERE (state = ANY (ARRAY['intent'::text, 'submitted'::text, 'collecting'::text]));

CREATE UNIQUE INDEX audit_executions_role_attempt_unique ON audit_executions USING btree (audit_id, COALESCE(round_id, ''::text), role, workflow_role, COALESCE(role_attempt, 0)) WHERE (role = ANY (ARRAY['discovery'::text, 'assessment'::text]));

CREATE UNIQUE INDEX audit_executions_run_unique ON audit_executions USING btree (run_id) WHERE (run_id IS NOT NULL);

CREATE TRIGGER audit_executions_project_admission BEFORE INSERT ON audit_executions FOR EACH ROW EXECUTE FUNCTION contractor_guard_audit_child_admission();

CREATE TRIGGER audit_executions_protect_collection BEFORE UPDATE ON audit_executions FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_execution_collection();

CREATE CONSTRAINT TRIGGER audit_executions_require_matching_run AFTER INSERT OR UPDATE ON audit_executions DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION contractor_require_matching_audit_run();

CREATE TRIGGER audit_executions_validate_preparation BEFORE INSERT OR UPDATE ON audit_executions FOR EACH ROW EXECUTE FUNCTION contractor_validate_audit_preparation_execution();

CREATE TRIGGER eval_audit_children_projection AFTER INSERT OR DELETE OR UPDATE ON audit_executions FOR EACH ROW EXECUTE FUNCTION contractor_eval_audit_child_changed();

CREATE TABLE audit_idempotency (
    owner_id text NOT NULL,
    operation text NOT NULL,
    idempotency_key text NOT NULL,
    request_digest text NOT NULL,
    audit_id text NOT NULL,
    resource_id text NOT NULL,
    response_snapshot jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT audit_idempotency_idempotency_key_check CHECK ((idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'::text)),
    CONSTRAINT audit_idempotency_operation_check CHECK ((operation ~ '^[a-z][a-z0-9_.-]{0,127}$'::text)),
    CONSTRAINT audit_idempotency_owner_id_check CHECK (((btrim(owner_id) <> ''::text) AND (octet_length(owner_id) <= 256))),
    CONSTRAINT audit_idempotency_request_digest_check CHECK ((request_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audit_idempotency_resource_id_check CHECK (((btrim(resource_id) <> ''::text) AND (octet_length(resource_id) <= 256))),
    CONSTRAINT audit_idempotency_response_snapshot_check CHECK (((jsonb_typeof(response_snapshot) = 'object'::text) AND (octet_length((response_snapshot)::text) <= 1048576)))
);

ALTER TABLE audit_idempotency
    ADD CONSTRAINT audit_idempotency_pkey PRIMARY KEY (owner_id, operation, idempotency_key);

CREATE INDEX audit_idempotency_audit_idx ON audit_idempotency USING btree (audit_id);

CREATE TABLE audit_items (
    item_id text NOT NULL,
    audit_id text NOT NULL,
    round_id text NOT NULL,
    item_key text NOT NULL,
    ordinal integer NOT NULL,
    kind text NOT NULL,
    subject_key text NOT NULL,
    task_ref jsonb NOT NULL,
    task_digest text NOT NULL,
    workflow_role text NOT NULL,
    state text NOT NULL,
    final_disposition text,
    accepted_result_ref jsonb,
    accepted_result_digest text,
    last_execution_item_id text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    origin jsonb NOT NULL,
    approval_kind text DEFAULT 'none'::text NOT NULL,
    approval_subject_digest text,
    coverage_status text NOT NULL,
    coverage_requested jsonb NOT NULL,
    coverage_completed jsonb NOT NULL,
    coverage_gaps jsonb NOT NULL,
    coverage_rationale text DEFAULT ''::text NOT NULL,
    coverage_result_ref jsonb,
    coverage_result_digest text,
    coverage_updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    proposal_receipt_id text,
    proposal_check_ordinal integer,
    proposal_ref jsonb,
    proposal_digest text,
    CONSTRAINT audit_items_accepted_result_digest_check CHECK (((accepted_result_digest IS NULL) OR (accepted_result_digest ~ '^sha256:[0-9a-f]{64}$'::text))),
    CONSTRAINT audit_items_accepted_result_ref_check CHECK (((accepted_result_ref IS NULL) OR ((jsonb_typeof(accepted_result_ref) = 'object'::text) AND (octet_length((accepted_result_ref)::text) <= 4096)))),
    CONSTRAINT audit_items_approval_kind_check CHECK ((approval_kind = ANY (ARRAY['none'::text, 'active-check-approval'::text, 'requirement-applicability'::text]))),
    CONSTRAINT audit_items_approval_shape CHECK (((approval_kind = 'none'::text) = (approval_subject_digest IS NULL))),
    CONSTRAINT audit_items_approval_subject_digest_check CHECK (((approval_subject_digest IS NULL) OR (approval_subject_digest ~ '^sha256:[0-9a-f]{64}$'::text))),
    CONSTRAINT audit_items_coverage_completed_check CHECK (((jsonb_typeof(coverage_completed) = 'array'::text) AND (octet_length((coverage_completed)::text) <= 1048576))),
    CONSTRAINT audit_items_coverage_gaps_check CHECK (((jsonb_typeof(coverage_gaps) = 'array'::text) AND (octet_length((coverage_gaps)::text) <= 1048576))),
    CONSTRAINT audit_items_coverage_rationale_check CHECK ((octet_length(coverage_rationale) <= 4096)),
    CONSTRAINT audit_items_coverage_requested_check CHECK (((jsonb_typeof(coverage_requested) = 'array'::text) AND (octet_length((coverage_requested)::text) <= 1048576))),
    CONSTRAINT audit_items_coverage_result_shape CHECK ((((coverage_result_ref IS NULL) = (coverage_result_digest IS NULL)) AND ((coverage_result_ref IS NULL) OR ((jsonb_typeof(coverage_result_ref) = 'object'::text) AND (octet_length((coverage_result_ref)::text) <= 4096) AND (coverage_result_digest ~ '^sha256:[0-9a-f]{64}$'::text))))),
    CONSTRAINT audit_items_coverage_status_check CHECK ((coverage_status = ANY (ARRAY['not-tested'::text, 'inconclusive'::text, 'satisfied'::text, 'violated'::text, 'not-applicable'::text, 'blocked'::text, 'excluded'::text, 'traced-complete'::text, 'traced-partial'::text, 'unmapped'::text]))),
    CONSTRAINT audit_items_final_disposition_check CHECK (((final_disposition IS NULL) OR (final_disposition = ANY (ARRAY['accepted-result'::text, 'missing-output'::text, 'invalid-result'::text, 'execution-failed'::text, 'execution-cancelled'::text, 'excluded'::text, 'not-applicable'::text])))),
    CONSTRAINT audit_items_item_id_check CHECK ((item_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT audit_items_item_key_check CHECK (((btrim(item_key) <> ''::text) AND (octet_length(item_key) <= 256))),
    CONSTRAINT audit_items_kind_check CHECK (((btrim(kind) <> ''::text) AND (octet_length(kind) <= 128))),
    CONSTRAINT audit_items_ordinal_check CHECK ((ordinal >= 0)),
    CONSTRAINT audit_items_origin_shape CHECK (((jsonb_typeof(origin) = 'object'::text) AND ((origin ->> 'schema'::text) = 'contractor.audit.item-origin.v1'::text) AND ((origin ->> 'entryKey'::text) = item_key) AND (octet_length((origin)::text) <= 16384) AND (((origin -> 'provenanceIncomplete'::text) = 'true'::jsonb) OR ((jsonb_typeof((origin -> 'sourceRef'::text)) = 'object'::text) AND ((origin ->> 'sourceContentDigest'::text) ~ '^sha256:[0-9a-f]{64}$'::text) AND ((origin ->> 'canonicalInventoryDigest'::text) ~ '^sha256:[0-9a-f]{64}$'::text) AND (btrim((origin ->> 'sourceMediaType'::text)) <> ''::text) AND (octet_length((origin ->> 'sourceMediaType'::text)) <= 256))))),
    CONSTRAINT audit_items_proposal_check_ordinal_check CHECK (((proposal_check_ordinal >= 0) AND (proposal_check_ordinal < 512))),
    CONSTRAINT audit_items_proposal_digest_check CHECK ((proposal_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audit_items_proposal_ref_check CHECK (((jsonb_typeof(proposal_ref) = 'object'::text) AND (octet_length((proposal_ref)::text) <= 4096))),
    CONSTRAINT audit_items_proposal_shape CHECK ((((proposal_receipt_id IS NULL) = (proposal_check_ordinal IS NULL)) AND ((proposal_receipt_id IS NULL) = (proposal_ref IS NULL)) AND ((proposal_receipt_id IS NULL) = (proposal_digest IS NULL)))),
    CONSTRAINT audit_items_result_shape CHECK ((((accepted_result_ref IS NULL) = (accepted_result_digest IS NULL)) AND ((accepted_result_ref IS NULL) OR (final_disposition = 'accepted-result'::text)))),
    CONSTRAINT audit_items_settlement_shape CHECK (((state = 'settled'::text) = (final_disposition IS NOT NULL))),
    CONSTRAINT audit_items_state_check CHECK ((state = ANY (ARRAY['pending'::text, 'awaiting_review'::text, 'ready'::text, 'submitted'::text, 'collecting'::text, 'settled'::text]))),
    CONSTRAINT audit_items_subject_key_check CHECK (((btrim(subject_key) <> ''::text) AND (octet_length(subject_key) <= 512))),
    CONSTRAINT audit_items_task_digest_check CHECK ((task_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audit_items_task_ref_check CHECK (((jsonb_typeof(task_ref) = 'object'::text) AND (octet_length((task_ref)::text) <= 4096))),
    CONSTRAINT audit_items_workflow_role_check CHECK (((btrim(workflow_role) <> ''::text) AND (octet_length(workflow_role) <= 128)))
);

ALTER TABLE audit_items
    ADD CONSTRAINT audit_items_audit_identity UNIQUE (item_id, audit_id, round_id);

ALTER TABLE audit_items
    ADD CONSTRAINT audit_items_pkey PRIMARY KEY (item_id);

ALTER TABLE audit_items
    ADD CONSTRAINT audit_items_proposal_check_once UNIQUE (audit_id, proposal_receipt_id, proposal_check_ordinal);

ALTER TABLE audit_items
    ADD CONSTRAINT audit_items_round_key UNIQUE (round_id, item_key);

ALTER TABLE audit_items
    ADD CONSTRAINT audit_items_round_ordinal UNIQUE (round_id, ordinal);

CREATE INDEX audit_items_dispatch_idx ON audit_items USING btree (audit_id, state, round_id, ordinal, item_id) WHERE (state <> 'settled'::text);

CREATE INDEX audit_items_pending_approval_idx ON audit_items USING btree (audit_id, state, approval_kind, item_id) WHERE (state = 'awaiting_review'::text);

CREATE TRIGGER audit_items_protect_source BEFORE UPDATE ON audit_items FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_item_source();

CREATE TABLE audit_rounds (
    round_id text NOT NULL,
    audit_id text NOT NULL,
    ordinal integer NOT NULL,
    manifest_ref jsonb NOT NULL,
    manifest_digest text NOT NULL,
    state text NOT NULL,
    expected_item_count integer NOT NULL,
    revision bigint DEFAULT 1 NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    acceptance_digest text,
    CONSTRAINT audit_rounds_acceptance_digest_check CHECK (((acceptance_digest IS NULL) OR (acceptance_digest ~ '^sha256:[0-9a-f]{64}$'::text))),
    CONSTRAINT audit_rounds_expected_item_count_check CHECK (((expected_item_count >= 0) AND (expected_item_count <= 10000))),
    CONSTRAINT audit_rounds_manifest_digest_check CHECK ((manifest_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audit_rounds_manifest_ref_check CHECK (((jsonb_typeof(manifest_ref) = 'object'::text) AND (octet_length((manifest_ref)::text) <= 4096))),
    CONSTRAINT audit_rounds_ordinal_check CHECK ((ordinal > 0)),
    CONSTRAINT audit_rounds_revision_check CHECK ((revision > 0)),
    CONSTRAINT audit_rounds_round_id_check CHECK ((round_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT audit_rounds_state_check CHECK ((state = ANY (ARRAY['proposed'::text, 'accepted'::text, 'executing'::text, 'assessing'::text, 'closed'::text])))
);

ALTER TABLE audit_rounds
    ADD CONSTRAINT audit_rounds_audit_id_ordinal_key UNIQUE (audit_id, ordinal);

ALTER TABLE audit_rounds
    ADD CONSTRAINT audit_rounds_pkey PRIMARY KEY (round_id);

ALTER TABLE audit_rounds
    ADD CONSTRAINT audit_rounds_round_id_audit_id_key UNIQUE (round_id, audit_id);

CREATE TRIGGER audit_rounds_project_admission BEFORE INSERT ON audit_rounds FOR EACH ROW EXECUTE FUNCTION contractor_guard_audit_child_admission();

CREATE TABLE audits (
    audit_id text NOT NULL,
    owner_id text NOT NULL,
    project_id text NOT NULL,
    profile_name text NOT NULL,
    profile_version text NOT NULL,
    profile_digest text NOT NULL,
    profile_snapshot jsonb NOT NULL,
    input_selection jsonb NOT NULL,
    baseline_snapshot jsonb,
    state text DEFAULT 'draft'::text NOT NULL,
    revision bigint DEFAULT 1 NOT NULL,
    current_round_id text,
    dispatch_state text DEFAULT 'open'::text NOT NULL,
    hold_state text DEFAULT 'pending'::text NOT NULL,
    deadline_at timestamp with time zone,
    max_rounds integer NOT NULL,
    batch_size integer NOT NULL,
    max_items_per_round integer NOT NULL,
    max_items_total integer NOT NULL,
    max_submitted_runs integer NOT NULL,
    max_item_run_attempts integer NOT NULL,
    max_evidence_bytes bigint NOT NULL,
    reserved_run_count integer DEFAULT 0 NOT NULL,
    submitted_run_count integer DEFAULT 0 NOT NULL,
    outstanding_run_count integer DEFAULT 0 NOT NULL,
    retained_evidence_bytes bigint DEFAULT 0 NOT NULL,
    next_event_sequence bigint DEFAULT 1 NOT NULL,
    stop_reason_code text,
    stop_reason_message text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    started_at timestamp with time zone,
    finished_at timestamp with time zone,
    deletion_requested_at timestamp with time zone,
    paused_at timestamp with time zone,
    phase text DEFAULT 'not-started'::text NOT NULL,
    CONSTRAINT audits_audit_id_check CHECK ((audit_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT audits_baseline_shape CHECK (((state = ANY (ARRAY['draft'::text, 'deleting'::text])) OR (baseline_snapshot IS NOT NULL))),
    CONSTRAINT audits_baseline_snapshot_check CHECK (((baseline_snapshot IS NULL) OR ((jsonb_typeof(baseline_snapshot) = 'object'::text) AND (octet_length((baseline_snapshot)::text) <= 16777216)))),
    CONSTRAINT audits_batch_size_check CHECK (((batch_size > 0) AND (batch_size <= 64))),
    CONSTRAINT audits_counter_bounds CHECK (((reserved_run_count <= max_submitted_runs) AND (submitted_run_count <= reserved_run_count) AND (outstanding_run_count <= reserved_run_count) AND (retained_evidence_bytes <= max_evidence_bytes))),
    CONSTRAINT audits_deletion_intent_shape CHECK ((((deletion_requested_at IS NULL) AND (state <> 'deleting'::text)) OR ((deletion_requested_at IS NOT NULL) AND (state = ANY (ARRAY['cancelling'::text, 'deleting'::text]))))),
    CONSTRAINT audits_dispatch_state_check CHECK ((dispatch_state = ANY (ARRAY['open'::text, 'closed'::text]))),
    CONSTRAINT audits_hold_state_check CHECK ((hold_state = ANY (ARRAY['pending'::text, 'held'::text, 'released'::text]))),
    CONSTRAINT audits_input_selection_check CHECK (((jsonb_typeof(input_selection) = 'object'::text) AND (octet_length((input_selection)::text) <= 8388608))),
    CONSTRAINT audits_max_evidence_bytes_check CHECK (((max_evidence_bytes > 0) AND (max_evidence_bytes <= 1073741824))),
    CONSTRAINT audits_max_item_run_attempts_check CHECK (((max_item_run_attempts > 0) AND (max_item_run_attempts <= 10))),
    CONSTRAINT audits_max_items_per_round_check CHECK (((max_items_per_round > 0) AND (max_items_per_round <= 10000))),
    CONSTRAINT audits_max_items_total_check CHECK (((max_items_total > 0) AND (max_items_total <= 100000))),
    CONSTRAINT audits_max_rounds_check CHECK (((max_rounds > 0) AND (max_rounds <= 32))),
    CONSTRAINT audits_max_submitted_runs_check CHECK (((max_submitted_runs > 0) AND (max_submitted_runs <= 1000000))),
    CONSTRAINT audits_next_event_sequence_check CHECK ((next_event_sequence > 0)),
    CONSTRAINT audits_outstanding_run_count_check CHECK ((outstanding_run_count >= 0)),
    CONSTRAINT audits_owner_id_check CHECK (((btrim(owner_id) <> ''::text) AND (octet_length(owner_id) <= 256))),
    CONSTRAINT audits_phase_shape CHECK (((phase = ANY (ARRAY['not-started'::text, 'preparing'::text, 'inventory'::text, 'rounds'::text])) AND ((phase <> ALL (ARRAY['preparing'::text, 'inventory'::text])) OR ((current_round_id IS NULL) AND (baseline_snapshot IS NOT NULL))))),
    CONSTRAINT audits_profile_digest_check CHECK ((profile_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audits_profile_name_check CHECK (((btrim(profile_name) <> ''::text) AND (octet_length(profile_name) <= 128))),
    CONSTRAINT audits_profile_snapshot_check CHECK (((jsonb_typeof(profile_snapshot) = 'object'::text) AND (octet_length((profile_snapshot)::text) <= 8388608))),
    CONSTRAINT audits_profile_version_check CHECK (((btrim(profile_version) <> ''::text) AND (octet_length(profile_version) <= 128))),
    CONSTRAINT audits_reserved_run_count_check CHECK ((reserved_run_count >= 0)),
    CONSTRAINT audits_retained_evidence_bytes_check CHECK ((retained_evidence_bytes >= 0)),
    CONSTRAINT audits_revision_check CHECK ((revision > 0)),
    CONSTRAINT audits_state_check CHECK ((state = ANY (ARRAY['draft'::text, 'active'::text, 'waiting_review'::text, 'paused'::text, 'finalizing'::text, 'cancelling'::text, 'completed'::text, 'cancelled'::text, 'failed'::text, 'deleting'::text]))),
    CONSTRAINT audits_stop_reason_code_check CHECK (((stop_reason_code IS NULL) OR ((btrim(stop_reason_code) <> ''::text) AND (octet_length(stop_reason_code) <= 128)))),
    CONSTRAINT audits_stop_reason_message_check CHECK (((stop_reason_message IS NULL) OR (octet_length(stop_reason_message) <= 4096))),
    CONSTRAINT audits_stop_reason_shape CHECK (((stop_reason_code IS NULL) = (stop_reason_message IS NULL))),
    CONSTRAINT audits_submitted_run_count_check CHECK ((submitted_run_count >= 0)),
    CONSTRAINT audits_time_shape CHECK ((((state = 'draft'::text) AND (started_at IS NULL) AND (finished_at IS NULL)) OR ((state = ANY (ARRAY['active'::text, 'waiting_review'::text, 'paused'::text, 'finalizing'::text, 'cancelling'::text])) AND (started_at IS NOT NULL) AND (finished_at IS NULL)) OR ((state = ANY (ARRAY['completed'::text, 'cancelled'::text, 'failed'::text])) AND (started_at IS NOT NULL) AND (finished_at IS NOT NULL)) OR (state = 'deleting'::text)))
);

ALTER TABLE audits
    ADD CONSTRAINT audits_pkey PRIMARY KEY (audit_id);

ALTER TABLE audits
    ADD CONSTRAINT audits_project_owner_identity UNIQUE (audit_id, project_id, owner_id);

CREATE INDEX audits_owner_created_idx ON audits USING btree (owner_id, created_at DESC, audit_id DESC);

CREATE INDEX audits_project_deletion_intent_idx ON audits USING btree (project_id, deletion_requested_at, created_at, audit_id);

CREATE INDEX audits_project_state_idx ON audits USING btree (project_id, state, created_at, audit_id);

CREATE INDEX audits_reconcile_idx ON audits USING btree (updated_at, audit_id) WHERE (state = ANY (ARRAY['active'::text, 'waiting_review'::text, 'paused'::text, 'finalizing'::text, 'cancelling'::text, 'deleting'::text]));

CREATE TRIGGER audits_eval_tombstone BEFORE DELETE ON audits FOR EACH ROW EXECUTE FUNCTION contractor_eval_retain_deleted_execution();

CREATE TRIGGER audits_project_admission BEFORE INSERT ON audits FOR EACH ROW EXECUTE FUNCTION contractor_guard_audit_admission();

CREATE TRIGGER audits_protect_preparation_baseline BEFORE UPDATE ON audits FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_preparation_baseline();

CREATE TRIGGER eval_audits_projection AFTER DELETE OR UPDATE OF state, finished_at, dispatch_state ON audits FOR EACH ROW EXECUTE FUNCTION contractor_eval_execution_changed();


-- Findings: proposals, deduplication, assessments and reviews

CREATE TABLE audit_finding_assessments (
    assessment_id text NOT NULL,
    finding_id text NOT NULL,
    audit_id text NOT NULL,
    receipt_id text NOT NULL,
    item_id text,
    execution_item_id text,
    collection_receipt_id text,
    semantic_assessment text NOT NULL,
    result_ref jsonb NOT NULL,
    result_digest text NOT NULL,
    direct_verification boolean DEFAULT false NOT NULL,
    contract_ref jsonb,
    contract_digest text,
    accepted_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT audit_finding_assessments_assessment_id_check CHECK ((assessment_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT audit_finding_assessments_contract_shape CHECK ((((contract_ref IS NULL) = (contract_digest IS NULL)) AND ((NOT direct_verification) OR (contract_ref IS NOT NULL)) AND ((contract_ref IS NULL) OR ((jsonb_typeof(contract_ref) = 'object'::text) AND (octet_length((contract_ref)::text) <= 4096) AND (contract_digest ~ '^sha256:[0-9a-f]{64}$'::text))))),
    CONSTRAINT audit_finding_assessments_item_shape CHECK ((((item_id IS NULL) = (execution_item_id IS NULL)) AND (direct_verification OR (item_id IS NOT NULL)))),
    CONSTRAINT audit_finding_assessments_result_digest_check CHECK ((result_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audit_finding_assessments_result_ref_check CHECK (((jsonb_typeof(result_ref) = 'object'::text) AND (octet_length((result_ref)::text) <= 4096))),
    CONSTRAINT audit_finding_assessments_semantic_assessment_check CHECK ((semantic_assessment = ANY (ARRAY['supported'::text, 'refuted'::text, 'inconclusive'::text, 'blocked'::text, 'satisfied'::text, 'violated'::text, 'not-tested'::text])))
);

ALTER TABLE audit_finding_assessments
    ADD CONSTRAINT audit_finding_assessments_pkey PRIMARY KEY (assessment_id);

ALTER TABLE audit_finding_assessments
    ADD CONSTRAINT audit_finding_assessments_receipt_id_execution_item_id_resu_key UNIQUE (receipt_id, execution_item_id, result_digest);

CREATE INDEX audit_finding_assessments_audit_idx ON audit_finding_assessments USING btree (audit_id);

CREATE INDEX audit_finding_assessments_collection_receipt_idx ON audit_finding_assessments USING btree (collection_receipt_id);

CREATE INDEX audit_finding_assessments_execution_item_idx ON audit_finding_assessments USING btree (execution_item_id, item_id);

CREATE TRIGGER audit_finding_assessments_protect_update BEFORE UPDATE ON audit_finding_assessments FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_finding_history();

CREATE TABLE audit_finding_contributions (
    finding_id text NOT NULL,
    audit_id text NOT NULL,
    receipt_id text NOT NULL,
    relation text NOT NULL,
    proposal_ref jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT audit_finding_contributions_proposal_ref_check CHECK (((jsonb_typeof(proposal_ref) = 'object'::text) AND (octet_length((proposal_ref)::text) <= 4096))),
    CONSTRAINT audit_finding_contributions_relation_check CHECK ((relation = ANY (ARRAY['first'::text, 'contributing'::text])))
);

ALTER TABLE audit_finding_contributions
    ADD CONSTRAINT audit_finding_contributions_audit_id_receipt_id_key UNIQUE (audit_id, receipt_id);

ALTER TABLE audit_finding_contributions
    ADD CONSTRAINT audit_finding_contributions_pkey PRIMARY KEY (finding_id, receipt_id);

CREATE TRIGGER audit_finding_contributions_protect_update BEFORE UPDATE ON audit_finding_contributions FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_finding_history();

CREATE TABLE audit_findings (
    finding_id text NOT NULL,
    audit_id text NOT NULL,
    first_receipt_id text NOT NULL,
    first_proposal_ref jsonb NOT NULL,
    state text DEFAULT 'proposed'::text NOT NULL,
    rejection_reason text,
    duplicate_target_id text,
    current_assessment_id text,
    current_decision_id text,
    revision bigint DEFAULT 1 NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT audit_findings_duplicate_shape CHECK (((state = 'duplicate'::text) = (duplicate_target_id IS NOT NULL))),
    CONSTRAINT audit_findings_effective_decision_shape CHECK (((state = ANY (ARRAY['confirmed'::text, 'rejected'::text])) = (current_decision_id IS NOT NULL))),
    CONSTRAINT audit_findings_finding_id_check CHECK ((finding_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT audit_findings_first_proposal_ref_check CHECK (((jsonb_typeof(first_proposal_ref) = 'object'::text) AND (octet_length((first_proposal_ref)::text) <= 4096))),
    CONSTRAINT audit_findings_rejection_reason_check CHECK (((rejection_reason IS NULL) OR (rejection_reason = ANY (ARRAY['false-positive'::text, 'policy'::text, 'out-of-scope'::text])))),
    CONSTRAINT audit_findings_rejection_shape CHECK (((state = 'rejected'::text) = (rejection_reason IS NOT NULL))),
    CONSTRAINT audit_findings_revision_check CHECK ((revision > 0)),
    CONSTRAINT audit_findings_state_check CHECK ((state = ANY (ARRAY['proposed'::text, 'confirmed'::text, 'rejected'::text, 'duplicate'::text, 'needs-evidence'::text])))
);

ALTER TABLE audit_findings
    ADD CONSTRAINT audit_findings_audit_id_first_receipt_id_key UNIQUE (audit_id, first_receipt_id);

ALTER TABLE audit_findings
    ADD CONSTRAINT audit_findings_finding_id_audit_id_key UNIQUE (finding_id, audit_id);

ALTER TABLE audit_findings
    ADD CONSTRAINT audit_findings_pkey PRIMARY KEY (finding_id);

CREATE INDEX audit_findings_global_created_idx ON audit_findings USING btree (created_at DESC, finding_id DESC);

CREATE INDEX audit_findings_list_idx ON audit_findings USING btree (audit_id, created_at, finding_id);

CREATE INDEX audit_findings_state_idx ON audit_findings USING btree (audit_id, state, created_at, finding_id);

CREATE TABLE audit_review_decisions (
    decision_id text NOT NULL,
    request_id text NOT NULL,
    audit_id text NOT NULL,
    finding_id text,
    actor_id text NOT NULL,
    verdict text,
    severity text,
    rationale text NOT NULL,
    duplicate_target_id text,
    subject_revision bigint NOT NULL,
    subject_digest text NOT NULL,
    idempotency_key text NOT NULL,
    request_digest text NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    action text NOT NULL,
    CONSTRAINT audit_review_decisions_action_check CHECK ((action = ANY (ARRAY['true_positive'::text, 'false_positive'::text, 'duplicate'::text, 'reopen'::text, 'needs_evidence'::text, 'approve'::text, 'reject'::text, 'not_applicable'::text]))),
    CONSTRAINT audit_review_decisions_actor_id_check CHECK (((btrim(actor_id) <> ''::text) AND (octet_length(actor_id) <= 256))),
    CONSTRAINT audit_review_decisions_decision_id_check CHECK ((decision_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT audit_review_decisions_idempotency_key_check CHECK ((idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'::text)),
    CONSTRAINT audit_review_decisions_rationale_check CHECK (((btrim(rationale) <> ''::text) AND (octet_length(rationale) <= 65536))),
    CONSTRAINT audit_review_decisions_request_digest_check CHECK ((request_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audit_review_decisions_severity_check CHECK (((severity IS NULL) OR (severity = ANY (ARRAY['informational'::text, 'low'::text, 'medium'::text, 'high'::text, 'critical'::text])))),
    CONSTRAINT audit_review_decisions_shape CHECK ((((finding_id IS NOT NULL) AND (verdict IS NOT NULL) AND (action = verdict) AND ((verdict = 'true_positive'::text) = (severity IS NOT NULL)) AND ((verdict = 'duplicate'::text) = (duplicate_target_id IS NOT NULL))) OR ((finding_id IS NULL) AND (verdict IS NULL) AND (severity IS NULL) AND (duplicate_target_id IS NULL) AND (action = ANY (ARRAY['approve'::text, 'reject'::text, 'not_applicable'::text]))))),
    CONSTRAINT audit_review_decisions_subject_digest_check CHECK ((subject_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audit_review_decisions_subject_revision_check CHECK ((subject_revision > 0)),
    CONSTRAINT audit_review_decisions_verdict_check CHECK ((verdict = ANY (ARRAY['true_positive'::text, 'false_positive'::text, 'duplicate'::text, 'reopen'::text, 'needs_evidence'::text])))
);

ALTER TABLE audit_review_decisions
    ADD CONSTRAINT audit_review_decisions_audit_id_idempotency_key_key UNIQUE (audit_id, idempotency_key);

ALTER TABLE audit_review_decisions
    ADD CONSTRAINT audit_review_decisions_pkey PRIMARY KEY (decision_id);

ALTER TABLE audit_review_decisions
    ADD CONSTRAINT audit_review_decisions_request_id_key UNIQUE (request_id);

CREATE TRIGGER audit_review_decisions_protect_update BEFORE UPDATE ON audit_review_decisions FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_finding_history();

CREATE TABLE audit_review_requests (
    request_id text NOT NULL,
    audit_id text NOT NULL,
    finding_id text,
    kind text NOT NULL,
    subject_revision bigint NOT NULL,
    subject_digest text NOT NULL,
    requested_actions jsonb NOT NULL,
    state text DEFAULT 'pending'::text NOT NULL,
    expires_at timestamp with time zone,
    revision bigint DEFAULT 1 NOT NULL,
    idempotency_key text NOT NULL,
    request_digest text NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    subject_kind text NOT NULL,
    subject_id text NOT NULL,
    report_round_id text,
    report_machine_link jsonb,
    report_summary_link jsonb,
    CONSTRAINT audit_review_requests_idempotency_key_check CHECK ((idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'::text)),
    CONSTRAINT audit_review_requests_kind_check CHECK ((kind = ANY (ARRAY['plan-approval'::text, 'active-check-approval'::text, 'provide-evidence'::text, 'finding-triage'::text, 'requirement-applicability'::text, 'report-acceptance'::text]))),
    CONSTRAINT audit_review_requests_report_candidate_shape CHECK ((((subject_kind = 'audit-report'::text) = (report_round_id IS NOT NULL)) AND ((report_round_id IS NULL) = (report_machine_link IS NULL)) AND ((report_round_id IS NULL) = (report_summary_link IS NULL)))),
    CONSTRAINT audit_review_requests_report_machine_link_check CHECK (((jsonb_typeof(report_machine_link) = 'object'::text) AND (octet_length((report_machine_link)::text) <= 16384))),
    CONSTRAINT audit_review_requests_report_summary_link_check CHECK (((jsonb_typeof(report_summary_link) = 'object'::text) AND (octet_length((report_summary_link)::text) <= 16384))),
    CONSTRAINT audit_review_requests_request_digest_check CHECK ((request_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audit_review_requests_request_id_check CHECK ((request_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT audit_review_requests_requested_actions_check CHECK (((jsonb_typeof(requested_actions) = 'array'::text) AND (octet_length((requested_actions)::text) <= 65536))),
    CONSTRAINT audit_review_requests_revision_check CHECK ((revision > 0)),
    CONSTRAINT audit_review_requests_state_check CHECK ((state = ANY (ARRAY['pending'::text, 'decided'::text, 'expired'::text]))),
    CONSTRAINT audit_review_requests_subject_digest_check CHECK ((subject_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audit_review_requests_subject_id_check CHECK ((subject_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT audit_review_requests_subject_kind_check CHECK ((subject_kind = ANY (ARRAY['finding'::text, 'audit-item-action'::text, 'audit-report'::text]))),
    CONSTRAINT audit_review_requests_subject_revision_check CHECK ((subject_revision > 0)),
    CONSTRAINT audit_review_requests_subject_shape CHECK ((((subject_kind = 'finding'::text) AND (finding_id = subject_id)) OR ((subject_kind <> 'finding'::text) AND (finding_id IS NULL))))
);

ALTER TABLE audit_review_requests
    ADD CONSTRAINT audit_review_requests_audit_id_idempotency_key_key UNIQUE (audit_id, idempotency_key);

ALTER TABLE audit_review_requests
    ADD CONSTRAINT audit_review_requests_pkey PRIMARY KEY (request_id);

ALTER TABLE audit_review_requests
    ADD CONSTRAINT audit_review_requests_request_audit_unique UNIQUE (request_id, audit_id);

CREATE INDEX audit_review_requests_global_created_idx ON audit_review_requests USING btree (created_at DESC, request_id DESC);

CREATE INDEX audit_review_requests_global_state_created_idx ON audit_review_requests USING btree (state, created_at DESC, request_id DESC);

CREATE INDEX audit_review_requests_list_idx ON audit_review_requests USING btree (audit_id, created_at, request_id);

CREATE UNIQUE INDEX audit_review_requests_pending_subject_unique ON audit_review_requests USING btree (audit_id, subject_kind, subject_id, kind, subject_revision, subject_digest) WHERE (state = 'pending'::text);

CREATE UNIQUE INDEX audit_review_requests_report_candidate_unique ON audit_review_requests USING btree (audit_id) WHERE (subject_kind = 'audit-report'::text);

CREATE TABLE finding_proposal_audit_holds (
    receipt_id text NOT NULL,
    audit_id text NOT NULL,
    project_id text NOT NULL,
    proposal_ref jsonb NOT NULL,
    evidence jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT finding_proposal_audit_holds_evidence_check CHECK (((jsonb_typeof(evidence) = 'array'::text) AND (octet_length((evidence)::text) <= 1048576))),
    CONSTRAINT finding_proposal_audit_holds_proposal_ref_check CHECK (((jsonb_typeof(proposal_ref) = 'object'::text) AND (octet_length((proposal_ref)::text) <= 4096)))
);

ALTER TABLE finding_proposal_audit_holds
    ADD CONSTRAINT finding_proposal_audit_holds_pkey PRIMARY KEY (receipt_id, audit_id);

CREATE INDEX finding_proposal_audit_holds_audit_idx ON finding_proposal_audit_holds USING btree (audit_id, created_at, receipt_id);

CREATE INDEX finding_proposal_audit_holds_project_idx ON finding_proposal_audit_holds USING btree (project_id);

CREATE TRIGGER finding_proposal_audit_holds_admit_finding AFTER INSERT ON finding_proposal_audit_holds FOR EACH ROW EXECUTE FUNCTION contractor_admit_audit_finding();

CREATE TRIGGER finding_proposal_audit_holds_protect_update BEFORE UPDATE ON finding_proposal_audit_holds FOR EACH ROW EXECUTE FUNCTION contractor_protect_finding_proposal_receipt();

CREATE TRIGGER finding_proposal_audit_holds_reconcile AFTER INSERT OR DELETE ON finding_proposal_audit_holds FOR EACH ROW EXECUTE FUNCTION contractor_reconcile_finding_retention();

CREATE TABLE finding_proposal_receipts (
    receipt_id text NOT NULL,
    proposal_id text NOT NULL,
    allocation_id text NOT NULL,
    runtime_agent_id text NOT NULL,
    runtime_instance_id text NOT NULL,
    stage_execution_id text NOT NULL,
    logical_agent_name text NOT NULL,
    invocation_id text NOT NULL,
    submission_id text NOT NULL,
    client_key text NOT NULL,
    request_digest text NOT NULL,
    run_id text NOT NULL,
    owner_id text NOT NULL,
    project_id text,
    audit_execution_id text,
    audit_id text,
    audit_role text,
    workflow_name text NOT NULL,
    workflow_version text NOT NULL,
    workflow_schema_version text NOT NULL,
    workflow_configuration_ref jsonb NOT NULL,
    workflow_closure_digest text NOT NULL,
    proposal_ref jsonb NOT NULL,
    proposal_digest text NOT NULL,
    proposal_media_type text NOT NULL,
    proposal_size_bytes bigint NOT NULL,
    evidence jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    retention_state text DEFAULT 'source-held'::text NOT NULL,
    source_run_deleted_at timestamp with time zone,
    discarded_at timestamp with time zone,
    retention_updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT finding_proposal_receipts_allocation_id_check CHECK (((btrim(allocation_id) <> ''::text) AND (octet_length(allocation_id) <= 256))),
    CONSTRAINT finding_proposal_receipts_audit_role_check CHECK (((audit_role IS NULL) OR (audit_role = ANY (ARRAY['discovery'::text, 'check'::text, 'assessment'::text])))),
    CONSTRAINT finding_proposal_receipts_audit_shape CHECK ((((audit_execution_id IS NULL) AND (audit_id IS NULL) AND (audit_role IS NULL)) OR ((audit_execution_id IS NOT NULL) AND (audit_id IS NOT NULL) AND (audit_role IS NOT NULL)))),
    CONSTRAINT finding_proposal_receipts_client_key_check CHECK ((client_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$'::text)),
    CONSTRAINT finding_proposal_receipts_evidence_check CHECK (((jsonb_typeof(evidence) = 'array'::text) AND (octet_length((evidence)::text) <= 1048576))),
    CONSTRAINT finding_proposal_receipts_invocation_id_check CHECK (((btrim(invocation_id) <> ''::text) AND (octet_length(invocation_id) <= 128))),
    CONSTRAINT finding_proposal_receipts_logical_agent_name_check CHECK (((btrim(logical_agent_name) <> ''::text) AND (octet_length(logical_agent_name) <= 128))),
    CONSTRAINT finding_proposal_receipts_owner_id_check CHECK (((btrim(owner_id) <> ''::text) AND (octet_length(owner_id) <= 256))),
    CONSTRAINT finding_proposal_receipts_proposal_digest_check CHECK ((proposal_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT finding_proposal_receipts_proposal_id_check CHECK ((proposal_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT finding_proposal_receipts_proposal_media_type_check CHECK ((proposal_media_type = 'application/json'::text)),
    CONSTRAINT finding_proposal_receipts_proposal_ref_check CHECK (((jsonb_typeof(proposal_ref) = 'object'::text) AND (octet_length((proposal_ref)::text) <= 4096))),
    CONSTRAINT finding_proposal_receipts_proposal_size_bytes_check CHECK (((proposal_size_bytes >= 0) AND (proposal_size_bytes <= 8388608))),
    CONSTRAINT finding_proposal_receipts_receipt_id_check CHECK ((receipt_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT finding_proposal_receipts_request_digest_check CHECK ((request_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT finding_proposal_receipts_retention_shape CHECK ((((retention_state = 'source-held'::text) AND (source_run_deleted_at IS NULL) AND (discarded_at IS NULL)) OR ((retention_state = 'audit-held'::text) AND (discarded_at IS NULL)) OR ((retention_state = 'discarded'::text) AND (source_run_deleted_at IS NOT NULL) AND (discarded_at IS NOT NULL)))),
    CONSTRAINT finding_proposal_receipts_retention_state_check CHECK ((retention_state = ANY (ARRAY['source-held'::text, 'audit-held'::text, 'discarded'::text]))),
    CONSTRAINT finding_proposal_receipts_run_id_check CHECK (((btrim(run_id) <> ''::text) AND (octet_length(run_id) <= 256))),
    CONSTRAINT finding_proposal_receipts_runtime_agent_id_check CHECK (((btrim(runtime_agent_id) <> ''::text) AND (octet_length(runtime_agent_id) <= 256))),
    CONSTRAINT finding_proposal_receipts_runtime_instance_id_check CHECK (((btrim(runtime_instance_id) <> ''::text) AND (octet_length(runtime_instance_id) <= 256))),
    CONSTRAINT finding_proposal_receipts_stage_execution_id_check CHECK (((btrim(stage_execution_id) <> ''::text) AND (octet_length(stage_execution_id) <= 256))),
    CONSTRAINT finding_proposal_receipts_submission_id_check CHECK (((btrim(submission_id) <> ''::text) AND (octet_length(submission_id) <= 128))),
    CONSTRAINT finding_proposal_receipts_workflow_closure_digest_check CHECK ((workflow_closure_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT finding_proposal_receipts_workflow_configuration_ref_check CHECK (((jsonb_typeof(workflow_configuration_ref) = 'object'::text) AND (octet_length((workflow_configuration_ref)::text) <= 4096))),
    CONSTRAINT finding_proposal_receipts_workflow_name_check CHECK (((btrim(workflow_name) <> ''::text) AND (octet_length(workflow_name) <= 128))),
    CONSTRAINT finding_proposal_receipts_workflow_schema_version_check CHECK (((btrim(workflow_schema_version) <> ''::text) AND (octet_length(workflow_schema_version) <= 128))),
    CONSTRAINT finding_proposal_receipts_workflow_version_check CHECK (((btrim(workflow_version) <> ''::text) AND (octet_length(workflow_version) <= 128)))
);

ALTER TABLE finding_proposal_receipts
    ADD CONSTRAINT finding_proposal_receipts_allocation_id_invocation_id_submi_key UNIQUE (allocation_id, invocation_id, submission_id);

ALTER TABLE finding_proposal_receipts
    ADD CONSTRAINT finding_proposal_receipts_pkey PRIMARY KEY (receipt_id);

ALTER TABLE finding_proposal_receipts
    ADD CONSTRAINT finding_proposal_receipts_proposal_id_key UNIQUE (proposal_id);

CREATE INDEX finding_proposal_receipts_audit_idx ON finding_proposal_receipts USING btree (audit_id, created_at, receipt_id) WHERE (audit_id IS NOT NULL);

CREATE INDEX finding_proposal_receipts_run_idx ON finding_proposal_receipts USING btree (run_id, created_at, receipt_id);

CREATE TRIGGER finding_proposal_receipts_protect_immutable BEFORE DELETE OR UPDATE ON finding_proposal_receipts FOR EACH ROW EXECUTE FUNCTION contractor_protect_finding_proposal_receipt_retention();


-- Evals: experiments, the case matrix, results and UI projections

CREATE TABLE eval_collections (
    owner_id text NOT NULL,
    project_id text NOT NULL,
    revision bigint DEFAULT 1 NOT NULL
);

ALTER TABLE eval_collections
    ADD CONSTRAINT eval_collections_pkey PRIMARY KEY (owner_id, project_id);

CREATE TABLE eval_commands (
    command_id text NOT NULL,
    experiment_id text NOT NULL,
    actor_id text NOT NULL,
    kind text NOT NULL,
    state text NOT NULL,
    accepted_revision bigint NOT NULL,
    diagnostic jsonb,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    finished_at timestamp with time zone,
    CONSTRAINT eval_commands_state_check CHECK ((state = ANY (ARRAY['accepted'::text, 'running'::text, 'succeeded'::text, 'failed'::text])))
);

ALTER TABLE eval_commands
    ADD CONSTRAINT eval_commands_pkey PRIMARY KEY (command_id);

CREATE INDEX eval_commands_pending_idx ON eval_commands USING btree (experiment_id, created_at, command_id) WHERE (state = ANY (ARRAY['accepted'::text, 'running'::text]));

CREATE TABLE eval_controller_claims (
    experiment_id text NOT NULL,
    epoch bigint DEFAULT 0 NOT NULL,
    holder_id text,
    expires_at timestamp with time zone,
    CONSTRAINT eval_controller_claims_check CHECK (((holder_id IS NULL) = (expires_at IS NULL))),
    CONSTRAINT eval_controller_claims_epoch_check CHECK ((epoch >= 0))
);

ALTER TABLE eval_controller_claims
    ADD CONSTRAINT eval_controller_claims_pkey PRIMARY KEY (experiment_id);

CREATE INDEX eval_claims_expiry_idx ON eval_controller_claims USING btree (expires_at, experiment_id);

CREATE TABLE eval_dataset_revisions (
    owner_id text NOT NULL,
    project_id text NOT NULL,
    dataset_id text NOT NULL,
    revision text NOT NULL,
    document bytea NOT NULL,
    document_sha256 text GENERATED ALWAYS AS (('sha256:'::text || encode(sha256(document), 'hex'::text))) STORED,
    metadata jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT eval_dataset_revisions_document_check CHECK (((octet_length(document) >= 2) AND (octet_length(document) <= 1048576))),
    CONSTRAINT eval_dataset_revisions_metadata_check CHECK ((jsonb_typeof(metadata) = 'object'::text))
);

ALTER TABLE eval_dataset_revisions
    ADD CONSTRAINT eval_dataset_revisions_pkey PRIMARY KEY (project_id, dataset_id, revision);

CREATE INDEX eval_datasets_owner_page_idx ON eval_dataset_revisions USING btree (owner_id, project_id, dataset_id, revision);

CREATE TRIGGER eval_datasets_collection AFTER INSERT OR DELETE ON eval_dataset_revisions FOR EACH ROW EXECUTE FUNCTION contractor_eval_collection_changed();

CREATE TRIGGER eval_datasets_immutable BEFORE DELETE OR UPDATE ON eval_dataset_revisions FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();

CREATE TABLE eval_evidence_refs (
    experiment_id text NOT NULL,
    member_id text NOT NULL,
    record_sha256 text NOT NULL,
    scope_kind text NOT NULL,
    scope_id text NOT NULL,
    namespace text NOT NULL,
    name text NOT NULL,
    revision text NOT NULL
);

ALTER TABLE eval_evidence_refs
    ADD CONSTRAINT eval_evidence_refs_pkey PRIMARY KEY (experiment_id, member_id, record_sha256, scope_kind, scope_id, namespace, name, revision);

CREATE INDEX eval_evidence_refs_artifact ON eval_evidence_refs USING btree (scope_kind, scope_id, namespace, name, revision);

CREATE TABLE eval_execution_tombstones (
    experiment_id text NOT NULL,
    member_id text NOT NULL,
    execution_kind text NOT NULL,
    execution_id text NOT NULL,
    terminal_state text NOT NULL,
    never_started boolean NOT NULL,
    deleted_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL
);

ALTER TABLE eval_execution_tombstones
    ADD CONSTRAINT eval_execution_tombstones_pkey PRIMARY KEY (experiment_id, member_id);

CREATE TRIGGER eval_tombstones_immutable BEFORE DELETE OR UPDATE ON eval_execution_tombstones FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();

CREATE TABLE eval_experiments (
    experiment_id text NOT NULL,
    owner_id text NOT NULL,
    project_id text NOT NULL,
    portable_id text NOT NULL,
    control_mode text NOT NULL,
    name text NOT NULL,
    state text NOT NULL,
    revision bigint DEFAULT 1 NOT NULL,
    draft bytea,
    dataset_id text,
    dataset_revision text,
    expected_count integer DEFAULT 0 NOT NULL,
    outstanding_count integer DEFAULT 0 NOT NULL,
    max_in_flight integer NOT NULL,
    wall_ms bigint NOT NULL,
    token_limit bigint,
    observed_tokens bigint DEFAULT 0 NOT NULL,
    started_at timestamp with time zone,
    deadline_at timestamp with time zone,
    last_producer_activity_at timestamp with time zone,
    deletion_requested_at timestamp with time zone,
    diagnostic jsonb,
    view_generation bigint DEFAULT 1 NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT eval_experiments_check CHECK (((outstanding_count >= 0) AND (outstanding_count <= expected_count))),
    CONSTRAINT eval_experiments_check1 CHECK (((dataset_id IS NULL) = (dataset_revision IS NULL))),
    CONSTRAINT eval_experiments_check2 CHECK ((((started_at IS NULL) AND (deadline_at IS NULL)) OR ((started_at IS NOT NULL) AND (deadline_at > started_at)))),
    CONSTRAINT eval_experiments_check3 CHECK (((draft IS NULL) OR ((control_mode = 'server'::text) AND ((octet_length(draft) >= 2) AND (octet_length(draft) <= 1048576))))),
    CONSTRAINT eval_experiments_control_mode_check CHECK ((control_mode = ANY (ARRAY['server'::text, 'external'::text]))),
    CONSTRAINT eval_experiments_expected_count_check CHECK (((expected_count >= 0) AND (expected_count <= 10000))),
    CONSTRAINT eval_experiments_max_in_flight_check CHECK (((max_in_flight >= 1) AND (max_in_flight <= 10000))),
    CONSTRAINT eval_experiments_name_check CHECK (((length(name) >= 1) AND (length(name) <= 256))),
    CONSTRAINT eval_experiments_observed_tokens_check CHECK ((observed_tokens >= 0)),
    CONSTRAINT eval_experiments_revision_check CHECK ((revision > 0)),
    CONSTRAINT eval_experiments_state_check CHECK ((state = ANY (ARRAY['draft'::text, 'preparing'::text, 'ready'::text, 'running'::text, 'settling'::text, 'finished'::text, 'pausing'::text, 'paused'::text, 'cancelling'::text, 'cancelled'::text]))),
    CONSTRAINT eval_experiments_token_limit_check CHECK ((token_limit >= 0)),
    CONSTRAINT eval_experiments_view_generation_check CHECK ((view_generation > 0)),
    CONSTRAINT eval_experiments_wall_ms_check CHECK ((wall_ms > 0))
);

ALTER TABLE eval_experiments
    ADD CONSTRAINT eval_experiments_owner_id_project_id_portable_id_key UNIQUE (owner_id, project_id, portable_id);

ALTER TABLE eval_experiments
    ADD CONSTRAINT eval_experiments_pkey PRIMARY KEY (experiment_id);

CREATE INDEX eval_experiments_claim_active_idx ON eval_experiments USING btree (experiment_id) WHERE (state = ANY (ARRAY['preparing'::text, 'running'::text, 'settling'::text, 'pausing'::text, 'cancelling'::text]));

CREATE INDEX eval_experiments_claim_paused_deadline_idx ON eval_experiments USING btree (deadline_at, experiment_id) WHERE (state = 'paused'::text);

CREATE INDEX eval_experiments_control_page_idx ON eval_experiments USING btree (owner_id, control_mode, experiment_id);

CREATE INDEX eval_experiments_dataset_page_idx ON eval_experiments USING btree (owner_id, dataset_id, experiment_id);

CREATE INDEX eval_experiments_owner_created_page_idx ON eval_experiments USING btree (owner_id, created_at DESC, experiment_id DESC);

CREATE INDEX eval_experiments_owner_page_idx ON eval_experiments USING btree (owner_id, experiment_id);

CREATE INDEX eval_experiments_project_created_page_idx ON eval_experiments USING btree (owner_id, project_id, created_at DESC, experiment_id DESC);

CREATE INDEX eval_experiments_project_page_idx ON eval_experiments USING btree (owner_id, project_id, experiment_id);

CREATE INDEX eval_experiments_state_page_idx ON eval_experiments USING btree (owner_id, state, experiment_id);

CREATE TRIGGER eval_experiments_collection_delete AFTER DELETE ON eval_experiments REFERENCING OLD TABLE AS changed_experiments FOR EACH STATEMENT EXECUTE FUNCTION contractor_eval_collections_changed();

CREATE TRIGGER eval_experiments_collection_insert AFTER INSERT ON eval_experiments REFERENCING NEW TABLE AS changed_experiments FOR EACH STATEMENT EXECUTE FUNCTION contractor_eval_collections_changed();

CREATE TRIGGER eval_experiments_collection_update AFTER UPDATE ON eval_experiments REFERENCING OLD TABLE AS old_experiments NEW TABLE AS new_experiments FOR EACH STATEMENT EXECUTE FUNCTION contractor_eval_collections_updated();

CREATE TRIGGER eval_experiments_protect BEFORE UPDATE ON eval_experiments FOR EACH ROW EXECUTE FUNCTION contractor_eval_protect_experiment();

CREATE TABLE eval_frozen_plans (
    experiment_id text NOT NULL,
    document_kind text NOT NULL,
    document bytea NOT NULL,
    document_sha256 text GENERATED ALWAYS AS (('sha256:'::text || encode(sha256(document), 'hex'::text))) STORED,
    plan_sha256 text NOT NULL,
    setup bytea NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT eval_frozen_plans_document_check CHECK (((octet_length(document) >= 2) AND (octet_length(document) <= 1048576))),
    CONSTRAINT eval_frozen_plans_document_kind_check CHECK ((document_kind = ANY (ARRAY['playground.plan/v1'::text, 'ExternalRegistration'::text]))),
    CONSTRAINT eval_frozen_plans_plan_sha256_check CHECK ((plan_sha256 ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT eval_frozen_plans_setup_check CHECK (((octet_length(setup) >= 2) AND (octet_length(setup) <= 1048576)))
);

ALTER TABLE eval_frozen_plans
    ADD CONSTRAINT eval_frozen_plans_pkey PRIMARY KEY (experiment_id);

CREATE CONSTRAINT TRIGGER eval_plans_complete_matrix AFTER INSERT ON eval_frozen_plans DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION contractor_eval_complete_matrix();

CREATE TRIGGER eval_plans_immutable BEFORE DELETE OR UPDATE ON eval_frozen_plans FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();

CREATE TABLE eval_member_projections (
    experiment_id text NOT NULL,
    member_id text NOT NULL,
    revision bigint DEFAULT 1 NOT NULL,
    projected_revision bigint DEFAULT 0 NOT NULL,
    document bytea,
    collection_complete boolean DEFAULT false NOT NULL,
    observed_document bytea,
    checked_at timestamp with time zone,
    CONSTRAINT eval_member_projections_document_check CHECK (((document IS NULL) OR ((octet_length(document) >= 2) AND (octet_length(document) <= 1048576)))),
    CONSTRAINT eval_member_projections_observed_document_check CHECK (((observed_document IS NULL) OR ((octet_length(observed_document) >= 2) AND (octet_length(observed_document) <= 1048576))))
);

ALTER TABLE eval_member_projections
    ADD CONSTRAINT eval_member_projections_pkey PRIMARY KEY (experiment_id, member_id);

CREATE INDEX eval_member_projections_dirty ON eval_member_projections USING btree (experiment_id, checked_at NULLS FIRST, member_id) WHERE (revision <> projected_revision);

CREATE TABLE eval_members (
    experiment_id text NOT NULL,
    member_id text NOT NULL,
    pair_id text NOT NULL,
    ordinal integer NOT NULL,
    suite_id text NOT NULL,
    case_id text NOT NULL,
    sample integer NOT NULL,
    variant_id text NOT NULL,
    case_sha256 text NOT NULL,
    binding_sha256 text NOT NULL,
    eligibility text NOT NULL,
    execution_kind text NOT NULL,
    recipe bytea NOT NULL,
    submission_key text NOT NULL,
    eligibility_reason text,
    CONSTRAINT eval_members_eligibility_check CHECK ((eligibility = ANY (ARRAY['eligible'::text, 'unsupported'::text, 'blocked'::text]))),
    CONSTRAINT eval_members_eligibility_reason_check CHECK ((length(eligibility_reason) <= 8192)),
    CONSTRAINT eval_members_execution_kind_check CHECK ((execution_kind = ANY (ARRAY['run'::text, 'audit'::text]))),
    CONSTRAINT eval_members_member_id_check CHECK ((member_id ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT eval_members_ordinal_check CHECK (((ordinal >= 0) AND (ordinal <= 9999))),
    CONSTRAINT eval_members_pair_id_check CHECK ((pair_id ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT eval_members_recipe_check CHECK (((octet_length(recipe) >= 2) AND (octet_length(recipe) <= 1048576))),
    CONSTRAINT eval_members_sample_check CHECK (((sample >= 1) AND (sample <= 100)))
);

ALTER TABLE eval_members
    ADD CONSTRAINT eval_members_experiment_id_member_id_execution_kind_key UNIQUE (experiment_id, member_id, execution_kind);

ALTER TABLE eval_members
    ADD CONSTRAINT eval_members_experiment_id_ordinal_key UNIQUE (experiment_id, ordinal);

ALTER TABLE eval_members
    ADD CONSTRAINT eval_members_experiment_id_suite_id_case_id_sample_variant__key UNIQUE (experiment_id, suite_id, case_id, sample, variant_id);

ALTER TABLE eval_members
    ADD CONSTRAINT eval_members_pkey PRIMARY KEY (experiment_id, member_id);

ALTER TABLE eval_members
    ADD CONSTRAINT eval_members_submission_key_key UNIQUE (submission_key);

CREATE INDEX eval_members_pair_idx ON eval_members USING btree (experiment_id, pair_id, variant_id);

CREATE TRIGGER eval_members_bound BEFORE INSERT ON eval_members FOR EACH ROW EXECUTE FUNCTION contractor_eval_member_bound();

CREATE TRIGGER eval_members_immutable BEFORE DELETE OR UPDATE ON eval_members FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();

CREATE TRIGGER eval_members_projection AFTER INSERT ON eval_members REFERENCING NEW TABLE AS inserted_members FOR EACH STATEMENT EXECUTE FUNCTION contractor_eval_members_inserted();

CREATE TABLE eval_mutation_receipts (
    owner_id text NOT NULL,
    project_id text NOT NULL,
    resource_id text NOT NULL,
    operation text NOT NULL,
    operation_key text NOT NULL,
    request_sha256 text NOT NULL,
    expected_revision bigint,
    response bytea NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT eval_mutation_receipts_operation_key_check CHECK (((length(operation_key) >= 1) AND (length(operation_key) <= 128))),
    CONSTRAINT eval_mutation_receipts_request_sha256_check CHECK ((request_sha256 ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT eval_mutation_receipts_response_check CHECK (((octet_length(response) >= 2) AND (octet_length(response) <= 1048576)))
);

ALTER TABLE eval_mutation_receipts
    ADD CONSTRAINT eval_mutation_receipts_pkey PRIMARY KEY (owner_id, project_id, resource_id, operation, operation_key);

CREATE INDEX eval_mutation_receipts_project_idx ON eval_mutation_receipts USING btree (project_id);

CREATE TRIGGER eval_receipts_immutable BEFORE DELETE OR UPDATE ON eval_mutation_receipts FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();

CREATE TABLE eval_plan_resources (
    experiment_id text NOT NULL,
    resource_path text NOT NULL,
    document_kind text NOT NULL,
    document bytea NOT NULL,
    document_sha256 text GENERATED ALWAYS AS (('sha256:'::text || encode(sha256(document), 'hex'::text))) STORED,
    CONSTRAINT eval_plan_resources_document_check CHECK (((octet_length(document) >= 2) AND (octet_length(document) <= 1048576))),
    CONSTRAINT eval_plan_resources_resource_path_check CHECK (((length(resource_path) >= 1) AND (length(resource_path) <= 1024)))
);

ALTER TABLE eval_plan_resources
    ADD CONSTRAINT eval_plan_resources_pkey PRIMARY KEY (experiment_id, resource_path);

CREATE TRIGGER eval_resources_immutable BEFORE DELETE OR UPDATE ON eval_plan_resources FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();

CREATE TABLE eval_progress_observations (
    experiment_id text NOT NULL,
    sequence bigint NOT NULL,
    observed_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    counts jsonb NOT NULL
);

ALTER TABLE eval_progress_observations ALTER COLUMN sequence ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME eval_progress_observations_sequence_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);

ALTER TABLE eval_progress_observations
    ADD CONSTRAINT eval_progress_observations_pkey PRIMARY KEY (experiment_id, sequence);

CREATE INDEX eval_progress_observations_time ON eval_progress_observations USING btree (experiment_id, observed_at, sequence);

CREATE TABLE eval_projection_queue (
    experiment_id text NOT NULL,
    revision bigint DEFAULT 1 NOT NULL,
    published_revision bigint DEFAULT 0 NOT NULL,
    snapshot_id text,
    generation bigint,
    summary bytea,
    suites bytea,
    pins_verified boolean,
    content_sha256 text,
    published_at timestamp with time zone,
    CONSTRAINT eval_projection_queue_content_sha256_check CHECK ((content_sha256 ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT eval_projection_queue_generation_check CHECK ((generation > 0)),
    CONSTRAINT eval_projection_queue_view_shape CHECK ((((snapshot_id IS NULL) = (generation IS NULL)) AND ((generation IS NULL) = (summary IS NULL)) AND ((generation IS NULL) = (suites IS NULL)) AND ((generation IS NULL) = (pins_verified IS NULL)) AND ((generation IS NULL) = (published_at IS NULL))))
);

ALTER TABLE eval_projection_queue
    ADD CONSTRAINT eval_projection_queue_pkey PRIMARY KEY (experiment_id);

CREATE INDEX eval_projection_queue_dirty_idx ON eval_projection_queue USING btree (experiment_id) WHERE (revision <> published_revision);

CREATE TABLE eval_records (
    experiment_id text NOT NULL,
    member_id text NOT NULL,
    kind text NOT NULL,
    record_sha256 text NOT NULL,
    document bytea NOT NULL,
    actor_id text NOT NULL,
    predecessor_sha256 text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT eval_records_check CHECK ((record_sha256 = ('sha256:'::text || encode(sha256(document), 'hex'::text)))),
    CONSTRAINT eval_records_document_check CHECK (((octet_length(document) >= 2) AND (octet_length(document) <= 1048576))),
    CONSTRAINT eval_records_kind_check CHECK ((kind = ANY (ARRAY['result'::text, 'assessment'::text])))
);

ALTER TABLE eval_records
    ADD CONSTRAINT eval_records_pkey PRIMARY KEY (experiment_id, member_id, kind, record_sha256);

CREATE INDEX eval_records_native_recent ON eval_records USING btree (experiment_id, member_id, kind, created_at DESC, record_sha256 DESC) WHERE (actor_id = 'system:eval-collector'::text);

CREATE INDEX eval_records_recent ON eval_records USING btree (experiment_id, member_id, kind, created_at DESC, record_sha256 DESC);

CREATE TRIGGER eval_records_immutable BEFORE DELETE OR UPDATE ON eval_records FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();

CREATE TABLE eval_selections (
    experiment_id text NOT NULL,
    member_id text NOT NULL,
    result_kind text DEFAULT 'result'::text NOT NULL,
    result_sha256 text NOT NULL,
    assessment_kind text DEFAULT 'assessment'::text NOT NULL,
    assessment_sha256 text,
    revision bigint DEFAULT 1 NOT NULL,
    actor_id text NOT NULL,
    CONSTRAINT eval_selections_assessment_kind_check CHECK ((assessment_kind = 'assessment'::text)),
    CONSTRAINT eval_selections_result_kind_check CHECK ((result_kind = 'result'::text))
);

ALTER TABLE eval_selections
    ADD CONSTRAINT eval_selections_pkey PRIMARY KEY (experiment_id, member_id);

CREATE TRIGGER eval_selections_projection AFTER INSERT OR UPDATE ON eval_selections FOR EACH ROW EXECUTE FUNCTION contractor_eval_member_changed();

CREATE TABLE eval_submissions (
    experiment_id text NOT NULL,
    member_id text NOT NULL,
    state text NOT NULL,
    actor_id text NOT NULL,
    execution_kind text NOT NULL,
    execution_id text,
    execution_project_id text,
    terminal_state text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    observed_tokens bigint DEFAULT 0 NOT NULL,
    CONSTRAINT eval_submissions_check CHECK (((state = ANY (ARRAY['accepted'::text, 'terminal'::text])) = (execution_id IS NOT NULL))),
    CONSTRAINT eval_submissions_check1 CHECK (((state = 'terminal'::text) = (terminal_state IS NOT NULL))),
    CONSTRAINT eval_submissions_observed_tokens_check CHECK ((observed_tokens >= 0)),
    CONSTRAINT eval_submissions_state_check CHECK ((state = ANY (ARRAY['intent'::text, 'accepted'::text, 'terminal'::text, 'rejected'::text])))
);

ALTER TABLE eval_submissions
    ADD CONSTRAINT eval_submissions_pkey PRIMARY KEY (experiment_id, member_id);

CREATE UNIQUE INDEX eval_submissions_execution_idx ON eval_submissions USING btree (execution_kind, execution_id) WHERE (execution_id IS NOT NULL);

CREATE INDEX eval_submissions_outstanding_idx ON eval_submissions USING btree (experiment_id, member_id) WHERE (state = ANY (ARRAY['intent'::text, 'accepted'::text]));

CREATE INDEX eval_submissions_reconcile_idx ON eval_submissions USING btree (experiment_id, updated_at, member_id) WHERE (state = ANY (ARRAY['intent'::text, 'accepted'::text]));

CREATE UNIQUE INDEX eval_submissions_workspace_idx ON eval_submissions USING btree (execution_project_id) WHERE (execution_project_id IS NOT NULL);

CREATE TRIGGER eval_submissions_projection AFTER INSERT ON eval_submissions FOR EACH ROW EXECUTE FUNCTION contractor_eval_member_changed();

CREATE TRIGGER eval_submissions_projection_update AFTER UPDATE OF state, execution_id, execution_project_id, terminal_state ON eval_submissions FOR EACH ROW WHEN (((((old.state IS DISTINCT FROM new.state) OR (old.execution_id IS DISTINCT FROM new.execution_id)) OR (old.execution_project_id IS DISTINCT FROM new.execution_project_id)) OR (old.terminal_state IS DISTINCT FROM new.terminal_state))) EXECUTE FUNCTION contractor_eval_member_changed();

CREATE TRIGGER eval_submissions_protect BEFORE UPDATE ON eval_submissions FOR EACH ROW EXECUTE FUNCTION contractor_eval_protect_submission();

CREATE TABLE eval_suboperations (
    experiment_id text NOT NULL,
    member_id text NOT NULL,
    kind text NOT NULL,
    operation_key text NOT NULL,
    request bytea NOT NULL,
    request_sha256 text GENERATED ALWAYS AS (('sha256:'::text || encode(sha256(request), 'hex'::text))) STORED,
    response bytea,
    state text DEFAULT 'intent'::text NOT NULL,
    CONSTRAINT eval_suboperations_check CHECK (((state = 'intent'::text) = (response IS NULL))),
    CONSTRAINT eval_suboperations_kind_check CHECK ((kind = ANY (ARRAY['project-create'::text, 'inputs'::text, 'run-create'::text, 'audit-create'::text, 'audit-start'::text, 'cancel'::text]))),
    CONSTRAINT eval_suboperations_request_check CHECK (((octet_length(request) >= 2) AND (octet_length(request) <= 1048576))),
    CONSTRAINT eval_suboperations_response_check CHECK (((octet_length(response) >= 2) AND (octet_length(response) <= 1048576))),
    CONSTRAINT eval_suboperations_state_check CHECK ((state = ANY (ARRAY['intent'::text, 'succeeded'::text, 'rejected'::text])))
);

ALTER TABLE eval_suboperations
    ADD CONSTRAINT eval_suboperations_operation_key_key UNIQUE (operation_key);

ALTER TABLE eval_suboperations
    ADD CONSTRAINT eval_suboperations_pkey PRIMARY KEY (experiment_id, member_id, kind);

CREATE TRIGGER eval_suboperations_protect BEFORE UPDATE ON eval_suboperations FOR EACH ROW EXECUTE FUNCTION contractor_eval_protect_suboperation();

CREATE TABLE eval_view_charts (
    experiment_id text NOT NULL,
    generation bigint NOT NULL,
    suite_id text NOT NULL,
    metric text NOT NULL,
    document bytea NOT NULL
);

ALTER TABLE eval_view_charts
    ADD CONSTRAINT eval_view_charts_pkey PRIMARY KEY (experiment_id, generation, suite_id, metric);

CREATE TRIGGER eval_view_charts_immutable BEFORE DELETE OR UPDATE ON eval_view_charts FOR EACH ROW EXECUTE FUNCTION contractor_eval_view_immutable();

CREATE TABLE eval_view_pairs (
    experiment_id text NOT NULL,
    generation bigint NOT NULL,
    ordinal integer NOT NULL,
    pair_id text NOT NULL,
    suite_id text NOT NULL,
    document bytea NOT NULL,
    regression boolean NOT NULL,
    unresolved boolean NOT NULL,
    tokens_a double precision,
    tokens_b double precision,
    duration_a double precision,
    duration_b double precision,
    complete_a boolean NOT NULL,
    complete_b boolean NOT NULL
);

ALTER TABLE eval_view_pairs
    ADD CONSTRAINT eval_view_pairs_experiment_id_generation_pair_id_key UNIQUE (experiment_id, generation, pair_id);

ALTER TABLE eval_view_pairs
    ADD CONSTRAINT eval_view_pairs_pkey PRIMARY KEY (experiment_id, generation, ordinal);

CREATE INDEX eval_view_pairs_suite ON eval_view_pairs USING btree (experiment_id, generation, suite_id, ordinal);

CREATE TRIGGER eval_view_pairs_immutable BEFORE DELETE OR UPDATE ON eval_view_pairs FOR EACH ROW EXECUTE FUNCTION contractor_eval_view_immutable();


-- Views

CREATE VIEW audit_live_item_reviews AS
 SELECT item.audit_id,
    item.item_id,
    request.request_id,
    (request.state = 'decided'::text) AS approved
   FROM (audit_items item
     JOIN audit_review_requests request ON (((request.audit_id = item.audit_id) AND (request.subject_kind = 'audit-item-action'::text) AND (request.subject_id = item.item_id) AND (request.kind = item.approval_kind) AND (request.subject_revision = 1) AND (request.subject_digest = item.approval_subject_digest))))
  WHERE (((request.expires_at IS NULL) OR (request.expires_at > clock_timestamp())) AND ((request.state = 'pending'::text) OR ((request.state = 'decided'::text) AND (EXISTS ( SELECT 1
           FROM audit_review_decisions decision
          WHERE ((decision.request_id = request.request_id) AND (decision.audit_id = request.audit_id) AND (decision.action = 'approve'::text) AND (decision.subject_revision = request.subject_revision) AND (decision.subject_digest = request.subject_digest)))))));


-- Foreign keys

ALTER TABLE allocation_execution_reports
    ADD CONSTRAINT allocation_execution_reports_stage_execution_id_allocation_fkey FOREIGN KEY (stage_execution_id, allocation_id, logical_agent_name) REFERENCES stage_allocations(stage_execution_id, allocation_id, logical_agent_name) ON DELETE CASCADE;

ALTER TABLE artifact_binding_revisions
    ADD CONSTRAINT artifact_binding_revisions_scope_kind_scope_id_fkey FOREIGN KEY (scope_kind, scope_id) REFERENCES artifact_scopes(scope_kind, scope_id);

ALTER TABLE artifact_binding_revisions
    ADD CONSTRAINT artifact_binding_revisions_version_id_fkey FOREIGN KEY (version_id) REFERENCES artifact_versions(version_id);

ALTER TABLE artifact_bindings
    ADD CONSTRAINT artifact_bindings_scope_kind_scope_id_fkey FOREIGN KEY (scope_kind, scope_id) REFERENCES artifact_scopes(scope_kind, scope_id);

ALTER TABLE artifact_bindings
    ADD CONSTRAINT artifact_bindings_scope_kind_scope_id_namespace_name_curre_fkey FOREIGN KEY (scope_kind, scope_id, namespace, name, current_revision) REFERENCES artifact_binding_revisions(scope_kind, scope_id, namespace, name, revision) DEFERRABLE INITIALLY DEFERRED;

ALTER TABLE artifact_lineage
    ADD CONSTRAINT artifact_lineage_source_scope_kind_source_scope_id_source__fkey FOREIGN KEY (source_scope_kind, source_scope_id, source_namespace, source_name, source_revision) REFERENCES artifact_binding_revisions(scope_kind, scope_id, namespace, name, revision);

ALTER TABLE artifact_lineage
    ADD CONSTRAINT artifact_lineage_target_scope_kind_target_scope_id_target__fkey FOREIGN KEY (target_scope_kind, target_scope_id, target_namespace, target_name, target_revision) REFERENCES artifact_binding_revisions(scope_kind, scope_id, namespace, name, revision);

ALTER TABLE artifact_scopes
    ADD CONSTRAINT artifact_scopes_project_id_fkey FOREIGN KEY (project_id) REFERENCES projects(project_id);

ALTER TABLE artifact_scopes
    ADD CONSTRAINT artifact_scopes_run_id_fkey FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id);

ALTER TABLE artifact_versions
    ADD CONSTRAINT artifact_versions_blob_sha256_fkey FOREIGN KEY (blob_sha256) REFERENCES artifact_blobs(sha256);

ALTER TABLE audit_artifact_links
    ADD CONSTRAINT audit_artifact_links_audit_id_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

ALTER TABLE audit_controller_claims
    ADD CONSTRAINT audit_controller_claims_audit_id_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

ALTER TABLE audit_events
    ADD CONSTRAINT audit_events_audit_id_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

ALTER TABLE audit_execution_items
    ADD CONSTRAINT audit_execution_items_execution_fkey FOREIGN KEY (execution_id, audit_id, round_id) REFERENCES audit_executions(execution_id, audit_id, round_id) ON DELETE CASCADE;

ALTER TABLE audit_execution_items
    ADD CONSTRAINT audit_execution_items_item_fkey FOREIGN KEY (item_id, audit_id, round_id) REFERENCES audit_items(item_id, audit_id, round_id) ON DELETE CASCADE;

ALTER TABLE audit_executions
    ADD CONSTRAINT audit_executions_audit_id_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

ALTER TABLE audit_executions
    ADD CONSTRAINT audit_executions_round_fkey FOREIGN KEY (round_id, audit_id) REFERENCES audit_rounds(round_id, audit_id) ON DELETE CASCADE;

ALTER TABLE audit_finding_assessments
    ADD CONSTRAINT audit_finding_assessments_audit_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

ALTER TABLE audit_finding_assessments
    ADD CONSTRAINT audit_finding_assessments_collection_fkey FOREIGN KEY (collection_receipt_id) REFERENCES audit_executions(collection_receipt_id) ON DELETE RESTRICT;

ALTER TABLE audit_finding_assessments
    ADD CONSTRAINT audit_finding_assessments_finding_fkey FOREIGN KEY (finding_id, audit_id) REFERENCES audit_findings(finding_id, audit_id) ON DELETE CASCADE;

ALTER TABLE audit_finding_assessments
    ADD CONSTRAINT audit_finding_assessments_item_fkey FOREIGN KEY (execution_item_id, item_id) REFERENCES audit_execution_items(execution_item_id, item_id) ON DELETE RESTRICT;

ALTER TABLE audit_finding_assessments
    ADD CONSTRAINT audit_finding_assessments_receipt_id_fkey FOREIGN KEY (receipt_id) REFERENCES finding_proposal_receipts(receipt_id) ON DELETE RESTRICT;

ALTER TABLE audit_finding_contributions
    ADD CONSTRAINT audit_finding_contributions_finding_fkey FOREIGN KEY (finding_id, audit_id) REFERENCES audit_findings(finding_id, audit_id) ON DELETE CASCADE;

ALTER TABLE audit_finding_contributions
    ADD CONSTRAINT audit_finding_contributions_receipt_id_fkey FOREIGN KEY (receipt_id) REFERENCES finding_proposal_receipts(receipt_id) ON DELETE RESTRICT;

ALTER TABLE audit_findings
    ADD CONSTRAINT audit_findings_audit_id_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

ALTER TABLE audit_findings
    ADD CONSTRAINT audit_findings_current_assessment_fkey FOREIGN KEY (current_assessment_id) REFERENCES audit_finding_assessments(assessment_id) DEFERRABLE INITIALLY DEFERRED;

ALTER TABLE audit_findings
    ADD CONSTRAINT audit_findings_current_decision_fkey FOREIGN KEY (current_decision_id) REFERENCES audit_review_decisions(decision_id) DEFERRABLE INITIALLY DEFERRED;

ALTER TABLE audit_findings
    ADD CONSTRAINT audit_findings_duplicate_target_fkey FOREIGN KEY (duplicate_target_id, audit_id) REFERENCES audit_findings(finding_id, audit_id) DEFERRABLE INITIALLY DEFERRED;

ALTER TABLE audit_findings
    ADD CONSTRAINT audit_findings_first_receipt_id_fkey FOREIGN KEY (first_receipt_id) REFERENCES finding_proposal_receipts(receipt_id) ON DELETE RESTRICT;

ALTER TABLE audit_idempotency
    ADD CONSTRAINT audit_idempotency_audit_id_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

ALTER TABLE audit_items
    ADD CONSTRAINT audit_items_last_execution_item_fkey FOREIGN KEY (last_execution_item_id, item_id) REFERENCES audit_execution_items(execution_item_id, item_id) DEFERRABLE INITIALLY DEFERRED;

ALTER TABLE audit_items
    ADD CONSTRAINT audit_items_proposal_hold_fkey FOREIGN KEY (proposal_receipt_id, audit_id) REFERENCES finding_proposal_audit_holds(receipt_id, audit_id) ON DELETE CASCADE;

ALTER TABLE audit_items
    ADD CONSTRAINT audit_items_round_fkey FOREIGN KEY (round_id, audit_id) REFERENCES audit_rounds(round_id, audit_id) ON DELETE CASCADE;

ALTER TABLE audit_review_decisions
    ADD CONSTRAINT audit_review_decisions_audit_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

ALTER TABLE audit_review_decisions
    ADD CONSTRAINT audit_review_decisions_duplicate_fkey FOREIGN KEY (duplicate_target_id, audit_id) REFERENCES audit_findings(finding_id, audit_id) DEFERRABLE INITIALLY DEFERRED;

ALTER TABLE audit_review_decisions
    ADD CONSTRAINT audit_review_decisions_request_fkey FOREIGN KEY (request_id, audit_id) REFERENCES audit_review_requests(request_id, audit_id) ON DELETE RESTRICT;

ALTER TABLE audit_review_requests
    ADD CONSTRAINT audit_review_requests_audit_id_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

ALTER TABLE audit_review_requests
    ADD CONSTRAINT audit_review_requests_finding_fkey FOREIGN KEY (finding_id, audit_id) REFERENCES audit_findings(finding_id, audit_id) ON DELETE CASCADE;

ALTER TABLE audit_review_requests
    ADD CONSTRAINT audit_review_requests_report_round_fkey FOREIGN KEY (report_round_id, audit_id) REFERENCES audit_rounds(round_id, audit_id) ON DELETE CASCADE;

ALTER TABLE audit_rounds
    ADD CONSTRAINT audit_rounds_audit_id_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

ALTER TABLE audits
    ADD CONSTRAINT audits_current_round_fkey FOREIGN KEY (current_round_id, audit_id) REFERENCES audit_rounds(round_id, audit_id) ON DELETE SET NULL (current_round_id) DEFERRABLE INITIALLY DEFERRED;

ALTER TABLE audits
    ADD CONSTRAINT audits_project_owner_fkey FOREIGN KEY (project_id, owner_id) REFERENCES projects(project_id, owner_id) ON DELETE RESTRICT;

ALTER TABLE eval_commands
    ADD CONSTRAINT eval_commands_experiment_id_fkey FOREIGN KEY (experiment_id) REFERENCES eval_experiments(experiment_id) ON DELETE CASCADE;

ALTER TABLE eval_controller_claims
    ADD CONSTRAINT eval_controller_claims_experiment_id_fkey FOREIGN KEY (experiment_id) REFERENCES eval_experiments(experiment_id) ON DELETE CASCADE;

ALTER TABLE eval_dataset_revisions
    ADD CONSTRAINT eval_dataset_revisions_owner_id_project_id_fkey FOREIGN KEY (owner_id, project_id) REFERENCES projects(owner_id, project_id) ON DELETE RESTRICT;

ALTER TABLE eval_evidence_refs
    ADD CONSTRAINT eval_evidence_refs_experiment_id_member_id_fkey FOREIGN KEY (experiment_id, member_id) REFERENCES eval_members(experiment_id, member_id) ON DELETE CASCADE;

ALTER TABLE eval_execution_tombstones
    ADD CONSTRAINT eval_execution_tombstones_experiment_id_member_id_fkey FOREIGN KEY (experiment_id, member_id) REFERENCES eval_members(experiment_id, member_id) ON DELETE CASCADE;

ALTER TABLE eval_experiments
    ADD CONSTRAINT eval_experiments_owner_id_project_id_fkey FOREIGN KEY (owner_id, project_id) REFERENCES projects(owner_id, project_id) ON DELETE RESTRICT;

ALTER TABLE eval_experiments
    ADD CONSTRAINT eval_experiments_project_id_dataset_id_dataset_revision_fkey FOREIGN KEY (project_id, dataset_id, dataset_revision) REFERENCES eval_dataset_revisions(project_id, dataset_id, revision) ON DELETE RESTRICT;

ALTER TABLE eval_frozen_plans
    ADD CONSTRAINT eval_frozen_plans_experiment_id_fkey FOREIGN KEY (experiment_id) REFERENCES eval_experiments(experiment_id) ON DELETE CASCADE;

ALTER TABLE eval_member_projections
    ADD CONSTRAINT eval_member_projections_experiment_id_member_id_fkey FOREIGN KEY (experiment_id, member_id) REFERENCES eval_members(experiment_id, member_id) ON DELETE CASCADE;

ALTER TABLE eval_members
    ADD CONSTRAINT eval_members_experiment_id_fkey FOREIGN KEY (experiment_id) REFERENCES eval_frozen_plans(experiment_id) ON DELETE CASCADE;

ALTER TABLE eval_mutation_receipts
    ADD CONSTRAINT eval_mutation_receipts_project_id_fkey FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE RESTRICT;

ALTER TABLE eval_plan_resources
    ADD CONSTRAINT eval_plan_resources_experiment_id_fkey FOREIGN KEY (experiment_id) REFERENCES eval_frozen_plans(experiment_id) ON DELETE CASCADE;

ALTER TABLE eval_progress_observations
    ADD CONSTRAINT eval_progress_observations_experiment_id_fkey FOREIGN KEY (experiment_id) REFERENCES eval_experiments(experiment_id) ON DELETE CASCADE;

ALTER TABLE eval_projection_queue
    ADD CONSTRAINT eval_projection_queue_experiment_id_fkey FOREIGN KEY (experiment_id) REFERENCES eval_experiments(experiment_id) ON DELETE CASCADE;

ALTER TABLE eval_records
    ADD CONSTRAINT eval_records_experiment_id_member_id_fkey FOREIGN KEY (experiment_id, member_id) REFERENCES eval_members(experiment_id, member_id) ON DELETE CASCADE;

ALTER TABLE eval_selections
    ADD CONSTRAINT eval_selections_experiment_id_member_id_assessment_kind_as_fkey FOREIGN KEY (experiment_id, member_id, assessment_kind, assessment_sha256) REFERENCES eval_records(experiment_id, member_id, kind, record_sha256);

ALTER TABLE eval_selections
    ADD CONSTRAINT eval_selections_experiment_id_member_id_result_kind_result_fkey FOREIGN KEY (experiment_id, member_id, result_kind, result_sha256) REFERENCES eval_records(experiment_id, member_id, kind, record_sha256);

ALTER TABLE eval_submissions
    ADD CONSTRAINT eval_submissions_experiment_id_member_id_execution_kind_fkey FOREIGN KEY (experiment_id, member_id, execution_kind) REFERENCES eval_members(experiment_id, member_id, execution_kind) ON DELETE CASCADE;

ALTER TABLE eval_suboperations
    ADD CONSTRAINT eval_suboperations_experiment_id_member_id_fkey FOREIGN KEY (experiment_id, member_id) REFERENCES eval_submissions(experiment_id, member_id) ON DELETE CASCADE;

ALTER TABLE eval_view_charts
    ADD CONSTRAINT eval_view_charts_experiment_id_fkey FOREIGN KEY (experiment_id) REFERENCES eval_projection_queue(experiment_id) ON DELETE CASCADE;

ALTER TABLE eval_view_pairs
    ADD CONSTRAINT eval_view_pairs_experiment_id_fkey FOREIGN KEY (experiment_id) REFERENCES eval_projection_queue(experiment_id) ON DELETE CASCADE;

ALTER TABLE finding_proposal_audit_holds
    ADD CONSTRAINT finding_proposal_audit_holds_audit_id_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

ALTER TABLE finding_proposal_audit_holds
    ADD CONSTRAINT finding_proposal_audit_holds_project_id_fkey FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE;

ALTER TABLE finding_proposal_audit_holds
    ADD CONSTRAINT finding_proposal_audit_holds_receipt_id_fkey FOREIGN KEY (receipt_id) REFERENCES finding_proposal_receipts(receipt_id) ON DELETE CASCADE;

ALTER TABLE gateway_allocation_routes
    ADD CONSTRAINT gateway_allocation_routes_allocation_id_fkey FOREIGN KEY (allocation_id) REFERENCES stage_allocations(allocation_id) ON DELETE CASCADE;

ALTER TABLE gateway_allocation_routes
    ADD CONSTRAINT gateway_allocation_routes_route_key_fkey FOREIGN KEY (route_key) REFERENCES gateway_recovery_routes(route_key);

ALTER TABLE gateway_allocation_routes
    ADD CONSTRAINT gateway_allocation_routes_run_id_fkey FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id) ON DELETE CASCADE;

ALTER TABLE gateway_recovery_failures
    ADD CONSTRAINT gateway_recovery_failures_route_key_fkey FOREIGN KEY (route_key) REFERENCES gateway_recovery_routes(route_key);

ALTER TABLE gateway_recovery_failures
    ADD CONSTRAINT gateway_recovery_failures_run_id_fkey FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id) ON DELETE CASCADE;

ALTER TABLE gateway_recovery_waits
    ADD CONSTRAINT gateway_recovery_waits_route_key_fkey FOREIGN KEY (route_key) REFERENCES gateway_recovery_routes(route_key);

ALTER TABLE gateway_recovery_waits
    ADD CONSTRAINT gateway_recovery_waits_run_id_fkey FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id) ON DELETE CASCADE;

ALTER TABLE gateway_run_routes
    ADD CONSTRAINT gateway_run_routes_route_key_fkey FOREIGN KEY (route_key) REFERENCES gateway_recovery_routes(route_key);

ALTER TABLE gateway_run_routes
    ADD CONSTRAINT gateway_run_routes_run_id_fkey FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id) ON DELETE CASCADE;

ALTER TABLE planner_execution_reports
    ADD CONSTRAINT planner_execution_reports_session_id_fkey FOREIGN KEY (session_id) REFERENCES planner_sessions(session_id) ON DELETE CASCADE;

ALTER TABLE planner_execution_reports
    ADD CONSTRAINT planner_execution_reports_stage_execution_id_fkey FOREIGN KEY (stage_execution_id) REFERENCES stage_executions(stage_execution_id) ON DELETE CASCADE;

ALTER TABLE planner_sessions
    ADD CONSTRAINT planner_sessions_stage_execution_id_fkey FOREIGN KEY (stage_execution_id) REFERENCES stage_executions(stage_execution_id) ON DELETE CASCADE;

ALTER TABLE projects
    ADD CONSTRAINT projects_http_target_credential_fkey FOREIGN KEY (http_target_credential_id, http_target_credential_kind) REFERENCES runtime_credentials(credential_id, credential_kind);

ALTER TABLE run_stage_resumptions
    ADD CONSTRAINT run_stage_resumptions_run_id_fkey FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id) ON DELETE CASCADE;

ALTER TABLE run_stage_resumptions
    ADD CONSTRAINT run_stage_resumptions_source_execution_id_fkey FOREIGN KEY (source_execution_id) REFERENCES stage_executions(stage_execution_id) ON DELETE CASCADE;

ALTER TABLE run_stage_resumptions
    ADD CONSTRAINT run_stage_resumptions_target_execution_id_fkey FOREIGN KEY (target_execution_id) REFERENCES stage_executions(stage_execution_id) ON DELETE CASCADE;

ALTER TABLE runtime_label_bindings
    ADD CONSTRAINT runtime_label_bindings_config_name_config_version_config_d_fkey FOREIGN KEY (config_name, config_version, config_digest) REFERENCES runtime_config_versions(name, version, digest);

ALTER TABLE stage_allocations
    ADD CONSTRAINT stage_allocations_stage_execution_id_fkey FOREIGN KEY (stage_execution_id) REFERENCES stage_executions(stage_execution_id) ON DELETE CASCADE;

ALTER TABLE stage_executions
    ADD CONSTRAINT stage_execution_lineage FOREIGN KEY (run_id, stage_name, previous_execution_id) REFERENCES stage_executions(run_id, stage_name, stage_execution_id) DEFERRABLE INITIALLY DEFERRED;

ALTER TABLE stage_executions
    ADD CONSTRAINT stage_execution_planner_session FOREIGN KEY (stage_execution_id, planner_session_id, planner_invocation_id) REFERENCES planner_sessions(stage_execution_id, session_id, invocation_id) DEFERRABLE INITIALLY DEFERRED;

ALTER TABLE stage_executions
    ADD CONSTRAINT stage_execution_resume_receipt FOREIGN KEY (stage_execution_id, resume_source_execution_id) REFERENCES run_stage_resumptions(target_execution_id, source_execution_id) DEFERRABLE INITIALLY DEFERRED;

ALTER TABLE stage_executions
    ADD CONSTRAINT stage_executions_run_id_fkey FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id) ON DELETE CASCADE;

ALTER TABLE stage_metrics
    ADD CONSTRAINT stage_metrics_stage_execution_id_fkey FOREIGN KEY (stage_execution_id) REFERENCES stage_executions(stage_execution_id) ON DELETE CASCADE;

ALTER TABLE stage_transition_decisions
    ADD CONSTRAINT stage_transition_source_run FOREIGN KEY (run_id, source_execution_id) REFERENCES stage_executions(run_id, stage_execution_id) ON DELETE CASCADE;

ALTER TABLE stage_transition_decisions
    ADD CONSTRAINT stage_transition_target_run_stage FOREIGN KEY (run_id, target_stage_name, target_execution_id) REFERENCES stage_executions(run_id, stage_name, stage_execution_id) DEFERRABLE INITIALLY DEFERRED;

ALTER TABLE workflow_run_events
    ADD CONSTRAINT workflow_run_events_run_id_fkey FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id) ON DELETE CASCADE;

ALTER TABLE workflow_run_output_publications
    ADD CONSTRAINT workflow_run_output_publicati_source_scope_kind_run_id_sou_fkey FOREIGN KEY (source_scope_kind, run_id, source_namespace, source_name, source_revision) REFERENCES artifact_binding_revisions(scope_kind, scope_id, namespace, name, revision);

ALTER TABLE workflow_run_output_publications
    ADD CONSTRAINT workflow_run_output_publicati_target_scope_kind_project_id_fkey FOREIGN KEY (target_scope_kind, project_id, target_namespace, target_name, target_revision) REFERENCES artifact_binding_revisions(scope_kind, scope_id, namespace, name, revision);

ALTER TABLE workflow_run_output_publications
    ADD CONSTRAINT workflow_run_output_publications_run_id_project_id_fkey FOREIGN KEY (run_id, project_id) REFERENCES workflow_runs(run_id, project_id);

ALTER TABLE workflow_runs
    ADD CONSTRAINT workflow_runs_audit_submission_fkey FOREIGN KEY (audit_execution_id, audit_submission_key) REFERENCES audit_executions(execution_id, submission_key) ON DELETE RESTRICT;

ALTER TABLE workflow_runs
    ADD CONSTRAINT workflow_runs_project_owner_fkey FOREIGN KEY (project_id, owner_id) REFERENCES projects(project_id, owner_id);


-- Rows every database starts with

INSERT INTO artifact_blob_settings (singleton, backend)
VALUES (true, NULL);

INSERT INTO runtime_config_versions (name, version, digest, canonical_document, built_in, actor_id, created_at, idempotency_key_digest, request_digest)
VALUES ('contractor-empty', '1', 'sha256:80a1754c01f8443c29fdc8f650a2254b2461694819918b204a55a7ad3425dc5f', '{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"contractor-empty","version":"1"},"spec":{}}', true, 'contractor-bootstrap', '1970-01-01 00:00:00+00', NULL, NULL);

INSERT INTO runtime_label_bindings (label, config_name, config_version, config_digest, revision, created_by, created_at, updated_by, updated_at)
VALUES ('default', 'contractor-empty', '1', 'sha256:80a1754c01f8443c29fdc8f650a2254b2461694819918b204a55a7ad3425dc5f', 1, 'contractor-bootstrap', '1970-01-01 00:00:00+00', 'contractor-bootstrap', '1970-01-01 00:00:00+00');

INSERT INTO scheduler_settings (singleton, max_concurrent_runs, revision)
VALUES (true, 1, 1);

RESET check_function_bodies;
