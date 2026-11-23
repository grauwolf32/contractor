-- Contractor schema baseline, folded on 2026-11-23.
-- Replaces the previous baseline and migrations 000002-000010.
-- Retired ledgers fail the unchanged strict drift check.
-- Objects follow PostgreSQL dependency order.
SET LOCAL check_function_bodies = false;

-- Function: contractor_advance_finding_correlation(text[])

CREATE FUNCTION contractor_advance_finding_correlation(affected_projects text[]) RETURNS void
    LANGUAGE plpgsql
    AS $$
DECLARE
 selected_project text;
BEGIN
 FOR selected_project IN SELECT DISTINCT unnest(affected_projects) ORDER BY 1 LOOP
  PERFORM contractor_lock_finding_correlation_fence(selected_project);
  UPDATE finding_correlation_fences SET revision = revision + 1 WHERE project_id = selected_project;
 END LOOP;
END;
$$;

-- Function: contractor_advance_hypothesis_revision()

CREATE FUNCTION contractor_advance_hypothesis_revision() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
 IF NEW.current_revision<>OLD.current_revision+1 THEN
  RAISE EXCEPTION 'hypothesis revisions advance exactly once' USING ERRCODE='23514';
 END IF;
 RETURN NEW;
END;
$$;

-- Function: contractor_append_audit_events(text, jsonb)

CREATE FUNCTION contractor_append_audit_events(target_audit_id text, events jsonb) RETURNS bigint
    LANGUAGE plpgsql
    AS $$
DECLARE
    event_count bigint;
    reserved_next bigint;
    last_sequence bigint;
BEGIN
    IF events IS NULL OR jsonb_typeof(events) <> 'array' THEN
        RAISE EXCEPTION 'Audit events must be a JSON array' USING ERRCODE = 'assert_failure';
    END IF;
    event_count := jsonb_array_length(events);

    SELECT audit.next_event_sequence INTO reserved_next
      FROM audits AS audit
     WHERE audit.audit_id = target_audit_id;
    IF reserved_next IS NULL THEN
        RAISE EXCEPTION 'Audit events have no Audit' USING ERRCODE = 'assert_failure';
    END IF;

    SELECT event.sequence_number INTO last_sequence
      FROM audit_events AS event
     WHERE event.audit_id = target_audit_id
     ORDER BY event.sequence_number DESC
     LIMIT 1;
    IF COALESCE(last_sequence, 0) + event_count <> reserved_next - 1 THEN
        RAISE EXCEPTION 'Audit event reservation does not continue the event sequence'
            USING ERRCODE = 'assert_failure',
                  DETAIL = format('last event %s, %s events, next sequence %s',
                                  COALESCE(last_sequence, 0), event_count, reserved_next);
    END IF;

    INSERT INTO audit_events (
        audit_id, sequence_number, kind, entity_id, entity_revision, summary
    )
    SELECT target_audit_id, reserved_next - event_count + event.position - 1,
           event.value->>'kind', event.value->>'entityId',
           (event.value->>'entityRevision')::bigint, event.value->'summary'
      FROM jsonb_array_elements(events) WITH ORDINALITY AS event(value, position);
    RETURN reserved_next - 1;
END;
$$;

-- Function: contractor_append_workflow_run_lifecycle_event(text, text, text, text)

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

-- Function: contractor_artifact_hold_apply(jsonb, jsonb, boolean)

CREATE FUNCTION contractor_artifact_hold_apply(acquired jsonb, released jsonb, strict boolean) RETURNS void
    LANGUAGE plpgsql
    AS $$
DECLARE
    kinds text[];
    ids text[];
    namespaces text[];
    names text[];
    revisions text[];
    deltas integer[];
    missing bigint;
BEGIN
    SELECT COALESCE(array_agg(change.scope_kind ORDER BY change.scope_kind, change.scope_id, change.namespace,
                                  change.name, change.revision), '{}'),
           COALESCE(array_agg(change.scope_id ORDER BY change.scope_kind, change.scope_id, change.namespace,
                                  change.name, change.revision), '{}'),
           COALESCE(array_agg(change.namespace ORDER BY change.scope_kind, change.scope_id, change.namespace,
                                  change.name, change.revision), '{}'),
           COALESCE(array_agg(change.name ORDER BY change.scope_kind, change.scope_id, change.namespace,
                                  change.name, change.revision), '{}'),
           COALESCE(array_agg(change.revision ORDER BY change.scope_kind, change.scope_id, change.namespace,
                                  change.name, change.revision), '{}'),
           COALESCE(array_agg(change.delta ORDER BY change.scope_kind, change.scope_id, change.namespace,
                                  change.name, change.revision), '{}')
    INTO kinds, ids, namespaces, names, revisions, deltas
    FROM (
        SELECT key.scope_kind, key.scope_id, key.namespace, key.name, key.revision,
               sum(key.delta)::integer AS delta
        FROM (
            SELECT held ->> 0 AS scope_kind, held ->> 1 AS scope_id, held ->> 2 AS namespace,
                   held ->> 3 AS name, held ->> 4 AS revision, 1 AS delta
            FROM jsonb_array_elements(acquired) AS held
            UNION ALL
            SELECT freed ->> 0, freed ->> 1, freed ->> 2, freed ->> 3, freed ->> 4, -1
            FROM jsonb_array_elements(released) AS freed
        ) AS key
        WHERE key.scope_kind IS NOT NULL AND key.scope_id IS NOT NULL AND key.namespace IS NOT NULL
          AND key.name IS NOT NULL AND key.revision IS NOT NULL
        GROUP BY key.scope_kind, key.scope_id, key.namespace, key.name, key.revision
        HAVING sum(key.delta) <> 0
    ) AS change;
    IF cardinality(deltas) = 0 THEN
        RETURN;
    END IF;
    PERFORM 1
    FROM artifact_binding_revisions AS revision
    JOIN unnest(kinds, ids, namespaces, names, revisions)
        AS change(scope_kind, scope_id, namespace, name, revision)
      ON (revision.scope_kind, revision.scope_id, revision.namespace, revision.name, revision.revision)
       = (change.scope_kind, change.scope_id, change.namespace, change.name, change.revision)
    ORDER BY revision.scope_kind, revision.scope_id, revision.namespace, revision.name, revision.revision
    FOR NO KEY UPDATE OF revision;
    UPDATE artifact_binding_revisions AS revision
    SET hold_count = revision.hold_count + change.delta
    FROM unnest(kinds, ids, namespaces, names, revisions, deltas)
        AS change(scope_kind, scope_id, namespace, name, revision, delta)
    WHERE (revision.scope_kind, revision.scope_id, revision.namespace, revision.name, revision.revision)
        = (change.scope_kind, change.scope_id, change.namespace, change.name, change.revision);
    IF strict THEN
        SELECT count(*) INTO missing
        FROM unnest(kinds, ids, namespaces, names, revisions, deltas)
            AS change(scope_kind, scope_id, namespace, name, revision, delta)
        WHERE change.delta > 0
          AND EXISTS (
              SELECT 1 FROM artifact_scopes AS scope
              WHERE scope.scope_kind = change.scope_kind AND scope.scope_id = change.scope_id
          )
          AND NOT EXISTS (
              SELECT 1 FROM artifact_binding_revisions AS revision
              WHERE (revision.scope_kind, revision.scope_id, revision.namespace, revision.name,
                     revision.revision)
                  = (change.scope_kind, change.scope_id, change.namespace, change.name, change.revision)
          );
        IF missing > 0 THEN
            RAISE EXCEPTION 'exact Artifact revision to hold does not exist'
                USING ERRCODE = '23503', CONSTRAINT = 'artifact_exact_hold';
        END IF;
    END IF;
END;
$$;

-- Function: contractor_artifact_key_set(jsonb[])

CREATE FUNCTION contractor_artifact_key_set(keys jsonb[]) RETURNS jsonb
    LANGUAGE sql IMMUTABLE
    BEGIN ATOMIC
 SELECT COALESCE(jsonb_agg(DISTINCT key.key ORDER BY key.key), '[]'::jsonb) AS "coalesce"
    FROM unnest(contractor_artifact_key_set.keys) key(key)
   WHERE (key.key IS NOT NULL);
END;

-- Function: contractor_artifact_ref_key(text, text, jsonb)

CREATE FUNCTION contractor_artifact_ref_key(scope_kind text, scope_id text, ref jsonb) RETURNS jsonb
    LANGUAGE sql IMMUTABLE
    BEGIN ATOMIC
 SELECT
         CASE
             WHEN ((scope_kind IS NOT NULL) AND (scope_id IS NOT NULL) AND (jsonb_typeof((ref -> 'namespace'::text)) = 'string'::text) AND (jsonb_typeof((ref -> 'name'::text)) = 'string'::text) AND (jsonb_typeof((ref -> 'revision'::text)) = 'string'::text)) THEN jsonb_build_array(scope_kind, scope_id, (ref -> 'namespace'::text), (ref -> 'name'::text), (ref -> 'revision'::text))
             ELSE NULL::jsonb
         END AS "case";
END;

-- Function: contractor_artifact_retention_enabled()

CREATE FUNCTION contractor_artifact_retention_enabled() RETURNS boolean
    LANGUAGE sql STABLE
    AS $$
SELECT COALESCE(current_setting('contractor.artifact_retention', true), '') = 'evict'
$$;

-- Function: contractor_audit_input_artifact_holds(text, jsonb, jsonb)

CREATE FUNCTION contractor_audit_input_artifact_holds(project_id text, selection jsonb, baseline jsonb) RETURNS jsonb
    LANGUAGE sql IMMUTABLE
    AS $$
SELECT contractor_artifact_key_set(ARRAY(
    SELECT contractor_artifact_ref_key('project', project_id, input -> 'ref')
    FROM contractor_jsonb_values(selection -> 'inputs') AS input
    UNION ALL
    SELECT contractor_artifact_ref_key('project', project_id, input -> 'ref')
    FROM contractor_jsonb_values(baseline -> 'inputs') AS input
))
$$;

-- Function: contractor_audit_run_input_holds(text, jsonb)

CREATE FUNCTION contractor_audit_run_input_holds(run_id text, progress jsonb) RETURNS jsonb
    LANGUAGE sql IMMUTABLE
    AS $$
SELECT contractor_artifact_key_set(ARRAY(
 SELECT contractor_artifact_ref_key('run',run_id,ref)
 FROM contractor_jsonb_elements(progress->'refs') AS ref
))
$$;

-- Function: contractor_capture_stage_execution_lifecycle_event()

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

-- Function: contractor_capture_workflow_run_lifecycle_event()

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

-- Function: contractor_check_llm_credential_exclusive_state()

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

-- Function: contractor_copy_finding_hold_receipt_time()

CREATE FUNCTION contractor_copy_finding_hold_receipt_time() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    receipt_time timestamp with time zone;
BEGIN
    SELECT receipt.created_at INTO receipt_time
      FROM finding_proposal_receipts AS receipt
     WHERE receipt.receipt_id = NEW.receipt_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'finding proposal Audit hold has no receipt' USING ERRCODE = '23503';
    END IF;
    IF NEW.receipt_created_at IS NOT NULL AND NEW.receipt_created_at <> receipt_time THEN
        RAISE EXCEPTION 'finding proposal Audit hold names another receipt time' USING ERRCODE = '23514';
    END IF;
    NEW.receipt_created_at := receipt_time;
    RETURN NEW;
END;
$$;

-- Function: contractor_count_run_artifact_revisions()

CREATE FUNCTION contractor_count_run_artifact_revisions() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    exceeded boolean;
BEGIN
    WITH added AS (
        SELECT scope_id, count(*) AS revisions
          FROM inserted_revisions
         WHERE scope_kind = 'run' AND namespace <> 'outputs'
         GROUP BY scope_id
    ), counted AS (
        UPDATE artifact_scopes AS scope
           SET revision_count = scope.revision_count + added.revisions
          FROM added
         WHERE scope.scope_kind = 'run' AND scope.scope_id = added.scope_id
        RETURNING scope.revision_count
    )
    SELECT COALESCE(bool_or(revision_count > 50000), false) INTO exceeded FROM counted;
    IF exceeded THEN
        RAISE EXCEPTION 'Run artifact revision limit exceeded'
            USING ERRCODE = '54000', CONSTRAINT = 'artifact_scopes_run_revision_limit';
    END IF;
    RETURN NULL;
END;
$$;

-- Function: contractor_empty_finding_checks(text, bigint)

CREATE FUNCTION contractor_empty_finding_checks(digest text, bytes bigint) RETURNS json
    LANGUAGE sql IMMUTABLE
    AS $$
 SELECT json_build_object('schema','contractor.audit.finding-checks-projection.v1',
  'proposal_digest',digest,'size_bytes',bytes,'discarded',true,
  'subject',NULL,'limitations',NULL,'checks','[]'::json)
$$;

-- Function: contractor_eval_audit_child_changed()

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

-- Function: contractor_eval_collection_changed()

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

-- Function: contractor_eval_collections_changed()

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

-- Function: contractor_eval_collections_updated()

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

-- Function: contractor_eval_complete_matrix()

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

-- Function: contractor_eval_dirty(text, text)

CREATE FUNCTION contractor_eval_dirty(selected_experiment text, selected_member text) RETURNS void
    LANGUAGE plpgsql
    AS $$
BEGIN
    INSERT INTO eval_projection_queue(experiment_id) VALUES (selected_experiment)
    ON CONFLICT (experiment_id)
    DO UPDATE SET revision = eval_projection_queue.revision + 1;

    INSERT INTO eval_member_projections(experiment_id, member_id)
    VALUES (selected_experiment, selected_member)
    ON CONFLICT (experiment_id, member_id)
    DO UPDATE SET revision = eval_member_projections.revision + 1;
END;
$$;

-- Function: contractor_eval_evidence_deleted()

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

-- Function: contractor_eval_execution_changed()

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

-- Function: contractor_eval_fence_project()

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

-- Function: contractor_eval_immutable()

CREATE FUNCTION contractor_eval_immutable() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'DELETE' AND current_setting('contractor.eval_purge', true) = 'on' THEN RETURN OLD; END IF;
    RAISE EXCEPTION 'Evaluation record is immutable' USING ERRCODE = '23514';
END; $$;

-- Function: contractor_eval_member_bound()

CREATE FUNCTION contractor_eval_member_bound() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM eval_experiments WHERE experiment_id=NEW.experiment_id
                   AND plan_document IS NOT NULL AND NEW.ordinal<expected_count) THEN
        RAISE EXCEPTION 'Evaluation member is outside frozen matrix' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END; $$;

-- Function: contractor_eval_member_changed()

CREATE FUNCTION contractor_eval_member_changed() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    PERFORM contractor_eval_dirty(NEW.experiment_id, NEW.member_id);
    RETURN NULL;
END;
$$;

-- Function: contractor_eval_members_inserted()

CREATE FUNCTION contractor_eval_members_inserted() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    INSERT INTO eval_projection_queue(experiment_id)
    SELECT DISTINCT experiment_id FROM inserted_members ORDER BY experiment_id
    ON CONFLICT (experiment_id)
    DO UPDATE SET revision = eval_projection_queue.revision + 1;

    INSERT INTO eval_member_projections(experiment_id, member_id)
    SELECT experiment_id, member_id FROM inserted_members ORDER BY experiment_id, member_id;
    RETURN NULL;
END;
$$;

-- Function: contractor_eval_metrics_changed()

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

    FOR r IN SELECT * FROM contractor_eval_run_members(selected_run) LOOP
        PERFORM contractor_eval_dirty(r.experiment_id, r.member_id);
    END LOOP;
    RETURN NULL;
END;
$$;

-- Function: contractor_eval_output_changed()

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

-- Function: contractor_eval_protect_experiment()

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
       OR (OLD.plan_document IS NOT NULL
           AND ROW(NEW.draft,NEW.dataset_id,NEW.dataset_revision,NEW.expected_count,NEW.max_in_flight,NEW.wall_ms,NEW.token_limit)
               IS DISTINCT FROM ROW(OLD.draft,OLD.dataset_id,OLD.dataset_revision,OLD.expected_count,OLD.max_in_flight,OLD.wall_ms,OLD.token_limit))
    THEN RAISE EXCEPTION 'Evaluation revision or immutable fields changed' USING ERRCODE = '23514'; END IF;
    RETURN NEW;
END; $$;

-- Function: contractor_eval_protect_submission()

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

-- Function: contractor_eval_protect_suboperation()

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

-- Function: contractor_eval_require_frozen_plan()

CREATE FUNCTION contractor_eval_require_frozen_plan() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM eval_experiments WHERE experiment_id=NEW.experiment_id AND plan_document IS NOT NULL) THEN
        RAISE EXCEPTION 'Evaluation plan is not frozen' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END; $$;

-- Function: contractor_eval_retain_deleted_execution()

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
        FROM mutation_receipts i JOIN eval_suboperations op ON op.operation_key=i.idempotency_key
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

-- Function: contractor_eval_run_members(text)

CREATE FUNCTION contractor_eval_run_members(selected_run text) RETURNS TABLE(experiment_id text, member_id text)
    LANGUAGE sql STABLE
    AS $$
    SELECT s.experiment_id, s.member_id
      FROM eval_submissions s
     WHERE s.execution_kind = 'run' AND s.execution_id = selected_run
    UNION
    SELECT s.experiment_id, s.member_id
      FROM audit_executions x
      JOIN eval_submissions s ON s.execution_kind = 'audit' AND s.execution_id = x.audit_id
     WHERE x.run_id = selected_run
$$;

-- Function: contractor_eval_view_immutable()

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

-- Function: contractor_finding_comparison_projection(text, text, text, text)

CREATE FUNCTION contractor_finding_comparison_projection(report text, fingerprint text, anchors text, retention text) RETURNS jsonb
    LANGUAGE plpgsql IMMUTABLE STRICT
    AS $$
DECLARE
 r jsonb := report::jsonb;
 f jsonb := fingerprint::jsonb;
 a jsonb := anchors::jsonb;
 families jsonb := '[]';
 family_available boolean := true;
 result jsonb;
BEGIN
 IF retention='discarded' THEN
  RETURN '{"schema":"contractor.finding.comparison-projection.v1","availability":"unavailable","reason":"discarded"}'::jsonb;
 END IF;
 IF coalesce(r->>'title','')='' THEN
  RETURN '{"schema":"contractor.finding.comparison-projection.v1","availability":"unavailable","reason":"not_recorded"}'::jsonb;
 END IF;
 IF jsonb_array_length(coalesce(r->'standard_refs','[]'))>32 OR EXISTS (
  SELECT 1 FROM jsonb_array_elements(coalesce(r->'standard_refs','[]')) AS ref
  WHERE octet_length(ref->>'scheme')>256 OR octet_length(ref->>'version')>160 OR octet_length(ref->>'requirement_id')>256
 ) THEN family_available:=false;
 ELSE
  SELECT coalesce(jsonb_agg(ref ORDER BY (ref->>'scheme') COLLATE "C",(ref->>'version') COLLATE "C",(ref->>'requirement_id') COLLATE "C"),'[]') INTO families
  FROM jsonb_array_elements(coalesce(r->'standard_refs','[]')) AS ref;
 END IF;
 result:=jsonb_build_object('schema','contractor.finding.comparison-projection.v1','availability','available',
  'preview',jsonb_build_object('title',left(r->>'title',256),'titleTruncated',length(r->>'title')>256,
   'severitySuggestion',coalesce(r->>'severity_suggestion','')),
  'hints',jsonb_build_object('subject',r->'subject','families',families,'familiesAvailable',family_available,
   'producerKey',CASE WHEN f->>'availability'='available' AND f->>'algorithm'='contractor/producer-issue/v1'
    AND f->>'adapter' IN ('pinned-sarif-native/v1','normalized-candidate/v1') THEN f->>'value' ELSE '' END,
   'sites',CASE WHEN a->>'availability'='available' THEN coalesce((SELECT jsonb_agg(k->>'siteKey') FROM jsonb_array_elements(a->'keys') AS k),'[]') ELSE '[]' END,
   'repository',coalesce(f->>'repository',a->>'repository',''),
   'commit',coalesce(f->>'commit',a->>'commit','')));
 RETURN result;
END;
$$;

-- Function: contractor_finding_correlation_source_deleted()

CREATE FUNCTION contractor_finding_correlation_source_deleted() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
 PERFORM contractor_advance_finding_correlation(ARRAY(
  SELECT DISTINCT n.project_id FROM updated_receipts AS n JOIN previous_receipts AS o USING(receipt_id)
  JOIN finding_correlation_fences AS fence ON fence.project_id = n.project_id
  WHERE n.source_run_deleted_at IS DISTINCT FROM o.source_run_deleted_at
 ));
 RETURN NULL;
END;
$$;

-- Function: contractor_finding_group_key(text, text, text)

CREATE FUNCTION contractor_finding_group_key(project_id text, algorithm text, value text) RETURNS text
    LANGUAGE sql IMMUTABLE STRICT
    BEGIN ATOMIC
 SELECT encode(sha256(convert_to((((((((('['::text || (to_json('contractor/finding-group/v1'::text))::text) || ','::text) || (to_json(project_id))::text) || ','::text) || (to_json(algorithm))::text) || ','::text) || (to_json(value))::text) || ']'::text), 'UTF8'::name)), 'hex'::text) AS encode;
END;

-- Function: contractor_finding_primary_key(text)

CREATE FUNCTION contractor_finding_primary_key(selected_receipt text) RETURNS TABLE(algorithm text, value text, reason text)
    LANGUAGE plpgsql
    AS $$
DECLARE
 fingerprint jsonb;
 anchors jsonb;
 issue_keys text[];
BEGIN
 SELECT fingerprint_projection::jsonb, anchor_projection::jsonb INTO fingerprint, anchors
 FROM finding_proposal_receipts WHERE receipt_id = selected_receipt;
 IF NOT FOUND THEN
  RAISE EXCEPTION 'finding first receipt is missing' USING ERRCODE='23503';
 END IF;
 IF fingerprint->>'availability' = 'available'
  AND fingerprint->>'algorithm' = 'contractor/fact-evidence/v1' THEN
  IF NOT EXISTS (SELECT 1 FROM finding_proposal_receipts
     WHERE receipt_id=selected_receipt AND report_projection::jsonb->>'severity_suggestion'='informational') THEN
   RAISE EXCEPTION 'fact equality requires informational producer severity' USING ERRCODE='23514';
  END IF;
  RETURN QUERY SELECT 'contractor/fact-evidence/v1'::text, fingerprint->>'value', ''::text;
  RETURN;
 END IF;
 IF fingerprint->>'availability' = 'available' THEN
  IF fingerprint->>'algorithm' IS DISTINCT FROM 'contractor/producer-issue/v1'
   OR fingerprint->>'adapter' NOT IN ('pinned-sarif-native/v1','normalized-candidate/v1') THEN
   RAISE EXCEPTION 'unsupported primary producer policy' USING ERRCODE='23514';
  END IF;
  RETURN QUERY SELECT 'contractor/producer-issue/v1'::text, fingerprint->>'value', ''::text;
  RETURN;
 END IF;
 IF anchors->>'availability' = 'available' THEN
  SELECT array_agg(key) INTO issue_keys FROM (
   SELECT DISTINCT item->>'issueKey' AS key FROM jsonb_array_elements(anchors->'keys') AS item
   WHERE item ? 'issueKey' LIMIT 2
  ) AS qualified;
 END IF;
 IF cardinality(issue_keys) = 1 THEN
  RETURN QUERY SELECT 'contractor/snapshot-issue-anchor/v1'::text, issue_keys[1], ''::text;
 ELSIF cardinality(issue_keys) > 1 THEN
  RETURN QUERY SELECT ''::text, ''::text, 'primary_key_ambiguous'::text;
 ELSIF fingerprint->>'reason' = 'not_recorded' AND anchors->>'reason' = 'not_recorded' THEN
  RETURN QUERY SELECT ''::text, ''::text, 'not_recorded'::text;
 ELSE
  RETURN QUERY SELECT ''::text, ''::text, 'no_qualified_key'::text;
 END IF;
END;
$$;

-- Function: contractor_finding_receipt_artifact_holds(text, jsonb, jsonb)

CREATE FUNCTION contractor_finding_receipt_artifact_holds(run_id text, proposal_ref jsonb, evidence jsonb) RETURNS jsonb
    LANGUAGE sql IMMUTABLE
    AS $$
SELECT contractor_artifact_key_set(ARRAY(
    SELECT contractor_artifact_ref_key('run', run_id, proposal_ref)
    UNION ALL
    SELECT contractor_artifact_ref_key('run', run_id, item -> 'ref')
    FROM contractor_jsonb_elements(evidence) AS item
))
$$;

-- Function: contractor_guard_audit_admission()

CREATE FUNCTION contractor_guard_audit_admission() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    PERFORM contractor_require_active_audit_project(NEW.project_id, NEW.owner_id);
    RETURN NEW;
END;
$$;

-- Function: contractor_guard_audit_artifact_admission()

CREATE FUNCTION contractor_guard_audit_artifact_admission() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    audit_state text;
    project text;
BEGIN
    SELECT state,project_id INTO audit_state,project FROM audits WHERE audit_id=NEW.scope_id FOR SHARE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Audit is missing' USING ERRCODE='23503';
    END IF;
    IF audit_state='deleting' THEN
        RAISE EXCEPTION 'Audit is deleting' USING ERRCODE='55000';
    END IF;
    -- Terminal collection during Project cancellation/drain retains its existing
    -- claim authority. New Run admission remains fenced by the owning service.
    PERFORM 1 FROM projects WHERE project_id=project FOR SHARE;
    RETURN NEW;
END;
$$;

-- Function: contractor_guard_audit_child_admission()

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

-- Function: contractor_guard_project_artifact_admission()

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

-- Function: contractor_guard_project_artifact_binding_admission()

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

-- Function: contractor_guard_project_artifact_scope_admission()

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

-- Function: contractor_guard_workflow_run_project_admission()

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

-- Function: contractor_hold_audit_artifacts()

CREATE FUNCTION contractor_hold_audit_artifacts() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    inputs_acquired jsonb := '[]';
    inputs_released jsonb := '[]';
    skills_acquired jsonb := '[]';
    skills_released jsonb := '[]';
BEGIN
    IF TG_OP <> 'DELETE' THEN
        inputs_acquired := contractor_audit_input_artifact_holds(NEW.project_id, NEW.input_selection,
            NEW.baseline_snapshot);
        skills_acquired := contractor_skill_source_artifact_holds(NEW.owner_id, NEW.baseline_snapshot -> 'skills');
    END IF;
    IF TG_OP <> 'INSERT' THEN
        inputs_released := contractor_audit_input_artifact_holds(OLD.project_id, OLD.input_selection,
            OLD.baseline_snapshot);
        skills_released := contractor_skill_source_artifact_holds(OLD.owner_id, OLD.baseline_snapshot -> 'skills');
    END IF;
    PERFORM contractor_artifact_hold_apply(inputs_acquired, inputs_released, true);
    PERFORM contractor_artifact_hold_apply(skills_acquired, skills_released, false);
    RETURN NULL;
END;
$$;

-- Function: contractor_hold_audit_run_inputs()

CREATE FUNCTION contractor_hold_audit_run_inputs() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE acquired jsonb:='[]'; released jsonb:='[]';
BEGIN
 IF TG_OP<>'DELETE' THEN acquired:=contractor_audit_run_input_holds(NEW.run_id,NEW.audit_input_materialization::jsonb); END IF;
 IF TG_OP<>'INSERT' THEN released:=contractor_audit_run_input_holds(OLD.run_id,OLD.audit_input_materialization::jsonb); END IF;
 PERFORM contractor_artifact_hold_apply(acquired,released,true);
 RETURN NULL;
END;
$$;

-- Function: contractor_hold_finding_receipt_artifacts()

CREATE FUNCTION contractor_hold_finding_receipt_artifacts() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    PERFORM contractor_artifact_hold_apply(
        contractor_finding_receipt_artifact_holds(NEW.run_id, NEW.proposal_ref, NEW.evidence), '[]', true);
    RETURN NULL;
END;
$$;

-- Function: contractor_hold_lineage_artifacts()

CREATE FUNCTION contractor_hold_lineage_artifacts() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        PERFORM contractor_artifact_hold_apply((
            SELECT COALESCE(jsonb_agg(key), '[]'::jsonb)
            FROM inserted_lineage AS edge,
                 LATERAL (VALUES
                     (jsonb_build_array(edge.target_scope_kind, edge.target_scope_id, edge.target_namespace,
                                        edge.target_name, edge.target_revision)),
                     (jsonb_build_array(edge.source_scope_kind, edge.source_scope_id, edge.source_namespace,
                                        edge.source_name, edge.source_revision))) AS end_key(key)
        ), '[]'::jsonb, true);
    ELSE
        PERFORM contractor_artifact_hold_apply('[]'::jsonb, (
            SELECT COALESCE(jsonb_agg(key), '[]'::jsonb)
            FROM deleted_lineage AS edge,
                 LATERAL (VALUES
                     (jsonb_build_array(edge.target_scope_kind, edge.target_scope_id, edge.target_namespace,
                                        edge.target_name, edge.target_revision)),
                     (jsonb_build_array(edge.source_scope_kind, edge.source_scope_id, edge.source_namespace,
                                        edge.source_name, edge.source_revision))) AS end_key(key)
        ), false);
    END IF;
    RETURN NULL;
END;
$$;

-- Function: contractor_hold_output_publication_artifacts()

CREATE FUNCTION contractor_hold_output_publication_artifacts() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        PERFORM contractor_artifact_hold_apply((
            SELECT COALESCE(jsonb_agg(key), '[]'::jsonb)
            FROM inserted_publications AS publication,
                 LATERAL (VALUES
                     (jsonb_build_array('run', publication.run_id, publication.source_namespace,
                                        publication.source_name, publication.source_revision)),
                     (CASE WHEN publication.target_revision IS NOT NULL THEN
                          jsonb_build_array('project', publication.project_id, publication.target_namespace,
                                            publication.target_name, publication.target_revision) END)
                 ) AS end_key(key)
            WHERE key IS NOT NULL
        ), '[]'::jsonb, true);
    ELSE
        PERFORM contractor_artifact_hold_apply('[]'::jsonb, (
            SELECT COALESCE(jsonb_agg(key), '[]'::jsonb)
            FROM deleted_publications AS publication,
                 LATERAL (VALUES
                     (jsonb_build_array('run', publication.run_id, publication.source_namespace,
                                        publication.source_name, publication.source_revision)),
                     (CASE WHEN publication.target_revision IS NOT NULL THEN
                          jsonb_build_array('project', publication.project_id, publication.target_namespace,
                                            publication.target_name, publication.target_revision) END)
                 ) AS end_key(key)
            WHERE key IS NOT NULL
        ), false);
    END IF;
    RETURN NULL;
END;
$$;

-- Function: contractor_hold_planner_session_artifacts()

CREATE FUNCTION contractor_hold_planner_session_artifacts() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    session_run_id text;
    previous jsonb := '[]';
BEGIN
    IF TG_OP = 'UPDATE' THEN
        previous := OLD.artifact_refs;
    END IF;
    SELECT stage.run_id INTO session_run_id
    FROM stage_executions AS stage
    WHERE stage.stage_execution_id = NEW.stage_execution_id;
    PERFORM contractor_artifact_hold_apply(
        contractor_run_artifact_keys(session_run_id, NEW.artifact_refs),
        contractor_run_artifact_keys(session_run_id, previous),
        false);
    RETURN NULL;
END;
$$;

-- Function: contractor_hold_run_skill_artifacts()

CREATE FUNCTION contractor_hold_run_skill_artifacts() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    acquired jsonb := '[]';
    released jsonb := '[]';
BEGIN
    IF TG_OP <> 'DELETE' THEN
        acquired := contractor_skill_source_artifact_holds(NEW.owner_id, NEW.skill_snapshot);
    END IF;
    IF TG_OP <> 'INSERT' THEN
        released := contractor_skill_source_artifact_holds(OLD.owner_id, OLD.skill_snapshot);
    END IF;
    PERFORM contractor_artifact_hold_apply(acquired, released, false);
    RETURN NULL;
END;
$$;

-- Function: contractor_hold_stage_execution_artifacts()

CREATE FUNCTION contractor_hold_stage_execution_artifacts() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    previous jsonb := '[]';
BEGIN
    IF TG_OP = 'UPDATE' THEN
        previous := contractor_stage_execution_artifact_holds(OLD.run_id, OLD.stage_context_snapshot,
            OLD.candidate_stage_result, OLD.accepted_stage_result);
    END IF;
    PERFORM contractor_artifact_hold_apply(
        contractor_stage_execution_artifact_holds(NEW.run_id, NEW.stage_context_snapshot,
            NEW.candidate_stage_result, NEW.accepted_stage_result),
        previous, true);
    RETURN NULL;
END;
$$;

-- Function: contractor_initialize_finding_comparison_projection()

CREATE FUNCTION contractor_initialize_finding_comparison_projection() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
 NEW.comparison_projection:=contractor_finding_comparison_projection(NEW.report_projection,NEW.fingerprint_projection,NEW.anchor_projection,NEW.retention_state);
 RETURN NEW;
END;
$$;

-- Function: contractor_initialize_project_finding_fence()

CREATE FUNCTION contractor_initialize_project_finding_fence() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.kind = 'project' THEN
        INSERT INTO finding_correlation_fences(project_id) VALUES (NEW.project_id);
    END IF;
    RETURN NEW;
END;
$$;

-- Function: contractor_jsonb_elements(jsonb)

CREATE FUNCTION contractor_jsonb_elements(value jsonb) RETURNS SETOF jsonb
    LANGUAGE sql IMMUTABLE
    BEGIN ATOMIC
 SELECT element.value AS element
    FROM jsonb_array_elements(
         CASE
             WHEN (jsonb_typeof(contractor_jsonb_elements.value) = 'array'::text) THEN contractor_jsonb_elements.value
             ELSE '[]'::jsonb
         END) element(value);
END;

-- Function: contractor_jsonb_values(jsonb)

CREATE FUNCTION contractor_jsonb_values(value jsonb) RETURNS SETOF jsonb
    LANGUAGE sql IMMUTABLE
    BEGIN ATOMIC
 SELECT member.value
    FROM jsonb_each(
         CASE
             WHEN (jsonb_typeof(contractor_jsonb_values.value) = 'object'::text) THEN contractor_jsonb_values.value
             ELSE '{}'::jsonb
         END) member(key, value);
END;

-- Function: contractor_lifecycle_purge_enabled()

CREATE FUNCTION contractor_lifecycle_purge_enabled() RETURNS boolean
    LANGUAGE sql STABLE
    BEGIN ATOMIC
 SELECT (COALESCE(current_setting('contractor.lifecycle_purge'::text, true), ''::text) = ANY (ARRAY['run'::text, 'project'::text, 'audit'::text]));
END;

-- Function: contractor_lock_finding_correlation_fence(text)

CREATE FUNCTION contractor_lock_finding_correlation_fence(selected_project text) RETURNS void
    LANGUAGE plpgsql
    AS $$
DECLARE
 fenced boolean;
BEGIN
 BEGIN
  PERFORM 1 FROM projects WHERE project_id = selected_project FOR SHARE NOWAIT;
  SELECT true INTO fenced FROM finding_correlation_fences WHERE project_id = selected_project
   FOR NO KEY UPDATE NOWAIT;
 EXCEPTION WHEN lock_not_available THEN
  RAISE EXCEPTION 'Project finding correlation is concurrently changing' USING ERRCODE='40001';
 END;
 IF fenced IS NULL THEN
  RAISE EXCEPTION 'Project % has no finding correlation fence', selected_project USING ERRCODE='P0004';
 END IF;
END;
$$;

-- Function: contractor_mark_allocation_report_received()

CREATE FUNCTION contractor_mark_allocation_report_received() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    UPDATE stage_allocations
       SET execution_report_received = true
     WHERE allocation_id = NEW.allocation_id AND NOT execution_report_received;
    RETURN NULL;
END;
$$;

-- Function: contractor_material_rank(text, text, boolean)

CREATE FUNCTION contractor_material_rank(namespace text, media_type text, git_source boolean) RETURNS smallint
    LANGUAGE sql IMMUTABLE PARALLEL SAFE
    AS $$
    SELECT CASE
        WHEN git_source THEN 0
        WHEN namespace = 'sources' THEN 0
        WHEN namespace = 'openapi' THEN 1
        WHEN namespace = 'engagement' THEN 2
        WHEN namespace = 'likec4' THEN 3
        WHEN namespace = 'docs' THEN 4
        WHEN namespace = 'diffs' THEN 5
        WHEN media_type IN ('application/vnd.oai.openapi', 'application/vnd.oai.openapi+json', 'application/vnd.oai.openapi+yaml') THEN 1
        WHEN media_type = 'application/vnd.contractor.engagement+json' THEN 2
        WHEN media_type = 'text/vnd.likec4' THEN 3
        WHEN media_type IN ('text/markdown', 'text/x-markdown', 'application/pdf') THEN 4
        WHEN media_type = 'text/x-diff' THEN 5
        ELSE 6
    END::smallint
$$;

-- Function: contractor_notify_workflow_run_event()

CREATE FUNCTION contractor_notify_workflow_run_event() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    PERFORM pg_notify('contractor_run_events_v1', NEW.run_id);
    RETURN NEW;
END;
$$;

-- Function: contractor_planner_session_artifact_refs(jsonb)

CREATE FUNCTION contractor_planner_session_artifact_refs(state jsonb) RETURNS jsonb
    LANGUAGE sql IMMUTABLE
    BEGIN ATOMIC
 SELECT contractor_artifact_key_set(ARRAY( SELECT contractor_artifact_ref_key('run'::text, ''::text, (((contractor_planner_session_artifact_refs.state -> 'completion'::text) -> 'result'::text) -> 'summaryArtifact'::text)) AS contractor_artifact_ref_key
         UNION ALL
          SELECT contractor_artifact_ref_key('run'::text, ''::text, slot.slot) AS contractor_artifact_ref_key
            FROM contractor_jsonb_values((((contractor_planner_session_artifact_refs.state -> 'completion'::text) -> 'result'::text) -> 'artifacts'::text)) slot(slot)
         UNION ALL
          SELECT contractor_artifact_ref_key('run'::text, ''::text, (((contractor_planner_session_artifact_refs.state -> 'scan'::text) -> 'state'::text) -> 'plan'::text)) AS contractor_artifact_ref_key
         UNION ALL
          SELECT contractor_artifact_ref_key('run'::text, ''::text, (job.job -> 'report'::text)) AS contractor_artifact_ref_key
            FROM contractor_jsonb_elements((((contractor_planner_session_artifact_refs.state -> 'scan'::text) -> 'state'::text) -> 'jobs'::text)) job(job)
         UNION ALL
          SELECT contractor_artifact_ref_key('run'::text, ''::text, input.input) AS contractor_artifact_ref_key
            FROM contractor_jsonb_elements((((contractor_planner_session_artifact_refs.state -> 'scan'::text) -> 'state'::text) -> 'jobs'::text)) job(job),
             LATERAL contractor_jsonb_values((job.job -> 'inputArtifacts'::text)) input(input)
         UNION ALL
          SELECT contractor_artifact_ref_key('run'::text, ''::text, selected.output) AS contractor_artifact_ref_key
            FROM contractor_jsonb_elements((contractor_planner_session_artifact_refs.state #> '{scan,state,jobs}'::text[])) job(job),
             LATERAL contractor_jsonb_values((job.job -> 'outputs'::text)) selected(output))) AS contractor_artifact_key_set;
END;

-- Function: contractor_protect_artifact_binding()

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
                SELECT 1 FROM stage_executions AS target
                JOIN workflow_runs AS run ON run.run_id = target.run_id
                WHERE target.run_id = OLD.scope_id
                  AND target.resume_source_execution_id IS NOT NULL
                  AND target.resume_transaction_id = pg_current_xact_id()
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

-- Function: contractor_protect_artifact_blob_immutable()

CREATE FUNCTION contractor_protect_artifact_blob_immutable() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        IF contractor_lifecycle_purge_enabled() OR contractor_artifact_retention_enabled() THEN
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

-- Function: contractor_protect_artifact_immutable()

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

-- Function: contractor_protect_artifact_revision_delete()

CREATE FUNCTION contractor_protect_artifact_revision_delete() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF contractor_lifecycle_purge_enabled() THEN
        RETURN OLD;
    END IF;
    -- The binding's foreign key to its current revision is deferred to the
    -- commit, so retention refuses the current revision here.
    IF contractor_artifact_retention_enabled() AND OLD.hold_count = 0 AND NOT EXISTS (
        SELECT 1 FROM artifact_bindings AS binding
        WHERE binding.scope_kind = OLD.scope_kind AND binding.scope_id = OLD.scope_id
          AND binding.namespace = OLD.namespace AND binding.name = OLD.name
          AND binding.current_revision = OLD.revision
    ) THEN
        RETURN OLD;
    END IF;
    RAISE EXCEPTION 'immutable ArtifactStore row cannot be changed' USING ERRCODE = '23514';
END;
$$;

-- Function: contractor_protect_artifact_scope_created_at()

CREATE FUNCTION contractor_protect_artifact_scope_created_at() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.created_at IS DISTINCT FROM OLD.created_at THEN
        RAISE EXCEPTION 'Artifact scope creation time cannot be changed' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

-- Function: contractor_protect_artifact_version()

CREATE FUNCTION contractor_protect_artifact_version() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'DELETE' AND (contractor_lifecycle_purge_enabled() OR contractor_artifact_retention_enabled()) THEN
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

-- Function: contractor_protect_audit_completion()

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

-- Function: contractor_protect_audit_event()

CREATE FUNCTION contractor_protect_audit_event() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    RAISE EXCEPTION 'Audit events are immutable' USING ERRCODE = '23514';
END;
$$;

-- Function: contractor_protect_audit_execution_collection()

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

-- Function: contractor_protect_audit_input_materialization()

CREATE FUNCTION contractor_protect_audit_input_materialization() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
 IF NEW.audit_input_materialization::text IS DISTINCT FROM OLD.audit_input_materialization::text THEN
  IF NEW.publication_mode<>'audit-managed' OR NEW.state<>'initializing'
   OR NEW.audit_input_materialization IS NULL THEN
   RAISE EXCEPTION 'Audit input progress is writable only during initialization' USING ERRCODE='23514';
  END IF;
  IF OLD.audit_input_materialization IS NOT NULL AND (
   (OLD.audit_input_materialization->>'complete')::boolean
   OR NEW.audit_input_materialization->>'identity' IS DISTINCT FROM OLD.audit_input_materialization->>'identity'
   OR NEW.audit_input_materialization->>'deadlineAt' IS DISTINCT FROM OLD.audit_input_materialization->>'deadlineAt'
   OR (NEW.audit_input_materialization->>'revision')::bigint<>(OLD.audit_input_materialization->>'revision')::bigint+1
   OR NOT ((NEW.audit_input_materialization::jsonb->'refs') @> (OLD.audit_input_materialization::jsonb->'refs'))) THEN
   RAISE EXCEPTION 'Audit input progress cannot change authority or release its closure' USING ERRCODE='23514';
  END IF;
 END IF;
 RETURN NEW;
END;
$$;

-- Function: contractor_protect_audit_prepared_source()

CREATE FUNCTION contractor_protect_audit_prepared_source() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF OLD.prepared_work_source IS NOT NULL AND (
        NEW.prepared_work_source IS NULL OR
        OLD.prepared_work_source::jsonb IS DISTINCT FROM NEW.prepared_work_source::jsonb OR
        OLD.prepared_work_count IS DISTINCT FROM NEW.prepared_work_count OR
        NEW.prepared_work_offset < OLD.prepared_work_offset OR
        (OLD.state='collected' AND NEW.prepared_work_offset <> OLD.prepared_work_offset)
    ) THEN
        RAISE EXCEPTION 'prepared source and committed progress are immutable' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END;
$$;

-- Function: contractor_protect_credential_operation()

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

-- Function: contractor_protect_execution_origin()

CREATE FUNCTION contractor_protect_execution_origin() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
 IF NEW.execution_origin IS DISTINCT FROM OLD.execution_origin THEN
  RAISE EXCEPTION 'artifact execution origin is immutable' USING ERRCODE='23514';
 END IF;
 RETURN NEW;
END;
$$;

-- Function: contractor_protect_finding_correlation_fence()

CREATE FUNCTION contractor_protect_finding_correlation_fence() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
 IF NEW.project_id <> OLD.project_id OR NEW.revision NOT IN (OLD.revision, OLD.revision + 1) THEN
  RAISE EXCEPTION 'finding correlation revision must advance monotonically' USING ERRCODE='23514';
 END IF;
 RETURN NEW;
END;
$$;

-- Function: contractor_protect_finding_execution_lane()

CREATE FUNCTION contractor_protect_finding_execution_lane() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
 IF ROW(NEW.operation_ref,NEW.scan_origin) IS DISTINCT FROM ROW(OLD.operation_ref,OLD.scan_origin) THEN
  RAISE EXCEPTION 'finding execution origin is immutable' USING ERRCODE='23514';
 END IF;
 RETURN NEW;
END;
$$;

-- Function: contractor_protect_finding_proposal_receipt()

CREATE FUNCTION contractor_protect_finding_proposal_receipt() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    RAISE EXCEPTION 'finding proposal receipts are immutable' USING ERRCODE = '23514';
END;
$$;

-- Function: contractor_protect_finding_proposal_receipt_retention()

CREATE FUNCTION contractor_protect_finding_proposal_receipt_retention() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    materials_unchanged boolean;
    evidence_index integer;
BEGIN
    IF TG_OP='UPDATE' THEN
        materials_unchanged := ROW(NEW.project_proposal,NEW.project_evidence)
            IS NOT DISTINCT FROM ROW(OLD.project_proposal,OLD.project_evidence);
        IF NOT materials_unchanged AND OLD.project_proposal IS NULL AND NEW.project_proposal IS NOT NULL
           AND NEW.project_proposal->>'digest'=OLD.proposal_digest
           AND NEW.project_proposal->>'mediaType'=OLD.proposal_media_type
           AND (NEW.project_proposal->>'sizeBytes')::bigint=OLD.proposal_size_bytes
           AND jsonb_array_length(NEW.project_evidence)=jsonb_array_length(OLD.evidence) THEN
            materials_unchanged := true;
            FOR evidence_index IN 0..jsonb_array_length(OLD.evidence)-1 LOOP
                IF (NEW.project_evidence->evidence_index)-'ref' IS DISTINCT FROM (OLD.evidence->evidence_index)-'ref' THEN
                    materials_unchanged := false;
                END IF;
            END LOOP;
        END IF;
        IF NEW.retention_state='discarded' THEN
            NEW.report_projection:='{}';
            NEW.anchor_projection:='{"schema":"contractor.finding.anchors-projection.v1","availability":"unavailable","reason":"discarded"}';
            NEW.fingerprint_projection:='{"schema":"contractor.finding.fingerprint-projection.v1","availability":"unavailable","reason":"discarded"}';
            NEW.checks_projection:=contractor_empty_finding_checks(NEW.proposal_digest,NEW.proposal_size_bytes);
        END IF;
        IF NEW.comparison_projection IS DISTINCT FROM OLD.comparison_projection
           AND NEW.comparison_projection IS DISTINCT FROM contractor_finding_comparison_projection(
               NEW.report_projection,NEW.fingerprint_projection,NEW.anchor_projection,NEW.retention_state) THEN
            RAISE EXCEPTION 'finding comparison projection does not match its immutable sources' USING ERRCODE='23514';
        END IF;
        NEW.comparison_projection:=contractor_finding_comparison_projection(
            NEW.report_projection,NEW.fingerprint_projection,NEW.anchor_projection,NEW.retention_state);
        IF materials_unchanged AND
           to_jsonb(NEW)-ARRAY['retention_state','source_run_deleted_at','discarded_at','retention_updated_at',
            'report_projection','fingerprint_projection','anchor_projection','comparison_projection',
            'checks_projection','checks_projection_bytes','checks_projection_count','project_proposal','project_evidence']
           =to_jsonb(OLD)-ARRAY['retention_state','source_run_deleted_at','discarded_at','retention_updated_at',
            'report_projection','fingerprint_projection','anchor_projection','comparison_projection',
            'checks_projection','checks_projection_bytes','checks_projection_count','project_proposal','project_evidence']
           AND (NEW.report_projection=OLD.report_projection OR NEW.retention_state='discarded')
           AND (NEW.anchor_projection=OLD.anchor_projection OR NEW.retention_state='discarded')
           AND (NEW.fingerprint_projection=OLD.fingerprint_projection OR NEW.retention_state='discarded')
           AND (NEW.checks_projection::text IS NOT DISTINCT FROM OLD.checks_projection::text
                OR NEW.retention_state='discarded') THEN RETURN NEW;
        END IF;
    END IF;
    IF TG_OP='DELETE' AND current_setting('contractor.lifecycle_purge',true)='project'
       AND EXISTS(SELECT 1 FROM projects WHERE project_id=OLD.project_id AND lifecycle_state='deleting') THEN RETURN OLD; END IF;
    RAISE EXCEPTION 'finding proposal receipts are immutable' USING ERRCODE='23514';
END;
$$;

-- Function: contractor_protect_immutable_credential_row()

CREATE FUNCTION contractor_protect_immutable_credential_row() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    RAISE EXCEPTION 'credential identity, active record, and tombstone rows are immutable'
        USING ERRCODE = '23514';
END;
$$;

-- Function: contractor_protect_immutable_history()

CREATE FUNCTION contractor_protect_immutable_history() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN RAISE EXCEPTION 'historical records are immutable' USING ERRCODE='23514'; END; $$;

-- Function: contractor_protect_model_selection_settings()

CREATE FUNCTION contractor_protect_model_selection_settings() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'Model selection singleton cannot be deleted' USING ERRCODE = '23514';
    END IF;
    IF NEW.singleton IS DISTINCT FROM OLD.singleton
       OR (NEW.default_model IS NOT DISTINCT FROM OLD.default_model
           AND NEW.role_models IS NOT DISTINCT FROM OLD.role_models)
       OR NEW.revision IS DISTINCT FROM OLD.revision + 1
       OR NEW.updated_at <= OLD.updated_at THEN
        RAISE EXCEPTION 'Model selection update must be one semantic revision' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

-- Function: contractor_protect_mutation_receipt()

CREATE FUNCTION contractor_protect_mutation_receipt() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
 IF TG_OP = 'DELETE' AND (
  OLD.key_scope[1] = 'audit'
  OR (OLD.key_scope[1] = 'eval' AND current_setting('contractor.eval_purge', true) = 'on')
  OR (OLD.key_scope[1] = 'project' AND current_setting('contractor.lifecycle_purge', true) = 'project'
   AND EXISTS (SELECT 1 FROM projects WHERE project_id=OLD.project_id AND lifecycle_state='deleting'))
 ) THEN RETURN OLD; END IF;
 RAISE EXCEPTION 'mutation receipt is immutable' USING ERRCODE = '23514';
END; $$;

-- Function: contractor_protect_owner_queue_control()

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

-- Function: contractor_protect_placement_labels()

CREATE FUNCTION contractor_protect_placement_labels() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    RAISE EXCEPTION 'Runtime placement labels are immutable' USING ERRCODE = '23514';
END;
$$;

-- Function: contractor_protect_project()

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
            OR NEW.active_checks IS DISTINCT FROM OLD.active_checks
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
        NEW.name, NEW.description, NEW.active_checks,
        NEW.http_target_url, NEW.http_target_credential_id, NEW.http_target_credential_kind,
        NEW.lifecycle_state, NEW.deletion_phase, NEW.deletion_requested_at
    ) IS DISTINCT FROM ROW(
        OLD.name, OLD.description, OLD.active_checks,
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

-- Function: contractor_protect_project_finding_identity()

CREATE FUNCTION contractor_protect_project_finding_identity() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP='INSERT' THEN
      IF NEW.owner_id IS DISTINCT FROM (SELECT owner_id FROM projects WHERE project_id=NEW.project_id) THEN
        RAISE EXCEPTION 'Finding owner must be its Project owner' USING ERRCODE='23514';
      END IF;
      RETURN NEW;
    END IF;
    IF ROW(NEW.owner_id, NEW.finding_id, NEW.project_id, NEW.first_receipt_id, NEW.first_proposal_ref, NEW.title, NEW.subject_kind, NEW.subject_key, NEW.subject_truncated, NEW.severity_suggestion,
           NEW.correlation_algorithm, NEW.correlation_value, NEW.correlation_group_key, NEW.correlation_reason,
           NEW.created_at)
       IS DISTINCT FROM ROW(OLD.owner_id, OLD.finding_id, OLD.project_id, OLD.first_receipt_id, OLD.first_proposal_ref, OLD.title, OLD.subject_kind, OLD.subject_key, OLD.subject_truncated, OLD.severity_suggestion,
           OLD.correlation_algorithm, OLD.correlation_value, OLD.correlation_group_key, OLD.correlation_reason,
           OLD.created_at) THEN
        RAISE EXCEPTION 'Project Finding identity is immutable' USING ERRCODE = '23514';
    END IF;
    IF NEW.revision <> OLD.revision + 1 THEN
        RAISE EXCEPTION 'Project Finding revision must advance once' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

-- Function: contractor_protect_runtime_agent_principal()

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

-- Function: contractor_protect_runtime_config_immutable()

CREATE FUNCTION contractor_protect_runtime_config_immutable() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    RAISE EXCEPTION 'immutable RuntimeConfig row cannot be changed'
        USING ERRCODE = '23514';
END;
$$;

-- Function: contractor_protect_runtime_credential_immutable()

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

-- Function: contractor_protect_runtime_label_binding()

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

-- Function: contractor_protect_scheduler_settings()

CREATE FUNCTION contractor_protect_scheduler_settings() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'Scheduler settings singleton cannot be deleted'
            USING ERRCODE = '23514';
    END IF;
    IF NEW.singleton IS DISTINCT FROM OLD.singleton
        OR (NEW.max_concurrent_runs IS NOT DISTINCT FROM OLD.max_concurrent_runs
            AND NEW.max_concurrent_tool_runs IS NOT DISTINCT FROM OLD.max_concurrent_tool_runs)
        OR NEW.revision IS DISTINCT FROM OLD.revision + 1
        OR NEW.updated_at <= OLD.updated_at
    THEN
        RAISE EXCEPTION 'Scheduler settings update must be one semantic revision'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

-- Function: contractor_protect_stage_allocation_identity()

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

-- Function: contractor_protect_stage_allocation_performance_policy()

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

-- Function: contractor_protect_stage_execution_immutable()

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

-- Function: contractor_protect_stage_transition()

CREATE FUNCTION contractor_protect_stage_transition() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF OLD.transition_action IS NOT NULL AND
       ROW(NEW.transition_action, NEW.transition_target_stage_name, NEW.transition_target_execution_id,
           NEW.transition_escalation_ordinal, NEW.transition_escalation_exhausted, NEW.transition_decided_at)
       IS DISTINCT FROM
       ROW(OLD.transition_action, OLD.transition_target_stage_name, OLD.transition_target_execution_id,
           OLD.transition_escalation_ordinal, OLD.transition_escalation_exhausted, OLD.transition_decided_at)
    THEN
        RAISE EXCEPTION 'Stage transition decisions are immutable' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

-- Function: contractor_protect_telemetry_report_immutable()

CREATE FUNCTION contractor_protect_telemetry_report_immutable() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    RAISE EXCEPTION 'telemetry report cannot be changed' USING ERRCODE = '23514';
END;
$$;

-- Function: contractor_protect_workflow_run_event_immutable()

CREATE FUNCTION contractor_protect_workflow_run_event_immutable() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    RAISE EXCEPTION 'WorkflowRun events are immutable' USING ERRCODE = '23514';
END;
$$;

-- Function: contractor_protect_workflow_run_immutable()

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

-- Function: contractor_protect_workflow_run_skill_snapshot()

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

-- Function: contractor_require_active_audit_project(text, text)

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

-- Function: contractor_require_active_project(text, text)

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

-- Function: contractor_require_allocation_completion()

CREATE FUNCTION contractor_require_allocation_completion() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE pinned jsonb; stage_name text;
BEGIN
 IF TG_OP = 'UPDATE' THEN
  IF NEW.completion_contract IS DISTINCT FROM OLD.completion_contract THEN
   RAISE EXCEPTION 'Allocation completion is immutable' USING ERRCODE = '23514';
  END IF;
  RETURN NEW;
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

-- Function: contractor_require_complete_audit_run_binding()

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

-- Function: contractor_require_lineage_source()

CREATE FUNCTION contractor_require_lineage_source() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    PERFORM 1 FROM artifact_binding_revisions
    WHERE scope_kind=NEW.source_scope_kind AND scope_id=NEW.source_scope_id
      AND namespace=NEW.source_namespace AND name=NEW.source_name AND revision=NEW.source_revision
    FOR KEY SHARE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Artifact source revision is missing'
            USING ERRCODE='23503', CONSTRAINT='artifact_exact_hold';
    END IF;
    RETURN NEW;
END;
$$;

-- Function: contractor_require_matching_audit_run()

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

-- Function: contractor_retain_finding_correlation_fence()

CREATE FUNCTION contractor_retain_finding_correlation_fence() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
 IF EXISTS (SELECT 1 FROM projects WHERE project_id = OLD.project_id) THEN
  RAISE EXCEPTION 'finding correlation fence is kept while its Project exists' USING ERRCODE='23514';
 END IF;
 RETURN OLD;
END;
$$;

-- Function: contractor_round_results_descriptor_bytes(text, jsonb, text, text, bigint, jsonb, text)

CREATE FUNCTION contractor_round_results_descriptor_bytes(key text, ref jsonb, digest text, media text, size bigint, provenance jsonb, display text) RETURNS bigint
    LANGUAGE sql IMMUTABLE
    BEGIN ATOMIC
 SELECT octet_length(replace(replace(replace(replace(replace((jsonb_strip_nulls(jsonb_build_object('logical_key', key, 'classification', 'accepted-attachment', 'artifact', jsonb_build_object('ref', ref, 'digest', digest, 'mediaType', media, 'sizeBytes', size), 'kind', ((provenance -> 'evidence'::text) ->> 'kind'::text), 'evidence_id', ((provenance -> 'evidence'::text) ->> 'id'::text), 'content_member_id', ((provenance -> 'evidence'::text) ->> 'content_member_id'::text), 'display_ref', display, 'original',
         CASE
             WHEN ((((provenance -> 'evidence'::text) -> 'artifact'::text) IS NOT NULL) AND (((provenance -> 'evidence'::text) -> 'artifact'::text) <> 'null'::jsonb)) THEN jsonb_build_object('ref', ((provenance -> 'evidence'::text) -> 'artifact'::text), 'digest', digest, 'mediaType', media, 'sizeBytes', size)
             ELSE NULL::jsonb
         END)))::text, '<'::text, '<'::text), '>'::text, '>'::text), '&'::text, '&'::text), ' '::text, ' '::text), ' '::text, ' '::text)) AS octet_length;
END;

-- Function: contractor_run_artifact_keys(text, jsonb)

CREATE FUNCTION contractor_run_artifact_keys(run_id text, refs jsonb) RETURNS jsonb
    LANGUAGE sql IMMUTABLE
    AS $$
SELECT COALESCE(jsonb_agg(jsonb_set(ref, '{1}', to_jsonb(run_id)) ORDER BY ref), '[]'::jsonb)
FROM jsonb_array_elements(refs) AS ref
WHERE run_id IS NOT NULL
$$;

-- Function: contractor_skill_source_artifact_holds(text, jsonb)

CREATE FUNCTION contractor_skill_source_artifact_holds(owner_id text, skills jsonb) RETURNS jsonb
    LANGUAGE sql IMMUTABLE
    AS $$
SELECT contractor_artifact_key_set(ARRAY(
    SELECT contractor_artifact_ref_key('user', owner_id, skill -> 'source')
    FROM contractor_jsonb_elements(skills) AS skill
))
$$;

-- Function: contractor_stage_execution_artifact_holds(text, jsonb, jsonb, jsonb)

CREATE FUNCTION contractor_stage_execution_artifact_holds(run_id text, context jsonb, candidate jsonb, accepted jsonb) RETURNS jsonb
    LANGUAGE sql IMMUTABLE
    AS $$
SELECT contractor_artifact_key_set(ARRAY(
    SELECT contractor_artifact_ref_key('run', run_id, pin -> 'artifact')
    FROM contractor_jsonb_values(context -> 'artifacts') AS pin
    UNION ALL
    SELECT contractor_artifact_ref_key('run', run_id, context -> 'continuation' -> 'summaryArtifact')
    UNION ALL
    SELECT contractor_artifact_ref_key('run', run_id, slot)
    FROM contractor_jsonb_values(context -> 'continuation' -> 'artifacts') AS slot
    UNION ALL
    SELECT contractor_artifact_ref_key('run', run_id, result.value -> 'summaryArtifact')
    FROM (VALUES (candidate), (accepted)) AS result(value)
    UNION ALL
    SELECT contractor_artifact_ref_key('run', run_id, slot)
    FROM (VALUES (candidate), (accepted)) AS result(value),
         contractor_jsonb_values(result.value -> 'artifacts') AS slot
))
$$;

-- Function: contractor_valid_allocation_runtime_configuration(jsonb)

CREATE FUNCTION contractor_valid_allocation_runtime_configuration(candidate jsonb) RETURNS boolean
    LANGUAGE plpgsql IMMUTABLE STRICT
    SET search_path TO 'pg_catalog'
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

-- Function: contractor_valid_audit_input_materialization(json)

CREATE FUNCTION contractor_valid_audit_input_materialization(progress json) RETURNS boolean
    LANGUAGE sql IMMUTABLE
    BEGIN ATOMIC
 SELECT COALESCE(((octet_length((contractor_valid_audit_input_materialization.progress)::text) <= 262144) AND (jsonb_typeof(parsed.value) = 'object'::text) AND (parsed.value ?& ARRAY['identity'::text, 'revision'::text, 'complete'::text, 'deadlineAt'::text, 'checkpoint'::text, 'refs'::text]) AND ((parsed.value - ARRAY['identity'::text, 'revision'::text, 'complete'::text, 'deadlineAt'::text, 'checkpoint'::text, 'refs'::text]) = '{}'::jsonb) AND (jsonb_typeof((parsed.value -> 'identity'::text)) = 'string'::text) AND ((parsed.value ->> 'identity'::text) ~ '^sha256:[0-9a-f]{64}$'::text) AND (jsonb_typeof((parsed.value -> 'revision'::text)) = 'number'::text) AND ((((parsed.value ->> 'revision'::text))::bigint >= 1) AND (((parsed.value ->> 'revision'::text))::bigint <= '4294967295'::bigint)) AND (jsonb_typeof((parsed.value -> 'complete'::text)) = 'boolean'::text) AND (jsonb_typeof((parsed.value -> 'deadlineAt'::text)) = 'string'::text) AND (jsonb_typeof((parsed.value -> 'checkpoint'::text)) = 'object'::text) AND (jsonb_typeof((parsed.value -> 'refs'::text)) = 'array'::text) AND (jsonb_array_length((parsed.value -> 'refs'::text)) <= 1024)), false) AS "coalesce"
    FROM ( SELECT (contractor_valid_audit_input_materialization.progress)::jsonb AS value) parsed;
END;

-- Function: contractor_valid_engagement_credential_ids(text[])

CREATE FUNCTION contractor_valid_engagement_credential_ids(ids text[]) RETURNS boolean
    LANGUAGE sql IMMUTABLE STRICT
    BEGIN ATOMIC
 SELECT ((cardinality(ids) <= 8) AND ((cardinality(ids) = 0) OR ((array_ndims(ids) = 1) AND (array_lower(ids, 1) = 1))) AND (NOT (EXISTS ( SELECT 1
            FROM ( SELECT item.id,
                     lag(item.id, 1, ''::text) OVER (ORDER BY item."position") AS previous
                    FROM unnest(contractor_valid_engagement_credential_ids.ids) WITH ORDINALITY item(id, "position")) ordered
           WHERE ((ordered.id IS NULL) OR ((ordered.id COLLATE "C") !~ '^[a-z][a-z0-9_-]*$'::text) OR (octet_length(ordered.id) > 128) OR ((ordered.id COLLATE "C") <= (ordered.previous COLLATE "C")))))));
END;

-- Function: contractor_valid_model_selection(text, jsonb)

CREATE FUNCTION contractor_valid_model_selection(default_model text, roles jsonb) RETURNS boolean
    LANGUAGE sql IMMUTABLE PARALLEL SAFE
    BEGIN ATOMIC
 SELECT
         CASE
             WHEN (jsonb_typeof(roles) IS DISTINCT FROM 'object'::text) THEN false
             ELSE ((default_model ~ '^[a-z0-9][a-z0-9.-]{0,74}$'::text) AND (NOT (EXISTS ( SELECT 1
                FROM jsonb_each(contractor_valid_model_selection.roles) r(key, value)
               WHERE ((r.key <> ALL (ARRAY['planner'::text, 'worker'::text, 'summarizer'::text, 'audit_completion_worker'::text])) OR (jsonb_typeof(r.value) IS DISTINCT FROM 'string'::text) OR ((r.value #>> '{}'::text[]) !~ '^[a-z0-9][a-z0-9.-]{0,74}$'::text))))))
         END AS "case";
END;

-- Function: contractor_valid_pinned_runtime_config(jsonb, text, boolean)

CREATE FUNCTION contractor_valid_pinned_runtime_config(candidate jsonb, expected_label text, explicit boolean) RETURNS boolean
    LANGUAGE sql IMMUTABLE STRICT
    BEGIN ATOMIC
 SELECT ((jsonb_typeof(candidate) = 'object'::text) AND (( SELECT array_agg(key.key ORDER BY key.key) AS array_agg
            FROM jsonb_object_keys(contractor_valid_pinned_runtime_config.candidate) key(key)) = ARRAY['bindingRevision'::text, 'config'::text, 'explicit'::text, 'label'::text]) AND ((candidate ->> 'label'::text) = expected_label) AND ((candidate -> 'explicit'::text) = to_jsonb(explicit)) AND (jsonb_typeof((candidate -> 'bindingRevision'::text)) = 'number'::text) AND (((candidate ->> 'bindingRevision'::text))::numeric >= (1)::numeric) AND (((candidate ->> 'bindingRevision'::text))::numeric <= '18446744073709551615'::numeric) AND (scale(((candidate ->> 'bindingRevision'::text))::numeric) = 0) AND (jsonb_typeof((candidate -> 'config'::text)) = 'object'::text) AND (( SELECT array_agg(key.key ORDER BY key.key) AS array_agg
            FROM jsonb_object_keys((contractor_valid_pinned_runtime_config.candidate -> 'config'::text)) key(key)) = ARRAY['digest'::text, 'name'::text, 'version'::text]) AND (((candidate -> 'config'::text) ->> 'name'::text) ~ '^[a-z][a-z0-9_-]*$'::text) AND (length(((candidate -> 'config'::text) ->> 'name'::text)) <= 63) AND (((candidate -> 'config'::text) ->> 'version'::text) ~ '^[A-Za-z0-9][A-Za-z0-9._-]*$'::text) AND (length(((candidate -> 'config'::text) ->> 'version'::text)) <= 128) AND (((candidate -> 'config'::text) ->> 'digest'::text) ~ '^sha256:[0-9a-f]{64}$'::text));
END;

-- Function: contractor_valid_placement_labels(text[])

CREATE FUNCTION contractor_valid_placement_labels(labels text[]) RETURNS boolean
    LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE
    SET search_path TO 'pg_catalog'
    AS $_$
DECLARE
    label text;
    previous text := '';
BEGIN
    IF labels IS NULL OR cardinality(labels) > 32
       OR (cardinality(labels) > 0 AND (array_ndims(labels) <> 1 OR array_lower(labels, 1) <> 1))
    THEN RETURN false; END IF;
    FOREACH label IN ARRAY labels LOOP
        IF label IS NULL OR octet_length(label) NOT BETWEEN 1 AND 63
           OR label !~ '^(cloud|local)(/[a-z][a-z0-9_-]*)*$'
           OR label ~ '[^a-z0-9_/-]'
        THEN RETURN false; END IF;
        IF previous <> '' AND left(label, length(previous) + 1) <> previous || '/'
        THEN RETURN false; END IF;
        previous := label;
    END LOOP;
    RETURN true;
END;
$_$;

-- Function: contractor_valid_run_cancellation(jsonb)

CREATE FUNCTION contractor_valid_run_cancellation(payload jsonb) RETURNS boolean
    LANGUAGE sql IMMUTABLE
    BEGIN ATOMIC
 SELECT COALESCE(((jsonb_typeof(payload) = 'object'::text) AND ((payload ->> 'code'::text) = 'user_cancelled'::text) AND (jsonb_typeof((payload -> 'requestedAt'::text)) = 'string'::text) AND (btrim((payload ->> 'requestedAt'::text)) <> ''::text) AND ((NOT (payload ? 'requestedBy'::text)) OR ((jsonb_typeof((payload -> 'requestedBy'::text)) = 'string'::text) AND (btrim((payload ->> 'requestedBy'::text)) <> ''::text))) AND ((NOT (payload ? 'reason'::text)) OR ((jsonb_typeof((payload -> 'reason'::text)) = 'string'::text) AND (btrim((payload ->> 'reason'::text)) <> ''::text)))), false) AS "coalesce";
END;

-- Function: contractor_valid_run_metadata_labels(jsonb)

CREATE FUNCTION contractor_valid_run_metadata_labels(labels jsonb) RETURNS boolean
LANGUAGE sql IMMUTABLE
BEGIN ATOMIC
    SELECT jsonb_typeof(labels) = 'object'
       AND (SELECT count(*) FROM jsonb_object_keys(labels)) <= 32
       AND NOT EXISTS (
           SELECT 1 FROM jsonb_each(labels) AS entry
           WHERE jsonb_typeof(entry.value) <> 'string'
              OR octet_length(entry.key) NOT BETWEEN 1 AND 63
              OR entry.key !~ '^[a-z][a-z0-9]*([._-][a-z0-9]+)*$'
              OR entry.key ~ '^contractor\.'
              OR octet_length(entry.value #>> '{}') > 256
       );
END;

-- Function: contractor_valid_run_runtime_snapshot(jsonb, text[])

CREATE FUNCTION contractor_valid_run_runtime_snapshot(candidate jsonb, labels text[]) RETURNS boolean
    LANGUAGE sql IMMUTABLE STRICT
    BEGIN ATOMIC
 SELECT
         CASE
             WHEN (jsonb_typeof(candidate) <> 'object'::text) THEN false
             WHEN (( SELECT array_agg(key.key ORDER BY (key.key COLLATE "C")) AS array_agg
                FROM jsonb_object_keys(contractor_valid_run_runtime_snapshot.candidate) key(key)) IS DISTINCT FROM ARRAY['default'::text, 'labels'::text, 'llmCredentialIds'::text, 'runtimeCredentialIds'::text]) THEN false
             WHEN ((jsonb_typeof((candidate -> 'labels'::text)) <> 'array'::text) OR (jsonb_typeof((candidate -> 'llmCredentialIds'::text)) <> 'array'::text) OR (jsonb_typeof((candidate -> 'runtimeCredentialIds'::text)) <> 'array'::text)) THEN false
             ELSE ((contractor_valid_pinned_runtime_config((candidate -> 'default'::text), 'default'::text, false) IS TRUE) AND (jsonb_array_length((candidate -> 'labels'::text)) = cardinality(labels)) AND (NOT (EXISTS ( SELECT 1
                FROM jsonb_array_elements((contractor_valid_run_runtime_snapshot.candidate -> 'labels'::text)) WITH ORDINALITY item(value, "position")
               WHERE (contractor_valid_pinned_runtime_config(item.value, (contractor_valid_run_runtime_snapshot.labels)[item."position"], true) IS NOT TRUE)))) AND (NOT (EXISTS ( SELECT 1
                FROM ( SELECT credentials.identifier,
                         lag(credentials.identifier, 1, ''::text) OVER (PARTITION BY credentials.list ORDER BY credentials."position") AS previous
                        FROM ( SELECT 'llm'::text AS list,
                                 (credential.value #>> '{}'::text[]) AS identifier,
                                 credential."position"
                                FROM jsonb_array_elements((contractor_valid_run_runtime_snapshot.candidate -> 'llmCredentialIds'::text)) WITH ORDINALITY credential(value, "position")
                             UNION ALL
                              SELECT 'runtime'::text AS text,
                                 (credential.value #>> '{}'::text[]),
                                 credential."position"
                                FROM jsonb_array_elements((contractor_valid_run_runtime_snapshot.candidate -> 'runtimeCredentialIds'::text)) WITH ORDINALITY credential(value, "position")) credentials) ordered
               WHERE ((ordered.identifier IS NULL) OR ((ordered.identifier COLLATE "C") !~ '^[a-z][a-z0-9_-]*$'::text) OR (length(ordered.identifier) > 128) OR ((ordered.identifier COLLATE "C") <= (ordered.previous COLLATE "C")))))))
         END AS "case";
END;

-- Function: contractor_valid_runtime_agent_labels(text[])

CREATE FUNCTION contractor_valid_runtime_agent_labels(candidate text[]) RETURNS boolean
    LANGUAGE sql IMMUTABLE STRICT
    BEGIN ATOMIC
 SELECT ((cardinality(candidate) <= 32) AND (candidate = ARRAY( SELECT value.value
            FROM unnest(contractor_valid_runtime_agent_labels.candidate) value(value)
           ORDER BY (value.value COLLATE "C"))) AND (NOT (EXISTS ( SELECT 1
            FROM unnest(contractor_valid_runtime_agent_labels.candidate) value(value)
           WHERE ((value.value !~ '^[a-z][a-z0-9_-]*$'::text) OR (length(value.value) > 63) OR (value.value = 'default'::text))))) AND (cardinality(candidate) = ( SELECT count(DISTINCT value.value) AS count
            FROM unnest(contractor_valid_runtime_agent_labels.candidate) value(value))));
END;

-- Function: contractor_valid_stage_result(jsonb)

CREATE FUNCTION contractor_valid_stage_result(payload jsonb) RETURNS boolean
    LANGUAGE sql IMMUTABLE
    BEGIN ATOMIC
 SELECT COALESCE(((jsonb_typeof(payload) = 'object'::text) AND (jsonb_typeof((payload -> 'apiVersion'::text)) = 'string'::text) AND (btrim((payload ->> 'apiVersion'::text)) <> ''::text) AND ((payload ->> 'outcome'::text) = ANY (ARRAY['succeeded'::text, 'failed'::text])) AND (jsonb_typeof((payload -> 'summary'::text)) = 'string'::text) AND (btrim((payload ->> 'summary'::text)) <> ''::text) AND (jsonb_typeof((payload -> 'artifacts'::text)) = 'object'::text) AND ((NOT (payload ? 'summarized'::text)) OR (jsonb_typeof((payload -> 'summarized'::text)) = 'boolean'::text)) AND ((COALESCE(((payload -> 'summarized'::text) = 'true'::jsonb), false) AND ((payload ->> 'outcome'::text) = 'failed'::text) AND (((payload -> 'error'::text) ->> 'code'::text) = 'worker_summarized'::text) AND (((payload -> 'error'::text) -> 'retryable'::text) = 'false'::jsonb)) OR ((NOT COALESCE(((payload -> 'summarized'::text) = 'true'::jsonb), false)) AND (((payload -> 'error'::text) ->> 'code'::text) IS DISTINCT FROM 'worker_summarized'::text))) AND ((NOT (payload ? 'summaryArtifact'::text)) OR (((payload -> 'summarized'::text) = 'true'::jsonb) AND (jsonb_typeof((payload -> 'summaryArtifact'::text)) = 'object'::text) AND (jsonb_typeof(((payload -> 'summaryArtifact'::text) -> 'namespace'::text)) = 'string'::text) AND (btrim(((payload -> 'summaryArtifact'::text) ->> 'namespace'::text)) <> ''::text) AND (jsonb_typeof(((payload -> 'summaryArtifact'::text) -> 'name'::text)) = 'string'::text) AND (btrim(((payload -> 'summaryArtifact'::text) ->> 'name'::text)) <> ''::text) AND (jsonb_typeof(((payload -> 'summaryArtifact'::text) -> 'revision'::text)) = 'string'::text) AND (btrim(((payload -> 'summaryArtifact'::text) ->> 'revision'::text)) <> ''::text))) AND ((((payload ->> 'outcome'::text) = 'succeeded'::text) AND ((NOT (payload ? 'error'::text)) OR ((payload -> 'error'::text) = 'null'::jsonb))) OR (((payload ->> 'outcome'::text) = 'failed'::text) AND (jsonb_typeof((payload -> 'error'::text)) = 'object'::text) AND (jsonb_typeof(((payload -> 'error'::text) -> 'code'::text)) = 'string'::text) AND (btrim(((payload -> 'error'::text) ->> 'code'::text)) <> ''::text) AND (jsonb_typeof(((payload -> 'error'::text) -> 'message'::text)) = 'string'::text) AND (btrim(((payload -> 'error'::text) ->> 'message'::text)) <> ''::text) AND (jsonb_typeof(((payload -> 'error'::text) -> 'retryable'::text)) = 'boolean'::text)))), false) AS "coalesce";
END;

-- Function: contractor_valid_stage_termination(jsonb)

CREATE FUNCTION contractor_valid_stage_termination(payload jsonb) RETURNS boolean
    LANGUAGE sql IMMUTABLE
    BEGIN ATOMIC
 SELECT COALESCE(((jsonb_typeof(payload) = 'object'::text) AND (jsonb_typeof((payload -> 'outcome'::text)) = 'string'::text) AND ((payload ->> 'outcome'::text) = ANY (ARRAY['cancelled'::text, 'interrupted'::text])) AND (jsonb_typeof((payload -> 'code'::text)) = 'string'::text) AND (btrim((payload ->> 'code'::text)) <> ''::text) AND (jsonb_typeof((payload -> 'message'::text)) = 'string'::text) AND (btrim((payload ->> 'message'::text)) <> ''::text) AND (jsonb_typeof((payload -> 'retryable'::text)) = 'boolean'::text) AND (jsonb_typeof((payload -> 'phase'::text)) = 'string'::text) AND ((payload ->> 'phase'::text) = ANY (ARRAY['preparing'::text, 'running'::text])) AND (jsonb_typeof((payload -> 'occurredAt'::text)) = 'string'::text) AND (btrim((payload ->> 'occurredAt'::text)) <> ''::text)), false) AS "coalesce";
END;

-- Function: contractor_validate_manual_stage_execution()

CREATE FUNCTION contractor_validate_manual_stage_execution() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        IF ROW(NEW.resume_source_execution_id, NEW.resume_requested_by, NEW.resume_requested_at, NEW.resume_transaction_id)
           IS DISTINCT FROM
           ROW(OLD.resume_source_execution_id, OLD.resume_requested_by, OLD.resume_requested_at, OLD.resume_transaction_id) THEN
            RAISE EXCEPTION 'manual continuation identity is immutable' USING ERRCODE = '23514';
        END IF;
        RETURN NEW;
    END IF;
    IF NEW.resume_source_execution_id IS NOT NULL AND (
        NEW.resume_transaction_id IS DISTINCT FROM pg_current_xact_id()
        OR NOT EXISTS (SELECT 1 FROM workflow_runs WHERE run_id = NEW.run_id AND owner_id = NEW.resume_requested_by)
    ) THEN
        RAISE EXCEPTION 'manual continuation receipt must be created by its owner in this transaction' USING ERRCODE = '23514';
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


SET default_tablespace = '';

SET default_table_access_method = heap;

-- Table: allocation_execution_reports

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

-- Table: artifact_binding_revisions

CREATE TABLE artifact_binding_revisions (
    scope_kind text NOT NULL,
    scope_id text NOT NULL,
    namespace text NOT NULL,
    name text NOT NULL,
    revision text NOT NULL,
    version_id text NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    hold_count integer DEFAULT 0 NOT NULL,
    CONSTRAINT artifact_binding_revisions_hold_count_check CHECK ((hold_count >= 0)),
    CONSTRAINT artifact_binding_revisions_name_check CHECK (((btrim(name) <> ''::text) AND (POSITION(('/'::text) IN (name)) = 0))),
    CONSTRAINT artifact_binding_revisions_namespace_check CHECK (((btrim(namespace) <> ''::text) AND (POSITION(('/'::text) IN (namespace)) = 0))),
    CONSTRAINT artifact_binding_revisions_revision_check CHECK ((btrim(revision) <> ''::text))
);

-- Table: artifact_bindings

CREATE TABLE artifact_bindings (
    scope_kind text NOT NULL,
    scope_id text NOT NULL,
    namespace text NOT NULL,
    name text NOT NULL,
    current_revision text NOT NULL,
    frozen boolean DEFAULT false NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    retained_revisions bigint NOT NULL,
    retained_bytes bigint NOT NULL,
    material_rank smallint DEFAULT 6 NOT NULL,
    CONSTRAINT artifact_bindings_current_revision_check CHECK ((btrim(current_revision) <> ''::text)),
    CONSTRAINT artifact_bindings_material_rank_check CHECK (((material_rank >= 0) AND (material_rank <= 6))),
    CONSTRAINT artifact_bindings_name_check CHECK (((btrim(name) <> ''::text) AND (POSITION(('/'::text) IN (name)) = 0))),
    CONSTRAINT artifact_bindings_namespace_check CHECK (((btrim(namespace) <> ''::text) AND (POSITION(('/'::text) IN (namespace)) = 0))),
    CONSTRAINT artifact_bindings_retained_bytes_check CHECK ((retained_bytes >= 0)),
    CONSTRAINT artifact_bindings_retained_revisions_check CHECK ((retained_revisions >= 1))
);

-- Table: artifact_blobs

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

-- Table: artifact_lineage

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
    CONSTRAINT artifact_lineage_lineage_kind_check CHECK ((lineage_kind = ANY (ARRAY['input_fork'::text, 'output_bind'::text, 'project_output_publish'::text, 'audit_import'::text, 'context_projection'::text, 'audit_output'::text, 'audit_input'::text, 'audit_publish'::text])))
);

-- Table: artifact_scopes

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
    revision_count bigint DEFAULT 0 NOT NULL,
    audit_id text GENERATED ALWAYS AS (
CASE
    WHEN (scope_kind = 'audit'::text) THEN scope_id
    ELSE NULL::text
END) STORED,
    CONSTRAINT artifact_scopes_revision_count_check CHECK (((revision_count >= 0) AND ((scope_kind = 'run'::text) OR (revision_count = 0)))),
    CONSTRAINT artifact_scopes_scope_id_check CHECK ((btrim(scope_id) <> ''::text)),
    CONSTRAINT artifact_scopes_scope_kind_check CHECK ((scope_kind = ANY (ARRAY['user'::text, 'project'::text, 'run'::text, 'audit'::text])))
);

-- Table: artifact_versions

CREATE TABLE artifact_versions (
    version_id text NOT NULL,
    blob_sha256 bytea NOT NULL,
    media_type text NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    git_repository_url text,
    git_requested_ref text,
    git_resolved_commit text,
    git_imported_at timestamp with time zone,
    git_analysis jsonb,
    execution_origin jsonb,
    CONSTRAINT artifact_versions_execution_origin_check CHECK (((execution_origin IS NULL) OR (((jsonb_typeof(execution_origin) = 'object'::text) AND (octet_length((execution_origin)::text) <= 2048) AND (execution_origin ?& ARRAY['runId'::text, 'stageExecutionId'::text, 'allocationId'::text, 'logicalAgentName'::text, 'artifact'::text]) AND ((execution_origin - ARRAY['runId'::text, 'stageExecutionId'::text, 'allocationId'::text, 'logicalAgentName'::text, 'artifact'::text]) = '{}'::jsonb) AND (jsonb_typeof((execution_origin -> 'artifact'::text)) = 'object'::text) AND (jsonb_typeof((execution_origin -> 'runId'::text)) = 'string'::text) AND (jsonb_typeof((execution_origin -> 'stageExecutionId'::text)) = 'string'::text) AND (jsonb_typeof((execution_origin -> 'allocationId'::text)) = 'string'::text) AND (jsonb_typeof((execution_origin -> 'logicalAgentName'::text)) = 'string'::text) AND ((execution_origin ->> 'runId'::text) ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text) AND ((execution_origin ->> 'stageExecutionId'::text) ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text) AND ((execution_origin ->> 'allocationId'::text) ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text) AND ((execution_origin ->> 'logicalAgentName'::text) ~ '^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$'::text)) IS TRUE))),
    CONSTRAINT artifact_versions_git_analysis CHECK (((git_analysis IS NULL) OR (((git_resolved_commit IS NOT NULL) AND (jsonb_typeof(git_analysis) = 'object'::text) AND (git_analysis ?& ARRAY['policy'::text, 'reportPath'::text, 'includedFiles'::text, 'excludedEntries'::text]) AND ((git_analysis - ARRAY['policy'::text, 'reportPath'::text, 'includedFiles'::text, 'excludedEntries'::text]) = '{}'::jsonb) AND ((git_analysis ->> 'policy'::text) = 'source-analysis-v1'::text) AND ((git_analysis ->> 'reportPath'::text) = '.contractor-import-report.json'::text) AND (jsonb_typeof((git_analysis -> 'includedFiles'::text)) = 'number'::text) AND ((git_analysis ->> 'includedFiles'::text) ~ '^[1-9][0-9]{0,4}$'::text) AND ((((git_analysis ->> 'includedFiles'::text))::numeric >= (1)::numeric) AND (((git_analysis ->> 'includedFiles'::text))::numeric <= (49999)::numeric)) AND (jsonb_typeof((git_analysis -> 'excludedEntries'::text)) = 'number'::text) AND ((git_analysis ->> 'excludedEntries'::text) ~ '^(0|[1-9][0-9]{0,4})$'::text) AND ((((git_analysis ->> 'excludedEntries'::text))::numeric >= (0)::numeric) AND (((git_analysis ->> 'excludedEntries'::text))::numeric <= (50000)::numeric))) IS TRUE))),
    CONSTRAINT artifact_versions_git_repository_url_check CHECK (((octet_length(git_repository_url) >= 1) AND (octet_length(git_repository_url) <= 12288))),
    CONSTRAINT artifact_versions_git_requested_ref_check CHECK (((octet_length(git_requested_ref) >= 1) AND (octet_length(git_requested_ref) <= 1024))),
    CONSTRAINT artifact_versions_git_resolved_commit_check CHECK ((git_resolved_commit ~ '^[0-9a-f]{40}$'::text)),
    CONSTRAINT artifact_versions_git_source CHECK ((((git_repository_url IS NULL) = (git_resolved_commit IS NULL)) AND ((git_repository_url IS NULL) = (git_imported_at IS NULL)) AND ((git_requested_ref IS NULL) OR (git_repository_url IS NOT NULL)))),
    CONSTRAINT artifact_versions_media_type_check CHECK (((media_type = lower(media_type)) AND (media_type ~ '^[a-z0-9][a-z0-9!#$%&''+.^_`|~-]*/[a-z0-9][a-z0-9!#$%&''+.^_`|~-]*$'::text))),
    CONSTRAINT artifact_versions_media_type_length CHECK (((length(media_type) >= 3) AND (length(media_type) <= 255))),
    CONSTRAINT artifact_versions_version_id_check CHECK ((btrim(version_id) <> ''::text))
);

-- Table: audit_artifact_links

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
    results_descriptor_bytes bigint GENERATED ALWAYS AS (contractor_round_results_descriptor_bytes(logical_key, artifact_ref, artifact_digest, media_type, size_bytes, source_provenance, display_ref)) STORED,
    artifact_scope text DEFAULT 'project'::text NOT NULL,
    material_ordinal bigint NOT NULL,
    CONSTRAINT audit_artifact_links_artifact_digest_check CHECK ((artifact_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audit_artifact_links_artifact_ref_check CHECK (((jsonb_typeof(artifact_ref) = 'object'::text) AND (octet_length((artifact_ref)::text) <= 4096))),
    CONSTRAINT audit_artifact_links_artifact_scope_check CHECK ((artifact_scope = ANY (ARRAY['project'::text, 'audit'::text]))),
    CONSTRAINT audit_artifact_links_display_ref_check CHECK ((octet_length(display_ref) <= 1024)),
    CONSTRAINT audit_artifact_links_logical_key_check CHECK (((btrim(logical_key) <> ''::text) AND (octet_length(logical_key) <= 512))),
    CONSTRAINT audit_artifact_links_media_type_check CHECK (((btrim(media_type) <> ''::text) AND (octet_length(media_type) <= 256))),
    CONSTRAINT audit_artifact_links_size_bytes_check CHECK ((size_bytes >= 0)),
    CONSTRAINT audit_artifact_links_source_provenance_check CHECK (((jsonb_typeof(source_provenance) = 'object'::text) AND (octet_length((source_provenance)::text) <= 1048576)))
);

-- Sequence: audit_artifact_links_material_ordinal_seq

ALTER TABLE audit_artifact_links ALTER COLUMN material_ordinal ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME audit_artifact_links_material_ordinal_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);

-- Table: audit_controller_claims

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

-- Table: audit_events

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

-- Table: audit_execution_items

CREATE TABLE audit_execution_items (
    execution_item_id text NOT NULL,
    execution_id text NOT NULL,
    audit_id text NOT NULL,
    batch_ordinal integer NOT NULL,
    item_attempt integer NOT NULL,
    state text DEFAULT 'submitted'::text NOT NULL,
    collection_disposition text,
    result_ref jsonb,
    result_digest text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    collected_at timestamp with time zone,
    task_id text NOT NULL,
    finding_result jsonb,
    stop_record jsonb,
    CONSTRAINT audit_execution_items_batch_ordinal_check CHECK ((batch_ordinal >= 0)),
    CONSTRAINT audit_execution_items_collection_disposition_check CHECK (((collection_disposition IS NULL) OR (collection_disposition = ANY (ARRAY['accepted-result'::text, 'missing-output'::text, 'invalid-result'::text, 'execution-failed'::text, 'execution-cancelled'::text, 'collection-contract-invalid'::text])))),
    CONSTRAINT audit_execution_items_collection_shape CHECK ((((state = ANY (ARRAY['submitted'::text, 'collecting'::text])) AND (collection_disposition IS NULL) AND (result_ref IS NULL) AND (result_digest IS NULL) AND (collected_at IS NULL) AND (stop_record IS NULL)) OR ((state = 'settled'::text) AND (collection_disposition IS NOT NULL) AND (collected_at IS NOT NULL) AND ((result_ref IS NULL) = (result_digest IS NULL)) AND ((stop_record IS NULL) OR ((result_ref IS NULL) AND (jsonb_typeof(stop_record) = 'object'::text) AND (octet_length((stop_record)::text) <= 1048576)))))),
    CONSTRAINT audit_execution_items_execution_item_id_check CHECK ((execution_item_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT audit_execution_items_item_attempt_check CHECK ((item_attempt > 0)),
    CONSTRAINT audit_execution_items_result_digest_check CHECK (((result_digest IS NULL) OR (result_digest ~ '^sha256:[0-9a-f]{64}$'::text))),
    CONSTRAINT audit_execution_items_result_ref_check CHECK (((result_ref IS NULL) OR ((jsonb_typeof(result_ref) = 'object'::text) AND (octet_length((result_ref)::text) <= 4096)))),
    CONSTRAINT audit_execution_items_state_check CHECK ((state = ANY (ARRAY['submitted'::text, 'collecting'::text, 'settled'::text]))),
    CONSTRAINT audit_item_findings_check CHECK (((finding_result IS NULL) OR (((state = 'settled'::text) AND (result_ref IS NULL) AND (jsonb_typeof(finding_result) = 'object'::text) AND (octet_length((finding_result)::text) <= 49152) AND (finding_result ?& ARRAY['ids'::text, 'truncated'::text]) AND ((finding_result - ARRAY['ids'::text, 'truncated'::text]) = '{}'::jsonb) AND (jsonb_typeof((finding_result -> 'ids'::text)) = 'array'::text) AND (jsonb_array_length((finding_result -> 'ids'::text)) <= 128) AND (jsonb_typeof((finding_result -> 'truncated'::text)) = 'boolean'::text)) IS TRUE)))
);

-- Table: audit_executions

CREATE TABLE audit_executions (
    execution_id text NOT NULL,
    audit_id text NOT NULL,
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
    lane_class text NOT NULL,
    prepared_work_source json,
    prepared_work_offset integer DEFAULT 0 NOT NULL,
    prepared_work_count integer,
    llm_cost jsonb,
    CONSTRAINT audit_executions_collection_disposition_check CHECK ((collection_disposition = ANY (ARRAY['accepted-result'::text, 'missing-output'::text, 'invalid-result'::text, 'execution-failed'::text, 'execution-cancelled'::text, 'collection-contract-invalid'::text]))),
    CONSTRAINT audit_executions_collection_error_code_check CHECK (((btrim(collection_error_code) <> ''::text) AND (octet_length(collection_error_code) <= 128))),
    CONSTRAINT audit_executions_collection_observation_shape CHECK (((collection_receipt_id IS NULL) OR ((run_id IS NOT NULL) AND (terminal_outcome IS NOT NULL) AND (terminal_run_generation IS NOT NULL) AND (btrim(terminal_run_generation) <> ''::text) AND (terminal_run_sequence >= 0)) OR ((run_id IS NULL) AND (terminal_outcome = 'submission-failed'::text) AND (terminal_run_generation IS NULL) AND (terminal_run_sequence IS NULL)))),
    CONSTRAINT audit_executions_collection_receipt_id_check CHECK ((collection_receipt_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT audit_executions_collection_request_digest_check CHECK ((collection_request_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audit_executions_collection_retained_refs_check CHECK (((jsonb_typeof(collection_retained_refs) = 'array'::text) AND (octet_length((collection_retained_refs)::text) <= 8388608))),
    CONSTRAINT audit_executions_collection_shape CHECK ((((collection_receipt_id IS NULL) = (collection_disposition IS NULL)) AND ((collection_receipt_id IS NULL) = (collection_retained_refs IS NULL)) AND ((collection_receipt_id IS NULL) = (collection_request_digest IS NULL)) AND ((collection_receipt_id IS NULL) = (collected_at IS NULL)) AND ((collection_receipt_id IS NOT NULL) OR ((collection_error_code IS NULL) AND (collection_source_output_ref IS NULL))))),
    CONSTRAINT audit_executions_collection_source_output_digest_check CHECK ((collection_source_output_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audit_executions_collection_source_output_ref_check CHECK (((jsonb_typeof(collection_source_output_ref) = 'object'::text) AND (octet_length((collection_source_output_ref)::text) <= 4096))),
    CONSTRAINT audit_executions_collection_source_shape CHECK ((((collection_source_output_ref IS NULL) = (collection_source_output_digest IS NULL)) AND ((collection_receipt_id IS NULL) OR ((collection_disposition = 'accepted-result'::text) OR ((collection_disposition = 'invalid-result'::text) = (collection_source_output_ref IS NOT NULL)))))),
    CONSTRAINT audit_executions_execution_id_check CHECK ((execution_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT audit_executions_lane_class_check CHECK ((lane_class = ANY (ARRAY['model'::text, 'tool'::text]))),
    CONSTRAINT audit_executions_llm_cost_check CHECK (((llm_cost IS NULL) OR (jsonb_typeof(llm_cost) = 'object'::text))),
    CONSTRAINT audit_executions_prepared_work_count_check CHECK (((prepared_work_count >= 0) AND (prepared_work_count <= 256))),
    CONSTRAINT audit_executions_prepared_work_offset_check CHECK (((prepared_work_offset >= 0) AND (prepared_work_offset <= 256))),
    CONSTRAINT audit_executions_prepared_work_shape CHECK ((((prepared_work_source IS NULL) AND (prepared_work_count IS NULL) AND (prepared_work_offset = 0)) OR ((prepared_work_source IS NOT NULL) AND (prepared_work_count IS NOT NULL) AND (prepared_work_offset <= prepared_work_count)))),
    CONSTRAINT audit_executions_prepared_work_source_check CHECK (((prepared_work_source IS NULL) OR ((json_typeof(prepared_work_source) = 'object'::text) AND (octet_length((prepared_work_source)::text) <= 4096)))),
    CONSTRAINT audit_executions_request_digest_check CHECK ((request_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
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

-- Table: audit_hypotheses

CREATE TABLE audit_hypotheses (
    hypothesis_id text NOT NULL,
    audit_id text NOT NULL,
    source_key text NOT NULL,
    fixed boolean NOT NULL,
    ordinal integer NOT NULL,
    current_bytes integer NOT NULL,
    current_revision bigint NOT NULL,
    CONSTRAINT audit_hypotheses_current_bytes_check CHECK (((current_bytes >= 1) AND (current_bytes <= 32768))),
    CONSTRAINT audit_hypotheses_current_revision_check CHECK (((current_revision >= 1) AND (current_revision <= '9007199254740991'::bigint))),
    CONSTRAINT audit_hypotheses_hypothesis_id_check CHECK ((hypothesis_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT audit_hypotheses_ordinal_check CHECK (((ordinal >= 1) AND (ordinal <= 10000))),
    CONSTRAINT audit_hypotheses_source_key_check CHECK (((octet_length(source_key) >= 1) AND (octet_length(source_key) <= 128) AND (source_key ~ '^[A-Za-z0-9][A-Za-z0-9_.-]*$'::text)))
);

-- Table: audit_hypothesis_revisions

CREATE TABLE audit_hypothesis_revisions (
    hypothesis_id text NOT NULL,
    audit_id text NOT NULL,
    revision bigint NOT NULL,
    document json NOT NULL,
    CONSTRAINT audit_hypothesis_revisions_check CHECK ((((document ->> 'id'::text) = hypothesis_id) AND (((document ->> 'revision'::text))::numeric = (revision)::numeric) AND (((document -> 'origin'::text) ->> 'audit_id'::text) = audit_id))),
    CONSTRAINT audit_hypothesis_revisions_document_check CHECK (((json_typeof(document) = 'object'::text) AND (octet_length((document)::text) <= 32768))),
    CONSTRAINT audit_hypothesis_revisions_document_check1 CHECK (((document ->> 'schema'::text) = 'contractor.audit.hypothesis.v1'::text)),
    CONSTRAINT audit_hypothesis_revisions_revision_check CHECK (((revision >= 1) AND (revision <= '9007199254740991'::bigint)))
);

-- Table: audit_tasks

CREATE TABLE audit_tasks (
    task_id text NOT NULL,
    audit_id text NOT NULL,
    source_key text NOT NULL,
    source json NOT NULL,
    document json NOT NULL,
    workflow_alias text GENERATED ALWAYS AS ((document ->> 'workflow'::text)) STORED NOT NULL,
    batch_digest text NOT NULL,
    ordinal integer NOT NULL,
    iteration integer NOT NULL,
    state text DEFAULT 'ready'::text NOT NULL,
    attempts integer DEFAULT 0 NOT NULL,
    last_execution_id text,
    result json,
    gap_code text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    attention boolean DEFAULT false NOT NULL,
    hypothesis_snapshot json,
    finding_result jsonb,
    metadata_bytes integer GENERATED ALWAYS AS ((((49152 + octet_length((document)::text)) + octet_length((source)::text)) + COALESCE(octet_length((hypothesis_snapshot)::text), 0))) STORED,
    stop_record jsonb,
    CONSTRAINT audit_task_findings_check CHECK (((finding_result IS NULL) OR (((state = 'settled'::text) AND (result IS NULL) AND (jsonb_typeof(finding_result) = 'object'::text) AND (octet_length((finding_result)::text) <= 49152) AND (finding_result ?& ARRAY['ids'::text, 'truncated'::text]) AND ((finding_result - ARRAY['ids'::text, 'truncated'::text]) = '{}'::jsonb) AND (jsonb_typeof((finding_result -> 'ids'::text)) = 'array'::text) AND (jsonb_array_length((finding_result -> 'ids'::text)) <= 128) AND (jsonb_typeof((finding_result -> 'truncated'::text)) = 'boolean'::text)) IS TRUE))),
    CONSTRAINT audit_tasks_attempts_check CHECK (((attempts >= 0) AND (attempts <= 10))),
    CONSTRAINT audit_tasks_batch_digest_check CHECK ((batch_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audit_tasks_check CHECK ((((document ->> 'audit_id'::text) = audit_id) AND ((document ->> 'task_id'::text) = task_id))),
    CONSTRAINT audit_tasks_check1 CHECK (((state = 'settled'::text) OR ((result IS NULL) AND (gap_code IS NULL)))),
    CONSTRAINT audit_tasks_document_check CHECK (((json_typeof(document) = 'object'::text) AND (octet_length((document)::text) <= 1048576))),
    CONSTRAINT audit_tasks_document_check1 CHECK (((document ->> 'schema'::text) = 'contractor.audit.task.v1'::text)),
    CONSTRAINT audit_tasks_gap_code_check CHECK (((gap_code IS NULL) OR ((btrim(gap_code) <> ''::text) AND (octet_length(gap_code) <= 128)))),
    CONSTRAINT audit_tasks_hypothesis_shape CHECK (((((document)::jsonb ? 'hypothesis'::text) = (hypothesis_snapshot IS NOT NULL)) AND ((hypothesis_snapshot IS NULL) OR ((((document -> 'hypothesis'::text) ->> 'id'::text) = (hypothesis_snapshot ->> 'id'::text)) AND ((((document -> 'hypothesis'::text) ->> 'revision'::text))::numeric = ((hypothesis_snapshot ->> 'revision'::text))::numeric))))),
    CONSTRAINT audit_tasks_hypothesis_snapshot_check CHECK (((hypothesis_snapshot IS NULL) OR ((json_typeof(hypothesis_snapshot) = 'object'::text) AND (octet_length((hypothesis_snapshot)::text) <= 32768)))),
    CONSTRAINT audit_tasks_iteration_check CHECK (((iteration >= 1) AND (iteration <= 32))),
    CONSTRAINT audit_tasks_ordinal_check CHECK (((ordinal >= 1) AND (ordinal <= 100000))),
    CONSTRAINT audit_tasks_result_check CHECK (((result IS NULL) OR ((json_typeof(result) = 'object'::text) AND (octet_length((result)::text) <= 4096)))),
    CONSTRAINT audit_tasks_source_check CHECK (((json_typeof(source) = 'object'::text) AND (octet_length((source)::text) <= 4096))),
    CONSTRAINT audit_tasks_source_key_check CHECK ((source_key ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audit_tasks_state_check CHECK ((state = ANY (ARRAY['ready'::text, 'selected'::text, 'settled'::text]))),
    CONSTRAINT audit_tasks_stop_record_check CHECK (((stop_record IS NULL) OR ((state = 'settled'::text) AND (result IS NULL) AND (jsonb_typeof(stop_record) = 'object'::text) AND (octet_length((stop_record)::text) <= 1048576)))),
    CONSTRAINT audit_tasks_task_id_check CHECK ((task_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text))
);

-- Table: audits

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
    outstanding_tool_run_count integer DEFAULT 0 NOT NULL,
    task_count integer DEFAULT 0 NOT NULL,
    task_iteration integer DEFAULT 1 NOT NULL,
    iteration_task_count integer DEFAULT 0 NOT NULL,
    initial_work_offset integer DEFAULT 0 NOT NULL,
    initial_hypothesis_offset integer DEFAULT 0 NOT NULL,
    hypothesis_count integer DEFAULT 0 NOT NULL,
    hypothesis_revision_count integer DEFAULT 0 NOT NULL,
    hypothesis_bytes bigint DEFAULT 0 NOT NULL,
    llm_cost jsonb DEFAULT '{}'::jsonb,
    CONSTRAINT audits_audit_id_check CHECK ((audit_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT audits_baseline_shape CHECK (((state = ANY (ARRAY['draft'::text, 'deleting'::text])) OR (baseline_snapshot IS NOT NULL))),
    CONSTRAINT audits_baseline_snapshot_check CHECK (((baseline_snapshot IS NULL) OR ((jsonb_typeof(baseline_snapshot) = 'object'::text) AND (octet_length((baseline_snapshot)::text) <= 16777216)))),
    CONSTRAINT audits_batch_size_check CHECK (((batch_size > 0) AND (batch_size <= 64))),
    CONSTRAINT audits_check CHECK (((task_count >= 0) AND (task_count <= max_items_total))),
    CONSTRAINT audits_check1 CHECK (((task_iteration >= 1) AND (task_iteration <= max_rounds))),
    CONSTRAINT audits_check2 CHECK (((iteration_task_count >= 0) AND (iteration_task_count <= max_items_per_round))),
    CONSTRAINT audits_counter_bounds CHECK (((reserved_run_count <= max_submitted_runs) AND (submitted_run_count <= reserved_run_count) AND (outstanding_run_count <= reserved_run_count) AND (retained_evidence_bytes <= max_evidence_bytes))),
    CONSTRAINT audits_deletion_intent_shape CHECK ((((deletion_requested_at IS NULL) AND (state <> 'deleting'::text)) OR ((deletion_requested_at IS NOT NULL) AND (state = ANY (ARRAY['cancelling'::text, 'deleting'::text]))))),
    CONSTRAINT audits_dispatch_state_check CHECK ((dispatch_state = ANY (ARRAY['open'::text, 'closed'::text]))),
    CONSTRAINT audits_hold_state_check CHECK ((hold_state = ANY (ARRAY['pending'::text, 'held'::text, 'released'::text]))),
    CONSTRAINT audits_hypothesis_bytes_check CHECK (((hypothesis_bytes >= 0) AND (hypothesis_bytes <= 268435456))),
    CONSTRAINT audits_hypothesis_count_check CHECK (((hypothesis_count >= 0) AND (hypothesis_count <= 10000))),
    CONSTRAINT audits_hypothesis_revision_count_check CHECK (((hypothesis_revision_count >= 0) AND (hypothesis_revision_count <= 50000))),
    CONSTRAINT audits_initial_hypothesis_offset_check CHECK (((initial_hypothesis_offset >= 0) AND (initial_hypothesis_offset <= 10000))),
    CONSTRAINT audits_initial_work_offset_check CHECK (((initial_work_offset >= 0) AND (initial_work_offset <= 10000))),
    CONSTRAINT audits_input_selection_check CHECK (((jsonb_typeof(input_selection) = 'object'::text) AND (octet_length((input_selection)::text) <= 8388608))),
    CONSTRAINT audits_llm_cost_check CHECK (((llm_cost IS NULL) OR (jsonb_typeof(llm_cost) = 'object'::text))),
    CONSTRAINT audits_max_evidence_bytes_check CHECK (((max_evidence_bytes > 0) AND (max_evidence_bytes <= 1073741824))),
    CONSTRAINT audits_max_item_run_attempts_check CHECK (((max_item_run_attempts > 0) AND (max_item_run_attempts <= 10))),
    CONSTRAINT audits_max_items_per_round_check CHECK (((max_items_per_round > 0) AND (max_items_per_round <= 10000))),
    CONSTRAINT audits_max_items_total_check CHECK (((max_items_total > 0) AND (max_items_total <= 100000))),
    CONSTRAINT audits_max_rounds_check CHECK (((max_rounds > 0) AND (max_rounds <= 32))),
    CONSTRAINT audits_max_submitted_runs_check CHECK (((max_submitted_runs > 0) AND (max_submitted_runs <= 1000000))),
    CONSTRAINT audits_next_event_sequence_check CHECK ((next_event_sequence > 0)),
    CONSTRAINT audits_outstanding_run_count_check CHECK ((outstanding_run_count >= 0)),
    CONSTRAINT audits_outstanding_tool_run_count_check CHECK (((outstanding_tool_run_count >= 0) AND (outstanding_tool_run_count <= outstanding_run_count))),
    CONSTRAINT audits_owner_id_check CHECK (((btrim(owner_id) <> ''::text) AND (octet_length(owner_id) <= 256))),
    CONSTRAINT audits_profile_digest_check CHECK ((profile_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT audits_profile_name_check CHECK (((btrim(profile_name) <> ''::text) AND (octet_length(profile_name) <= 128))),
    CONSTRAINT audits_profile_snapshot_check CHECK (((jsonb_typeof(profile_snapshot) = 'object'::text) AND (octet_length((profile_snapshot)::text) <= 8388608))),
    CONSTRAINT audits_profile_version_check CHECK (((btrim(profile_version) <> ''::text) AND (octet_length(profile_version) <= 128))),
    CONSTRAINT audits_reserved_run_count_check CHECK ((reserved_run_count >= 0)),
    CONSTRAINT audits_retained_evidence_bytes_check CHECK ((retained_evidence_bytes >= 0)),
    CONSTRAINT audits_revision_check CHECK ((revision > 0)),
    CONSTRAINT audits_state_check CHECK ((state = ANY (ARRAY['draft'::text, 'active'::text, 'paused'::text, 'finalizing'::text, 'cancelling'::text, 'completed'::text, 'cancelled'::text, 'failed'::text, 'deleting'::text]))),
    CONSTRAINT audits_stop_reason_code_check CHECK (((stop_reason_code IS NULL) OR ((btrim(stop_reason_code) <> ''::text) AND (octet_length(stop_reason_code) <= 128)))),
    CONSTRAINT audits_stop_reason_message_check CHECK (((stop_reason_message IS NULL) OR (octet_length(stop_reason_message) <= 4096))),
    CONSTRAINT audits_stop_reason_shape CHECK (((stop_reason_code IS NULL) = (stop_reason_message IS NULL))),
    CONSTRAINT audits_submitted_run_count_check CHECK ((submitted_run_count >= 0)),
    CONSTRAINT audits_time_shape CHECK ((((state = 'draft'::text) AND (started_at IS NULL) AND (finished_at IS NULL)) OR ((state = ANY (ARRAY['active'::text, 'paused'::text, 'finalizing'::text, 'cancelling'::text])) AND (started_at IS NOT NULL) AND (finished_at IS NULL)) OR ((state = ANY (ARRAY['completed'::text, 'cancelled'::text, 'failed'::text])) AND (started_at IS NOT NULL) AND (finished_at IS NOT NULL)) OR (state = 'deleting'::text)))
);

-- Table: credential_operations

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

-- Table: eval_collections

CREATE TABLE eval_collections (
    owner_id text NOT NULL,
    project_id text NOT NULL,
    revision bigint DEFAULT 1 NOT NULL
);

-- Table: eval_commands

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

-- Table: eval_controller_claims

CREATE TABLE eval_controller_claims (
    experiment_id text NOT NULL,
    epoch bigint DEFAULT 0 NOT NULL,
    holder_id text,
    expires_at timestamp with time zone,
    claimed_at timestamp with time zone,
    CONSTRAINT eval_controller_claim_shape CHECK ((((holder_id IS NULL) AND (claimed_at IS NULL) AND (expires_at IS NULL)) OR ((holder_id IS NOT NULL) AND (claimed_at IS NOT NULL) AND (expires_at > claimed_at)))),
    CONSTRAINT eval_controller_claims_epoch_check CHECK ((epoch >= 0)),
    CONSTRAINT eval_controller_claims_holder_id_check CHECK (((holder_id IS NULL) OR ((btrim(holder_id) <> ''::text) AND (octet_length(holder_id) <= 256))))
);

-- Table: eval_dataset_revisions

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

-- Table: eval_evidence_refs

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

-- Table: eval_execution_tombstones

CREATE TABLE eval_execution_tombstones (
    experiment_id text NOT NULL,
    member_id text NOT NULL,
    execution_kind text NOT NULL,
    execution_id text NOT NULL,
    terminal_state text NOT NULL,
    never_started boolean NOT NULL,
    deleted_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL
);

-- Table: eval_experiments

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
    plan_kind text,
    plan_document bytea,
    plan_document_sha256 text GENERATED ALWAYS AS (
CASE
    WHEN (plan_document IS NOT NULL) THEN ('sha256:'::text || encode(sha256(plan_document), 'hex'::text))
    ELSE NULL::text
END) STORED,
    plan_sha256 text,
    plan_setup bytea,
    plan_created_at timestamp with time zone,
    plan_readiness bytea,
    CONSTRAINT eval_experiments_check CHECK (((outstanding_count >= 0) AND (outstanding_count <= expected_count))),
    CONSTRAINT eval_experiments_check1 CHECK (((dataset_id IS NULL) = (dataset_revision IS NULL))),
    CONSTRAINT eval_experiments_check2 CHECK ((((started_at IS NULL) AND (deadline_at IS NULL)) OR ((started_at IS NOT NULL) AND (deadline_at > started_at)))),
    CONSTRAINT eval_experiments_check3 CHECK (((draft IS NULL) OR ((control_mode = 'server'::text) AND ((octet_length(draft) >= 2) AND (octet_length(draft) <= 1048576))))),
    CONSTRAINT eval_experiments_control_mode_check CHECK ((control_mode = ANY (ARRAY['server'::text, 'external'::text]))),
    CONSTRAINT eval_experiments_expected_count_check CHECK (((expected_count >= 0) AND (expected_count <= 10000))),
    CONSTRAINT eval_experiments_max_in_flight_check CHECK (((max_in_flight >= 1) AND (max_in_flight <= 10000))),
    CONSTRAINT eval_experiments_name_check CHECK (((length(name) >= 1) AND (length(name) <= 256))),
    CONSTRAINT eval_experiments_observed_tokens_check CHECK ((observed_tokens >= 0)),
    CONSTRAINT eval_experiments_plan_document_check CHECK (((octet_length(plan_document) >= 2) AND (octet_length(plan_document) <= 1048576))),
    CONSTRAINT eval_experiments_plan_kind CHECK ((plan_kind = ANY (ARRAY['playground.plan/v1'::text, 'ExternalRegistration'::text]))),
    CONSTRAINT eval_experiments_plan_readiness_check CHECK (((octet_length(plan_readiness) >= 2) AND (octet_length(plan_readiness) <= 1048576))),
    CONSTRAINT eval_experiments_plan_setup_check CHECK (((octet_length(plan_setup) >= 2) AND (octet_length(plan_setup) <= 1048576))),
    CONSTRAINT eval_experiments_plan_sha256 CHECK ((plan_sha256 ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT eval_experiments_plan_shape CHECK ((((plan_document IS NULL) AND (plan_document_sha256 IS NULL) AND (plan_kind IS NULL) AND (plan_sha256 IS NULL) AND (plan_setup IS NULL) AND (plan_created_at IS NULL) AND (plan_readiness IS NULL)) OR ((plan_document IS NOT NULL) AND (plan_document_sha256 IS NOT NULL) AND (plan_kind IS NOT NULL) AND (plan_sha256 IS NOT NULL) AND (plan_setup IS NOT NULL) AND (plan_created_at IS NOT NULL) AND ((plan_kind = 'playground.plan/v1'::text) = (plan_readiness IS NOT NULL))))),
    CONSTRAINT eval_experiments_revision_check CHECK ((revision > 0)),
    CONSTRAINT eval_experiments_state_check CHECK ((state = ANY (ARRAY['draft'::text, 'preparing'::text, 'ready'::text, 'running'::text, 'settling'::text, 'finished'::text, 'pausing'::text, 'paused'::text, 'cancelling'::text, 'cancelled'::text]))),
    CONSTRAINT eval_experiments_token_limit_check CHECK ((token_limit >= 0)),
    CONSTRAINT eval_experiments_view_generation_check CHECK ((view_generation > 0)),
    CONSTRAINT eval_experiments_wall_ms_check CHECK ((wall_ms > 0))
);

-- Table: eval_member_projections

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

-- Table: eval_members

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
    selected_result_kind text GENERATED ALWAYS AS ('result'::text) STORED,
    selected_result_sha256 text,
    selected_assessment_kind text GENERATED ALWAYS AS ('assessment'::text) STORED,
    selected_assessment_sha256 text,
    selection_revision bigint DEFAULT 0 NOT NULL,
    selected_by text,
    CONSTRAINT eval_members_eligibility_check CHECK ((eligibility = ANY (ARRAY['eligible'::text, 'unsupported'::text, 'blocked'::text]))),
    CONSTRAINT eval_members_eligibility_reason_check CHECK ((length(eligibility_reason) <= 8192)),
    CONSTRAINT eval_members_execution_kind_check CHECK ((execution_kind = ANY (ARRAY['run'::text, 'audit'::text]))),
    CONSTRAINT eval_members_member_id_check CHECK ((member_id ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT eval_members_ordinal_check CHECK (((ordinal >= 0) AND (ordinal <= 9999))),
    CONSTRAINT eval_members_pair_id_check CHECK ((pair_id ~ '^[0-9a-f]{64}$'::text)),
    CONSTRAINT eval_members_recipe_check CHECK (((octet_length(recipe) >= 2) AND (octet_length(recipe) <= 1048576))),
    CONSTRAINT eval_members_sample_check CHECK (((sample >= 1) AND (sample <= 100))),
    CONSTRAINT eval_members_selection_shape CHECK ((((selected_result_sha256 IS NULL) AND (selected_assessment_sha256 IS NULL) AND (selection_revision = 0) AND (selected_by IS NULL)) OR ((selected_result_sha256 IS NOT NULL) AND (selection_revision > 0) AND (selected_by IS NOT NULL))))
);

-- Table: eval_plan_resources

CREATE TABLE eval_plan_resources (
    experiment_id text NOT NULL,
    resource_path text NOT NULL,
    document_kind text NOT NULL,
    document bytea NOT NULL,
    document_sha256 text GENERATED ALWAYS AS (('sha256:'::text || encode(sha256(document), 'hex'::text))) STORED,
    CONSTRAINT eval_plan_resources_document_check CHECK (((octet_length(document) >= 2) AND (octet_length(document) <= 1048576))),
    CONSTRAINT eval_plan_resources_resource_path_check CHECK (((length(resource_path) >= 1) AND (length(resource_path) <= 1024)))
);

-- Table: eval_progress_observations

CREATE TABLE eval_progress_observations (
    experiment_id text NOT NULL,
    sequence bigint NOT NULL,
    observed_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    counts jsonb NOT NULL
);

-- Sequence: eval_progress_observations_sequence_seq

ALTER TABLE eval_progress_observations ALTER COLUMN sequence ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME eval_progress_observations_sequence_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);

-- Table: eval_projection_queue

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

-- Table: eval_records

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

-- Table: eval_submissions

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

-- Table: eval_suboperations

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

-- Table: eval_view_charts

CREATE TABLE eval_view_charts (
    experiment_id text NOT NULL,
    generation bigint NOT NULL,
    suite_id text NOT NULL,
    metric text NOT NULL,
    document bytea NOT NULL
);

-- Table: eval_view_pairs

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
    complete_b boolean NOT NULL,
    member_ordinal_a integer NOT NULL,
    member_ordinal_b integer NOT NULL,
    variant_a text NOT NULL,
    variant_b text NOT NULL,
    eligibility_a text NOT NULL,
    eligibility_b text NOT NULL,
    execution_state_a text,
    execution_state_b text,
    assessment_a text NOT NULL,
    assessment_b text NOT NULL,
    conflicting_a boolean NOT NULL,
    conflicting_b boolean NOT NULL
);

-- Table: finding_correlation_fences

CREATE TABLE finding_correlation_fences (
    project_id text NOT NULL,
    revision bigint DEFAULT 1 NOT NULL,
    CONSTRAINT finding_correlation_fences_revision_check CHECK (((revision >= 1) AND (revision <= '9007199254740991'::bigint)))
);

-- Table: finding_proposal_audit_holds

CREATE TABLE finding_proposal_audit_holds (
    receipt_id text NOT NULL,
    audit_id text NOT NULL,
    project_id text NOT NULL,
    proposal_ref jsonb NOT NULL,
    evidence jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    receipt_created_at timestamp with time zone NOT NULL,
    CONSTRAINT finding_proposal_audit_holds_evidence_check CHECK (((jsonb_typeof(evidence) = 'array'::text) AND (octet_length((evidence)::text) <= 1048576))),
    CONSTRAINT finding_proposal_audit_holds_proposal_ref_check CHECK (((jsonb_typeof(proposal_ref) = 'object'::text) AND (octet_length((proposal_ref)::text) <= 4096)))
);

-- Table: finding_proposal_receipts

CREATE TABLE finding_proposal_receipts (
    receipt_id text NOT NULL,
    proposal_id text NOT NULL,
    allocation_id text,
    runtime_agent_id text,
    runtime_instance_id text,
    stage_execution_id text NOT NULL,
    logical_agent_name text,
    invocation_id text NOT NULL,
    submission_id text NOT NULL,
    client_key text NOT NULL,
    request_digest text NOT NULL,
    run_id text NOT NULL,
    owner_id text NOT NULL,
    project_id text,
    audit_execution_id text,
    audit_id text,
    audit_task_id text,
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
    report_projection text NOT NULL,
    fingerprint_projection text DEFAULT '{"schema":"contractor.finding.fingerprint-projection.v1","availability":"unavailable","reason":"not_recorded"}'::text NOT NULL,
    anchor_projection text DEFAULT '{"schema":"contractor.finding.anchors-projection.v1","availability":"unavailable","reason":"not_recorded"}'::text NOT NULL,
    comparison_projection jsonb DEFAULT '{"reason": "not_recorded", "schema": "contractor.finding.comparison-projection.v1", "availability": "unavailable"}'::jsonb NOT NULL,
    checks_projection json NOT NULL,
    checks_projection_bytes integer GENERATED ALWAYS AS (octet_length((checks_projection)::text)) STORED,
    checks_projection_count integer GENERATED ALWAYS AS (json_array_length((checks_projection -> 'checks'::text))) STORED,
    publication_requested boolean DEFAULT false NOT NULL,
    publication_error text DEFAULT ''::text NOT NULL,
    project_proposal jsonb,
    project_evidence jsonb,
    audit_hypothesis jsonb,
    audit_hypothesis_snapshot jsonb,
    operation_ref text DEFAULT ''::text NOT NULL,
    scan_origin jsonb,
    CONSTRAINT finding_anchor_projection_shape CHECK ((((octet_length(anchor_projection) >= 2) AND (octet_length(anchor_projection) <= 32768) AND (jsonb_typeof((anchor_projection)::jsonb) = 'object'::text) AND (((anchor_projection)::jsonb ->> 'schema'::text) = 'contractor.finding.anchors-projection.v1'::text) AND (((anchor_projection)::jsonb ->> 'availability'::text) = ANY (ARRAY['available'::text, 'unavailable'::text]))) IS TRUE)),
    CONSTRAINT finding_checks_projection_discarded CHECK (((retention_state <> 'discarded'::text) OR COALESCE(((checks_projection IS NOT NULL) AND ((checks_projection ->> 'discarded'::text) = 'true'::text) AND (json_array_length((checks_projection -> 'checks'::text)) = 0) AND ((checks_projection ->> 'subject'::text) IS NULL) AND ((checks_projection ->> 'limitations'::text) IS NULL)), false))),
    CONSTRAINT finding_checks_projection_shape CHECK (((checks_projection IS NULL) OR COALESCE(((octet_length((checks_projection)::text) <= 262144) AND (json_typeof(checks_projection) = 'object'::text) AND ((checks_projection ->> 'schema'::text) = 'contractor.audit.finding-checks-projection.v1'::text) AND ((checks_projection ->> 'proposal_digest'::text) = proposal_digest) AND (((checks_projection ->> 'size_bytes'::text))::bigint = proposal_size_bytes) AND (json_typeof((checks_projection -> 'checks'::text)) = 'array'::text) AND (json_array_length((checks_projection -> 'checks'::text)) <= 512) AND (json_typeof((checks_projection -> 'discarded'::text)) = 'boolean'::text) AND (((checks_projection ->> 'discarded'::text))::boolean = (retention_state = 'discarded'::text))), false))),
    CONSTRAINT finding_comparison_projection_shape CHECK (((jsonb_typeof(comparison_projection) = 'object'::text) AND (octet_length((comparison_projection)::text) <= 32768) AND ((comparison_projection ->> 'schema'::text) = 'contractor.finding.comparison-projection.v1'::text) AND ((comparison_projection ->> 'availability'::text) = ANY (ARRAY['available'::text, 'unavailable'::text])))),
    CONSTRAINT finding_fact_fingerprint_shape CHECK (((((fingerprint_projection)::jsonb ->> 'algorithm'::text) IS DISTINCT FROM 'contractor/fact-evidence/v1'::text) OR COALESCE(((((fingerprint_projection)::jsonb ->> 'availability'::text) = 'available'::text) AND ((fingerprint_projection)::jsonb ?& ARRAY['schema'::text, 'availability'::text, 'algorithm'::text, 'value'::text]) AND (((fingerprint_projection)::jsonb - ARRAY['schema'::text, 'availability'::text, 'algorithm'::text, 'value'::text]) = '{}'::jsonb) AND (jsonb_typeof(((fingerprint_projection)::jsonb -> 'value'::text)) = 'string'::text) AND (((fingerprint_projection)::jsonb ->> 'value'::text) ~ '^[0-9a-f]{64}$'::text) AND (((report_projection)::jsonb ->> 'severity_suggestion'::text) = 'informational'::text) AND ((jsonb_array_length(evidence) >= 1) AND (jsonb_array_length(evidence) <= 4))), false))),
    CONSTRAINT finding_fingerprint_projection_shape CHECK ((((octet_length(fingerprint_projection) >= 2) AND (octet_length(fingerprint_projection) <= 16384) AND (jsonb_typeof((fingerprint_projection)::jsonb) = 'object'::text) AND (((fingerprint_projection)::jsonb ->> 'schema'::text) = 'contractor.finding.fingerprint-projection.v1'::text) AND (((fingerprint_projection)::jsonb ->> 'availability'::text) = ANY (ARRAY['available'::text, 'unavailable'::text]))) IS TRUE)),
    CONSTRAINT finding_project_material_shape CHECK ((((project_proposal IS NULL) = (project_evidence IS NULL)) AND ((project_proposal IS NULL) OR ((project_id IS NOT NULL) AND (jsonb_typeof(project_proposal) = 'object'::text) AND (octet_length((project_proposal)::text) <= 4096) AND (jsonb_typeof(project_evidence) = 'array'::text) AND (octet_length((project_evidence)::text) <= 1048576))))),
    CONSTRAINT finding_proposal_receipts_allocation_id_check CHECK (((btrim(allocation_id) <> ''::text) AND (octet_length(allocation_id) <= 256))),
    CONSTRAINT finding_proposal_receipts_audit_shape CHECK ((((audit_execution_id IS NULL) AND (audit_id IS NULL) AND (audit_task_id IS NULL)) OR ((audit_execution_id IS NOT NULL) AND (audit_id IS NOT NULL)))),
    CONSTRAINT finding_proposal_receipts_client_key_check CHECK ((client_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$'::text)),
    CONSTRAINT finding_proposal_receipts_discarded_report_check CHECK (((retention_state <> 'discarded'::text) OR (report_projection = '{}'::text))),
    CONSTRAINT finding_proposal_receipts_evidence_check CHECK (((jsonb_typeof(evidence) = 'array'::text) AND (octet_length((evidence)::text) <= 1048576))),
    CONSTRAINT finding_proposal_receipts_hypothesis_check CHECK (((audit_hypothesis IS NULL) OR ((audit_id IS NOT NULL) AND (jsonb_typeof(audit_hypothesis) = 'object'::text) AND (audit_hypothesis ?& ARRAY['id'::text, 'revision'::text]) AND ((audit_hypothesis - ARRAY['id'::text, 'revision'::text]) = '{}'::jsonb) AND (jsonb_typeof((audit_hypothesis -> 'id'::text)) = 'string'::text) AND ((audit_hypothesis ->> 'id'::text) ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text) AND (jsonb_typeof((audit_hypothesis -> 'revision'::text)) = 'number'::text) AND ((audit_hypothesis ->> 'revision'::text) ~ '^[1-9][0-9]*$'::text) AND (((audit_hypothesis ->> 'revision'::text))::numeric <= ('9007199254740991'::bigint)::numeric)))),
    CONSTRAINT finding_proposal_receipts_hypothesis_snapshot_check CHECK (((audit_hypothesis_snapshot IS NULL) OR ((audit_hypothesis IS NOT NULL) AND (project_proposal IS NOT NULL) AND (jsonb_typeof(audit_hypothesis_snapshot) = 'object'::text) AND (octet_length((audit_hypothesis_snapshot)::text) <= 4096)))),
    CONSTRAINT finding_proposal_receipts_hypothesis_task CHECK (((audit_hypothesis IS NULL) OR (audit_task_id IS NOT NULL))),
    CONSTRAINT finding_proposal_receipts_invocation_id_check CHECK (((btrim(invocation_id) <> ''::text) AND (octet_length(invocation_id) <= 128))),
    CONSTRAINT finding_proposal_receipts_logical_agent_name_check CHECK (((btrim(logical_agent_name) <> ''::text) AND (octet_length(logical_agent_name) <= 128))),
    CONSTRAINT finding_proposal_receipts_owner_id_check CHECK (((btrim(owner_id) <> ''::text) AND (octet_length(owner_id) <= 256))),
    CONSTRAINT finding_proposal_receipts_proposal_digest_check CHECK ((proposal_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT finding_proposal_receipts_proposal_id_check CHECK ((proposal_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT finding_proposal_receipts_proposal_media_type_check CHECK ((proposal_media_type = 'application/json'::text)),
    CONSTRAINT finding_proposal_receipts_proposal_ref_check CHECK (((jsonb_typeof(proposal_ref) = 'object'::text) AND (octet_length((proposal_ref)::text) <= 4096))),
    CONSTRAINT finding_proposal_receipts_proposal_size_bytes_check CHECK (((proposal_size_bytes >= 0) AND (proposal_size_bytes <= 8388608))),
    CONSTRAINT finding_proposal_receipts_receipt_id_check CHECK ((receipt_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT finding_proposal_receipts_report_preconditions_check CHECK (((retention_state = 'discarded'::text) OR (NOT (jsonb_typeof(((report_projection)::jsonb -> 'preconditions'::text)) IS DISTINCT FROM 'array'::text)))),
    CONSTRAINT finding_proposal_receipts_report_projection_check CHECK (((octet_length(report_projection) >= 2) AND (octet_length(report_projection) <= 8388608) AND (jsonb_typeof((report_projection)::jsonb) = 'object'::text))),
    CONSTRAINT finding_proposal_receipts_request_digest_check CHECK ((request_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT finding_proposal_receipts_retention_shape CHECK ((((retention_state = 'source-held'::text) AND (project_proposal IS NULL) AND (source_run_deleted_at IS NULL) AND (discarded_at IS NULL)) OR ((retention_state = 'project-held'::text) AND (project_proposal IS NOT NULL) AND (discarded_at IS NULL)) OR ((retention_state = 'discarded'::text) AND (project_proposal IS NULL) AND (source_run_deleted_at IS NOT NULL) AND (discarded_at IS NOT NULL)))),
    CONSTRAINT finding_proposal_receipts_retention_state_check CHECK ((retention_state = ANY (ARRAY['source-held'::text, 'project-held'::text, 'discarded'::text]))),
    CONSTRAINT finding_proposal_receipts_run_id_check CHECK (((btrim(run_id) <> ''::text) AND (octet_length(run_id) <= 256))),
    CONSTRAINT finding_proposal_receipts_runtime_agent_id_check CHECK (((btrim(runtime_agent_id) <> ''::text) AND (octet_length(runtime_agent_id) <= 256))),
    CONSTRAINT finding_proposal_receipts_runtime_instance_id_check CHECK (((btrim(runtime_instance_id) <> ''::text) AND (octet_length(runtime_instance_id) <= 256))),
    CONSTRAINT finding_proposal_receipts_stage_execution_id_check CHECK (((btrim(stage_execution_id) <> ''::text) AND (octet_length(stage_execution_id) <= 256))),
    CONSTRAINT finding_proposal_receipts_submission_id_check CHECK (((btrim(submission_id) <> ''::text) AND (octet_length(submission_id) <= 128))),
    CONSTRAINT finding_proposal_receipts_task_id_check CHECK (((audit_task_id IS NULL) OR (audit_task_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text))),
    CONSTRAINT finding_proposal_receipts_workflow_closure_digest_check CHECK ((workflow_closure_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT finding_proposal_receipts_workflow_configuration_ref_check CHECK (((jsonb_typeof(workflow_configuration_ref) = 'object'::text) AND (octet_length((workflow_configuration_ref)::text) <= 4096))),
    CONSTRAINT finding_proposal_receipts_workflow_name_check CHECK (((btrim(workflow_name) <> ''::text) AND (octet_length(workflow_name) <= 128))),
    CONSTRAINT finding_proposal_receipts_workflow_schema_version_check CHECK (((btrim(workflow_schema_version) <> ''::text) AND (octet_length(workflow_schema_version) <= 128))),
    CONSTRAINT finding_proposal_receipts_workflow_version_check CHECK (((btrim(workflow_version) <> ''::text) AND (octet_length(workflow_version) <= 128))),
    CONSTRAINT finding_publication_error_shape CHECK (((publication_error = ''::text) OR (publication_requested AND (publication_error = ANY (ARRAY['problem_evidence_required'::text, 'problem_project_required'::text]))))),
    CONSTRAINT finding_receipt_execution_lane CHECK ((((operation_ref = ''::text) AND (scan_origin IS NULL) AND (allocation_id IS NOT NULL) AND (runtime_agent_id IS NOT NULL) AND (runtime_instance_id IS NOT NULL) AND (logical_agent_name IS NOT NULL)) OR ((operation_ref = 'publish-findings@1'::text) AND (scan_origin IS NOT NULL) AND (jsonb_typeof(scan_origin) = 'object'::text) AND (octet_length((scan_origin)::text) <= 2048) AND (allocation_id IS NULL) AND (runtime_agent_id IS NULL) AND (runtime_instance_id IS NULL) AND (logical_agent_name IS NULL))))
);

-- Table: git_ssh_keys

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

-- Table: llm_credentials

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

-- Table: model_selection_settings

CREATE TABLE model_selection_settings (
    singleton boolean DEFAULT true NOT NULL,
    default_model text NOT NULL,
    role_models jsonb NOT NULL,
    revision numeric(20,0) DEFAULT 1 NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT model_selection_settings_revision_check CHECK (((revision >= (1)::numeric) AND (revision <= '18446744073709551615'::numeric))),
    CONSTRAINT model_selection_settings_singleton_check CHECK (singleton),
    CONSTRAINT model_selection_settings_value CHECK (contractor_valid_model_selection(default_model, role_models))
);

-- Table: mutation_receipts

CREATE TABLE mutation_receipts (
    key_scope text[] NOT NULL,
    idempotency_key text NOT NULL,
    operation text NOT NULL,
    request_digest text NOT NULL,
    resource_id text NOT NULL,
    response bytea NOT NULL,
    owner_id text,
    project_id text,
    audit_id text,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT mutation_receipts_family_check CHECK (
CASE key_scope[1]
    WHEN 'audit'::text THEN ((owner_id IS NOT NULL) AND (audit_id IS NOT NULL) AND (project_id IS NULL) AND (key_scope = ARRAY['audit'::text, owner_id, operation]) AND (idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'::text))
    WHEN 'eval'::text THEN ((owner_id IS NOT NULL) AND (project_id IS NOT NULL) AND (audit_id IS NULL) AND (key_scope = ARRAY['eval'::text, owner_id, project_id, resource_id, operation]))
    WHEN 'runtime-management'::text THEN ((owner_id IS NULL) AND (project_id IS NULL) AND (audit_id IS NULL) AND (key_scope = ARRAY['runtime-management'::text]) AND (idempotency_key ~ '^sha256:[0-9a-f]{64}$'::text))
    WHEN 'project'::text THEN ((owner_id IS NOT NULL) AND (project_id IS NOT NULL) AND (audit_id IS NULL) AND (key_scope = ARRAY['project'::text, owner_id, project_id, resource_id, operation]) AND (idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'::text))
    ELSE false
END),
    CONSTRAINT mutation_receipts_idempotency_key_check CHECK (((octet_length(idempotency_key) >= 1) AND (octet_length(idempotency_key) <= 128))),
    CONSTRAINT mutation_receipts_operation_check CHECK ((operation ~ '^[a-z][a-z0-9_.-]{0,127}$'::text)),
    CONSTRAINT mutation_receipts_owner_id_check CHECK (((owner_id IS NULL) OR ((btrim(owner_id) <> ''::text) AND (octet_length(owner_id) <= 256)))),
    CONSTRAINT mutation_receipts_request_digest_check CHECK ((request_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT mutation_receipts_resource_id_check CHECK (((btrim(resource_id) <> ''::text) AND (octet_length(resource_id) <= 1024))),
    CONSTRAINT mutation_receipts_response_check CHECK (((octet_length(response) >= 2) AND (octet_length(response) <= 1048576)))
);

-- Table: owner_queue_controls

CREATE TABLE owner_queue_controls (
    owner_id text NOT NULL,
    paused boolean DEFAULT false NOT NULL,
    revision bigint DEFAULT 1 NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT owner_queue_controls_owner_id_check CHECK (((btrim(owner_id) <> ''::text) AND (octet_length(owner_id) <= 256))),
    CONSTRAINT owner_queue_controls_revision_check CHECK ((revision > 0))
);

-- Table: performance_minutes

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

-- Table: planner_execution_reports

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

-- Table: planner_sessions

CREATE TABLE planner_sessions (
    session_id text NOT NULL,
    stage_execution_id text NOT NULL,
    invocation_id text NOT NULL,
    state_schema_version text NOT NULL,
    state jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    next_event_sequence bigint DEFAULT 1 NOT NULL,
    artifact_refs jsonb GENERATED ALWAYS AS (contractor_planner_session_artifact_refs(state)) STORED,
    CONSTRAINT planner_session_next_event_sequence_positive CHECK ((next_event_sequence > 0)),
    CONSTRAINT planner_session_state_size CHECK ((octet_length((state)::text) <= 2097152)),
    CONSTRAINT planner_sessions_invocation_id_check CHECK ((btrim(invocation_id) <> ''::text)),
    CONSTRAINT planner_sessions_session_id_check CHECK ((btrim(session_id) <> ''::text)),
    CONSTRAINT planner_sessions_state_check CHECK ((jsonb_typeof(state) = 'object'::text)),
    CONSTRAINT planner_sessions_state_schema_version_check CHECK ((btrim(state_schema_version) <> ''::text))
);

-- Table: project_finding_audits

CREATE TABLE project_finding_audits (
    audit_id text NOT NULL,
    finding_id text NOT NULL,
    project_id text NOT NULL,
    finding_created_at timestamp with time zone NOT NULL,
    correlation_group_key text NOT NULL
);

-- Table: project_finding_sources

CREATE TABLE project_finding_sources (
    receipt_id text NOT NULL,
    project_id text NOT NULL,
    finding_id text NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL
);

-- Table: project_findings

CREATE TABLE project_findings (
    owner_id text NOT NULL,
    finding_id text NOT NULL,
    project_id text NOT NULL,
    first_receipt_id text NOT NULL,
    first_proposal_ref jsonb NOT NULL,
    title text NOT NULL,
    subject_kind text NOT NULL,
    subject_key text NOT NULL,
    subject_truncated boolean NOT NULL,
    severity_suggestion text NOT NULL,
    correlation_algorithm text NOT NULL,
    correlation_value text NOT NULL,
    correlation_group_key text NOT NULL,
    correlation_reason text NOT NULL,
    state text,
    analyst_verdict text,
    analyst_severity text,
    rationale text,
    actor_id text,
    assessed_at timestamp with time zone,
    duplicate_target_id text,
    revision bigint DEFAULT 1 NOT NULL,
    created_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT project_finding_assessment_shape CHECK ((((state IS NULL) AND (analyst_verdict IS NULL) AND (analyst_severity IS NULL) AND (rationale IS NULL) AND (actor_id IS NULL) AND (assessed_at IS NULL) AND (duplicate_target_id IS NULL)) OR ((state IS NOT NULL) AND (analyst_verdict IS NOT NULL) AND (rationale IS NOT NULL) AND (actor_id IS NOT NULL) AND (assessed_at IS NOT NULL) AND (state =
CASE analyst_verdict
    WHEN 'true_positive'::text THEN 'confirmed'::text
    WHEN 'false_positive'::text THEN 'rejected'::text
    WHEN 'duplicate'::text THEN 'duplicate'::text
    WHEN 'reopen'::text THEN 'proposed'::text
    WHEN 'needs_evidence'::text THEN 'needs-evidence'::text
    ELSE NULL::text
END) AND ((analyst_verdict = 'true_positive'::text) = (analyst_severity IS NOT NULL)) AND ((analyst_verdict = 'duplicate'::text) = (duplicate_target_id IS NOT NULL))))),
    CONSTRAINT project_finding_primary_shape CHECK ((((correlation_algorithm = ''::text) AND (correlation_value = ''::text) AND (correlation_group_key = ''::text) AND (correlation_reason = ANY (ARRAY['not_recorded'::text, 'no_qualified_key'::text, 'primary_key_ambiguous'::text]))) OR ((correlation_algorithm = ANY (ARRAY['contractor/fact-evidence/v1'::text, 'contractor/producer-issue/v1'::text, 'contractor/snapshot-issue-anchor/v1'::text])) AND (correlation_value ~ '^[0-9a-f]{64}$'::text) AND (correlation_reason = ''::text) AND (correlation_group_key = contractor_finding_group_key(project_id, correlation_algorithm, correlation_value))))),
    CONSTRAINT project_findings_actor_id_check CHECK (((octet_length(actor_id) >= 1) AND (octet_length(actor_id) <= 256) AND (btrim(actor_id) <> ''::text))),
    CONSTRAINT project_findings_analyst_severity_check CHECK ((analyst_severity = ANY (ARRAY['informational'::text, 'low'::text, 'medium'::text, 'high'::text, 'critical'::text]))),
    CONSTRAINT project_findings_analyst_verdict_check CHECK ((analyst_verdict = ANY (ARRAY['true_positive'::text, 'false_positive'::text, 'duplicate'::text, 'reopen'::text, 'needs_evidence'::text]))),
    CONSTRAINT project_findings_check CHECK (((duplicate_target_id IS NULL) OR (duplicate_target_id <> finding_id))),
    CONSTRAINT project_findings_finding_id_check CHECK ((finding_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT project_findings_first_proposal_ref_check CHECK (((jsonb_typeof(first_proposal_ref) = 'object'::text) AND (octet_length((first_proposal_ref)::text) <= 4096))),
    CONSTRAINT project_findings_owner_id_check CHECK (((octet_length(owner_id) >= 1) AND (octet_length(owner_id) <= 256) AND (btrim(owner_id) <> ''::text))),
    CONSTRAINT project_findings_rationale_check CHECK (((octet_length(rationale) >= 1) AND (octet_length(rationale) <= 65536) AND (btrim(rationale) <> ''::text))),
    CONSTRAINT project_findings_revision_check CHECK (((revision >= 1) AND (revision <= '9007199254740991'::bigint))),
    CONSTRAINT project_findings_severity_suggestion_check CHECK ((severity_suggestion = ANY (ARRAY[''::text, 'informational'::text, 'low'::text, 'medium'::text, 'high'::text, 'critical'::text]))),
    CONSTRAINT project_findings_state_check CHECK ((state = ANY (ARRAY['proposed'::text, 'confirmed'::text, 'rejected'::text, 'duplicate'::text, 'needs-evidence'::text]))),
    CONSTRAINT project_findings_subject_key_check CHECK ((octet_length(subject_key) <= 1024)),
    CONSTRAINT project_findings_subject_kind_check CHECK ((octet_length(subject_kind) <= 256)),
    CONSTRAINT project_findings_title_check CHECK (((octet_length(title) >= 1) AND (octet_length(title) <= 65536) AND (btrim(title) <> ''::text)))
);

-- Table: projects

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
    active_checks text DEFAULT 'automatic'::text NOT NULL,
    llm_cost jsonb DEFAULT '{}'::jsonb,
    CONSTRAINT projects_active_checks_policy CHECK ((active_checks = ANY (ARRAY['automatic'::text, 'prohibited'::text]))),
    CONSTRAINT projects_deletion_phase_check CHECK ((deletion_phase = ANY (ARRAY['cancelling'::text, 'draining'::text, 'purging_runs'::text, 'purging_artifacts'::text]))),
    CONSTRAINT projects_deletion_shape CHECK ((((lifecycle_state = 'active'::text) AND (deletion_phase IS NULL) AND (deletion_requested_at IS NULL) AND (deletion_claim_id IS NULL) AND (deletion_claimed_at IS NULL) AND (deletion_claim_expires_at IS NULL)) OR ((lifecycle_state = 'deleting'::text) AND (deletion_phase IS NOT NULL) AND (deletion_requested_at IS NOT NULL) AND (((deletion_claim_id IS NULL) AND (deletion_claimed_at IS NULL) AND (deletion_claim_expires_at IS NULL)) OR ((deletion_claim_id IS NOT NULL) AND (deletion_claimed_at IS NOT NULL) AND (deletion_claim_expires_at > deletion_claimed_at)))))),
    CONSTRAINT projects_description_check CHECK ((octet_length(description) <= 4096)),
    CONSTRAINT projects_http_target_shape CHECK ((((http_target_url IS NULL) AND (http_target_credential_id IS NULL) AND (http_target_credential_kind IS NULL)) OR ((http_target_url IS NOT NULL) AND (btrim(http_target_url) = http_target_url) AND ((octet_length(http_target_url) >= 1) AND (octet_length(http_target_url) <= 2048)) AND (((http_target_credential_id IS NULL) AND (http_target_credential_kind IS NULL)) OR ((http_target_credential_id IS NOT NULL) AND (http_target_credential_kind = ANY (ARRAY['http-origin-basic@1'::text, 'http-origin-bearer@1'::text]))))))),
    CONSTRAINT projects_kind_check CHECK ((kind = ANY (ARRAY['project'::text, 'evaluation'::text]))),
    CONSTRAINT projects_lifecycle_state_check CHECK ((lifecycle_state = ANY (ARRAY['active'::text, 'deleting'::text]))),
    CONSTRAINT projects_llm_cost_check CHECK (((llm_cost IS NULL) OR (jsonb_typeof(llm_cost) = 'object'::text))),
    CONSTRAINT projects_name_check CHECK (((btrim(name) <> ''::text) AND (octet_length(name) <= 160))),
    CONSTRAINT projects_owner_id_check CHECK (((btrim(owner_id) <> ''::text) AND (octet_length(owner_id) <= 256))),
    CONSTRAINT projects_project_id_check CHECK ((project_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'::text)),
    CONSTRAINT projects_request_digest_check CHECK ((request_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT projects_request_idempotency_key_check CHECK ((request_idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'::text)),
    CONSTRAINT projects_revision_check CHECK ((revision > 0))
);

-- Table: runtime_agent_principals

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

-- Table: runtime_config_versions

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

-- Table: runtime_credentials

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
    CONSTRAINT runtime_credentials_credential_kind_check CHECK ((credential_kind = ANY (ARRAY['otlp-headers@1'::text, 'http-proxy-basic@1'::text, 'http-proxy-bearer@1'::text, 'caido-bearer@1'::text, 'interactsh-token@1'::text, 'http-origin-basic@1'::text, 'http-origin-bearer@1'::text, 'target-basic@1'::text, 'target-bearer@1'::text, 'target-cookies@1'::text, 'target-api-keys@1'::text]))),
    CONSTRAINT runtime_credentials_deleted_by_check CHECK (((btrim(deleted_by) <> ''::text) AND (octet_length(deleted_by) <= 256))),
    CONSTRAINT runtime_credentials_deletion_marker CHECK (((deleted_by IS NULL) = (deleted_at IS NULL))),
    CONSTRAINT runtime_credentials_encryption_schema_version_check CHECK ((encryption_schema_version = 'contractor.runtime-credentials/v1'::text)),
    CONSTRAINT runtime_credentials_idempotency_key_digest_check CHECK ((idempotency_key_digest ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT runtime_credentials_key_id_check CHECK ((key_id ~ '^sha256:[0-9a-f]{64}$'::text)),
    CONSTRAINT runtime_credentials_nonce_check CHECK ((octet_length(nonce) = 12)),
    CONSTRAINT runtime_credentials_request_mac_check CHECK ((octet_length(request_mac) = 32))
);

-- Table: runtime_label_bindings

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

-- Table: scheduler_settings

CREATE TABLE scheduler_settings (
    singleton boolean DEFAULT true NOT NULL,
    max_concurrent_runs integer NOT NULL,
    revision numeric(20,0) NOT NULL,
    updated_at timestamp with time zone DEFAULT clock_timestamp() NOT NULL,
    max_concurrent_tool_runs integer NOT NULL,
    CONSTRAINT scheduler_settings_max_concurrent_runs_check CHECK (((max_concurrent_runs >= 1) AND (max_concurrent_runs <= 32))),
    CONSTRAINT scheduler_settings_max_concurrent_tool_runs_check CHECK (((max_concurrent_tool_runs >= 1) AND (max_concurrent_tool_runs <= 32))),
    CONSTRAINT scheduler_settings_revision_check CHECK (((revision >= (1)::numeric) AND (revision <= '18446744073709551615'::numeric))),
    CONSTRAINT scheduler_settings_singleton_check CHECK (singleton)
);

-- Table: stage_allocations

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
    execution_report_received boolean DEFAULT false NOT NULL,
    runtime_placement_labels text[] DEFAULT '{}'::text[] NOT NULL,
    llm_cost jsonb,
    CONSTRAINT stage_allocation_performance_collection_policy CHECK (((performance_collection_policy IS NULL) OR (performance_collection_policy = ANY (ARRAY['requested'::text, 'disabled'::text, 'unsupported'::text])))),
    CONSTRAINT stage_allocation_release_timestamps CHECK (((release_completed_at IS NULL) OR ((release_attempted_at IS NOT NULL) AND (release_completed_at >= release_attempted_at)))),
    CONSTRAINT stage_allocation_runtime_configuration_shape CHECK ((((runtime_agent_id IS NULL) AND (runtime_agent_label_revision IS NULL) AND (runtime_configuration_schema_version IS NULL) AND (runtime_configuration IS NULL)) OR ((runtime_agent_id ~ '^[0-9a-f]{64}$'::text) AND (runtime_agent_label_revision >= (1)::numeric) AND (runtime_agent_label_revision <= '18446744073709551615'::numeric) AND (runtime_configuration_schema_version = 'contractor.runtime-config-provenance/v2'::text) AND contractor_valid_allocation_runtime_configuration(runtime_configuration)))),
    CONSTRAINT stage_allocations_agent_template_ref_check CHECK ((jsonb_typeof(agent_template_ref) = 'object'::text)),
    CONSTRAINT stage_allocations_allocation_id_check CHECK ((btrim(allocation_id) <> ''::text)),
    CONSTRAINT stage_allocations_completion_contract_check CHECK (((completion_contract IS NULL) OR ((jsonb_typeof(completion_contract) = 'object'::text) AND ((completion_contract ->> 'kind'::text) = 'audit-check-results@1'::text)))),
    CONSTRAINT stage_allocations_llm_cost_check CHECK (((llm_cost IS NULL) OR (jsonb_typeof(llm_cost) = 'object'::text))),
    CONSTRAINT stage_allocations_logical_agent_name_check CHECK ((btrim(logical_agent_name) <> ''::text)),
    CONSTRAINT stage_allocations_namespace_check CHECK (((btrim(namespace) <> ''::text) AND (POSITION(('/'::text) IN (namespace)) = 0))),
    CONSTRAINT stage_allocations_placement_labels_valid CHECK (contractor_valid_placement_labels(runtime_placement_labels)),
    CONSTRAINT stage_allocations_runtime_agent_instance_id_check CHECK ((btrim(runtime_agent_instance_id) <> ''::text)),
    CONSTRAINT stage_allocations_worker_runtime_ref_check CHECK ((jsonb_typeof(worker_runtime_ref) = 'object'::text))
);

-- Table: stage_executions

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
    output_publication_failures integer DEFAULT 0 NOT NULL,
    transition_action text,
    transition_target_stage_name text,
    transition_target_execution_id text,
    transition_escalation_ordinal integer,
    transition_escalation_exhausted boolean DEFAULT false NOT NULL,
    transition_decided_at timestamp with time zone,
    resume_requested_by text,
    resume_requested_at timestamp with time zone,
    resume_transaction_id xid8,
    planner_llm_cost jsonb,
    CONSTRAINT stage_accepted_shape CHECK ((((accepted_result_schema_version IS NULL) = (accepted_stage_result IS NULL)) AND ((accepted_stage_result IS NULL) OR ((btrim(accepted_result_schema_version) <> ''::text) AND contractor_valid_stage_result(accepted_stage_result))))),
    CONSTRAINT stage_attempt_lineage_shape CHECK ((((attempt = 1) AND (previous_execution_id IS NULL)) OR ((attempt > 1) AND (previous_execution_id IS NOT NULL)))),
    CONSTRAINT stage_candidate_shape CHECK ((((candidate_result_schema_version IS NULL) = (candidate_stage_result IS NULL)) AND ((candidate_stage_result IS NULL) OR ((btrim(candidate_result_schema_version) <> ''::text) AND contractor_valid_stage_result(candidate_stage_result))))),
    CONSTRAINT stage_execution_config_variant_shape CHECK ((((execution_config_variant = 'base'::text) AND (escalation_ordinal IS NULL) AND (NOT (stage_context_snapshot ? 'continuation'::text))) OR ((execution_config_variant = ANY (ARRAY['failed_escalation'::text, 'interrupted_escalation'::text])) AND (escalation_ordinal IS NOT NULL) AND (escalation_ordinal > 0) AND (NOT (stage_context_snapshot ? 'continuation'::text))) OR ((execution_config_variant = 'summary_escalation'::text) AND (escalation_ordinal IS NOT NULL) AND (escalation_ordinal = 1) AND (stage_context_snapshot ? 'continuation'::text) AND (jsonb_typeof((stage_context_snapshot -> 'continuation'::text)) = 'object'::text) AND ((stage_context_snapshot -> 'continuation'::text) ? 'sourceStageExecutionId'::text) AND (jsonb_typeof(((stage_context_snapshot -> 'continuation'::text) -> 'sourceStageExecutionId'::text)) = 'string'::text)))),
    CONSTRAINT stage_execution_resume_previous CHECK (((resume_source_execution_id IS NULL) OR ((previous_execution_id IS NOT NULL) AND (resume_source_execution_id = previous_execution_id)))),
    CONSTRAINT stage_execution_resume_receipt_shape CHECK ((((resume_source_execution_id IS NULL) AND (resume_requested_by IS NULL) AND (resume_requested_at IS NULL) AND (resume_transaction_id IS NULL)) OR ((resume_source_execution_id IS NOT NULL) AND (resume_requested_by IS NOT NULL) AND (resume_requested_at IS NOT NULL) AND (resume_transaction_id IS NOT NULL)))),
    CONSTRAINT stage_execution_state_shape CHECK ((((state = 'preparing'::text) AND (planner_session_id IS NULL) AND (candidate_stage_result IS NULL) AND (accepted_stage_result IS NULL) AND (stage_termination IS NULL) AND (finalization_id IS NULL) AND (finalization_deadline IS NULL) AND (abort_id IS NULL) AND (abort_deadline IS NULL) AND (terminal_at IS NULL) AND (NOT (stage_spec_snapshot ? 'operation'::text))) OR ((state = 'running'::text) AND (planner_session_id IS NOT NULL) AND (candidate_stage_result IS NULL) AND (accepted_stage_result IS NULL) AND (stage_termination IS NULL) AND (finalization_id IS NULL) AND (finalization_deadline IS NULL) AND (abort_id IS NULL) AND (abort_deadline IS NULL) AND (planner_started_at IS NOT NULL) AND (terminal_at IS NULL) AND (NOT (stage_spec_snapshot ? 'operation'::text))) OR ((state = 'finalizing'::text) AND (planner_session_id IS NOT NULL) AND (candidate_stage_result IS NOT NULL) AND (accepted_stage_result IS NULL) AND (stage_termination IS NULL) AND (finalization_id IS NOT NULL) AND (btrim(finalization_id) <> ''::text) AND (finalization_deadline IS NOT NULL) AND (abort_id IS NULL) AND (abort_deadline IS NULL) AND (planner_started_at IS NOT NULL) AND (terminal_at IS NULL) AND (NOT (stage_spec_snapshot ? 'operation'::text))) OR ((state = 'aborting'::text) AND (candidate_stage_result IS NULL) AND (accepted_stage_result IS NULL) AND (stage_termination IS NOT NULL) AND (finalization_id IS NULL) AND (finalization_deadline IS NULL) AND (abort_id IS NOT NULL) AND (btrim(abort_id) <> ''::text) AND (abort_deadline IS NOT NULL) AND (terminal_at IS NULL) AND ((((stage_termination ->> 'phase'::text) = 'preparing'::text) AND (planner_session_id IS NULL)) OR (((stage_termination ->> 'phase'::text) = 'running'::text) AND (planner_session_id IS NOT NULL) AND (planner_started_at IS NOT NULL))) AND (NOT (stage_spec_snapshot ? 'operation'::text))) OR ((state = ANY (ARRAY['succeeded'::text, 'failed'::text])) AND (planner_session_id IS NOT NULL) AND (candidate_stage_result IS NOT NULL) AND (accepted_stage_result = candidate_stage_result) AND (accepted_result_schema_version = candidate_result_schema_version) AND ((accepted_stage_result ->> 'outcome'::text) = state) AND (stage_termination IS NULL) AND (finalization_id IS NOT NULL) AND (btrim(finalization_id) <> ''::text) AND (finalization_deadline IS NOT NULL) AND (abort_id IS NULL) AND (abort_deadline IS NULL) AND (planner_started_at IS NOT NULL) AND (terminal_at IS NOT NULL) AND (NOT (stage_spec_snapshot ? 'operation'::text))) OR ((state = ANY (ARRAY['interrupted'::text, 'cancelled'::text])) AND (candidate_stage_result IS NULL) AND (accepted_stage_result IS NULL) AND (stage_termination IS NOT NULL) AND ((stage_termination ->> 'outcome'::text) = state) AND (finalization_id IS NULL) AND (finalization_deadline IS NULL) AND (abort_id IS NOT NULL) AND (btrim(abort_id) <> ''::text) AND (abort_deadline IS NOT NULL) AND (terminal_at IS NOT NULL) AND ((((stage_termination ->> 'phase'::text) = 'preparing'::text) AND (planner_session_id IS NULL)) OR (((stage_termination ->> 'phase'::text) = 'running'::text) AND (planner_session_id IS NOT NULL) AND (planner_started_at IS NOT NULL))) AND (NOT (stage_spec_snapshot ? 'operation'::text))) OR ((state = 'preparing'::text) AND (candidate_stage_result IS NULL) AND (accepted_stage_result IS NULL) AND (stage_termination IS NULL) AND (finalization_id IS NULL) AND (finalization_deadline IS NULL) AND (abort_id IS NULL) AND (abort_deadline IS NULL) AND (terminal_at IS NULL) AND (stage_spec_snapshot ? 'operation'::text) AND (jsonb_typeof((stage_spec_snapshot -> 'operation'::text)) = 'object'::text) AND (NOT (stage_spec_snapshot ? 'planner'::text)) AND (planner_session_id IS NULL) AND (planner_invocation_id IS NULL) AND (planner_started_at IS NULL)) OR ((state = 'running'::text) AND (admitted_at IS NOT NULL) AND (candidate_stage_result IS NULL) AND (accepted_stage_result IS NULL) AND (stage_termination IS NULL) AND (finalization_id IS NULL) AND (finalization_deadline IS NULL) AND (abort_id IS NULL) AND (abort_deadline IS NULL) AND (terminal_at IS NULL) AND (stage_spec_snapshot ? 'operation'::text) AND (jsonb_typeof((stage_spec_snapshot -> 'operation'::text)) = 'object'::text) AND (NOT (stage_spec_snapshot ? 'planner'::text)) AND (planner_session_id IS NULL) AND (planner_invocation_id IS NULL) AND (planner_started_at IS NULL)) OR ((state = 'finalizing'::text) AND (admitted_at IS NOT NULL) AND (candidate_stage_result IS NOT NULL) AND (accepted_stage_result IS NULL) AND (stage_termination IS NULL) AND (finalization_id IS NOT NULL) AND (btrim(finalization_id) <> ''::text) AND (finalization_deadline IS NOT NULL) AND (abort_id IS NULL) AND (abort_deadline IS NULL) AND (terminal_at IS NULL) AND (stage_spec_snapshot ? 'operation'::text) AND (jsonb_typeof((stage_spec_snapshot -> 'operation'::text)) = 'object'::text) AND (NOT (stage_spec_snapshot ? 'planner'::text)) AND (planner_session_id IS NULL) AND (planner_invocation_id IS NULL) AND (planner_started_at IS NULL)) OR ((state = ANY (ARRAY['succeeded'::text, 'failed'::text])) AND (admitted_at IS NOT NULL) AND (candidate_stage_result IS NOT NULL) AND (accepted_stage_result = candidate_stage_result) AND (accepted_result_schema_version = candidate_result_schema_version) AND ((accepted_stage_result ->> 'outcome'::text) = state) AND (stage_termination IS NULL) AND (finalization_id IS NOT NULL) AND (btrim(finalization_id) <> ''::text) AND (finalization_deadline IS NOT NULL) AND (abort_id IS NULL) AND (abort_deadline IS NULL) AND (terminal_at IS NOT NULL) AND (stage_spec_snapshot ? 'operation'::text) AND (jsonb_typeof((stage_spec_snapshot -> 'operation'::text)) = 'object'::text) AND (NOT (stage_spec_snapshot ? 'planner'::text)) AND (planner_session_id IS NULL) AND (planner_invocation_id IS NULL) AND (planner_started_at IS NULL)) OR ((state = 'aborting'::text) AND (candidate_stage_result IS NULL) AND (accepted_stage_result IS NULL) AND (stage_termination IS NOT NULL) AND (finalization_id IS NULL) AND (finalization_deadline IS NULL) AND (abort_id IS NOT NULL) AND (btrim(abort_id) <> ''::text) AND (abort_deadline IS NOT NULL) AND ((((stage_termination ->> 'phase'::text) = 'running'::text) AND (admitted_at IS NOT NULL)) OR ((stage_termination ->> 'phase'::text) = 'preparing'::text)) AND (terminal_at IS NULL) AND (stage_spec_snapshot ? 'operation'::text) AND (jsonb_typeof((stage_spec_snapshot -> 'operation'::text)) = 'object'::text) AND (NOT (stage_spec_snapshot ? 'planner'::text)) AND (planner_session_id IS NULL) AND (planner_invocation_id IS NULL) AND (planner_started_at IS NULL)) OR ((state = ANY (ARRAY['interrupted'::text, 'cancelled'::text])) AND (candidate_stage_result IS NULL) AND (accepted_stage_result IS NULL) AND (stage_termination IS NOT NULL) AND (finalization_id IS NULL) AND (finalization_deadline IS NULL) AND (abort_id IS NOT NULL) AND (btrim(abort_id) <> ''::text) AND (abort_deadline IS NOT NULL) AND ((((stage_termination ->> 'phase'::text) = 'running'::text) AND (admitted_at IS NOT NULL)) OR ((stage_termination ->> 'phase'::text) = 'preparing'::text)) AND (terminal_at IS NOT NULL) AND ((stage_termination ->> 'outcome'::text) = state) AND (stage_spec_snapshot ? 'operation'::text) AND (jsonb_typeof((stage_spec_snapshot -> 'operation'::text)) = 'object'::text) AND (NOT (stage_spec_snapshot ? 'planner'::text)) AND (planner_session_id IS NULL) AND (planner_invocation_id IS NULL) AND (planner_started_at IS NULL)))),
    CONSTRAINT stage_execution_transition_presence CHECK ((((transition_action IS NULL) AND (transition_target_stage_name IS NULL) AND (transition_target_execution_id IS NULL) AND (transition_decided_at IS NULL) AND (transition_escalation_ordinal IS NULL) AND (NOT transition_escalation_exhausted)) OR ((transition_action IS NOT NULL) AND (transition_decided_at IS NOT NULL)))),
    CONSTRAINT stage_executions_attempt_check CHECK ((attempt > 0)),
    CONSTRAINT stage_executions_output_publication_failures_check CHECK ((output_publication_failures >= 0)),
    CONSTRAINT stage_executions_planner_llm_cost_check CHECK (((planner_llm_cost IS NULL) OR (jsonb_typeof(planner_llm_cost) = 'object'::text))),
    CONSTRAINT stage_executions_stage_context_schema_version_check CHECK ((btrim(stage_context_schema_version) <> ''::text)),
    CONSTRAINT stage_executions_stage_context_snapshot_check CHECK ((jsonb_typeof(stage_context_snapshot) = 'object'::text)),
    CONSTRAINT stage_executions_stage_execution_id_check CHECK ((btrim(stage_execution_id) <> ''::text)),
    CONSTRAINT stage_executions_stage_name_check CHECK ((btrim(stage_name) <> ''::text)),
    CONSTRAINT stage_executions_stage_spec_schema_version_check CHECK ((btrim(stage_spec_schema_version) <> ''::text)),
    CONSTRAINT stage_executions_stage_spec_snapshot_check CHECK ((jsonb_typeof(stage_spec_snapshot) = 'object'::text)),
    CONSTRAINT stage_executions_state_check CHECK ((state = ANY (ARRAY['preparing'::text, 'running'::text, 'finalizing'::text, 'aborting'::text, 'succeeded'::text, 'failed'::text, 'interrupted'::text, 'cancelled'::text]))),
    CONSTRAINT stage_executions_state_reason_code_check CHECK ((btrim(state_reason_code) <> ''::text)),
    CONSTRAINT stage_planner_ids_shape CHECK (((planner_session_id IS NULL) = (planner_invocation_id IS NULL))),
    CONSTRAINT stage_termination_payload_shape CHECK ((((termination_schema_version IS NULL) = (stage_termination IS NULL)) AND ((stage_termination IS NULL) OR ((btrim(termination_schema_version) <> ''::text) AND contractor_valid_stage_termination(stage_termination))))),
    CONSTRAINT stage_transition_action CHECK ((transition_action = ANY (ARRAY['next'::text, 'retry'::text, 'escalate'::text, 'succeed'::text, 'fail'::text]))),
    CONSTRAINT stage_transition_decision_shape CHECK ((((transition_action = ANY (ARRAY['next'::text, 'retry'::text, 'escalate'::text])) AND (transition_target_stage_name IS NOT NULL) AND (btrim(transition_target_stage_name) <> ''::text) AND (transition_target_execution_id IS NOT NULL) AND (btrim(transition_target_execution_id) <> ''::text)) OR ((transition_action = ANY (ARRAY['succeed'::text, 'fail'::text])) AND (transition_target_stage_name IS NULL) AND (transition_target_execution_id IS NULL)))),
    CONSTRAINT stage_transition_escalation_shape CHECK ((((transition_action = 'escalate'::text) AND (transition_escalation_ordinal IS NOT NULL) AND (transition_escalation_ordinal > 0) AND (transition_escalation_exhausted = false)) OR ((transition_action = ANY (ARRAY['next'::text, 'fail'::text])) AND (transition_escalation_ordinal IS NOT NULL) AND (transition_escalation_ordinal > 0) AND (transition_escalation_exhausted = true)) OR ((transition_action <> 'escalate'::text) AND (transition_escalation_ordinal IS NULL) AND (transition_escalation_exhausted = false))))
);

-- Table: stage_metrics

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

-- Table: workflow_run_events

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

-- Table: workflow_run_output_publications

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

-- Table: workflow_runs

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
    scheduler_claim_tier smallint GENERATED ALWAYS AS (
CASE
    WHEN (state = 'cancelling'::text) THEN 0
    WHEN (state = ANY (ARRAY['running'::text, 'waiting'::text])) THEN 1
    WHEN ((state = 'pending'::text) OR ((state = 'initializing'::text) AND (state_reason_code = 'skill_initialization_pending'::text))) THEN 2
    ELSE NULL::integer
END) STORED,
    scheduler_lane_class text NOT NULL,
    required_placement_labels text[] DEFAULT '{}'::text[] NOT NULL,
    audit_input_materialization json,
    engagement_credential_ids text[] DEFAULT '{}'::text[] NOT NULL,
    llm_cost jsonb DEFAULT '{}'::jsonb,
    CONSTRAINT workflow_run_cancellation_shape CHECK ((((state = ANY (ARRAY['cancelling'::text, 'cancelled'::text])) = ((cancellation_schema_version IS NOT NULL) AND (btrim(cancellation_schema_version) <> ''::text) AND (run_cancellation IS NOT NULL) AND contractor_valid_run_cancellation(run_cancellation))) AND ((state = ANY (ARRAY['cancelling'::text, 'cancelled'::text])) OR ((cancellation_schema_version IS NULL) AND (run_cancellation IS NULL))))),
    CONSTRAINT workflow_run_claim_shape CHECK ((((scheduler_claim_id IS NULL) AND (scheduler_claimed_at IS NULL) AND (scheduler_claim_expires_at IS NULL)) OR ((scheduler_claim_id IS NOT NULL) AND (btrim(scheduler_claim_id) <> ''::text) AND (scheduler_claimed_at IS NOT NULL) AND (scheduler_claim_expires_at > scheduler_claimed_at)))),
    CONSTRAINT workflow_run_terminal_shape CHECK (((state = ANY (ARRAY['succeeded'::text, 'failed'::text, 'cancelled'::text])) = (finished_at IS NOT NULL))),
    CONSTRAINT workflow_run_terminal_unclaimed CHECK (((state <> ALL (ARRAY['succeeded'::text, 'failed'::text, 'cancelled'::text])) OR (scheduler_claim_id IS NULL))),
    CONSTRAINT workflow_runs_audit_authority_shape CHECK ((((publication_mode = 'ordinary'::text) AND (audit_execution_id IS NULL) AND (audit_submission_key IS NULL)) OR ((publication_mode = 'audit-managed'::text) AND (project_id IS NOT NULL) AND (audit_execution_id IS NOT NULL) AND (audit_submission_key IS NOT NULL)))),
    CONSTRAINT workflow_runs_audit_input_materialization_check CHECK (((audit_input_materialization IS NULL) OR ((publication_mode = 'audit-managed'::text) AND contractor_valid_audit_input_materialization(audit_input_materialization)))),
    CONSTRAINT workflow_runs_check CHECK (((audit_completion IS NULL) OR ((publication_mode = 'audit-managed'::text) AND (jsonb_typeof(audit_completion) = 'object'::text) AND ((audit_completion #>> '{contract,kind}'::text[]) = 'audit-check-results@1'::text)))),
    CONSTRAINT workflow_runs_engagement_credentials_valid CHECK (contractor_valid_engagement_credential_ids(engagement_credential_ids)),
    CONSTRAINT workflow_runs_llm_cost_check CHECK (((llm_cost IS NULL) OR (jsonb_typeof(llm_cost) = 'object'::text))),
    CONSTRAINT workflow_runs_metadata_labels_check CHECK (contractor_valid_run_metadata_labels(metadata_labels)),
    CONSTRAINT workflow_runs_next_run_event_sequence_check CHECK ((next_run_event_sequence > 0)),
    CONSTRAINT workflow_runs_owner_id_check CHECK ((btrim(owner_id) <> ''::text)),
    CONSTRAINT workflow_runs_parameters_check CHECK ((jsonb_typeof(parameters) = 'object'::text)),
    CONSTRAINT workflow_runs_placement_labels_valid CHECK (contractor_valid_placement_labels(required_placement_labels)),
    CONSTRAINT workflow_runs_project_http_target_shape CHECK (((project_http_target_snapshot IS NULL) OR ((project_id IS NOT NULL) AND (jsonb_typeof(project_http_target_snapshot) = 'object'::text) AND (jsonb_typeof((project_http_target_snapshot -> 'url'::text)) = 'string'::text) AND (btrim((project_http_target_snapshot ->> 'url'::text)) = (project_http_target_snapshot ->> 'url'::text)) AND ((octet_length((project_http_target_snapshot ->> 'url'::text)) >= 1) AND (octet_length((project_http_target_snapshot ->> 'url'::text)) <= 2048)) AND ((NOT (project_http_target_snapshot ? 'credential'::text)) OR ((jsonb_typeof((project_http_target_snapshot -> 'credential'::text)) = 'object'::text) AND (jsonb_typeof((project_http_target_snapshot #> '{credential,credentialId}'::text[])) = 'string'::text) AND ((project_http_target_snapshot #>> '{credential,kind}'::text[]) = ANY (ARRAY['http-origin-basic@1'::text, 'http-origin-bearer@1'::text]))))))),
    CONSTRAINT workflow_runs_publication_mode_check CHECK ((publication_mode = ANY (ARRAY['ordinary'::text, 'audit-managed'::text]))),
    CONSTRAINT workflow_runs_request_idempotency_shape CHECK ((((request_idempotency_key IS NULL) AND (request_digest IS NULL)) OR ((request_idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'::text) AND (request_digest ~ '^sha256:[0-9a-f]{64}$'::text)))),
    CONSTRAINT workflow_runs_run_event_generation_check CHECK ((btrim(run_event_generation) <> ''::text)),
    CONSTRAINT workflow_runs_run_id_check CHECK ((btrim(run_id) <> ''::text)),
    CONSTRAINT workflow_runs_runtime_config_snapshot_shape CHECK (contractor_valid_run_runtime_snapshot(runtime_config_snapshot, runtime_labels)),
    CONSTRAINT workflow_runs_runtime_labels_check CHECK (contractor_valid_runtime_agent_labels(runtime_labels)),
    CONSTRAINT workflow_runs_scheduler_lane_class_check CHECK ((scheduler_lane_class = ANY (ARRAY['model'::text, 'tool'::text]))),
    CONSTRAINT workflow_runs_skill_snapshot_array CHECK ((jsonb_typeof(skill_snapshot) = 'array'::text)),
    CONSTRAINT workflow_runs_state_check CHECK ((state = ANY (ARRAY['initializing'::text, 'pending'::text, 'running'::text, 'waiting'::text, 'cancelling'::text, 'succeeded'::text, 'failed'::text, 'cancelled'::text]))),
    CONSTRAINT workflow_runs_state_reason_code_check CHECK ((btrim(state_reason_code) <> ''::text)),
    CONSTRAINT workflow_runs_workflow_name_check CHECK ((btrim(workflow_name) <> ''::text)),
    CONSTRAINT workflow_runs_workflow_schema_version_check CHECK ((btrim(workflow_schema_version) <> ''::text)),
    CONSTRAINT workflow_runs_workflow_snapshot_check CHECK ((jsonb_typeof(workflow_snapshot) = 'object'::text)),
    CONSTRAINT workflow_runs_workflow_version_check CHECK ((btrim(workflow_version) <> ''::text))
);


--
-- Data for Name: allocation_execution_reports; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: artifact_binding_revisions; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: artifact_bindings; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: artifact_blobs; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: artifact_lineage; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: artifact_scopes; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: artifact_versions; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: audit_artifact_links; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: audit_controller_claims; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: audit_events; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: audit_execution_items; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: audit_executions; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: audit_hypotheses; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: audit_hypothesis_revisions; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: audit_tasks; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: audits; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: credential_operations; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: eval_collections; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: eval_commands; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: eval_controller_claims; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: eval_dataset_revisions; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: eval_evidence_refs; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: eval_execution_tombstones; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: eval_experiments; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: eval_member_projections; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: eval_members; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: eval_plan_resources; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: eval_progress_observations; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: eval_projection_queue; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: eval_records; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: eval_submissions; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: eval_suboperations; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: eval_view_charts; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: eval_view_pairs; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: finding_correlation_fences; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: finding_proposal_audit_holds; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: finding_proposal_receipts; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: git_ssh_keys; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: llm_credentials; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: model_selection_settings; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: mutation_receipts; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: owner_queue_controls; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: performance_minutes; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: planner_execution_reports; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: planner_sessions; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: project_finding_audits; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: project_finding_sources; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: project_findings; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: projects; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: runtime_agent_principals; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: runtime_config_versions; Type: TABLE DATA; Schema: public; Owner: -
--

INSERT INTO runtime_config_versions VALUES ('contractor-empty', '1', 'sha256:80a1754c01f8443c29fdc8f650a2254b2461694819918b204a55a7ad3425dc5f', '{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"contractor-empty","version":"1"},"spec":{}}', true, 'contractor-bootstrap', '1970-01-01 03:00:00+03', NULL, NULL);


--
-- Data for Name: runtime_credentials; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: runtime_label_bindings; Type: TABLE DATA; Schema: public; Owner: -
--

INSERT INTO runtime_label_bindings VALUES ('default', 'contractor-empty', '1', 'sha256:80a1754c01f8443c29fdc8f650a2254b2461694819918b204a55a7ad3425dc5f', 1, 'contractor-bootstrap', '1970-01-01 03:00:00+03', 'contractor-bootstrap', '1970-01-01 03:00:00+03');


--
-- Data for Name: scheduler_settings; Type: TABLE DATA; Schema: public; Owner: -
--

INSERT INTO scheduler_settings (singleton, max_concurrent_runs, revision, max_concurrent_tool_runs) VALUES (true, 1, 1, 2);


--
-- Data for Name: stage_allocations; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: stage_executions; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: stage_metrics; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: workflow_run_events; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: workflow_run_output_publications; Type: TABLE DATA; Schema: public; Owner: -
--



--
-- Data for Name: workflow_runs; Type: TABLE DATA; Schema: public; Owner: -
--

-- Sequence Set: audit_artifact_links_material_ordinal_seq

SELECT pg_catalog.setval('audit_artifact_links_material_ordinal_seq', 1, false);

-- Sequence Set: eval_progress_observations_sequence_seq

SELECT pg_catalog.setval('eval_progress_observations_sequence_seq', 1, false);

-- Constraint: allocation_execution_reports allocation_execution_reports_pkey

ALTER TABLE ONLY allocation_execution_reports
    ADD CONSTRAINT allocation_execution_reports_pkey PRIMARY KEY (report_id);

-- Constraint: artifact_binding_revisions artifact_binding_revisions_pkey

ALTER TABLE ONLY artifact_binding_revisions
    ADD CONSTRAINT artifact_binding_revisions_pkey PRIMARY KEY (scope_kind, scope_id, namespace, name, revision);

-- Constraint: artifact_bindings artifact_bindings_pkey

ALTER TABLE ONLY artifact_bindings
    ADD CONSTRAINT artifact_bindings_pkey PRIMARY KEY (scope_kind, scope_id, namespace, name);

-- Constraint: artifact_blobs artifact_blobs_pkey

ALTER TABLE ONLY artifact_blobs
    ADD CONSTRAINT artifact_blobs_pkey PRIMARY KEY (sha256);

-- Constraint: artifact_lineage artifact_lineage_pkey

ALTER TABLE ONLY artifact_lineage
    ADD CONSTRAINT artifact_lineage_pkey PRIMARY KEY (target_scope_kind, target_scope_id, target_namespace, target_name, target_revision);

-- Constraint: artifact_scopes artifact_scopes_pkey

ALTER TABLE ONLY artifact_scopes
    ADD CONSTRAINT artifact_scopes_pkey PRIMARY KEY (scope_kind, scope_id);

-- Constraint: artifact_versions artifact_versions_pkey

ALTER TABLE ONLY artifact_versions
    ADD CONSTRAINT artifact_versions_pkey PRIMARY KEY (version_id);

-- Constraint: audit_artifact_links audit_artifact_links_pkey

ALTER TABLE ONLY audit_artifact_links
    ADD CONSTRAINT audit_artifact_links_pkey PRIMARY KEY (audit_id, logical_key);

-- Constraint: audit_controller_claims audit_controller_claims_pkey

ALTER TABLE ONLY audit_controller_claims
    ADD CONSTRAINT audit_controller_claims_pkey PRIMARY KEY (audit_id);

-- Constraint: audit_events audit_events_pkey

ALTER TABLE ONLY audit_events
    ADD CONSTRAINT audit_events_pkey PRIMARY KEY (audit_id, sequence_number);

-- Constraint: audit_execution_items audit_execution_items_common_member

ALTER TABLE ONLY audit_execution_items
    ADD CONSTRAINT audit_execution_items_common_member UNIQUE (execution_id, task_id);

-- Constraint: audit_execution_items audit_execution_items_execution_ordinal

ALTER TABLE ONLY audit_execution_items
    ADD CONSTRAINT audit_execution_items_execution_ordinal UNIQUE (execution_id, batch_ordinal);

-- Constraint: audit_execution_items audit_execution_items_pkey

ALTER TABLE ONLY audit_execution_items
    ADD CONSTRAINT audit_execution_items_pkey PRIMARY KEY (execution_item_id);

-- Constraint: audit_executions audit_executions_audit_identity

ALTER TABLE ONLY audit_executions
    ADD CONSTRAINT audit_executions_audit_identity UNIQUE (execution_id, audit_id);

-- Constraint: audit_executions audit_executions_collection_receipt_id_key

ALTER TABLE ONLY audit_executions
    ADD CONSTRAINT audit_executions_collection_receipt_id_key UNIQUE (collection_receipt_id);

-- Constraint: audit_executions audit_executions_pkey

ALTER TABLE ONLY audit_executions
    ADD CONSTRAINT audit_executions_pkey PRIMARY KEY (execution_id);

-- Constraint: audit_executions audit_executions_submission_identity

ALTER TABLE ONLY audit_executions
    ADD CONSTRAINT audit_executions_submission_identity UNIQUE (execution_id, submission_key);

-- Constraint: audit_executions audit_executions_submission_key

ALTER TABLE ONLY audit_executions
    ADD CONSTRAINT audit_executions_submission_key UNIQUE (submission_key);

-- Constraint: audit_hypotheses audit_hypotheses_audit_id_fixed_source_key_key

ALTER TABLE ONLY audit_hypotheses
    ADD CONSTRAINT audit_hypotheses_audit_id_fixed_source_key_key UNIQUE (audit_id, fixed, source_key);

-- Constraint: audit_hypotheses audit_hypotheses_audit_id_ordinal_key

ALTER TABLE ONLY audit_hypotheses
    ADD CONSTRAINT audit_hypotheses_audit_id_ordinal_key UNIQUE (audit_id, ordinal);

-- Constraint: audit_hypotheses audit_hypotheses_hypothesis_id_audit_id_key

ALTER TABLE ONLY audit_hypotheses
    ADD CONSTRAINT audit_hypotheses_hypothesis_id_audit_id_key UNIQUE (hypothesis_id, audit_id);

-- Constraint: audit_hypotheses audit_hypotheses_pkey

ALTER TABLE ONLY audit_hypotheses
    ADD CONSTRAINT audit_hypotheses_pkey PRIMARY KEY (hypothesis_id);

-- Constraint: audit_hypothesis_revisions audit_hypothesis_revisions_pkey

ALTER TABLE ONLY audit_hypothesis_revisions
    ADD CONSTRAINT audit_hypothesis_revisions_pkey PRIMARY KEY (hypothesis_id, revision);

-- Constraint: audit_tasks audit_tasks_audit_id_ordinal_key

ALTER TABLE ONLY audit_tasks
    ADD CONSTRAINT audit_tasks_audit_id_ordinal_key UNIQUE (audit_id, ordinal);

-- Constraint: audit_tasks audit_tasks_audit_id_source_key_key

ALTER TABLE ONLY audit_tasks
    ADD CONSTRAINT audit_tasks_audit_id_source_key_key UNIQUE (audit_id, source_key);

-- Constraint: audit_tasks audit_tasks_pkey

ALTER TABLE ONLY audit_tasks
    ADD CONSTRAINT audit_tasks_pkey PRIMARY KEY (task_id);

-- Constraint: audit_tasks audit_tasks_task_id_audit_id_key

ALTER TABLE ONLY audit_tasks
    ADD CONSTRAINT audit_tasks_task_id_audit_id_key UNIQUE (task_id, audit_id);

-- Constraint: audits audits_correlation_scope

ALTER TABLE ONLY audits
    ADD CONSTRAINT audits_correlation_scope UNIQUE (audit_id, project_id);

-- Constraint: audits audits_pkey

ALTER TABLE ONLY audits
    ADD CONSTRAINT audits_pkey PRIMARY KEY (audit_id);

-- Constraint: credential_operations credential_operations_operation_kind_idempotency_key_key

ALTER TABLE ONLY credential_operations
    ADD CONSTRAINT credential_operations_operation_kind_idempotency_key_key UNIQUE (operation_kind, idempotency_key);

-- Constraint: credential_operations credential_operations_pkey

ALTER TABLE ONLY credential_operations
    ADD CONSTRAINT credential_operations_pkey PRIMARY KEY (operation_id);

-- Constraint: eval_collections eval_collections_pkey

ALTER TABLE ONLY eval_collections
    ADD CONSTRAINT eval_collections_pkey PRIMARY KEY (owner_id, project_id);

-- Constraint: eval_commands eval_commands_pkey

ALTER TABLE ONLY eval_commands
    ADD CONSTRAINT eval_commands_pkey PRIMARY KEY (command_id);

-- Constraint: eval_controller_claims eval_controller_claims_pkey

ALTER TABLE ONLY eval_controller_claims
    ADD CONSTRAINT eval_controller_claims_pkey PRIMARY KEY (experiment_id);

-- Constraint: eval_dataset_revisions eval_dataset_revisions_pkey

ALTER TABLE ONLY eval_dataset_revisions
    ADD CONSTRAINT eval_dataset_revisions_pkey PRIMARY KEY (project_id, dataset_id, revision);

-- Constraint: eval_evidence_refs eval_evidence_refs_pkey

ALTER TABLE ONLY eval_evidence_refs
    ADD CONSTRAINT eval_evidence_refs_pkey PRIMARY KEY (experiment_id, member_id, record_sha256, scope_kind, scope_id, namespace, name, revision);

-- Constraint: eval_execution_tombstones eval_execution_tombstones_pkey

ALTER TABLE ONLY eval_execution_tombstones
    ADD CONSTRAINT eval_execution_tombstones_pkey PRIMARY KEY (experiment_id, member_id);

-- Constraint: eval_experiments eval_experiments_owner_id_project_id_portable_id_key

ALTER TABLE ONLY eval_experiments
    ADD CONSTRAINT eval_experiments_owner_id_project_id_portable_id_key UNIQUE (owner_id, project_id, portable_id);

-- Constraint: eval_experiments eval_experiments_pkey

ALTER TABLE ONLY eval_experiments
    ADD CONSTRAINT eval_experiments_pkey PRIMARY KEY (experiment_id);

-- Constraint: eval_member_projections eval_member_projections_pkey

ALTER TABLE ONLY eval_member_projections
    ADD CONSTRAINT eval_member_projections_pkey PRIMARY KEY (experiment_id, member_id);

-- Constraint: eval_members eval_members_experiment_id_member_id_execution_kind_key

ALTER TABLE ONLY eval_members
    ADD CONSTRAINT eval_members_experiment_id_member_id_execution_kind_key UNIQUE (experiment_id, member_id, execution_kind);

-- Constraint: eval_members eval_members_experiment_id_ordinal_key

ALTER TABLE ONLY eval_members
    ADD CONSTRAINT eval_members_experiment_id_ordinal_key UNIQUE (experiment_id, ordinal);

-- Constraint: eval_members eval_members_experiment_id_suite_id_case_id_sample_variant__key

ALTER TABLE ONLY eval_members
    ADD CONSTRAINT eval_members_experiment_id_suite_id_case_id_sample_variant__key UNIQUE (experiment_id, suite_id, case_id, sample, variant_id);

-- Constraint: eval_members eval_members_pkey

ALTER TABLE ONLY eval_members
    ADD CONSTRAINT eval_members_pkey PRIMARY KEY (experiment_id, member_id);

-- Constraint: eval_members eval_members_submission_key_key

ALTER TABLE ONLY eval_members
    ADD CONSTRAINT eval_members_submission_key_key UNIQUE (submission_key);

-- Constraint: eval_plan_resources eval_plan_resources_pkey

ALTER TABLE ONLY eval_plan_resources
    ADD CONSTRAINT eval_plan_resources_pkey PRIMARY KEY (experiment_id, resource_path);

-- Constraint: eval_progress_observations eval_progress_observations_pkey

ALTER TABLE ONLY eval_progress_observations
    ADD CONSTRAINT eval_progress_observations_pkey PRIMARY KEY (experiment_id, sequence);

-- Constraint: eval_projection_queue eval_projection_queue_pkey

ALTER TABLE ONLY eval_projection_queue
    ADD CONSTRAINT eval_projection_queue_pkey PRIMARY KEY (experiment_id);

-- Constraint: eval_records eval_records_pkey

ALTER TABLE ONLY eval_records
    ADD CONSTRAINT eval_records_pkey PRIMARY KEY (experiment_id, member_id, kind, record_sha256);

-- Constraint: eval_submissions eval_submissions_pkey

ALTER TABLE ONLY eval_submissions
    ADD CONSTRAINT eval_submissions_pkey PRIMARY KEY (experiment_id, member_id);

-- Constraint: eval_suboperations eval_suboperations_operation_key_key

ALTER TABLE ONLY eval_suboperations
    ADD CONSTRAINT eval_suboperations_operation_key_key UNIQUE (operation_key);

-- Constraint: eval_suboperations eval_suboperations_pkey

ALTER TABLE ONLY eval_suboperations
    ADD CONSTRAINT eval_suboperations_pkey PRIMARY KEY (experiment_id, member_id, kind);

-- Constraint: eval_view_charts eval_view_charts_pkey

ALTER TABLE ONLY eval_view_charts
    ADD CONSTRAINT eval_view_charts_pkey PRIMARY KEY (experiment_id, generation, suite_id, metric);

-- Constraint: eval_view_pairs eval_view_pairs_experiment_id_generation_pair_id_key

ALTER TABLE ONLY eval_view_pairs
    ADD CONSTRAINT eval_view_pairs_experiment_id_generation_pair_id_key UNIQUE (experiment_id, generation, pair_id);

-- Constraint: eval_view_pairs eval_view_pairs_pkey

ALTER TABLE ONLY eval_view_pairs
    ADD CONSTRAINT eval_view_pairs_pkey PRIMARY KEY (experiment_id, generation, ordinal);

-- Constraint: finding_correlation_fences finding_correlation_fences_pkey

ALTER TABLE ONLY finding_correlation_fences
    ADD CONSTRAINT finding_correlation_fences_pkey PRIMARY KEY (project_id);

-- Constraint: finding_proposal_audit_holds finding_proposal_audit_holds_pkey

ALTER TABLE ONLY finding_proposal_audit_holds
    ADD CONSTRAINT finding_proposal_audit_holds_pkey PRIMARY KEY (receipt_id, audit_id);

-- Constraint: finding_proposal_receipts finding_proposal_receipts_allocation_id_invocation_id_submi_key

ALTER TABLE ONLY finding_proposal_receipts
    ADD CONSTRAINT finding_proposal_receipts_allocation_id_invocation_id_submi_key UNIQUE (allocation_id, invocation_id, submission_id);

-- Constraint: finding_proposal_receipts finding_proposal_receipts_pkey

ALTER TABLE ONLY finding_proposal_receipts
    ADD CONSTRAINT finding_proposal_receipts_pkey PRIMARY KEY (receipt_id);

-- Constraint: finding_proposal_receipts finding_proposal_receipts_proposal_id_key

ALTER TABLE ONLY finding_proposal_receipts
    ADD CONSTRAINT finding_proposal_receipts_proposal_id_key UNIQUE (proposal_id);

-- Constraint: git_ssh_keys git_ssh_keys_pkey

ALTER TABLE ONLY git_ssh_keys
    ADD CONSTRAINT git_ssh_keys_pkey PRIMARY KEY (owner_id);

-- Constraint: llm_credentials llm_credentials_llm_gateway_digest_remote_key_id_key

ALTER TABLE ONLY llm_credentials
    ADD CONSTRAINT llm_credentials_llm_gateway_digest_remote_key_id_key UNIQUE (llm_gateway_digest, remote_key_id);

-- Constraint: llm_credentials llm_credentials_pkey

ALTER TABLE ONLY llm_credentials
    ADD CONSTRAINT llm_credentials_pkey PRIMARY KEY (credential_id);

-- Constraint: model_selection_settings model_selection_settings_pkey

ALTER TABLE ONLY model_selection_settings
    ADD CONSTRAINT model_selection_settings_pkey PRIMARY KEY (singleton);

-- Constraint: mutation_receipts mutation_receipts_pkey

ALTER TABLE ONLY mutation_receipts
    ADD CONSTRAINT mutation_receipts_pkey PRIMARY KEY (key_scope, idempotency_key);

-- Constraint: owner_queue_controls owner_queue_controls_pkey

ALTER TABLE ONLY owner_queue_controls
    ADD CONSTRAINT owner_queue_controls_pkey PRIMARY KEY (owner_id);

-- Constraint: performance_minutes performance_minutes_pkey

ALTER TABLE ONLY performance_minutes
    ADD CONSTRAINT performance_minutes_pkey PRIMARY KEY (server_generation, minute_start);

-- Constraint: planner_execution_reports planner_execution_reports_pkey

ALTER TABLE ONLY planner_execution_reports
    ADD CONSTRAINT planner_execution_reports_pkey PRIMARY KEY (report_id);

-- Constraint: planner_sessions planner_sessions_invocation_id_key

ALTER TABLE ONLY planner_sessions
    ADD CONSTRAINT planner_sessions_invocation_id_key UNIQUE (invocation_id);

-- Constraint: planner_sessions planner_sessions_pkey

ALTER TABLE ONLY planner_sessions
    ADD CONSTRAINT planner_sessions_pkey PRIMARY KEY (session_id);

-- Constraint: planner_sessions planner_sessions_stage_execution_id_key

ALTER TABLE ONLY planner_sessions
    ADD CONSTRAINT planner_sessions_stage_execution_id_key UNIQUE (stage_execution_id);

-- Constraint: planner_sessions planner_sessions_stage_execution_id_session_id_invocation_i_key

ALTER TABLE ONLY planner_sessions
    ADD CONSTRAINT planner_sessions_stage_execution_id_session_id_invocation_i_key UNIQUE (stage_execution_id, session_id, invocation_id);

-- Constraint: project_finding_audits project_finding_audits_pkey

ALTER TABLE ONLY project_finding_audits
    ADD CONSTRAINT project_finding_audits_pkey PRIMARY KEY (audit_id, finding_id);

-- Constraint: project_finding_sources project_finding_sources_pkey

ALTER TABLE ONLY project_finding_sources
    ADD CONSTRAINT project_finding_sources_pkey PRIMARY KEY (receipt_id);

-- Constraint: project_findings project_findings_pkey

ALTER TABLE ONLY project_findings
    ADD CONSTRAINT project_findings_pkey PRIMARY KEY (finding_id);

-- Constraint: project_findings project_findings_project_id_finding_id_key

ALTER TABLE ONLY project_findings
    ADD CONSTRAINT project_findings_project_id_finding_id_key UNIQUE (project_id, finding_id);

-- Constraint: projects projects_owner_id_request_idempotency_key_key

ALTER TABLE ONLY projects
    ADD CONSTRAINT projects_owner_id_request_idempotency_key_key UNIQUE (owner_id, request_idempotency_key);

-- Constraint: projects projects_owner_identity

ALTER TABLE ONLY projects
    ADD CONSTRAINT projects_owner_identity UNIQUE (owner_id, project_id);

-- Constraint: projects projects_pkey

ALTER TABLE ONLY projects
    ADD CONSTRAINT projects_pkey PRIMARY KEY (project_id);

-- Constraint: runtime_agent_principals runtime_agent_principals_pkey

ALTER TABLE ONLY runtime_agent_principals
    ADD CONSTRAINT runtime_agent_principals_pkey PRIMARY KEY (runtime_agent_id);

-- Constraint: runtime_config_versions runtime_config_versions_idempotency_key_digest_key

ALTER TABLE ONLY runtime_config_versions
    ADD CONSTRAINT runtime_config_versions_idempotency_key_digest_key UNIQUE (idempotency_key_digest);

-- Constraint: runtime_config_versions runtime_config_versions_name_version_digest_key

ALTER TABLE ONLY runtime_config_versions
    ADD CONSTRAINT runtime_config_versions_name_version_digest_key UNIQUE (name, version, digest);

-- Constraint: runtime_config_versions runtime_config_versions_pkey

ALTER TABLE ONLY runtime_config_versions
    ADD CONSTRAINT runtime_config_versions_pkey PRIMARY KEY (name, version);

-- Constraint: runtime_credentials runtime_credentials_credential_id_credential_kind_key

ALTER TABLE ONLY runtime_credentials
    ADD CONSTRAINT runtime_credentials_credential_id_credential_kind_key UNIQUE (credential_id, credential_kind);

-- Constraint: runtime_credentials runtime_credentials_idempotency_key_digest_key

ALTER TABLE ONLY runtime_credentials
    ADD CONSTRAINT runtime_credentials_idempotency_key_digest_key UNIQUE (idempotency_key_digest);

-- Constraint: runtime_credentials runtime_credentials_pkey

ALTER TABLE ONLY runtime_credentials
    ADD CONSTRAINT runtime_credentials_pkey PRIMARY KEY (credential_id);

-- Constraint: runtime_label_bindings runtime_label_bindings_pkey

ALTER TABLE ONLY runtime_label_bindings
    ADD CONSTRAINT runtime_label_bindings_pkey PRIMARY KEY (label);

-- Constraint: scheduler_settings scheduler_settings_pkey

ALTER TABLE ONLY scheduler_settings
    ADD CONSTRAINT scheduler_settings_pkey PRIMARY KEY (singleton);

-- Constraint: stage_allocations stage_allocation_report_identity

ALTER TABLE ONLY stage_allocations
    ADD CONSTRAINT stage_allocation_report_identity UNIQUE (stage_execution_id, allocation_id, logical_agent_name);

-- Constraint: stage_allocations stage_allocations_pkey

ALTER TABLE ONLY stage_allocations
    ADD CONSTRAINT stage_allocations_pkey PRIMARY KEY (allocation_id);

-- Constraint: stage_allocations stage_allocations_stage_execution_id_logical_agent_name_key

ALTER TABLE ONLY stage_allocations
    ADD CONSTRAINT stage_allocations_stage_execution_id_logical_agent_name_key UNIQUE (stage_execution_id, logical_agent_name);

-- Constraint: stage_executions stage_attempt_identity

ALTER TABLE ONLY stage_executions
    ADD CONSTRAINT stage_attempt_identity UNIQUE (run_id, stage_name, attempt);

-- Constraint: stage_executions stage_execution_lineage_key

ALTER TABLE ONLY stage_executions
    ADD CONSTRAINT stage_execution_lineage_key UNIQUE (run_id, stage_name, stage_execution_id);

-- Constraint: stage_executions stage_execution_resume_source_once

ALTER TABLE ONLY stage_executions
    ADD CONSTRAINT stage_execution_resume_source_once UNIQUE (resume_source_execution_id);

-- Constraint: stage_executions stage_execution_run_identity

ALTER TABLE ONLY stage_executions
    ADD CONSTRAINT stage_execution_run_identity UNIQUE (run_id, stage_execution_id);

-- Constraint: stage_executions stage_execution_transition_target_once

ALTER TABLE ONLY stage_executions
    ADD CONSTRAINT stage_execution_transition_target_once UNIQUE (transition_target_execution_id);

-- Constraint: stage_executions stage_executions_pkey

ALTER TABLE ONLY stage_executions
    ADD CONSTRAINT stage_executions_pkey PRIMARY KEY (stage_execution_id);

-- Constraint: stage_metrics stage_metrics_pkey

ALTER TABLE ONLY stage_metrics
    ADD CONSTRAINT stage_metrics_pkey PRIMARY KEY (stage_execution_id);

-- Constraint: workflow_run_events workflow_run_events_event_id_key

ALTER TABLE ONLY workflow_run_events
    ADD CONSTRAINT workflow_run_events_event_id_key UNIQUE (event_id);

-- Constraint: workflow_run_events workflow_run_events_pkey

ALTER TABLE ONLY workflow_run_events
    ADD CONSTRAINT workflow_run_events_pkey PRIMARY KEY (run_id, sequence_number);

-- Constraint: workflow_run_output_publications workflow_run_output_publications_pkey

ALTER TABLE ONLY workflow_run_output_publications
    ADD CONSTRAINT workflow_run_output_publications_pkey PRIMARY KEY (run_id, output_name);

-- Constraint: workflow_runs workflow_runs_audit_execution_unique

ALTER TABLE ONLY workflow_runs
    ADD CONSTRAINT workflow_runs_audit_execution_unique UNIQUE (audit_execution_id);

-- Constraint: workflow_runs workflow_runs_audit_submission_unique

ALTER TABLE ONLY workflow_runs
    ADD CONSTRAINT workflow_runs_audit_submission_unique UNIQUE (audit_submission_key);

-- Constraint: workflow_runs workflow_runs_pkey

ALTER TABLE ONLY workflow_runs
    ADD CONSTRAINT workflow_runs_pkey PRIMARY KEY (run_id);

-- Constraint: workflow_runs workflow_runs_run_project_key

ALTER TABLE ONLY workflow_runs
    ADD CONSTRAINT workflow_runs_run_project_key UNIQUE (run_id, project_id);

-- Index: allocation_execution_reports_allocation_idx

CREATE INDEX allocation_execution_reports_allocation_idx ON allocation_execution_reports USING btree (allocation_id, received_at DESC);

-- Index: allocation_execution_reports_complete_identity

CREATE UNIQUE INDEX allocation_execution_reports_complete_identity ON allocation_execution_reports USING btree (stage_execution_id, allocation_id, logical_agent_name) WHERE ((((report -> 'worker'::text) ->> 'complete'::text))::boolean AND (((report -> 'runtime'::text) ->> 'complete'::text))::boolean);

-- Index: allocation_execution_reports_expiry_idx

CREATE INDEX allocation_execution_reports_expiry_idx ON allocation_execution_reports USING btree (expires_at, stage_execution_id);

-- Index: allocation_execution_reports_stage_idx

CREATE INDEX allocation_execution_reports_stage_idx ON allocation_execution_reports USING btree (stage_execution_id, logical_agent_name);

-- Index: artifact_binding_revisions_eligible_idx

CREATE INDEX artifact_binding_revisions_eligible_idx ON artifact_binding_revisions USING btree (scope_kind, scope_id, namespace, name, created_at, revision) WHERE (hold_count = 0);

-- Index: artifact_binding_revisions_history_idx

CREATE INDEX artifact_binding_revisions_history_idx ON artifact_binding_revisions USING btree (scope_kind, scope_id, namespace, name, created_at DESC, revision DESC);

-- Index: artifact_binding_revisions_version_idx

CREATE INDEX artifact_binding_revisions_version_idx ON artifact_binding_revisions USING btree (version_id);

-- Index: artifact_bindings_project_listing_idx

CREATE INDEX artifact_bindings_project_listing_idx ON artifact_bindings USING btree (scope_id, namespace, name) WHERE ((scope_kind = 'project'::text) AND (namespace !~~ 'audit-%'::text));

-- Index: artifact_bindings_project_material_namespace_order_idx

CREATE INDEX artifact_bindings_project_material_namespace_order_idx ON artifact_bindings USING btree (scope_id, namespace, material_rank, name) WHERE ((scope_kind = 'project'::text) AND (namespace !~~ 'audit-%'::text));

-- Index: artifact_bindings_project_material_order_idx

CREATE INDEX artifact_bindings_project_material_order_idx ON artifact_bindings USING btree (scope_id, material_rank, namespace, name) WHERE ((scope_kind = 'project'::text) AND (namespace !~~ 'audit-%'::text) AND (namespace !~~ 'historical-%'::text));

-- Index: artifact_bindings_project_materials_idx

CREATE INDEX artifact_bindings_project_materials_idx ON artifact_bindings USING btree (scope_id, namespace, name) WHERE ((scope_kind = 'project'::text) AND (namespace !~~ 'audit-%'::text) AND (namespace !~~ 'historical-%'::text));

-- Index: artifact_bindings_project_prefix_idx

CREATE INDEX artifact_bindings_project_prefix_idx ON artifact_bindings USING btree (scope_id, namespace, name COLLATE "C") WHERE (scope_kind = 'project'::text);

-- Index: artifact_bindings_user_listing_idx

CREATE INDEX artifact_bindings_user_listing_idx ON artifact_bindings USING btree (scope_id, namespace, name) WHERE ((scope_kind = 'user'::text) AND (namespace <> 'audit-standards'::text));

-- Index: artifact_blobs_object_key_idx

CREATE UNIQUE INDEX artifact_blobs_object_key_idx ON artifact_blobs USING btree (object_key) WHERE (object_key IS NOT NULL);

-- Index: artifact_lineage_source_idx

CREATE INDEX artifact_lineage_source_idx ON artifact_lineage USING btree (source_scope_kind, source_scope_id, source_namespace, source_name, source_revision);

-- Index: artifact_scopes_audit_id_idx

CREATE INDEX artifact_scopes_audit_id_idx ON artifact_scopes USING btree (audit_id) WHERE (audit_id IS NOT NULL);

-- Index: artifact_scopes_project_idx

CREATE INDEX artifact_scopes_project_idx ON artifact_scopes USING btree (project_id) WHERE (project_id IS NOT NULL);

-- Index: artifact_scopes_run_idx

CREATE INDEX artifact_scopes_run_idx ON artifact_scopes USING btree (run_id) WHERE (run_id IS NOT NULL);

-- Index: artifact_versions_blob_idx

CREATE INDEX artifact_versions_blob_idx ON artifact_versions USING btree (blob_sha256);

-- Index: audit_artifact_links_digest_idx

CREATE INDEX audit_artifact_links_digest_idx ON audit_artifact_links USING btree (audit_id, artifact_digest);

-- Index: audit_artifact_links_exact_dependency_idx

CREATE INDEX audit_artifact_links_exact_dependency_idx ON audit_artifact_links USING btree (audit_id, ((artifact_ref ->> 'namespace'::text)), ((artifact_ref ->> 'name'::text)), ((artifact_ref ->> 'revision'::text)), artifact_digest);

-- Index: audit_artifact_links_execution_item_idx

CREATE INDEX audit_artifact_links_execution_item_idx ON audit_artifact_links USING btree (audit_id, COALESCE((source_provenance ->> 'executionItemId'::text), ((source_provenance -> 'source'::text) ->> 'executionItemId'::text)), logical_key);

-- Index: audit_artifact_links_material_idx

CREATE INDEX audit_artifact_links_material_idx ON audit_artifact_links USING btree (audit_id, ((artifact_ref ->> 'namespace'::text)), ((artifact_ref ->> 'name'::text)), ((artifact_ref ->> 'revision'::text)), logical_key) WHERE (artifact_scope = 'audit'::text);

-- Index: audit_artifact_links_ordinal_idx

CREATE UNIQUE INDEX audit_artifact_links_ordinal_idx ON audit_artifact_links USING btree (audit_id, material_ordinal);

-- Index: audit_artifact_links_receipt_order

CREATE INDEX audit_artifact_links_receipt_order ON audit_artifact_links USING btree (audit_id, logical_key COLLATE "C");

-- Index: audit_controller_claim_expiry_idx

CREATE INDEX audit_controller_claim_expiry_idx ON audit_controller_claims USING btree (expires_at, audit_id);

-- Index: audit_events_finding_proposal_rejected_idx

CREATE INDEX audit_events_finding_proposal_rejected_idx ON audit_events USING btree (audit_id, entity_id) WHERE (kind = 'finding.proposal_rejected'::text);

-- Index: audit_execution_items_audit_state_idx

CREATE INDEX audit_execution_items_audit_state_idx ON audit_execution_items USING btree (audit_id, state, execution_id, batch_ordinal);

-- Index: audit_execution_items_task_attempt_idx

CREATE UNIQUE INDEX audit_execution_items_task_attempt_idx ON audit_execution_items USING btree (task_id, audit_id, item_attempt) WHERE (task_id IS NOT NULL);

-- Index: audit_executions_audit_run_cleanup_idx

CREATE INDEX audit_executions_audit_run_cleanup_idx ON audit_executions USING btree (audit_id, run_deleted_at, created_at, execution_id) WHERE (run_id IS NOT NULL);

-- Index: audit_executions_inventory_idx

CREATE INDEX audit_executions_inventory_idx ON audit_executions USING btree (audit_id, execution_id);

-- Index: audit_executions_reconcile_idx

CREATE INDEX audit_executions_reconcile_idx ON audit_executions USING btree (audit_id, state, created_at, execution_id) WHERE (state = ANY (ARRAY['intent'::text, 'submitted'::text, 'collecting'::text]));

-- Index: audit_executions_run_unique

CREATE UNIQUE INDEX audit_executions_run_unique ON audit_executions USING btree (run_id) WHERE (run_id IS NOT NULL);

-- Index: audit_hypothesis_revisions_purge_idx

CREATE INDEX audit_hypothesis_revisions_purge_idx ON audit_hypothesis_revisions USING btree (audit_id, hypothesis_id, revision);

-- Index: audit_tasks_continuation_cursor

CREATE INDEX audit_tasks_continuation_cursor ON audit_tasks USING btree (audit_id, workflow_alias, ordinal DESC) WHERE ((source ->> 'kind'::text) = 'continuation'::text);

-- Index: audit_tasks_hypothesis_idx

CREATE INDEX audit_tasks_hypothesis_idx ON audit_tasks USING btree (audit_id, (((document -> 'hypothesis'::text) ->> 'id'::text)), ordinal) WHERE ((document -> 'hypothesis'::text) IS NOT NULL);

-- Index: audit_tasks_last_execution_idx

CREATE INDEX audit_tasks_last_execution_idx ON audit_tasks USING btree (last_execution_id, audit_id) WHERE (last_execution_id IS NOT NULL);

-- Index: audit_tasks_ready_workflow_idx

CREATE INDEX audit_tasks_ready_workflow_idx ON audit_tasks USING btree (audit_id, workflow_alias, attention DESC, ordinal) INCLUDE (task_id, metadata_bytes) WHERE (state = 'ready'::text);

-- Index: audit_tasks_state_idx

CREATE INDEX audit_tasks_state_idx ON audit_tasks USING btree (audit_id, state, ordinal);

-- Index: audit_tasks_workflow_external_ordinal

CREATE INDEX audit_tasks_workflow_external_ordinal ON audit_tasks USING btree (audit_id, workflow_alias, ordinal) WHERE ((state = 'settled'::text) AND ((source ->> 'kind'::text) <> 'continuation'::text));

-- Index: audits_held_llm_credentials_idx

CREATE INDEX audits_held_llm_credentials_idx ON audits USING gin (((baseline_snapshot -> 'llmCredentialIds'::text))) WHERE (hold_state = 'held'::text);

-- Index: audits_held_runtime_credentials_idx

CREATE INDEX audits_held_runtime_credentials_idx ON audits USING gin (((baseline_snapshot -> 'runtimeCredentialIds'::text))) WHERE (hold_state = 'held'::text);

-- Index: audits_owner_created_idx

CREATE INDEX audits_owner_created_idx ON audits USING btree (owner_id, created_at DESC, audit_id DESC);

-- Index: audits_project_deletion_intent_idx

CREATE INDEX audits_project_deletion_intent_idx ON audits USING btree (project_id, deletion_requested_at, created_at, audit_id);

-- Index: audits_project_state_idx

CREATE INDEX audits_project_state_idx ON audits USING btree (project_id, state, created_at, audit_id);

-- Index: audits_reconcile_idx

CREATE INDEX audits_reconcile_idx ON audits USING btree (updated_at, audit_id) WHERE (state = ANY (ARRAY['active'::text, 'paused'::text, 'finalizing'::text, 'cancelling'::text, 'deleting'::text]));

-- Index: credential_operations_create_identity

CREATE UNIQUE INDEX credential_operations_create_identity ON credential_operations USING btree (credential_id) WHERE (operation_kind = 'create'::text);

-- Index: credential_operations_prepared_idx

CREATE INDEX credential_operations_prepared_idx ON credential_operations USING btree (created_at, operation_id) WHERE (phase = 'prepared'::text);

-- Index: eval_claims_expiry_idx

CREATE INDEX eval_claims_expiry_idx ON eval_controller_claims USING btree (expires_at, experiment_id);

-- Index: eval_commands_experiment_idx

CREATE INDEX eval_commands_experiment_idx ON eval_commands USING btree (experiment_id);

-- Index: eval_commands_pending_idx

CREATE INDEX eval_commands_pending_idx ON eval_commands USING btree (experiment_id, created_at, command_id) WHERE (state = ANY (ARRAY['accepted'::text, 'running'::text]));

-- Index: eval_datasets_owner_page_idx

CREATE INDEX eval_datasets_owner_page_idx ON eval_dataset_revisions USING btree (owner_id, project_id, dataset_id, revision);

-- Index: eval_evidence_refs_artifact

CREATE INDEX eval_evidence_refs_artifact ON eval_evidence_refs USING btree (scope_kind, scope_id, namespace, name, revision);

-- Index: eval_experiments_claim_active_idx

CREATE INDEX eval_experiments_claim_active_idx ON eval_experiments USING btree (experiment_id) WHERE (state = ANY (ARRAY['preparing'::text, 'running'::text, 'settling'::text, 'pausing'::text, 'cancelling'::text]));

-- Index: eval_experiments_claim_paused_deadline_idx

CREATE INDEX eval_experiments_claim_paused_deadline_idx ON eval_experiments USING btree (deadline_at, experiment_id) WHERE (state = 'paused'::text);

-- Index: eval_experiments_dataset_revision_idx

CREATE INDEX eval_experiments_dataset_revision_idx ON eval_experiments USING btree (project_id, dataset_id, dataset_revision) WHERE (dataset_id IS NOT NULL);

-- Index: eval_experiments_owner_created_page_idx

CREATE INDEX eval_experiments_owner_created_page_idx ON eval_experiments USING btree (owner_id, created_at DESC, experiment_id DESC);

-- Index: eval_experiments_project_created_page_idx

CREATE INDEX eval_experiments_project_created_page_idx ON eval_experiments USING btree (owner_id, project_id, created_at DESC, experiment_id DESC);

-- Index: eval_experiments_project_page_idx

CREATE INDEX eval_experiments_project_page_idx ON eval_experiments USING btree (owner_id, project_id, experiment_id);

-- Index: eval_member_projections_dirty

CREATE INDEX eval_member_projections_dirty ON eval_member_projections USING btree (experiment_id, checked_at NULLS FIRST, member_id) WHERE (revision <> projected_revision);

-- Index: eval_members_pair_idx

CREATE INDEX eval_members_pair_idx ON eval_members USING btree (experiment_id, pair_id, variant_id);

-- Index: eval_members_selected_idx

CREATE INDEX eval_members_selected_idx ON eval_members USING btree (experiment_id, member_id) WHERE (selected_result_sha256 IS NOT NULL);

-- Index: eval_progress_observations_time

CREATE INDEX eval_progress_observations_time ON eval_progress_observations USING btree (experiment_id, observed_at, sequence);

-- Index: eval_projection_queue_dirty_idx

CREATE INDEX eval_projection_queue_dirty_idx ON eval_projection_queue USING btree (experiment_id) WHERE (revision <> published_revision);

-- Index: eval_records_native_recent

CREATE INDEX eval_records_native_recent ON eval_records USING btree (experiment_id, member_id, kind, created_at DESC, record_sha256 DESC) WHERE (actor_id = 'system:eval-collector'::text);

-- Index: eval_records_recent

CREATE INDEX eval_records_recent ON eval_records USING btree (experiment_id, member_id, kind, created_at DESC, record_sha256 DESC);

-- Index: eval_submissions_execution_idx

CREATE UNIQUE INDEX eval_submissions_execution_idx ON eval_submissions USING btree (execution_kind, execution_id) WHERE (execution_id IS NOT NULL);

-- Index: eval_submissions_outstanding_idx

CREATE INDEX eval_submissions_outstanding_idx ON eval_submissions USING btree (experiment_id, member_id) WHERE (state = ANY (ARRAY['intent'::text, 'accepted'::text]));

-- Index: eval_submissions_reconcile_idx

CREATE INDEX eval_submissions_reconcile_idx ON eval_submissions USING btree (experiment_id, updated_at, member_id) WHERE (state = ANY (ARRAY['intent'::text, 'accepted'::text]));

-- Index: eval_submissions_workspace_idx

CREATE UNIQUE INDEX eval_submissions_workspace_idx ON eval_submissions USING btree (execution_project_id) WHERE (execution_project_id IS NOT NULL);

-- Index: eval_view_pairs_suite

CREATE INDEX eval_view_pairs_suite ON eval_view_pairs USING btree (experiment_id, generation, suite_id, ordinal);

-- Index: finding_holds_admission_cutoff_idx

CREATE INDEX finding_holds_admission_cutoff_idx ON finding_proposal_audit_holds USING btree (audit_id, created_at, receipt_id);

-- Index: finding_proposal_audit_holds_audit_receipt_idx

CREATE INDEX finding_proposal_audit_holds_audit_receipt_idx ON finding_proposal_audit_holds USING btree (audit_id, receipt_created_at, receipt_id);

-- Index: finding_proposal_audit_holds_project_idx

CREATE INDEX finding_proposal_audit_holds_project_idx ON finding_proposal_audit_holds USING btree (project_id);

-- Index: finding_proposal_receipts_audit_idx

CREATE INDEX finding_proposal_receipts_audit_idx ON finding_proposal_receipts USING btree (audit_id, created_at, receipt_id) WHERE (audit_id IS NOT NULL);

-- Index: finding_proposal_receipts_execution_task_idx

CREATE INDEX finding_proposal_receipts_execution_task_idx ON finding_proposal_receipts USING btree (audit_execution_id, audit_task_id, created_at, receipt_id) WHERE (audit_id IS NOT NULL);

-- Index: finding_proposal_receipts_project_idx

CREATE INDEX finding_proposal_receipts_project_idx ON finding_proposal_receipts USING btree (project_id, receipt_id) WHERE (project_id IS NOT NULL);

-- Index: finding_proposal_receipts_run_idx

CREATE INDEX finding_proposal_receipts_run_idx ON finding_proposal_receipts USING btree (run_id, created_at, receipt_id);

-- Index: finding_receipts_project_history_idx

CREATE INDEX finding_receipts_project_history_idx ON finding_proposal_receipts USING btree (project_id, created_at, receipt_id COLLATE "C") WHERE (project_proposal IS NOT NULL);

-- Index: llm_credentials_created_idx

CREATE INDEX llm_credentials_created_idx ON llm_credentials USING btree (created_at, credential_id);

-- Index: mutation_receipts_audit_idx

CREATE INDEX mutation_receipts_audit_idx ON mutation_receipts USING btree (audit_id) WHERE (audit_id IS NOT NULL);

-- Index: mutation_receipts_owner_resource_idx

CREATE INDEX mutation_receipts_owner_resource_idx ON mutation_receipts USING btree (owner_id, resource_id, operation, idempotency_key) WHERE (owner_id IS NOT NULL);

-- Index: mutation_receipts_project_idx

CREATE INDEX mutation_receipts_project_idx ON mutation_receipts USING btree (project_id) WHERE (project_id IS NOT NULL);

-- Index: mutation_receipts_project_mutations_idx

CREATE INDEX mutation_receipts_project_mutations_idx ON mutation_receipts USING btree (project_id) WHERE (key_scope[1] = 'project'::text);

-- Index: performance_minutes_expiry_idx

CREATE INDEX performance_minutes_expiry_idx ON performance_minutes USING btree (expires_at, server_generation, minute_start);

-- Index: performance_minutes_range_idx

CREATE INDEX performance_minutes_range_idx ON performance_minutes USING btree (minute_start, server_generation);

-- Index: planner_execution_reports_complete_identity

CREATE UNIQUE INDEX planner_execution_reports_complete_identity ON planner_execution_reports USING btree (stage_execution_id, session_id) WHERE ((report ->> 'complete'::text))::boolean;

-- Index: planner_execution_reports_expiry_idx

CREATE INDEX planner_execution_reports_expiry_idx ON planner_execution_reports USING btree (expires_at, stage_execution_id);

-- Index: planner_execution_reports_session_idx

CREATE INDEX planner_execution_reports_session_idx ON planner_execution_reports USING btree (session_id);

-- Index: planner_execution_reports_stage_idx

CREATE INDEX planner_execution_reports_stage_idx ON planner_execution_reports USING btree (stage_execution_id, received_at DESC);

-- Index: project_finding_audits_finding_idx

CREATE INDEX project_finding_audits_finding_idx ON project_finding_audits USING btree (project_id, finding_id, audit_id);

-- Index: project_finding_audits_group_idx

CREATE UNIQUE INDEX project_finding_audits_group_idx ON project_finding_audits USING btree (audit_id, correlation_group_key COLLATE "C") WHERE (correlation_group_key <> ''::text);

-- Index: project_finding_audits_page_idx

CREATE INDEX project_finding_audits_page_idx ON project_finding_audits USING btree (audit_id, finding_created_at, finding_id COLLATE "C");

-- Index: project_finding_audits_unqualified_idx

CREATE INDEX project_finding_audits_unqualified_idx ON project_finding_audits USING btree (audit_id, finding_id COLLATE "C") WHERE (correlation_group_key = ''::text);

-- Index: project_finding_sources_owner_idx

CREATE INDEX project_finding_sources_owner_idx ON project_finding_sources USING btree (project_id, finding_id);

-- Index: project_finding_sources_page_idx

CREATE INDEX project_finding_sources_page_idx ON project_finding_sources USING btree (finding_id, created_at, receipt_id COLLATE "C");

-- Index: project_finding_sources_project_idx

CREATE INDEX project_finding_sources_project_idx ON project_finding_sources USING btree (project_id, receipt_id);

-- Index: project_findings_duplicate_idx

CREATE INDEX project_findings_duplicate_idx ON project_findings USING btree (project_id, duplicate_target_id) WHERE (duplicate_target_id IS NOT NULL);

-- Index: project_findings_first_receipt_idx

CREATE INDEX project_findings_first_receipt_idx ON project_findings USING btree (first_receipt_id);

-- Index: project_findings_owner_page_idx

CREATE INDEX project_findings_owner_page_idx ON project_findings USING btree (owner_id, created_at DESC, finding_id COLLATE "C" DESC);

-- Index: project_findings_page_idx

CREATE INDEX project_findings_page_idx ON project_findings USING btree (project_id, created_at, finding_id COLLATE "C");

-- Index: project_findings_qualified_key_idx

CREATE UNIQUE INDEX project_findings_qualified_key_idx ON project_findings USING btree (project_id, correlation_group_key) WHERE (correlation_group_key <> ''::text);

-- Index: project_findings_severity_page_idx

CREATE INDEX project_findings_severity_page_idx ON project_findings USING btree (project_id, analyst_severity, created_at, finding_id COLLATE "C");

-- Index: project_findings_state_page_idx

CREATE INDEX project_findings_state_page_idx ON project_findings USING btree (project_id, state, created_at, finding_id COLLATE "C");

-- Index: project_findings_verdict_page_idx

CREATE INDEX project_findings_verdict_page_idx ON project_findings USING btree (project_id, analyst_verdict, created_at, finding_id COLLATE "C");

-- Index: projects_deletion_claim_idx

CREATE INDEX projects_deletion_claim_idx ON projects USING btree (deletion_requested_at, project_id) WHERE (lifecycle_state = 'deleting'::text);

-- Index: projects_http_target_credential_idx

CREATE INDEX projects_http_target_credential_idx ON projects USING btree (http_target_credential_id, project_id) WHERE (http_target_credential_id IS NOT NULL);

-- Index: projects_owner_kind_created_idx

CREATE INDEX projects_owner_kind_created_idx ON projects USING btree (owner_id, kind, created_at DESC, project_id DESC);

-- Index: runtime_agent_principals_labels_idx

CREATE INDEX runtime_agent_principals_labels_idx ON runtime_agent_principals USING gin (labels);

-- Index: runtime_agent_principals_updated_idx

CREATE INDEX runtime_agent_principals_updated_idx ON runtime_agent_principals USING btree (updated_at, runtime_agent_id);

-- Index: runtime_config_versions_created_at_idx

CREATE INDEX runtime_config_versions_created_at_idx ON runtime_config_versions USING btree (created_at, name, version);

-- Index: runtime_credentials_created_idx

CREATE INDEX runtime_credentials_created_idx ON runtime_credentials USING btree (created_at, credential_id);

-- Index: runtime_label_bindings_config_idx

CREATE INDEX runtime_label_bindings_config_idx ON runtime_label_bindings USING btree (config_name, config_version, config_digest, label);

-- Index: stage_allocations_live_llm_credential_idx

CREATE INDEX stage_allocations_live_llm_credential_idx ON stage_allocations USING btree (((runtime_configuration #>> '{provenance,llmCredential,credentialId}'::text[]))) WHERE ((release_completed_at IS NULL) AND (runtime_configuration IS NOT NULL));

-- Index: stage_allocations_live_runtime_agent_idx

CREATE INDEX stage_allocations_live_runtime_agent_idx ON stage_allocations USING btree (runtime_agent_id, allocation_id) WHERE ((runtime_agent_id IS NOT NULL) AND (release_completed_at IS NULL));

-- Index: stage_allocations_live_runtime_credentials_idx

CREATE INDEX stage_allocations_live_runtime_credentials_idx ON stage_allocations USING gin ((((runtime_configuration -> 'provenance'::text) -> 'runtimeCredentialRefs'::text))) WHERE ((release_completed_at IS NULL) AND (runtime_configuration IS NOT NULL));

-- Index: stage_allocations_pending_release_idx

CREATE INDEX stage_allocations_pending_release_idx ON stage_allocations USING btree (release_attempted_at, stage_execution_id, allocation_id) WHERE (release_completed_at IS NULL);

-- Index: stage_execution_escalation_ordinal_unique

CREATE UNIQUE INDEX stage_execution_escalation_ordinal_unique ON stage_executions USING btree (run_id, stage_name, execution_config_variant, escalation_ordinal) WHERE ((execution_config_variant <> 'base'::text) AND (resume_source_execution_id IS NULL) AND ((execution_config_variant <> 'summary_escalation'::text) OR (((stage_context_snapshot -> 'continuation'::text) ->> 'sourceStageExecutionId'::text) = previous_execution_id)));

-- Index: stage_execution_transition_run_idx

CREATE INDEX stage_execution_transition_run_idx ON stage_executions USING btree (run_id, transition_decided_at, stage_execution_id) WHERE (transition_action IS NOT NULL);

-- Index: stage_executions_run_active_idx

CREATE INDEX stage_executions_run_active_idx ON stage_executions USING btree (run_id, created_at, stage_execution_id) WHERE (state = ANY (ARRAY['preparing'::text, 'running'::text, 'finalizing'::text, 'aborting'::text]));

-- Index: stage_executions_run_created_idx

CREATE INDEX stage_executions_run_created_idx ON stage_executions USING btree (run_id, created_at, stage_execution_id);

-- Index: stage_executions_run_terminal_history_idx

CREATE INDEX stage_executions_run_terminal_history_idx ON stage_executions USING btree (run_id, terminal_at DESC, stage_execution_id DESC) WHERE (terminal_at IS NOT NULL);

-- Index: stage_executions_terminal_history_idx

CREATE INDEX stage_executions_terminal_history_idx ON stage_executions USING btree (terminal_at DESC, stage_execution_id DESC) WHERE (terminal_at IS NOT NULL);

-- Index: stage_metrics_expiry_idx

CREATE INDEX stage_metrics_expiry_idx ON stage_metrics USING btree (expires_at, stage_execution_id);

-- Index: workflow_run_output_publications_project_idx

CREATE INDEX workflow_run_output_publications_project_idx ON workflow_run_output_publications USING btree (project_id, created_at DESC, run_id, output_name);

-- Index: workflow_runs_claim_idx

CREATE INDEX workflow_runs_claim_idx ON workflow_runs USING btree (scheduler_lane_class, scheduler_claim_tier, updated_at, created_at, run_id) WHERE (scheduler_claim_tier IS NOT NULL);

-- Index: workflow_runs_deferred_claim_idx

CREATE INDEX workflow_runs_deferred_claim_idx ON workflow_runs USING btree (scheduler_lane_class, updated_at, created_at, run_id) WHERE (scheduler_deferred AND (scheduler_claim_tier > 0));

-- Index: workflow_runs_engagement_credentials_idx

CREATE INDEX workflow_runs_engagement_credentials_idx ON workflow_runs USING gin (engagement_credential_ids) WHERE (state = ANY (ARRAY['initializing'::text, 'pending'::text, 'running'::text, 'waiting'::text, 'cancelling'::text]));

-- Index: workflow_runs_metadata_labels_idx

CREATE INDEX workflow_runs_metadata_labels_idx ON workflow_runs USING gin (metadata_labels jsonb_path_ops);

-- Index: workflow_runs_owner_created_idx

CREATE INDEX workflow_runs_owner_created_idx ON workflow_runs USING btree (owner_id, created_at, run_id);

-- Index: workflow_runs_owner_idempotency_key

CREATE UNIQUE INDEX workflow_runs_owner_idempotency_key ON workflow_runs USING btree (owner_id, request_idempotency_key) WHERE (request_idempotency_key IS NOT NULL);

-- Index: workflow_runs_owner_nonterminal_queue_idx

CREATE INDEX workflow_runs_owner_nonterminal_queue_idx ON workflow_runs USING btree (owner_id, created_at, run_id) WHERE (state = ANY (ARRAY['initializing'::text, 'pending'::text, 'running'::text, 'waiting'::text, 'cancelling'::text]));

-- Index: workflow_runs_owner_project_active_idx

CREATE INDEX workflow_runs_owner_project_active_idx ON workflow_runs USING btree (owner_id, project_id) WHERE ((project_id IS NOT NULL) AND (state = ANY (ARRAY['initializing'::text, 'pending'::text, 'running'::text, 'waiting'::text, 'cancelling'::text])));

-- Index: workflow_runs_owner_project_created_idx

CREATE INDEX workflow_runs_owner_project_created_idx ON workflow_runs USING btree (owner_id, project_id, created_at DESC, run_id DESC) WHERE (project_id IS NOT NULL);

-- Index: workflow_runs_project_http_target_credential_idx

CREATE INDEX workflow_runs_project_http_target_credential_idx ON workflow_runs USING btree (((project_http_target_snapshot #>> '{credential,credentialId}'::text[])), created_at, run_id) WHERE ((state = ANY (ARRAY['initializing'::text, 'pending'::text, 'running'::text, 'waiting'::text, 'cancelling'::text])) AND (project_http_target_snapshot ? 'credential'::text));

-- Index: workflow_runs_runtime_credentials_idx

CREATE INDEX workflow_runs_runtime_credentials_idx ON workflow_runs USING gin (((runtime_config_snapshot -> 'runtimeCredentialIds'::text))) WHERE (state = ANY (ARRAY['initializing'::text, 'pending'::text, 'running'::text, 'waiting'::text, 'cancelling'::text]));

-- Index: workflow_runs_runtime_llm_credentials_idx

CREATE INDEX workflow_runs_runtime_llm_credentials_idx ON workflow_runs USING gin (((runtime_config_snapshot -> 'llmCredentialIds'::text))) WHERE (state = ANY (ARRAY['initializing'::text, 'pending'::text, 'running'::text, 'waiting'::text, 'cancelling'::text]));

-- Index: workflow_runs_workflow_credentials_idx

CREATE INDEX workflow_runs_workflow_credentials_idx ON workflow_runs USING gin (jsonb_path_query_array(workflow_snapshot, '$.**."credentialId".**'::jsonpath, '{}'::jsonb, true)) WHERE (state = ANY (ARRAY['initializing'::text, 'pending'::text, 'running'::text, 'waiting'::text, 'cancelling'::text]));

-- Trigger: allocation_execution_reports allocation_execution_reports_immutable

CREATE TRIGGER allocation_execution_reports_immutable BEFORE UPDATE ON allocation_execution_reports FOR EACH ROW EXECUTE FUNCTION contractor_protect_telemetry_report_immutable();

-- Trigger: allocation_execution_reports allocation_execution_reports_mark_received

CREATE TRIGGER allocation_execution_reports_mark_received AFTER INSERT ON allocation_execution_reports FOR EACH ROW EXECUTE FUNCTION contractor_mark_allocation_report_received();

-- Trigger: artifact_binding_revisions artifact_binding_revisions_delete

CREATE TRIGGER artifact_binding_revisions_delete BEFORE DELETE ON artifact_binding_revisions FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_revision_delete();

-- Trigger: artifact_binding_revisions artifact_binding_revisions_immutable

CREATE TRIGGER artifact_binding_revisions_immutable BEFORE UPDATE OF scope_kind, scope_id, namespace, name, revision, version_id, created_at ON artifact_binding_revisions FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_immutable();

-- Trigger: artifact_binding_revisions artifact_binding_revisions_run_limit

CREATE TRIGGER artifact_binding_revisions_run_limit AFTER INSERT ON artifact_binding_revisions REFERENCING NEW TABLE AS inserted_revisions FOR EACH STATEMENT EXECUTE FUNCTION contractor_count_run_artifact_revisions();

-- Trigger: artifact_bindings artifact_bindings_audit_admission

CREATE TRIGGER artifact_bindings_audit_admission BEFORE INSERT OR UPDATE ON artifact_bindings FOR EACH ROW WHEN ((new.scope_kind = 'audit'::text)) EXECUTE FUNCTION contractor_guard_audit_artifact_admission();

-- Trigger: artifact_bindings artifact_bindings_project_admission

CREATE TRIGGER artifact_bindings_project_admission BEFORE INSERT OR UPDATE ON artifact_bindings FOR EACH ROW WHEN ((new.scope_kind = 'project'::text)) EXECUTE FUNCTION contractor_guard_project_artifact_binding_admission();

-- Trigger: artifact_bindings artifact_bindings_protect_identity

CREATE TRIGGER artifact_bindings_protect_identity BEFORE UPDATE OF scope_kind, scope_id, namespace, name, created_at, current_revision, frozen ON artifact_bindings FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_binding();

-- Trigger: artifact_blobs artifact_blobs_immutable

CREATE TRIGGER artifact_blobs_immutable BEFORE DELETE OR UPDATE ON artifact_blobs FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_blob_immutable();

-- Trigger: artifact_lineage artifact_lineage_hold

CREATE TRIGGER artifact_lineage_hold AFTER INSERT ON artifact_lineage REFERENCING NEW TABLE AS inserted_lineage FOR EACH STATEMENT EXECUTE FUNCTION contractor_hold_lineage_artifacts();

-- Trigger: artifact_lineage artifact_lineage_immutable

CREATE TRIGGER artifact_lineage_immutable BEFORE DELETE OR UPDATE ON artifact_lineage FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_immutable();

-- Trigger: artifact_lineage artifact_lineage_release

CREATE TRIGGER artifact_lineage_release AFTER DELETE ON artifact_lineage REFERENCING OLD TABLE AS deleted_lineage FOR EACH STATEMENT EXECUTE FUNCTION contractor_hold_lineage_artifacts();

-- Trigger: artifact_lineage artifact_lineage_require_source

CREATE TRIGGER artifact_lineage_require_source BEFORE INSERT ON artifact_lineage FOR EACH ROW EXECUTE FUNCTION contractor_require_lineage_source();

-- Trigger: artifact_scopes artifact_scopes_audit_admission

CREATE TRIGGER artifact_scopes_audit_admission BEFORE INSERT ON artifact_scopes FOR EACH ROW WHEN ((new.scope_kind = 'audit'::text)) EXECUTE FUNCTION contractor_guard_audit_artifact_admission();

-- Trigger: artifact_scopes artifact_scopes_project_admission

CREATE TRIGGER artifact_scopes_project_admission BEFORE INSERT ON artifact_scopes FOR EACH ROW EXECUTE FUNCTION contractor_guard_project_artifact_scope_admission();

-- Trigger: artifact_scopes artifact_scopes_protect_creation

CREATE TRIGGER artifact_scopes_protect_creation BEFORE UPDATE OF created_at ON artifact_scopes FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_scope_created_at();

-- Trigger: artifact_versions artifact_versions_execution_origin_immutable

CREATE TRIGGER artifact_versions_execution_origin_immutable BEFORE UPDATE OF execution_origin ON artifact_versions FOR EACH ROW EXECUTE FUNCTION contractor_protect_execution_origin();

-- Trigger: artifact_versions artifact_versions_immutable

CREATE TRIGGER artifact_versions_immutable BEFORE DELETE OR UPDATE ON artifact_versions FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_version();

-- Trigger: audit_events audit_events_protect_immutable

CREATE TRIGGER audit_events_protect_immutable BEFORE UPDATE ON audit_events FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_event();

-- Trigger: audit_executions audit_execution_protect_prepared_source

CREATE TRIGGER audit_execution_protect_prepared_source BEFORE UPDATE OF prepared_work_source, prepared_work_count, prepared_work_offset ON audit_executions FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_prepared_source();

-- Trigger: audit_executions audit_executions_project_admission

CREATE TRIGGER audit_executions_project_admission BEFORE INSERT ON audit_executions FOR EACH ROW EXECUTE FUNCTION contractor_guard_audit_child_admission();

-- Trigger: audit_executions audit_executions_protect_collection

CREATE TRIGGER audit_executions_protect_collection BEFORE UPDATE OF execution_id, audit_id, run_id, terminal_outcome, terminal_run_generation, terminal_run_sequence, collection_receipt_id, collection_disposition, collection_source_output_ref, collection_source_output_digest, collection_retained_refs, collection_error_code, collection_request_digest, collected_at ON audit_executions FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_execution_collection();

-- Trigger: audit_executions audit_executions_require_matching_run

CREATE CONSTRAINT TRIGGER audit_executions_require_matching_run AFTER INSERT ON audit_executions DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION contractor_require_matching_audit_run();

-- Trigger: audit_executions audit_executions_require_matching_run_on_update

CREATE CONSTRAINT TRIGGER audit_executions_require_matching_run_on_update AFTER UPDATE ON audit_executions DEFERRABLE INITIALLY DEFERRED FOR EACH ROW WHEN (((old.run_id IS DISTINCT FROM new.run_id) OR (old.run_deleted_at IS DISTINCT FROM new.run_deleted_at) OR (old.execution_id IS DISTINCT FROM new.execution_id) OR (old.submission_key IS DISTINCT FROM new.submission_key))) EXECUTE FUNCTION contractor_require_matching_audit_run();

-- Trigger: audit_hypotheses audit_hypotheses_identity_immutable

CREATE TRIGGER audit_hypotheses_identity_immutable BEFORE UPDATE OF hypothesis_id, audit_id, source_key, fixed, ordinal ON audit_hypotheses FOR EACH ROW EXECUTE FUNCTION contractor_protect_immutable_history();

-- Trigger: audit_hypotheses audit_hypothesis_revision_advance

CREATE TRIGGER audit_hypothesis_revision_advance BEFORE UPDATE OF current_revision, current_bytes ON audit_hypotheses FOR EACH ROW EXECUTE FUNCTION contractor_advance_hypothesis_revision();

-- Trigger: audit_hypothesis_revisions audit_hypothesis_revisions_immutable

CREATE TRIGGER audit_hypothesis_revisions_immutable BEFORE UPDATE ON audit_hypothesis_revisions FOR EACH ROW EXECUTE FUNCTION contractor_protect_immutable_history();

-- Trigger: audit_execution_items audit_items_protect_settled_findings

CREATE TRIGGER audit_items_protect_settled_findings BEFORE UPDATE OF finding_result ON audit_execution_items FOR EACH ROW WHEN ((old.state = 'settled'::text)) EXECUTE FUNCTION contractor_protect_immutable_history();

-- Trigger: audit_execution_items audit_items_protect_settled_stop

CREATE TRIGGER audit_items_protect_settled_stop BEFORE UPDATE OF stop_record ON audit_execution_items FOR EACH ROW WHEN ((old.state = 'settled'::text)) EXECUTE FUNCTION contractor_protect_immutable_history();

-- Trigger: audit_execution_items audit_task_attempt_protect_identity

CREATE TRIGGER audit_task_attempt_protect_identity BEFORE UPDATE OF execution_item_id, execution_id, audit_id, task_id, batch_ordinal, item_attempt ON audit_execution_items FOR EACH ROW WHEN (((new.execution_item_id IS DISTINCT FROM old.execution_item_id) OR (new.execution_id IS DISTINCT FROM old.execution_id) OR (new.audit_id IS DISTINCT FROM old.audit_id) OR (new.task_id IS DISTINCT FROM old.task_id) OR (new.batch_ordinal IS DISTINCT FROM old.batch_ordinal) OR (new.item_attempt IS DISTINCT FROM old.item_attempt))) EXECUTE FUNCTION contractor_protect_immutable_history();

-- Trigger: audit_executions audit_task_execution_protect_identity

CREATE TRIGGER audit_task_execution_protect_identity BEFORE UPDATE OF execution_id, audit_id, workflow_role, submission_key, request_digest, lane_class ON audit_executions FOR EACH ROW WHEN (((new.execution_id IS DISTINCT FROM old.execution_id) OR (new.audit_id IS DISTINCT FROM old.audit_id) OR (new.workflow_role IS DISTINCT FROM old.workflow_role) OR (new.submission_key IS DISTINCT FROM old.submission_key) OR (new.request_digest IS DISTINCT FROM old.request_digest) OR (new.lane_class IS DISTINCT FROM old.lane_class))) EXECUTE FUNCTION contractor_protect_immutable_history();

-- Trigger: audit_tasks audit_task_protect_settled_result

CREATE TRIGGER audit_task_protect_settled_result BEFORE UPDATE OF state, attempts, last_execution_id, result, gap_code ON audit_tasks FOR EACH ROW WHEN ((old.state = 'settled'::text)) EXECUTE FUNCTION contractor_protect_immutable_history();

-- Trigger: audit_tasks audit_tasks_hypothesis_immutable

CREATE TRIGGER audit_tasks_hypothesis_immutable BEFORE UPDATE OF hypothesis_snapshot, attention ON audit_tasks FOR EACH ROW EXECUTE FUNCTION contractor_protect_immutable_history();

-- Trigger: audit_tasks audit_tasks_protect_identity

CREATE TRIGGER audit_tasks_protect_identity BEFORE UPDATE OF task_id, audit_id, source_key, source, document, batch_digest, ordinal, iteration ON audit_tasks FOR EACH ROW EXECUTE FUNCTION contractor_protect_immutable_history();

-- Trigger: audit_tasks audit_tasks_protect_settled_findings

CREATE TRIGGER audit_tasks_protect_settled_findings BEFORE UPDATE OF finding_result ON audit_tasks FOR EACH ROW WHEN ((old.state = 'settled'::text)) EXECUTE FUNCTION contractor_protect_immutable_history();

-- Trigger: audit_tasks audit_tasks_protect_settled_stop

CREATE TRIGGER audit_tasks_protect_settled_stop BEFORE UPDATE OF stop_record ON audit_tasks FOR EACH ROW WHEN ((old.state = 'settled'::text)) EXECUTE FUNCTION contractor_protect_immutable_history();

-- Trigger: audits audits_eval_tombstone

CREATE TRIGGER audits_eval_tombstone BEFORE DELETE ON audits FOR EACH ROW EXECUTE FUNCTION contractor_eval_retain_deleted_execution();

-- Trigger: audits audits_hold_artifacts

CREATE TRIGGER audits_hold_artifacts AFTER INSERT ON audits FOR EACH ROW EXECUTE FUNCTION contractor_hold_audit_artifacts();

-- Trigger: audits audits_hold_artifacts_on_update

CREATE TRIGGER audits_hold_artifacts_on_update AFTER UPDATE OF input_selection, baseline_snapshot ON audits FOR EACH ROW WHEN (((old.input_selection IS DISTINCT FROM new.input_selection) OR (old.baseline_snapshot IS DISTINCT FROM new.baseline_snapshot))) EXECUTE FUNCTION contractor_hold_audit_artifacts();

-- Trigger: audits audits_project_admission

CREATE TRIGGER audits_project_admission BEFORE INSERT ON audits FOR EACH ROW EXECUTE FUNCTION contractor_guard_audit_admission();

-- Trigger: audits audits_release_artifacts

CREATE TRIGGER audits_release_artifacts AFTER DELETE ON audits FOR EACH ROW EXECUTE FUNCTION contractor_hold_audit_artifacts();

-- Trigger: credential_operations credential_operations_protect_delete

CREATE TRIGGER credential_operations_protect_delete BEFORE DELETE ON credential_operations FOR EACH ROW EXECUTE FUNCTION contractor_protect_immutable_credential_row();

-- Trigger: credential_operations credential_operations_protect_transition

CREATE TRIGGER credential_operations_protect_transition BEFORE UPDATE ON credential_operations FOR EACH ROW EXECUTE FUNCTION contractor_protect_credential_operation();

-- Trigger: artifact_binding_revisions eval_artifact_projection

CREATE TRIGGER eval_artifact_projection BEFORE DELETE ON artifact_binding_revisions FOR EACH ROW EXECUTE FUNCTION contractor_eval_evidence_deleted();

-- Trigger: audit_executions eval_audit_children_projection

CREATE TRIGGER eval_audit_children_projection AFTER INSERT OR DELETE OR UPDATE ON audit_executions FOR EACH ROW EXECUTE FUNCTION contractor_eval_audit_child_changed();

-- Trigger: audit_artifact_links eval_audit_outputs_projection

CREATE TRIGGER eval_audit_outputs_projection AFTER INSERT OR DELETE OR UPDATE ON audit_artifact_links FOR EACH ROW EXECUTE FUNCTION contractor_eval_audit_child_changed();

-- Trigger: audits eval_audits_projection

CREATE TRIGGER eval_audits_projection AFTER DELETE OR UPDATE OF state, finished_at, dispatch_state ON audits FOR EACH ROW EXECUTE FUNCTION contractor_eval_execution_changed();

-- Trigger: eval_dataset_revisions eval_datasets_collection

CREATE TRIGGER eval_datasets_collection AFTER INSERT OR DELETE ON eval_dataset_revisions FOR EACH ROW EXECUTE FUNCTION contractor_eval_collection_changed();

-- Trigger: eval_dataset_revisions eval_datasets_immutable

CREATE TRIGGER eval_datasets_immutable BEFORE DELETE OR UPDATE ON eval_dataset_revisions FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();

-- Trigger: eval_experiments eval_experiments_collection_delete

CREATE TRIGGER eval_experiments_collection_delete AFTER DELETE ON eval_experiments REFERENCING OLD TABLE AS changed_experiments FOR EACH STATEMENT EXECUTE FUNCTION contractor_eval_collections_changed();

-- Trigger: eval_experiments eval_experiments_collection_insert

CREATE TRIGGER eval_experiments_collection_insert AFTER INSERT ON eval_experiments REFERENCING NEW TABLE AS changed_experiments FOR EACH STATEMENT EXECUTE FUNCTION contractor_eval_collections_changed();

-- Trigger: eval_experiments eval_experiments_collection_update

CREATE TRIGGER eval_experiments_collection_update AFTER UPDATE ON eval_experiments REFERENCING OLD TABLE AS old_experiments NEW TABLE AS new_experiments FOR EACH STATEMENT EXECUTE FUNCTION contractor_eval_collections_updated();

-- Trigger: eval_experiments eval_experiments_protect

CREATE TRIGGER eval_experiments_protect AFTER UPDATE ON eval_experiments FOR EACH ROW EXECUTE FUNCTION contractor_eval_protect_experiment();

-- Trigger: eval_members eval_members_bound

CREATE TRIGGER eval_members_bound BEFORE INSERT ON eval_members FOR EACH ROW EXECUTE FUNCTION contractor_eval_member_bound();

-- Trigger: eval_members eval_members_immutable

CREATE TRIGGER eval_members_immutable BEFORE DELETE OR UPDATE OF experiment_id, member_id, pair_id, ordinal, suite_id, case_id, sample, variant_id, case_sha256, binding_sha256, eligibility, execution_kind, recipe, submission_key, eligibility_reason ON eval_members FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();

-- Trigger: eval_members eval_members_projection

CREATE TRIGGER eval_members_projection AFTER INSERT ON eval_members REFERENCING NEW TABLE AS inserted_members FOR EACH STATEMENT EXECUTE FUNCTION contractor_eval_members_inserted();

-- Trigger: eval_members eval_members_selection_projection

CREATE TRIGGER eval_members_selection_projection AFTER UPDATE OF selected_result_sha256, selected_assessment_sha256, selection_revision, selected_by ON eval_members FOR EACH ROW WHEN ((new.selected_result_sha256 IS NOT NULL)) EXECUTE FUNCTION contractor_eval_member_changed();

-- Trigger: stage_metrics eval_metrics_projection

CREATE TRIGGER eval_metrics_projection AFTER INSERT OR DELETE OR UPDATE OF metrics_schema_version, metrics, summary ON stage_metrics FOR EACH ROW EXECUTE FUNCTION contractor_eval_metrics_changed();

-- Trigger: artifact_bindings eval_outputs_projection

CREATE TRIGGER eval_outputs_projection AFTER INSERT ON artifact_bindings FOR EACH ROW WHEN (((new.scope_kind = 'run'::text) AND (new.namespace = 'outputs'::text))) EXECUTE FUNCTION contractor_eval_output_changed();

-- Trigger: artifact_bindings eval_outputs_projection_on_delete

CREATE TRIGGER eval_outputs_projection_on_delete AFTER DELETE ON artifact_bindings FOR EACH ROW WHEN (((old.scope_kind = 'run'::text) AND (old.namespace = 'outputs'::text))) EXECUTE FUNCTION contractor_eval_output_changed();

-- Trigger: artifact_bindings eval_outputs_projection_on_update

CREATE TRIGGER eval_outputs_projection_on_update AFTER UPDATE ON artifact_bindings FOR EACH ROW WHEN (((new.scope_kind = 'run'::text) AND (new.namespace = 'outputs'::text) AND ((old.current_revision IS DISTINCT FROM new.current_revision) OR (old.frozen IS DISTINCT FROM new.frozen)))) EXECUTE FUNCTION contractor_eval_output_changed();

-- Trigger: eval_experiments eval_plans_complete_matrix

CREATE CONSTRAINT TRIGGER eval_plans_complete_matrix AFTER INSERT OR UPDATE OF plan_document ON eval_experiments DEFERRABLE INITIALLY DEFERRED FOR EACH ROW WHEN ((new.plan_document IS NOT NULL)) EXECUTE FUNCTION contractor_eval_complete_matrix();

-- Trigger: eval_experiments eval_plans_immutable

CREATE TRIGGER eval_plans_immutable AFTER UPDATE OF plan_kind, plan_document, plan_document_sha256, plan_sha256, plan_setup, plan_created_at, plan_readiness ON eval_experiments FOR EACH ROW WHEN ((old.plan_document IS NOT NULL)) EXECUTE FUNCTION contractor_eval_immutable();

-- Trigger: eval_records eval_records_immutable

CREATE TRIGGER eval_records_immutable BEFORE DELETE OR UPDATE ON eval_records FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();

-- Trigger: eval_plan_resources eval_resources_frozen_plan

CREATE TRIGGER eval_resources_frozen_plan BEFORE INSERT ON eval_plan_resources FOR EACH ROW EXECUTE FUNCTION contractor_eval_require_frozen_plan();

-- Trigger: eval_plan_resources eval_resources_immutable

CREATE TRIGGER eval_resources_immutable BEFORE DELETE OR UPDATE ON eval_plan_resources FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();

-- Trigger: workflow_runs eval_runs_projection

CREATE TRIGGER eval_runs_projection AFTER DELETE OR UPDATE OF state, finished_at ON workflow_runs FOR EACH ROW EXECUTE FUNCTION contractor_eval_execution_changed();

-- Trigger: stage_executions eval_stages_projection

CREATE TRIGGER eval_stages_projection AFTER INSERT OR DELETE OR UPDATE OF state ON stage_executions FOR EACH ROW EXECUTE FUNCTION contractor_eval_metrics_changed();

-- Trigger: eval_submissions eval_submissions_projection

CREATE TRIGGER eval_submissions_projection AFTER INSERT ON eval_submissions FOR EACH ROW EXECUTE FUNCTION contractor_eval_member_changed();

-- Trigger: eval_submissions eval_submissions_projection_update

CREATE TRIGGER eval_submissions_projection_update AFTER UPDATE OF state, execution_id, execution_project_id, terminal_state ON eval_submissions FOR EACH ROW WHEN (((old.state IS DISTINCT FROM new.state) OR (old.execution_id IS DISTINCT FROM new.execution_id) OR (old.execution_project_id IS DISTINCT FROM new.execution_project_id) OR (old.terminal_state IS DISTINCT FROM new.terminal_state))) EXECUTE FUNCTION contractor_eval_member_changed();

-- Trigger: eval_submissions eval_submissions_protect

CREATE TRIGGER eval_submissions_protect BEFORE UPDATE ON eval_submissions FOR EACH ROW EXECUTE FUNCTION contractor_eval_protect_submission();

-- Trigger: eval_suboperations eval_suboperations_protect

CREATE TRIGGER eval_suboperations_protect BEFORE UPDATE ON eval_suboperations FOR EACH ROW EXECUTE FUNCTION contractor_eval_protect_suboperation();

-- Trigger: eval_execution_tombstones eval_tombstones_immutable

CREATE TRIGGER eval_tombstones_immutable BEFORE DELETE OR UPDATE ON eval_execution_tombstones FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();

-- Trigger: eval_view_charts eval_view_charts_immutable

CREATE TRIGGER eval_view_charts_immutable BEFORE DELETE OR UPDATE ON eval_view_charts FOR EACH ROW EXECUTE FUNCTION contractor_eval_view_immutable();

-- Trigger: eval_view_pairs eval_view_pairs_immutable

CREATE TRIGGER eval_view_pairs_immutable BEFORE DELETE OR UPDATE ON eval_view_pairs FOR EACH ROW EXECUTE FUNCTION contractor_eval_view_immutable();

-- Trigger: finding_correlation_fences finding_correlation_fences_protect_revision

CREATE TRIGGER finding_correlation_fences_protect_revision BEFORE UPDATE ON finding_correlation_fences FOR EACH ROW EXECUTE FUNCTION contractor_protect_finding_correlation_fence();

-- Trigger: finding_correlation_fences finding_correlation_fences_retain

CREATE TRIGGER finding_correlation_fences_retain BEFORE DELETE ON finding_correlation_fences FOR EACH ROW EXECUTE FUNCTION contractor_retain_finding_correlation_fence();

-- Trigger: finding_proposal_audit_holds finding_proposal_audit_holds_copy_receipt_time

CREATE TRIGGER finding_proposal_audit_holds_copy_receipt_time BEFORE INSERT ON finding_proposal_audit_holds FOR EACH ROW EXECUTE FUNCTION contractor_copy_finding_hold_receipt_time();

-- Trigger: finding_proposal_audit_holds finding_proposal_audit_holds_protect_update

CREATE TRIGGER finding_proposal_audit_holds_protect_update BEFORE UPDATE ON finding_proposal_audit_holds FOR EACH ROW EXECUTE FUNCTION contractor_protect_finding_proposal_receipt();

-- Trigger: finding_proposal_receipts finding_proposal_receipts_hold_artifacts

CREATE TRIGGER finding_proposal_receipts_hold_artifacts AFTER INSERT ON finding_proposal_receipts FOR EACH ROW EXECUTE FUNCTION contractor_hold_finding_receipt_artifacts();

-- Trigger: finding_proposal_receipts finding_proposal_receipts_protect_immutable

CREATE TRIGGER finding_proposal_receipts_protect_immutable BEFORE DELETE OR UPDATE ON finding_proposal_receipts FOR EACH ROW EXECUTE FUNCTION contractor_protect_finding_proposal_receipt_retention();

-- Trigger: finding_proposal_receipts finding_receipt_hypothesis_immutable

CREATE TRIGGER finding_receipt_hypothesis_immutable BEFORE UPDATE OF audit_hypothesis_snapshot ON finding_proposal_receipts FOR EACH ROW EXECUTE FUNCTION contractor_protect_immutable_history();

-- Trigger: finding_proposal_receipts finding_receipts_correlation_source_deleted

CREATE TRIGGER finding_receipts_correlation_source_deleted AFTER UPDATE ON finding_proposal_receipts REFERENCING OLD TABLE AS previous_receipts NEW TABLE AS updated_receipts FOR EACH STATEMENT EXECUTE FUNCTION contractor_finding_correlation_source_deleted();

-- Trigger: finding_proposal_receipts finding_receipts_execution_lane_immutable

CREATE TRIGGER finding_receipts_execution_lane_immutable BEFORE UPDATE OF operation_ref, scan_origin ON finding_proposal_receipts FOR EACH ROW EXECUTE FUNCTION contractor_protect_finding_execution_lane();

-- Trigger: finding_proposal_receipts finding_receipts_initialize_comparison

CREATE TRIGGER finding_receipts_initialize_comparison BEFORE INSERT ON finding_proposal_receipts FOR EACH ROW EXECUTE FUNCTION contractor_initialize_finding_comparison_projection();

-- Trigger: llm_credentials llm_credentials_exclusive_state

CREATE TRIGGER llm_credentials_exclusive_state BEFORE INSERT ON llm_credentials FOR EACH ROW EXECUTE FUNCTION contractor_check_llm_credential_exclusive_state();

-- Trigger: llm_credentials llm_credentials_protect_update

CREATE TRIGGER llm_credentials_protect_update BEFORE UPDATE ON llm_credentials FOR EACH ROW EXECUTE FUNCTION contractor_protect_immutable_credential_row();

-- Trigger: model_selection_settings model_selection_settings_protect_mutation

CREATE TRIGGER model_selection_settings_protect_mutation BEFORE DELETE OR UPDATE ON model_selection_settings FOR EACH ROW EXECUTE FUNCTION contractor_protect_model_selection_settings();

-- Trigger: mutation_receipts mutation_receipts_protect_immutable

CREATE TRIGGER mutation_receipts_protect_immutable BEFORE DELETE OR UPDATE ON mutation_receipts FOR EACH ROW EXECUTE FUNCTION contractor_protect_mutation_receipt();

-- Trigger: owner_queue_controls owner_queue_controls_protect_mutation

CREATE TRIGGER owner_queue_controls_protect_mutation BEFORE UPDATE ON owner_queue_controls FOR EACH ROW EXECUTE FUNCTION contractor_protect_owner_queue_control();

-- Trigger: planner_execution_reports planner_execution_reports_immutable

CREATE TRIGGER planner_execution_reports_immutable BEFORE UPDATE ON planner_execution_reports FOR EACH ROW EXECUTE FUNCTION contractor_protect_telemetry_report_immutable();

-- Trigger: planner_sessions planner_sessions_hold_artifacts

CREATE TRIGGER planner_sessions_hold_artifacts AFTER INSERT ON planner_sessions FOR EACH ROW WHEN ((new.artifact_refs <> '[]'::jsonb)) EXECUTE FUNCTION contractor_hold_planner_session_artifacts();

-- Trigger: planner_sessions planner_sessions_hold_artifacts_on_update

CREATE TRIGGER planner_sessions_hold_artifacts_on_update AFTER UPDATE OF state ON planner_sessions FOR EACH ROW WHEN ((old.artifact_refs IS DISTINCT FROM new.artifact_refs)) EXECUTE FUNCTION contractor_hold_planner_session_artifacts();

-- Trigger: project_finding_sources project_finding_sources_immutable

CREATE TRIGGER project_finding_sources_immutable BEFORE UPDATE ON project_finding_sources FOR EACH ROW EXECUTE FUNCTION contractor_protect_immutable_history();

-- Trigger: project_findings project_findings_protect_identity

CREATE TRIGGER project_findings_protect_identity BEFORE INSERT OR UPDATE ON project_findings FOR EACH ROW EXECUTE FUNCTION contractor_protect_project_finding_identity();

-- Trigger: projects projects_eval_fence

CREATE TRIGGER projects_eval_fence AFTER UPDATE OF lifecycle_state ON projects FOR EACH ROW EXECUTE FUNCTION contractor_eval_fence_project();

-- Trigger: projects projects_initialize_finding_fence

CREATE TRIGGER projects_initialize_finding_fence AFTER INSERT ON projects FOR EACH ROW EXECUTE FUNCTION contractor_initialize_project_finding_fence();

-- Trigger: projects projects_protect_identity

CREATE TRIGGER projects_protect_identity BEFORE UPDATE ON projects FOR EACH ROW EXECUTE FUNCTION contractor_protect_project();

-- Trigger: runtime_agent_principals runtime_agent_principals_protect_mutation

CREATE TRIGGER runtime_agent_principals_protect_mutation BEFORE UPDATE ON runtime_agent_principals FOR EACH ROW EXECUTE FUNCTION contractor_protect_runtime_agent_principal();

-- Trigger: runtime_config_versions runtime_config_versions_protect_immutable

CREATE TRIGGER runtime_config_versions_protect_immutable BEFORE DELETE OR UPDATE ON runtime_config_versions FOR EACH ROW EXECUTE FUNCTION contractor_protect_runtime_config_immutable();

-- Trigger: runtime_credentials runtime_credentials_protect_mutation

CREATE TRIGGER runtime_credentials_protect_mutation BEFORE DELETE OR UPDATE ON runtime_credentials FOR EACH ROW EXECUTE FUNCTION contractor_protect_runtime_credential_immutable();

-- Trigger: runtime_label_bindings runtime_label_bindings_protect_mutation

CREATE TRIGGER runtime_label_bindings_protect_mutation BEFORE DELETE OR UPDATE ON runtime_label_bindings FOR EACH ROW EXECUTE FUNCTION contractor_protect_runtime_label_binding();

-- Trigger: scheduler_settings scheduler_settings_protect_mutation

CREATE TRIGGER scheduler_settings_protect_mutation BEFORE DELETE OR UPDATE ON scheduler_settings FOR EACH ROW EXECUTE FUNCTION contractor_protect_scheduler_settings();

-- Trigger: stage_allocations stage_allocations_protect_identity

CREATE TRIGGER stage_allocations_protect_identity BEFORE UPDATE OF allocation_id, stage_execution_id, logical_agent_name, namespace, agent_template_ref, worker_runtime_ref, runtime_agent_instance_id, runtime_agent_id, runtime_agent_label_revision, runtime_configuration_schema_version, runtime_configuration, created_at ON stage_allocations FOR EACH ROW EXECUTE FUNCTION contractor_protect_stage_allocation_identity();

-- Trigger: stage_allocations stage_allocations_protect_performance_policy

CREATE TRIGGER stage_allocations_protect_performance_policy BEFORE UPDATE OF performance_collection_policy ON stage_allocations FOR EACH ROW EXECUTE FUNCTION contractor_protect_stage_allocation_performance_policy();

-- Trigger: stage_allocations stage_allocations_protect_placement_labels

CREATE TRIGGER stage_allocations_protect_placement_labels BEFORE UPDATE OF runtime_placement_labels ON stage_allocations FOR EACH ROW WHEN ((new.runtime_placement_labels IS DISTINCT FROM old.runtime_placement_labels)) EXECUTE FUNCTION contractor_protect_placement_labels();

-- Trigger: stage_allocations stage_allocations_require_completion

CREATE TRIGGER stage_allocations_require_completion BEFORE INSERT OR UPDATE OF completion_contract ON stage_allocations FOR EACH ROW EXECUTE FUNCTION contractor_require_allocation_completion();

-- Trigger: stage_executions stage_executions_capture_lifecycle_event

CREATE TRIGGER stage_executions_capture_lifecycle_event AFTER INSERT ON stage_executions FOR EACH ROW EXECUTE FUNCTION contractor_capture_stage_execution_lifecycle_event();

-- Trigger: stage_executions stage_executions_capture_lifecycle_event_on_update

CREATE TRIGGER stage_executions_capture_lifecycle_event_on_update AFTER UPDATE ON stage_executions FOR EACH ROW WHEN ((old.state IS DISTINCT FROM new.state)) EXECUTE FUNCTION contractor_capture_stage_execution_lifecycle_event();

-- Trigger: stage_executions stage_executions_hold_artifacts

CREATE TRIGGER stage_executions_hold_artifacts AFTER INSERT ON stage_executions FOR EACH ROW EXECUTE FUNCTION contractor_hold_stage_execution_artifacts();

-- Trigger: stage_executions stage_executions_hold_artifacts_on_update

CREATE TRIGGER stage_executions_hold_artifacts_on_update AFTER UPDATE OF stage_context_snapshot, candidate_stage_result, accepted_stage_result ON stage_executions FOR EACH ROW WHEN (((old.stage_context_snapshot IS DISTINCT FROM new.stage_context_snapshot) OR (old.candidate_stage_result IS DISTINCT FROM new.candidate_stage_result) OR (old.accepted_stage_result IS DISTINCT FROM new.accepted_stage_result))) EXECUTE FUNCTION contractor_hold_stage_execution_artifacts();

-- Trigger: stage_executions stage_executions_manual_continuation

CREATE TRIGGER stage_executions_manual_continuation BEFORE INSERT OR UPDATE OF resume_source_execution_id, resume_requested_by, resume_requested_at, resume_transaction_id ON stage_executions FOR EACH ROW EXECUTE FUNCTION contractor_validate_manual_stage_execution();

-- Trigger: stage_executions stage_executions_protect_immutable

CREATE TRIGGER stage_executions_protect_immutable BEFORE UPDATE OF stage_execution_id, run_id, stage_name, attempt, previous_execution_id, execution_config_variant, escalation_ordinal, stage_spec_schema_version, stage_spec_snapshot, stage_context_schema_version, stage_context_snapshot, created_at, planner_session_id, planner_invocation_id, candidate_stage_result, candidate_result_schema_version, accepted_stage_result, accepted_result_schema_version, stage_termination, termination_schema_version, finalization_id, finalization_deadline, abort_id, abort_deadline ON stage_executions FOR EACH ROW EXECUTE FUNCTION contractor_protect_stage_execution_immutable();

-- Trigger: stage_executions stage_executions_transition_immutable

CREATE TRIGGER stage_executions_transition_immutable BEFORE UPDATE OF transition_action, transition_target_stage_name, transition_target_execution_id, transition_escalation_ordinal, transition_escalation_exhausted, transition_decided_at ON stage_executions FOR EACH ROW EXECUTE FUNCTION contractor_protect_stage_transition();

-- Trigger: workflow_run_events workflow_run_events_notify_after_commit

CREATE TRIGGER workflow_run_events_notify_after_commit AFTER INSERT ON workflow_run_events FOR EACH ROW EXECUTE FUNCTION contractor_notify_workflow_run_event();

-- Trigger: workflow_run_events workflow_run_events_protect_immutable

CREATE TRIGGER workflow_run_events_protect_immutable BEFORE UPDATE ON workflow_run_events FOR EACH ROW EXECUTE FUNCTION contractor_protect_workflow_run_event_immutable();

-- Trigger: workflow_run_output_publications workflow_run_output_publications_hold

CREATE TRIGGER workflow_run_output_publications_hold AFTER INSERT ON workflow_run_output_publications REFERENCING NEW TABLE AS inserted_publications FOR EACH STATEMENT EXECUTE FUNCTION contractor_hold_output_publication_artifacts();

-- Trigger: workflow_run_output_publications workflow_run_output_publications_immutable

CREATE TRIGGER workflow_run_output_publications_immutable BEFORE DELETE OR UPDATE ON workflow_run_output_publications FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_immutable();

-- Trigger: workflow_run_output_publications workflow_run_output_publications_release

CREATE TRIGGER workflow_run_output_publications_release AFTER DELETE ON workflow_run_output_publications REFERENCING OLD TABLE AS deleted_publications FOR EACH STATEMENT EXECUTE FUNCTION contractor_hold_output_publication_artifacts();

-- Trigger: workflow_runs workflow_runs_capture_lifecycle_event

CREATE TRIGGER workflow_runs_capture_lifecycle_event AFTER INSERT ON workflow_runs FOR EACH ROW EXECUTE FUNCTION contractor_capture_workflow_run_lifecycle_event();

-- Trigger: workflow_runs workflow_runs_capture_lifecycle_event_on_update

CREATE TRIGGER workflow_runs_capture_lifecycle_event_on_update AFTER UPDATE ON workflow_runs FOR EACH ROW WHEN ((old.state IS DISTINCT FROM new.state)) EXECUTE FUNCTION contractor_capture_workflow_run_lifecycle_event();

-- Trigger: workflow_runs workflow_runs_eval_tombstone

CREATE TRIGGER workflow_runs_eval_tombstone BEFORE DELETE ON workflow_runs FOR EACH ROW WHEN (((old.project_id IS NOT NULL) AND (old.request_idempotency_key IS NOT NULL) AND (old.request_digest IS NOT NULL))) EXECUTE FUNCTION contractor_eval_retain_deleted_execution();

-- Trigger: workflow_runs workflow_runs_hold_audit_inputs_on_insert

CREATE TRIGGER workflow_runs_hold_audit_inputs_on_insert AFTER INSERT ON workflow_runs FOR EACH ROW WHEN ((new.audit_input_materialization IS NOT NULL)) EXECUTE FUNCTION contractor_hold_audit_run_inputs();

-- Trigger: workflow_runs workflow_runs_hold_audit_inputs_on_update

CREATE TRIGGER workflow_runs_hold_audit_inputs_on_update AFTER UPDATE OF audit_input_materialization ON workflow_runs FOR EACH ROW WHEN (((old.audit_input_materialization)::text IS DISTINCT FROM (new.audit_input_materialization)::text)) EXECUTE FUNCTION contractor_hold_audit_run_inputs();

-- Trigger: workflow_runs workflow_runs_hold_skill_sources

CREATE TRIGGER workflow_runs_hold_skill_sources AFTER INSERT ON workflow_runs FOR EACH ROW WHEN ((new.skill_snapshot <> '[]'::jsonb)) EXECUTE FUNCTION contractor_hold_run_skill_artifacts();

-- Trigger: workflow_runs workflow_runs_hold_skill_sources_on_update

CREATE TRIGGER workflow_runs_hold_skill_sources_on_update AFTER UPDATE OF skill_snapshot ON workflow_runs FOR EACH ROW WHEN ((old.skill_snapshot IS DISTINCT FROM new.skill_snapshot)) EXECUTE FUNCTION contractor_hold_run_skill_artifacts();

-- Trigger: workflow_runs workflow_runs_project_admission

CREATE TRIGGER workflow_runs_project_admission BEFORE INSERT ON workflow_runs FOR EACH ROW EXECUTE FUNCTION contractor_guard_workflow_run_project_admission();

-- Trigger: workflow_runs workflow_runs_protect_audit_completion

CREATE TRIGGER workflow_runs_protect_audit_completion BEFORE UPDATE OF audit_completion ON workflow_runs FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_completion();

-- Trigger: workflow_runs workflow_runs_protect_audit_input_materialization

CREATE TRIGGER workflow_runs_protect_audit_input_materialization BEFORE UPDATE OF audit_input_materialization ON workflow_runs FOR EACH ROW WHEN (((old.audit_input_materialization)::text IS DISTINCT FROM (new.audit_input_materialization)::text)) EXECUTE FUNCTION contractor_protect_audit_input_materialization();

-- Trigger: workflow_runs workflow_runs_protect_engagement_credentials

CREATE TRIGGER workflow_runs_protect_engagement_credentials BEFORE UPDATE OF engagement_credential_ids ON workflow_runs FOR EACH ROW WHEN ((old.engagement_credential_ids IS DISTINCT FROM new.engagement_credential_ids)) EXECUTE FUNCTION contractor_protect_artifact_immutable();

-- Trigger: workflow_runs workflow_runs_protect_immutable

CREATE TRIGGER workflow_runs_protect_immutable BEFORE UPDATE OF run_id, owner_id, project_id, workflow_name, workflow_version, workflow_schema_version, workflow_snapshot, parameters, metadata_labels, request_idempotency_key, request_digest, run_event_generation, runtime_labels, runtime_config_snapshot, project_http_target_snapshot, publication_mode, audit_execution_id, audit_submission_key, created_at, cancellation_schema_version, run_cancellation ON workflow_runs FOR EACH ROW EXECUTE FUNCTION contractor_protect_workflow_run_immutable();

-- Trigger: workflow_runs workflow_runs_protect_placement_labels

CREATE TRIGGER workflow_runs_protect_placement_labels BEFORE UPDATE OF required_placement_labels ON workflow_runs FOR EACH ROW WHEN ((new.required_placement_labels IS DISTINCT FROM old.required_placement_labels)) EXECUTE FUNCTION contractor_protect_placement_labels();

-- Trigger: workflow_runs workflow_runs_protect_skill_snapshot

CREATE TRIGGER workflow_runs_protect_skill_snapshot BEFORE UPDATE OF skill_snapshot ON workflow_runs FOR EACH ROW EXECUTE FUNCTION contractor_protect_workflow_run_skill_snapshot();

-- Trigger: workflow_runs workflow_runs_release_audit_inputs

CREATE TRIGGER workflow_runs_release_audit_inputs AFTER DELETE ON workflow_runs FOR EACH ROW WHEN ((old.audit_input_materialization IS NOT NULL)) EXECUTE FUNCTION contractor_hold_audit_run_inputs();

-- Trigger: workflow_runs workflow_runs_release_skill_sources

CREATE TRIGGER workflow_runs_release_skill_sources AFTER DELETE ON workflow_runs FOR EACH ROW WHEN ((old.skill_snapshot <> '[]'::jsonb)) EXECUTE FUNCTION contractor_hold_run_skill_artifacts();

-- Trigger: workflow_runs workflow_runs_require_complete_audit_binding

CREATE CONSTRAINT TRIGGER workflow_runs_require_complete_audit_binding AFTER INSERT OR UPDATE ON workflow_runs DEFERRABLE INITIALLY DEFERRED FOR EACH ROW WHEN ((new.publication_mode = 'audit-managed'::text)) EXECUTE FUNCTION contractor_require_complete_audit_run_binding();

-- Fk Constraint: allocation_execution_reports allocation_execution_reports_stage_execution_id_allocation_fkey

ALTER TABLE ONLY allocation_execution_reports
    ADD CONSTRAINT allocation_execution_reports_stage_execution_id_allocation_fkey FOREIGN KEY (stage_execution_id, allocation_id, logical_agent_name) REFERENCES stage_allocations(stage_execution_id, allocation_id, logical_agent_name) ON DELETE CASCADE;

-- Fk Constraint: artifact_binding_revisions artifact_binding_revisions_scope_kind_scope_id_fkey

ALTER TABLE ONLY artifact_binding_revisions
    ADD CONSTRAINT artifact_binding_revisions_scope_kind_scope_id_fkey FOREIGN KEY (scope_kind, scope_id) REFERENCES artifact_scopes(scope_kind, scope_id);

-- Fk Constraint: artifact_binding_revisions artifact_binding_revisions_version_id_fkey

ALTER TABLE ONLY artifact_binding_revisions
    ADD CONSTRAINT artifact_binding_revisions_version_id_fkey FOREIGN KEY (version_id) REFERENCES artifact_versions(version_id);

-- Fk Constraint: artifact_bindings artifact_bindings_scope_kind_scope_id_fkey

ALTER TABLE ONLY artifact_bindings
    ADD CONSTRAINT artifact_bindings_scope_kind_scope_id_fkey FOREIGN KEY (scope_kind, scope_id) REFERENCES artifact_scopes(scope_kind, scope_id);

-- Fk Constraint: artifact_bindings artifact_bindings_scope_kind_scope_id_namespace_name_curre_fkey

ALTER TABLE ONLY artifact_bindings
    ADD CONSTRAINT artifact_bindings_scope_kind_scope_id_namespace_name_curre_fkey FOREIGN KEY (scope_kind, scope_id, namespace, name, current_revision) REFERENCES artifact_binding_revisions(scope_kind, scope_id, namespace, name, revision) DEFERRABLE INITIALLY DEFERRED;

-- Fk Constraint: artifact_lineage artifact_lineage_target_scope_kind_target_scope_id_target__fkey

ALTER TABLE ONLY artifact_lineage
    ADD CONSTRAINT artifact_lineage_target_scope_kind_target_scope_id_target__fkey FOREIGN KEY (target_scope_kind, target_scope_id, target_namespace, target_name, target_revision) REFERENCES artifact_binding_revisions(scope_kind, scope_id, namespace, name, revision);

-- Fk Constraint: artifact_scopes artifact_scopes_audit_id_fkey

ALTER TABLE ONLY artifact_scopes
    ADD CONSTRAINT artifact_scopes_audit_id_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id);

-- Fk Constraint: artifact_scopes artifact_scopes_project_id_fkey

ALTER TABLE ONLY artifact_scopes
    ADD CONSTRAINT artifact_scopes_project_id_fkey FOREIGN KEY (project_id) REFERENCES projects(project_id);

-- Fk Constraint: artifact_scopes artifact_scopes_run_id_fkey

ALTER TABLE ONLY artifact_scopes
    ADD CONSTRAINT artifact_scopes_run_id_fkey FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id);

-- Fk Constraint: artifact_versions artifact_versions_blob_sha256_fkey

ALTER TABLE ONLY artifact_versions
    ADD CONSTRAINT artifact_versions_blob_sha256_fkey FOREIGN KEY (blob_sha256) REFERENCES artifact_blobs(sha256);

-- Fk Constraint: audit_artifact_links audit_artifact_links_audit_id_fkey

ALTER TABLE ONLY audit_artifact_links
    ADD CONSTRAINT audit_artifact_links_audit_id_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

-- Fk Constraint: audit_controller_claims audit_controller_claims_audit_id_fkey

ALTER TABLE ONLY audit_controller_claims
    ADD CONSTRAINT audit_controller_claims_audit_id_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

-- Fk Constraint: audit_events audit_events_audit_id_fkey

ALTER TABLE ONLY audit_events
    ADD CONSTRAINT audit_events_audit_id_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

-- Fk Constraint: audit_execution_items audit_execution_items_common_execution_fk

ALTER TABLE ONLY audit_execution_items
    ADD CONSTRAINT audit_execution_items_common_execution_fk FOREIGN KEY (execution_id, audit_id) REFERENCES audit_executions(execution_id, audit_id) ON DELETE CASCADE;

-- Fk Constraint: audit_execution_items audit_execution_items_task_fk

ALTER TABLE ONLY audit_execution_items
    ADD CONSTRAINT audit_execution_items_task_fk FOREIGN KEY (task_id, audit_id) REFERENCES audit_tasks(task_id, audit_id) ON DELETE CASCADE;

-- Fk Constraint: audit_executions audit_executions_audit_id_fkey

ALTER TABLE ONLY audit_executions
    ADD CONSTRAINT audit_executions_audit_id_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

-- Fk Constraint: audit_hypotheses audit_hypotheses_audit_id_fkey

ALTER TABLE ONLY audit_hypotheses
    ADD CONSTRAINT audit_hypotheses_audit_id_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

-- Fk Constraint: audit_hypothesis_revisions audit_hypothesis_revisions_hypothesis_id_audit_id_fkey

ALTER TABLE ONLY audit_hypothesis_revisions
    ADD CONSTRAINT audit_hypothesis_revisions_hypothesis_id_audit_id_fkey FOREIGN KEY (hypothesis_id, audit_id) REFERENCES audit_hypotheses(hypothesis_id, audit_id) ON DELETE CASCADE;

-- Fk Constraint: audit_tasks audit_tasks_audit_id_fkey

ALTER TABLE ONLY audit_tasks
    ADD CONSTRAINT audit_tasks_audit_id_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

-- Fk Constraint: audit_tasks audit_tasks_last_execution_fk

ALTER TABLE ONLY audit_tasks
    ADD CONSTRAINT audit_tasks_last_execution_fk FOREIGN KEY (last_execution_id, audit_id) REFERENCES audit_executions(execution_id, audit_id) DEFERRABLE INITIALLY DEFERRED;

-- Fk Constraint: audits audits_project_owner_fkey

ALTER TABLE ONLY audits
    ADD CONSTRAINT audits_project_owner_fkey FOREIGN KEY (project_id, owner_id) REFERENCES projects(project_id, owner_id) ON DELETE RESTRICT;

-- Fk Constraint: eval_commands eval_commands_experiment_id_fkey

ALTER TABLE ONLY eval_commands
    ADD CONSTRAINT eval_commands_experiment_id_fkey FOREIGN KEY (experiment_id) REFERENCES eval_experiments(experiment_id) ON DELETE CASCADE;

-- Fk Constraint: eval_controller_claims eval_controller_claims_experiment_id_fkey

ALTER TABLE ONLY eval_controller_claims
    ADD CONSTRAINT eval_controller_claims_experiment_id_fkey FOREIGN KEY (experiment_id) REFERENCES eval_experiments(experiment_id) ON DELETE CASCADE;

-- Fk Constraint: eval_dataset_revisions eval_dataset_revisions_owner_id_project_id_fkey

ALTER TABLE ONLY eval_dataset_revisions
    ADD CONSTRAINT eval_dataset_revisions_owner_id_project_id_fkey FOREIGN KEY (owner_id, project_id) REFERENCES projects(owner_id, project_id) ON DELETE RESTRICT;

-- Fk Constraint: eval_evidence_refs eval_evidence_refs_experiment_id_member_id_fkey

ALTER TABLE ONLY eval_evidence_refs
    ADD CONSTRAINT eval_evidence_refs_experiment_id_member_id_fkey FOREIGN KEY (experiment_id, member_id) REFERENCES eval_members(experiment_id, member_id) ON DELETE CASCADE;

-- Fk Constraint: eval_execution_tombstones eval_execution_tombstones_experiment_id_member_id_fkey

ALTER TABLE ONLY eval_execution_tombstones
    ADD CONSTRAINT eval_execution_tombstones_experiment_id_member_id_fkey FOREIGN KEY (experiment_id, member_id) REFERENCES eval_members(experiment_id, member_id) ON DELETE CASCADE;

-- Fk Constraint: eval_experiments eval_experiments_owner_id_project_id_fkey

ALTER TABLE ONLY eval_experiments
    ADD CONSTRAINT eval_experiments_owner_id_project_id_fkey FOREIGN KEY (owner_id, project_id) REFERENCES projects(owner_id, project_id) ON DELETE RESTRICT;

-- Fk Constraint: eval_experiments eval_experiments_project_id_dataset_id_dataset_revision_fkey

ALTER TABLE ONLY eval_experiments
    ADD CONSTRAINT eval_experiments_project_id_dataset_id_dataset_revision_fkey FOREIGN KEY (project_id, dataset_id, dataset_revision) REFERENCES eval_dataset_revisions(project_id, dataset_id, revision) ON DELETE RESTRICT;

-- Fk Constraint: eval_member_projections eval_member_projections_experiment_id_member_id_fkey

ALTER TABLE ONLY eval_member_projections
    ADD CONSTRAINT eval_member_projections_experiment_id_member_id_fkey FOREIGN KEY (experiment_id, member_id) REFERENCES eval_members(experiment_id, member_id) ON DELETE CASCADE;

-- Fk Constraint: eval_members eval_members_experiment_id_fkey

ALTER TABLE ONLY eval_members
    ADD CONSTRAINT eval_members_experiment_id_fkey FOREIGN KEY (experiment_id) REFERENCES eval_experiments(experiment_id) ON DELETE CASCADE;

-- Fk Constraint: eval_members eval_members_selected_assessment_fk

ALTER TABLE ONLY eval_members
    ADD CONSTRAINT eval_members_selected_assessment_fk FOREIGN KEY (experiment_id, member_id, selected_assessment_kind, selected_assessment_sha256) REFERENCES eval_records(experiment_id, member_id, kind, record_sha256);

-- Fk Constraint: eval_members eval_members_selected_result_fk

ALTER TABLE ONLY eval_members
    ADD CONSTRAINT eval_members_selected_result_fk FOREIGN KEY (experiment_id, member_id, selected_result_kind, selected_result_sha256) REFERENCES eval_records(experiment_id, member_id, kind, record_sha256);

-- Fk Constraint: eval_plan_resources eval_plan_resources_experiment_id_fkey

ALTER TABLE ONLY eval_plan_resources
    ADD CONSTRAINT eval_plan_resources_experiment_id_fkey FOREIGN KEY (experiment_id) REFERENCES eval_experiments(experiment_id) ON DELETE CASCADE;

-- Fk Constraint: eval_progress_observations eval_progress_observations_experiment_id_fkey

ALTER TABLE ONLY eval_progress_observations
    ADD CONSTRAINT eval_progress_observations_experiment_id_fkey FOREIGN KEY (experiment_id) REFERENCES eval_experiments(experiment_id) ON DELETE CASCADE;

-- Fk Constraint: eval_projection_queue eval_projection_queue_experiment_id_fkey

ALTER TABLE ONLY eval_projection_queue
    ADD CONSTRAINT eval_projection_queue_experiment_id_fkey FOREIGN KEY (experiment_id) REFERENCES eval_experiments(experiment_id) ON DELETE CASCADE;

-- Fk Constraint: eval_records eval_records_experiment_id_member_id_fkey

ALTER TABLE ONLY eval_records
    ADD CONSTRAINT eval_records_experiment_id_member_id_fkey FOREIGN KEY (experiment_id, member_id) REFERENCES eval_members(experiment_id, member_id) ON DELETE CASCADE;

-- Fk Constraint: eval_submissions eval_submissions_experiment_id_member_id_execution_kind_fkey

ALTER TABLE ONLY eval_submissions
    ADD CONSTRAINT eval_submissions_experiment_id_member_id_execution_kind_fkey FOREIGN KEY (experiment_id, member_id, execution_kind) REFERENCES eval_members(experiment_id, member_id, execution_kind) ON DELETE CASCADE;

-- Fk Constraint: eval_suboperations eval_suboperations_experiment_id_member_id_fkey

ALTER TABLE ONLY eval_suboperations
    ADD CONSTRAINT eval_suboperations_experiment_id_member_id_fkey FOREIGN KEY (experiment_id, member_id) REFERENCES eval_submissions(experiment_id, member_id) ON DELETE CASCADE;

-- Fk Constraint: eval_view_charts eval_view_charts_experiment_id_fkey

ALTER TABLE ONLY eval_view_charts
    ADD CONSTRAINT eval_view_charts_experiment_id_fkey FOREIGN KEY (experiment_id) REFERENCES eval_projection_queue(experiment_id) ON DELETE CASCADE;

-- Fk Constraint: eval_view_pairs eval_view_pairs_experiment_id_fkey

ALTER TABLE ONLY eval_view_pairs
    ADD CONSTRAINT eval_view_pairs_experiment_id_fkey FOREIGN KEY (experiment_id) REFERENCES eval_projection_queue(experiment_id) ON DELETE CASCADE;

-- Fk Constraint: finding_correlation_fences finding_correlation_fences_project_id_fkey

ALTER TABLE ONLY finding_correlation_fences
    ADD CONSTRAINT finding_correlation_fences_project_id_fkey FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE;

-- Fk Constraint: finding_proposal_audit_holds finding_proposal_audit_holds_audit_id_fkey

ALTER TABLE ONLY finding_proposal_audit_holds
    ADD CONSTRAINT finding_proposal_audit_holds_audit_id_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

-- Fk Constraint: finding_proposal_audit_holds finding_proposal_audit_holds_project_id_fkey

ALTER TABLE ONLY finding_proposal_audit_holds
    ADD CONSTRAINT finding_proposal_audit_holds_project_id_fkey FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE;

-- Fk Constraint: finding_proposal_audit_holds finding_proposal_audit_holds_receipt_id_fkey

ALTER TABLE ONLY finding_proposal_audit_holds
    ADD CONSTRAINT finding_proposal_audit_holds_receipt_id_fkey FOREIGN KEY (receipt_id) REFERENCES finding_proposal_receipts(receipt_id) ON DELETE CASCADE;

-- Fk Constraint: mutation_receipts mutation_receipts_audit_id_fkey

ALTER TABLE ONLY mutation_receipts
    ADD CONSTRAINT mutation_receipts_audit_id_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

-- Fk Constraint: mutation_receipts mutation_receipts_project_id_fkey

ALTER TABLE ONLY mutation_receipts
    ADD CONSTRAINT mutation_receipts_project_id_fkey FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE RESTRICT;

-- Fk Constraint: planner_execution_reports planner_execution_reports_session_id_fkey

ALTER TABLE ONLY planner_execution_reports
    ADD CONSTRAINT planner_execution_reports_session_id_fkey FOREIGN KEY (session_id) REFERENCES planner_sessions(session_id) ON DELETE CASCADE;

-- Fk Constraint: planner_execution_reports planner_execution_reports_stage_execution_id_fkey

ALTER TABLE ONLY planner_execution_reports
    ADD CONSTRAINT planner_execution_reports_stage_execution_id_fkey FOREIGN KEY (stage_execution_id) REFERENCES stage_executions(stage_execution_id) ON DELETE CASCADE;

-- Fk Constraint: planner_sessions planner_sessions_stage_execution_id_fkey

ALTER TABLE ONLY planner_sessions
    ADD CONSTRAINT planner_sessions_stage_execution_id_fkey FOREIGN KEY (stage_execution_id) REFERENCES stage_executions(stage_execution_id) ON DELETE CASCADE;

-- Fk Constraint: project_finding_audits project_finding_audits_audit_id_fkey

ALTER TABLE ONLY project_finding_audits
    ADD CONSTRAINT project_finding_audits_audit_id_fkey FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

-- Fk Constraint: project_finding_audits project_finding_audits_project_id_finding_id_fkey

ALTER TABLE ONLY project_finding_audits
    ADD CONSTRAINT project_finding_audits_project_id_finding_id_fkey FOREIGN KEY (project_id, finding_id) REFERENCES project_findings(project_id, finding_id) ON DELETE CASCADE;

-- Fk Constraint: project_finding_sources project_finding_sources_project_id_finding_id_fkey

ALTER TABLE ONLY project_finding_sources
    ADD CONSTRAINT project_finding_sources_project_id_finding_id_fkey FOREIGN KEY (project_id, finding_id) REFERENCES project_findings(project_id, finding_id) ON DELETE RESTRICT;

-- Fk Constraint: project_finding_sources project_finding_sources_receipt_id_fkey

ALTER TABLE ONLY project_finding_sources
    ADD CONSTRAINT project_finding_sources_receipt_id_fkey FOREIGN KEY (receipt_id) REFERENCES finding_proposal_receipts(receipt_id) ON DELETE RESTRICT;

-- Fk Constraint: project_findings project_findings_first_receipt_id_fkey

ALTER TABLE ONLY project_findings
    ADD CONSTRAINT project_findings_first_receipt_id_fkey FOREIGN KEY (first_receipt_id) REFERENCES finding_proposal_receipts(receipt_id) ON DELETE RESTRICT;

-- Fk Constraint: project_findings project_findings_project_id_duplicate_target_id_fkey

ALTER TABLE ONLY project_findings
    ADD CONSTRAINT project_findings_project_id_duplicate_target_id_fkey FOREIGN KEY (project_id, duplicate_target_id) REFERENCES project_findings(project_id, finding_id) DEFERRABLE INITIALLY DEFERRED;

-- Fk Constraint: project_findings project_findings_project_id_fkey

ALTER TABLE ONLY project_findings
    ADD CONSTRAINT project_findings_project_id_fkey FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE RESTRICT;

-- Fk Constraint: projects projects_http_target_credential_fkey

ALTER TABLE ONLY projects
    ADD CONSTRAINT projects_http_target_credential_fkey FOREIGN KEY (http_target_credential_id, http_target_credential_kind) REFERENCES runtime_credentials(credential_id, credential_kind);

-- Fk Constraint: runtime_label_bindings runtime_label_bindings_config_name_config_version_config_d_fkey

ALTER TABLE ONLY runtime_label_bindings
    ADD CONSTRAINT runtime_label_bindings_config_name_config_version_config_d_fkey FOREIGN KEY (config_name, config_version, config_digest) REFERENCES runtime_config_versions(name, version, digest);

-- Fk Constraint: stage_allocations stage_allocations_stage_execution_id_fkey

ALTER TABLE ONLY stage_allocations
    ADD CONSTRAINT stage_allocations_stage_execution_id_fkey FOREIGN KEY (stage_execution_id) REFERENCES stage_executions(stage_execution_id) ON DELETE CASCADE;

-- Fk Constraint: stage_executions stage_execution_lineage

ALTER TABLE ONLY stage_executions
    ADD CONSTRAINT stage_execution_lineage FOREIGN KEY (run_id, stage_name, previous_execution_id) REFERENCES stage_executions(run_id, stage_name, stage_execution_id) DEFERRABLE INITIALLY DEFERRED;

-- Fk Constraint: stage_executions stage_execution_planner_session

ALTER TABLE ONLY stage_executions
    ADD CONSTRAINT stage_execution_planner_session FOREIGN KEY (stage_execution_id, planner_session_id, planner_invocation_id) REFERENCES planner_sessions(stage_execution_id, session_id, invocation_id) DEFERRABLE INITIALLY DEFERRED;

-- Fk Constraint: stage_executions stage_executions_run_id_fkey

ALTER TABLE ONLY stage_executions
    ADD CONSTRAINT stage_executions_run_id_fkey FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id) ON DELETE CASCADE;

-- Fk Constraint: stage_metrics stage_metrics_stage_execution_id_fkey

ALTER TABLE ONLY stage_metrics
    ADD CONSTRAINT stage_metrics_stage_execution_id_fkey FOREIGN KEY (stage_execution_id) REFERENCES stage_executions(stage_execution_id) ON DELETE CASCADE;

-- Fk Constraint: stage_executions stage_transition_target_run_stage

ALTER TABLE ONLY stage_executions
    ADD CONSTRAINT stage_transition_target_run_stage FOREIGN KEY (run_id, transition_target_stage_name, transition_target_execution_id) REFERENCES stage_executions(run_id, stage_name, stage_execution_id) DEFERRABLE INITIALLY DEFERRED;

-- Fk Constraint: workflow_run_events workflow_run_events_run_id_fkey

ALTER TABLE ONLY workflow_run_events
    ADD CONSTRAINT workflow_run_events_run_id_fkey FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id) ON DELETE CASCADE;

-- Fk Constraint: workflow_run_output_publications workflow_run_output_publicati_source_scope_kind_run_id_sou_fkey

ALTER TABLE ONLY workflow_run_output_publications
    ADD CONSTRAINT workflow_run_output_publicati_source_scope_kind_run_id_sou_fkey FOREIGN KEY (source_scope_kind, run_id, source_namespace, source_name, source_revision) REFERENCES artifact_binding_revisions(scope_kind, scope_id, namespace, name, revision);

-- Fk Constraint: workflow_run_output_publications workflow_run_output_publicati_target_scope_kind_project_id_fkey

ALTER TABLE ONLY workflow_run_output_publications
    ADD CONSTRAINT workflow_run_output_publicati_target_scope_kind_project_id_fkey FOREIGN KEY (target_scope_kind, project_id, target_namespace, target_name, target_revision) REFERENCES artifact_binding_revisions(scope_kind, scope_id, namespace, name, revision);

-- Fk Constraint: workflow_run_output_publications workflow_run_output_publications_run_id_project_id_fkey

ALTER TABLE ONLY workflow_run_output_publications
    ADD CONSTRAINT workflow_run_output_publications_run_id_project_id_fkey FOREIGN KEY (run_id, project_id) REFERENCES workflow_runs(run_id, project_id);

-- Fk Constraint: workflow_runs workflow_runs_audit_submission_fkey

ALTER TABLE ONLY workflow_runs
    ADD CONSTRAINT workflow_runs_audit_submission_fkey FOREIGN KEY (audit_execution_id, audit_submission_key) REFERENCES audit_executions(execution_id, submission_key) ON DELETE RESTRICT;

-- Fk Constraint: workflow_runs workflow_runs_project_owner_fkey

ALTER TABLE ONLY workflow_runs
    ADD CONSTRAINT workflow_runs_project_owner_fkey FOREIGN KEY (project_id, owner_id) REFERENCES projects(project_id, owner_id);
