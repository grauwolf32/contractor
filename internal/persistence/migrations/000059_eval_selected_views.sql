-- Rebuildable projections have their own revisions: observations never consume
-- the user's experiment CAS revision. Only explicit authority mutations do.
CREATE TABLE eval_projection_queue (
    experiment_id text PRIMARY KEY REFERENCES eval_experiments ON DELETE CASCADE,
    revision bigint NOT NULL DEFAULT 1,
    published_revision bigint NOT NULL DEFAULT 0,
    snapshot_id text
);
CREATE TABLE eval_member_projections (
    experiment_id text NOT NULL,
    member_id text NOT NULL,
    revision bigint NOT NULL DEFAULT 1,
    projected_revision bigint NOT NULL DEFAULT 0,
    document bytea,
    collection_complete boolean NOT NULL DEFAULT false,
    observed_document bytea,
    checked_at timestamptz,
    PRIMARY KEY (experiment_id, member_id),
    FOREIGN KEY (experiment_id, member_id) REFERENCES eval_members ON DELETE CASCADE,
    CHECK (document IS NULL OR octet_length(document) BETWEEN 2 AND 1048576),
    CHECK (observed_document IS NULL OR octet_length(observed_document) BETWEEN 2 AND 1048576)
);
CREATE INDEX eval_member_projections_dirty ON eval_member_projections(experiment_id, checked_at NULLS FIRST, member_id) WHERE revision <> projected_revision;
CREATE TABLE eval_view_members (
    experiment_id text NOT NULL,
    generation bigint NOT NULL,
    ordinal integer NOT NULL,
    member_id text NOT NULL,
    pair_id text NOT NULL,
    document bytea NOT NULL,
    collection_complete boolean NOT NULL,
    PRIMARY KEY (experiment_id, generation, ordinal),
    FOREIGN KEY (experiment_id, generation) REFERENCES eval_view_generations ON DELETE CASCADE,
    UNIQUE (experiment_id, generation, member_id)
);
CREATE INDEX eval_view_members_pairs ON eval_view_members(experiment_id, generation, pair_id);
-- Public pages use only complete safe generations, with stable ordinal keys.
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
    PRIMARY KEY (experiment_id, generation, ordinal),
    FOREIGN KEY (experiment_id, generation) REFERENCES eval_view_generations ON DELETE CASCADE,
    UNIQUE (experiment_id, generation, pair_id)
);
CREATE INDEX eval_view_pairs_suite ON eval_view_pairs(experiment_id, generation, suite_id, ordinal);
CREATE TABLE eval_view_charts (
    experiment_id text NOT NULL,
    generation bigint NOT NULL,
    suite_id text NOT NULL,
    metric text NOT NULL,
    document bytea NOT NULL,
    PRIMARY KEY (experiment_id, generation, suite_id, metric),
    FOREIGN KEY (experiment_id, generation) REFERENCES eval_view_generations ON DELETE CASCADE
);

CREATE TRIGGER eval_view_pairs_immutable BEFORE UPDATE OR DELETE ON eval_view_pairs
FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();

CREATE TRIGGER eval_view_charts_immutable BEFORE UPDATE OR DELETE ON eval_view_charts
FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();

CREATE TABLE eval_selection_history (
    experiment_id text NOT NULL REFERENCES eval_experiments ON DELETE CASCADE,
    member_id text NOT NULL,
    experiment_revision bigint NOT NULL,
    selection_revision bigint NOT NULL,
    result_sha256 text NOT NULL,
    assessment_sha256 text,
    actor_id text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (experiment_id, experiment_revision, member_id)
);

CREATE TRIGGER eval_selection_history_immutable BEFORE UPDATE OR DELETE ON eval_selection_history
FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();

CREATE TRIGGER eval_views_immutable BEFORE UPDATE OR DELETE ON eval_view_generations
FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();

CREATE TRIGGER eval_view_members_immutable BEFORE UPDATE OR DELETE ON eval_view_members
FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();
CREATE TABLE eval_evidence_refs (
    experiment_id text NOT NULL,
    member_id text NOT NULL,
    record_sha256 text NOT NULL,
    scope_kind text NOT NULL,
    scope_id text NOT NULL,
    namespace text NOT NULL,
    name text NOT NULL,
    revision text NOT NULL,
    PRIMARY KEY (experiment_id, member_id, record_sha256, scope_kind, scope_id, namespace, name, revision),
    FOREIGN KEY (experiment_id, member_id) REFERENCES eval_members ON DELETE CASCADE
);
CREATE INDEX eval_evidence_refs_artifact ON eval_evidence_refs(scope_kind, scope_id, namespace, name, revision);
CREATE INDEX eval_submissions_execution ON eval_submissions(execution_kind, execution_id) WHERE execution_id IS NOT NULL;
CREATE INDEX eval_records_recent ON eval_records(experiment_id, member_id, kind, created_at DESC, record_sha256 DESC);
CREATE INDEX eval_records_native_recent ON eval_records(experiment_id, member_id, kind, created_at DESC, record_sha256 DESC)
WHERE actor_id = 'system:eval-collector';
CREATE INDEX eval_progress_observations_time ON eval_progress_observations(experiment_id, observed_at, sequence);
ALTER TABLE eval_view_generations ADD COLUMN suites bytea NOT NULL DEFAULT convert_to('{}','UTF8');
ALTER TABLE eval_view_generations ADD COLUMN pins_verified boolean NOT NULL DEFAULT false;
ALTER TABLE eval_view_generations ADD COLUMN source_revision bigint NOT NULL DEFAULT 0;

CREATE FUNCTION contractor_eval_dirty(selected_experiment text, selected_member text)
RETURNS void LANGUAGE plpgsql AS $$
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

CREATE FUNCTION contractor_eval_member_changed()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    PERFORM contractor_eval_dirty(NEW.experiment_id, NEW.member_id);
    RETURN NULL;
END;
$$;

-- A bulk frozen matrix invalidates its experiment once, rather than rewriting
-- the same queue row for every member in a COPY/INSERT statement.
CREATE FUNCTION contractor_eval_members_inserted()
RETURNS trigger LANGUAGE plpgsql AS $$
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


CREATE TRIGGER eval_members_projection AFTER INSERT ON eval_members
REFERENCING NEW TABLE AS inserted_members
FOR EACH STATEMENT EXECUTE FUNCTION contractor_eval_members_inserted();

CREATE TRIGGER eval_submissions_projection AFTER INSERT OR UPDATE ON eval_submissions
FOR EACH ROW EXECUTE FUNCTION contractor_eval_member_changed();

CREATE TRIGGER eval_selections_projection AFTER INSERT OR UPDATE ON eval_selections
FOR EACH ROW EXECUTE FUNCTION contractor_eval_member_changed();

CREATE FUNCTION contractor_eval_execution_changed()
RETURNS trigger LANGUAGE plpgsql AS $$
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


CREATE TRIGGER eval_runs_projection AFTER UPDATE OF state, finished_at OR DELETE ON workflow_runs
FOR EACH ROW EXECUTE FUNCTION contractor_eval_execution_changed();

CREATE TRIGGER eval_audits_projection AFTER UPDATE OF state, finished_at, dispatch_state OR DELETE ON audits
FOR EACH ROW EXECUTE FUNCTION contractor_eval_execution_changed();

CREATE FUNCTION contractor_eval_audit_child_changed()
RETURNS trigger LANGUAGE plpgsql AS $$
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


CREATE TRIGGER eval_audit_children_projection AFTER INSERT OR UPDATE OR DELETE ON audit_executions
FOR EACH ROW EXECUTE FUNCTION contractor_eval_audit_child_changed();

CREATE TRIGGER eval_audit_outputs_projection AFTER INSERT OR UPDATE OR DELETE ON audit_artifact_links
FOR EACH ROW EXECUTE FUNCTION contractor_eval_audit_child_changed();

CREATE FUNCTION contractor_eval_metrics_changed()
RETURNS trigger LANGUAGE plpgsql AS $$
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


CREATE TRIGGER eval_metrics_projection AFTER INSERT OR UPDATE OR DELETE ON stage_metrics
FOR EACH ROW EXECUTE FUNCTION contractor_eval_metrics_changed();

CREATE TRIGGER eval_stages_projection AFTER INSERT OR UPDATE OF state OR DELETE ON stage_executions
FOR EACH ROW EXECUTE FUNCTION contractor_eval_metrics_changed();

CREATE FUNCTION contractor_eval_output_changed()
RETURNS trigger LANGUAGE plpgsql AS $$
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


CREATE TRIGGER eval_outputs_projection AFTER INSERT OR UPDATE OR DELETE ON artifact_bindings
FOR EACH ROW EXECUTE FUNCTION contractor_eval_output_changed();

CREATE FUNCTION contractor_eval_evidence_deleted()
RETURNS trigger LANGUAGE plpgsql AS $$
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


CREATE TRIGGER eval_artifact_projection BEFORE DELETE ON artifact_binding_revisions
FOR EACH ROW EXECUTE FUNCTION contractor_eval_evidence_deleted();

-- Existing prepared experiments become eligible for initial collection.
INSERT INTO eval_member_projections(experiment_id, member_id)
SELECT experiment_id, member_id FROM eval_members;
INSERT INTO eval_projection_queue(experiment_id)
SELECT DISTINCT experiment_id FROM eval_members;
