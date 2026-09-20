-- Managed Evals owns private data and submission authority, never a Worker queue.
ALTER TABLE projects ADD CONSTRAINT projects_owner_identity UNIQUE (owner_id, project_id);

CREATE TABLE eval_dataset_revisions (
    owner_id text NOT NULL,
    project_id text NOT NULL,
    dataset_id text NOT NULL,
    revision text NOT NULL,
    document bytea NOT NULL CHECK (octet_length(document) BETWEEN 2 AND 1048576),
    document_sha256 text GENERATED ALWAYS AS ('sha256:' || encode(sha256(document), 'hex')) STORED,
    metadata jsonb NOT NULL CHECK (jsonb_typeof(metadata) = 'object'),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (project_id, dataset_id, revision),
    FOREIGN KEY (owner_id, project_id) REFERENCES projects(owner_id, project_id) ON DELETE RESTRICT
);
CREATE INDEX eval_datasets_owner_page_idx ON eval_dataset_revisions (owner_id, project_id, dataset_id, revision);

CREATE TABLE eval_experiments (
    experiment_id text PRIMARY KEY,
    owner_id text NOT NULL,
    project_id text NOT NULL,
    portable_id text NOT NULL,
    control_mode text NOT NULL CHECK (control_mode IN ('server', 'external')),
    name text NOT NULL CHECK (length(name) BETWEEN 1 AND 256),
    state text NOT NULL CHECK (state IN ('draft','preparing','ready','running','settling','finished','pausing','paused','cancelling','cancelled','interrupted')),
    revision bigint NOT NULL DEFAULT 1 CHECK (revision > 0),
    draft bytea,
    dataset_id text,
    dataset_revision text,
    expected_count integer NOT NULL DEFAULT 0 CHECK (expected_count BETWEEN 0 AND 10000),
    outstanding_count integer NOT NULL DEFAULT 0 CHECK (outstanding_count BETWEEN 0 AND expected_count),
    max_in_flight integer NOT NULL CHECK (max_in_flight BETWEEN 1 AND 10000),
    wall_ms bigint NOT NULL CHECK (wall_ms > 0),
    token_limit bigint CHECK (token_limit >= 0),
    observed_tokens bigint NOT NULL DEFAULT 0 CHECK (observed_tokens >= 0),
    started_at timestamptz,
    deadline_at timestamptz,
    last_producer_activity_at timestamptz,
    deletion_requested_at timestamptz,
    diagnostic jsonb,
    view_generation bigint NOT NULL DEFAULT 1 CHECK (view_generation > 0),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (owner_id, project_id, portable_id),
    FOREIGN KEY (owner_id, project_id) REFERENCES projects(owner_id, project_id) ON DELETE RESTRICT,
    FOREIGN KEY (project_id, dataset_id, dataset_revision) REFERENCES eval_dataset_revisions(project_id, dataset_id, revision) ON DELETE RESTRICT,
    CHECK ((dataset_id IS NULL) = (dataset_revision IS NULL)),
    CHECK ((started_at IS NULL AND deadline_at IS NULL) OR (started_at IS NOT NULL AND deadline_at > started_at)),
    CHECK (draft IS NULL OR (control_mode = 'server' AND octet_length(draft) BETWEEN 2 AND 1048576))
);
CREATE INDEX eval_experiments_owner_page_idx ON eval_experiments (owner_id, experiment_id);
CREATE INDEX eval_experiments_project_page_idx ON eval_experiments (owner_id, project_id, experiment_id);
CREATE INDEX eval_experiments_state_page_idx ON eval_experiments (owner_id, state, experiment_id);
CREATE INDEX eval_experiments_dataset_page_idx ON eval_experiments (owner_id, dataset_id, experiment_id);
CREATE INDEX eval_experiments_control_page_idx ON eval_experiments (owner_id, control_mode, experiment_id);

CREATE TABLE eval_frozen_plans (
    experiment_id text PRIMARY KEY REFERENCES eval_experiments ON DELETE CASCADE,
    document_kind text NOT NULL CHECK (document_kind IN ('playground.plan/v1', 'ExternalRegistration')),
    document bytea NOT NULL CHECK (octet_length(document) BETWEEN 2 AND 1048576),
    document_sha256 text GENERATED ALWAYS AS ('sha256:' || encode(sha256(document), 'hex')) STORED,
    plan_sha256 text NOT NULL CHECK (plan_sha256 ~ '^sha256:[0-9a-f]{64}$'),
    setup bytea NOT NULL CHECK (octet_length(setup) BETWEEN 2 AND 1048576),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE eval_members (
    experiment_id text NOT NULL REFERENCES eval_frozen_plans ON DELETE CASCADE,
    member_id text NOT NULL CHECK (member_id ~ '^[0-9a-f]{64}$'),
    pair_id text NOT NULL CHECK (pair_id ~ '^[0-9a-f]{64}$'),
    ordinal integer NOT NULL CHECK (ordinal BETWEEN 0 AND 9999),
    suite_id text NOT NULL,
    case_id text NOT NULL,
    sample integer NOT NULL CHECK (sample BETWEEN 1 AND 100),
    variant_id text NOT NULL,
    case_sha256 text NOT NULL,
    binding_sha256 text NOT NULL,
    eligibility text NOT NULL CHECK (eligibility IN ('eligible','unsupported','blocked')),
    execution_kind text NOT NULL CHECK (execution_kind IN ('run','audit')),
    recipe bytea NOT NULL CHECK (octet_length(recipe) BETWEEN 2 AND 1048576),
    submission_key text NOT NULL UNIQUE,
    PRIMARY KEY (experiment_id, member_id),
    UNIQUE (experiment_id, member_id, execution_kind),
    UNIQUE (experiment_id, ordinal),
    UNIQUE (experiment_id, suite_id, case_id, sample, variant_id)
);
CREATE INDEX eval_members_pair_idx ON eval_members(experiment_id, pair_id, variant_id);

CREATE TABLE eval_controller_claims (
    experiment_id text PRIMARY KEY REFERENCES eval_experiments ON DELETE CASCADE,
    epoch bigint NOT NULL DEFAULT 0 CHECK (epoch >= 0),
    holder_id text,
    expires_at timestamptz,
    CHECK ((holder_id IS NULL) = (expires_at IS NULL))
);
CREATE INDEX eval_claims_expiry_idx ON eval_controller_claims(expires_at, experiment_id);

CREATE TABLE eval_commands (
    command_id text PRIMARY KEY,
    experiment_id text NOT NULL REFERENCES eval_experiments ON DELETE CASCADE,
    actor_id text NOT NULL,
    kind text NOT NULL,
    state text NOT NULL CHECK (state IN ('accepted','running','succeeded','failed')),
    accepted_revision bigint NOT NULL,
    diagnostic jsonb,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    finished_at timestamptz
);
CREATE INDEX eval_commands_pending_idx ON eval_commands(experiment_id, created_at, command_id) WHERE state IN ('accepted','running');
CREATE TABLE eval_mutation_receipts (
    owner_id text NOT NULL,
    project_id text NOT NULL REFERENCES projects ON DELETE RESTRICT,
    resource_id text NOT NULL,
    operation text NOT NULL,
    operation_key text NOT NULL CHECK (length(operation_key) BETWEEN 1 AND 128),
    request_sha256 text NOT NULL CHECK (request_sha256 ~ '^sha256:[0-9a-f]{64}$'),
    expected_revision bigint,
    response bytea NOT NULL CHECK (octet_length(response) BETWEEN 2 AND 1048576),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (owner_id, project_id, resource_id, operation, operation_key)
);

CREATE TABLE eval_submissions (
    experiment_id text NOT NULL,
    member_id text NOT NULL,
    state text NOT NULL CHECK (state IN ('intent','accepted','terminal','rejected')),
    actor_id text NOT NULL,
    execution_kind text NOT NULL,
    execution_id text,
    execution_project_id text,
    terminal_state text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (experiment_id, member_id),
    FOREIGN KEY (experiment_id, member_id,execution_kind) REFERENCES eval_members(experiment_id,member_id,execution_kind) ON DELETE CASCADE,
    CHECK ((state IN ('accepted','terminal')) = (execution_id IS NOT NULL)),
    CHECK ((state = 'terminal') = (terminal_state IS NOT NULL))
);
CREATE UNIQUE INDEX eval_submissions_execution_idx ON eval_submissions(execution_kind,execution_id) WHERE execution_id IS NOT NULL;
CREATE UNIQUE INDEX eval_submissions_workspace_idx ON eval_submissions(execution_project_id) WHERE execution_project_id IS NOT NULL;
CREATE INDEX eval_submissions_outstanding_idx ON eval_submissions(experiment_id, member_id) WHERE state IN ('intent','accepted');

-- These receipts survive an uncertain remote outcome. Absence is never success.
CREATE TABLE eval_suboperations (
    experiment_id text NOT NULL,
    member_id text NOT NULL,
    kind text NOT NULL CHECK (kind IN ('project-create','inputs','run-create','audit-create','audit-start','cancel')),
    operation_key text NOT NULL UNIQUE,
    request bytea NOT NULL CHECK (octet_length(request) BETWEEN 2 AND 1048576),
    request_sha256 text GENERATED ALWAYS AS ('sha256:' || encode(sha256(request), 'hex')) STORED,
    response bytea CHECK (octet_length(response) BETWEEN 2 AND 1048576),
    state text NOT NULL DEFAULT 'intent' CHECK (state IN ('intent','succeeded','rejected')),
    PRIMARY KEY (experiment_id, member_id, kind),
    FOREIGN KEY (experiment_id, member_id) REFERENCES eval_submissions ON DELETE CASCADE,
    CHECK ((state = 'intent') = (response IS NULL))
);

-- Exact child workspace dependencies remain as tombstones after normal deletion.
CREATE TABLE eval_project_dependencies (
    project_id text PRIMARY KEY,
    experiment_id text NOT NULL,
    member_id text NOT NULL,
    owner_id text NOT NULL,
    FOREIGN KEY (experiment_id, member_id) REFERENCES eval_submissions ON DELETE CASCADE,
    UNIQUE (experiment_id, member_id)
);
CREATE INDEX eval_dependencies_experiment_idx ON eval_project_dependencies(experiment_id, project_id);

CREATE TABLE eval_records (
    experiment_id text NOT NULL,
    member_id text NOT NULL,
    kind text NOT NULL CHECK (kind IN ('result','assessment')),
    record_sha256 text NOT NULL,
    document bytea NOT NULL CHECK (octet_length(document) BETWEEN 2 AND 1048576),
    actor_id text NOT NULL,
    predecessor_sha256 text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (experiment_id, member_id, kind, record_sha256),
    FOREIGN KEY (experiment_id, member_id) REFERENCES eval_members ON DELETE CASCADE,
    CHECK (record_sha256 = 'sha256:' || encode(sha256(document), 'hex'))
);
CREATE TABLE eval_selections (
    experiment_id text NOT NULL,
    member_id text NOT NULL,
    result_kind text NOT NULL DEFAULT 'result' CHECK (result_kind = 'result'),
    result_sha256 text NOT NULL,
    assessment_kind text NOT NULL DEFAULT 'assessment' CHECK (assessment_kind = 'assessment'),
    assessment_sha256 text,
    revision bigint NOT NULL DEFAULT 1,
    actor_id text NOT NULL,
    PRIMARY KEY (experiment_id, member_id),
    FOREIGN KEY (experiment_id, member_id, result_kind, result_sha256) REFERENCES eval_records,
    FOREIGN KEY (experiment_id, member_id, assessment_kind, assessment_sha256) REFERENCES eval_records
);
CREATE TABLE eval_view_generations (
    experiment_id text NOT NULL REFERENCES eval_experiments ON DELETE CASCADE,
    generation bigint NOT NULL,
    snapshot_id text NOT NULL UNIQUE,
    summary bytea NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (experiment_id, generation)
);
CREATE TABLE eval_progress_observations (
    experiment_id text NOT NULL REFERENCES eval_experiments ON DELETE CASCADE,
    sequence bigint GENERATED ALWAYS AS IDENTITY,
    observed_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    counts jsonb NOT NULL,
    PRIMARY KEY (experiment_id, sequence)
);
CREATE TABLE eval_collections (
    owner_id text NOT NULL,
    project_id text NOT NULL,
    revision bigint NOT NULL DEFAULT 1,
    PRIMARY KEY (owner_id, project_id)
);
CREATE FUNCTION contractor_eval_collection_changed() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE selected_owner text; selected_project text;
BEGIN
    IF TG_OP = 'DELETE' THEN selected_owner := OLD.owner_id; selected_project := OLD.project_id;
    ELSE selected_owner := NEW.owner_id; selected_project := NEW.project_id; END IF;
    INSERT INTO eval_collections(owner_id, project_id) VALUES (selected_owner, ''), (selected_owner, selected_project)
    ON CONFLICT (owner_id, project_id) DO UPDATE SET revision = eval_collections.revision + 1;
    RETURN NULL;
END; $$;
CREATE TRIGGER eval_experiments_collection AFTER INSERT OR UPDATE OR DELETE ON eval_experiments FOR EACH ROW EXECUTE FUNCTION contractor_eval_collection_changed();
CREATE TRIGGER eval_datasets_collection AFTER INSERT OR DELETE ON eval_dataset_revisions FOR EACH ROW EXECUTE FUNCTION contractor_eval_collection_changed();

CREATE FUNCTION contractor_eval_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' AND current_setting('contractor.eval_purge', true) = 'on' THEN RETURN OLD; END IF;
    RAISE EXCEPTION 'Evaluation record is immutable' USING ERRCODE = '23514';
END; $$;
CREATE TRIGGER eval_datasets_immutable BEFORE UPDATE OR DELETE ON eval_dataset_revisions FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();
CREATE TRIGGER eval_plans_immutable BEFORE UPDATE OR DELETE ON eval_frozen_plans FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();
CREATE TRIGGER eval_members_immutable BEFORE UPDATE OR DELETE ON eval_members FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();
CREATE TRIGGER eval_records_immutable BEFORE UPDATE OR DELETE ON eval_records FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();
CREATE TRIGGER eval_receipts_immutable BEFORE UPDATE OR DELETE ON eval_mutation_receipts FOR EACH ROW EXECUTE FUNCTION contractor_eval_immutable();

CREATE FUNCTION contractor_eval_protect_experiment() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF ROW(NEW.experiment_id,NEW.owner_id,NEW.project_id,NEW.portable_id,NEW.control_mode,NEW.created_at)
       IS DISTINCT FROM ROW(OLD.experiment_id,OLD.owner_id,OLD.project_id,OLD.portable_id,OLD.control_mode,OLD.created_at)
       OR NEW.revision <> OLD.revision + 1 OR NEW.updated_at <= OLD.updated_at
       OR NEW.observed_tokens < OLD.observed_tokens
       OR (OLD.started_at IS NOT NULL AND ROW(NEW.started_at,NEW.deadline_at) IS DISTINCT FROM ROW(OLD.started_at,OLD.deadline_at))
       OR (OLD.deletion_requested_at IS NOT NULL AND NEW.deletion_requested_at IS DISTINCT FROM OLD.deletion_requested_at)
       OR (EXISTS (SELECT 1 FROM eval_frozen_plans WHERE experiment_id=OLD.experiment_id)
           AND ROW(NEW.draft,NEW.dataset_id,NEW.dataset_revision,NEW.expected_count,NEW.max_in_flight,NEW.wall_ms,NEW.token_limit)
               IS DISTINCT FROM ROW(OLD.draft,OLD.dataset_id,OLD.dataset_revision,OLD.expected_count,OLD.max_in_flight,OLD.wall_ms,OLD.token_limit))
    THEN RAISE EXCEPTION 'Evaluation revision or immutable fields changed' USING ERRCODE = '23514'; END IF;
    RETURN NEW;
END; $$;
CREATE TRIGGER eval_experiments_protect BEFORE UPDATE ON eval_experiments FOR EACH ROW EXECUTE FUNCTION contractor_eval_protect_experiment();

-- Same Project row lock as ordinary Run/Audit admission. Admission locks the
-- workspace before the experiment; deletion takes the same order.
CREATE FUNCTION contractor_eval_fence_project() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.lifecycle_state = 'deleting' AND OLD.lifecycle_state = 'active' THEN
        UPDATE eval_experiments SET deletion_requested_at=CASE WHEN project_id=NEW.project_id THEN COALESCE(deletion_requested_at, clock_timestamp()) ELSE deletion_requested_at END,
            state='cancelling', revision=revision+1,
            updated_at=GREATEST(clock_timestamp(), updated_at+interval '1 microsecond')
        WHERE project_id=NEW.project_id OR experiment_id IN (
            SELECT experiment_id FROM eval_project_dependencies WHERE project_id=NEW.project_id
        );
    END IF;
    RETURN NEW;
END; $$;
CREATE TRIGGER projects_eval_fence AFTER UPDATE OF lifecycle_state ON projects FOR EACH ROW EXECUTE FUNCTION contractor_eval_fence_project();

CREATE FUNCTION contractor_eval_complete_matrix() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE expected integer; actual integer;
BEGIN
    SELECT expected_count INTO expected FROM eval_experiments WHERE experiment_id=NEW.experiment_id;
    SELECT count(*) INTO actual FROM eval_members WHERE experiment_id=NEW.experiment_id;
    IF expected IS NULL OR expected <> actual OR actual < 1 THEN
        RAISE EXCEPTION 'Evaluation expected matrix is incomplete' USING ERRCODE = '23514';
    END IF;
    RETURN NULL;
END; $$;
CREATE CONSTRAINT TRIGGER eval_plans_complete_matrix AFTER INSERT ON eval_frozen_plans
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION contractor_eval_complete_matrix();
CREATE FUNCTION contractor_eval_member_bound() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS(SELECT 1 FROM eval_experiments WHERE experiment_id=NEW.experiment_id AND NEW.ordinal<expected_count) THEN
        RAISE EXCEPTION 'Evaluation member is outside frozen matrix' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END; $$;
CREATE TRIGGER eval_members_bound BEFORE INSERT ON eval_members FOR EACH ROW EXECUTE FUNCTION contractor_eval_member_bound();
CREATE FUNCTION contractor_eval_protect_suboperation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF ROW(NEW.experiment_id,NEW.member_id,NEW.kind,NEW.operation_key,NEW.request)
       IS DISTINCT FROM ROW(OLD.experiment_id,OLD.member_id,OLD.kind,OLD.operation_key,OLD.request)
       OR (OLD.state <> 'intent' AND ROW(NEW.state,NEW.response) IS DISTINCT FROM ROW(OLD.state,OLD.response))
    THEN RAISE EXCEPTION 'Evaluation suboperation is immutable' USING ERRCODE = '23514'; END IF;
    RETURN NEW;
END; $$;
CREATE TRIGGER eval_suboperations_protect BEFORE UPDATE ON eval_suboperations FOR EACH ROW EXECUTE FUNCTION contractor_eval_protect_suboperation();
CREATE FUNCTION contractor_eval_protect_submission() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF ROW(NEW.experiment_id,NEW.member_id,NEW.actor_id,NEW.created_at,NEW.execution_kind)
       IS DISTINCT FROM ROW(OLD.experiment_id,OLD.member_id,OLD.actor_id,OLD.created_at,OLD.execution_kind)
       OR (OLD.execution_id IS NOT NULL AND NEW.execution_id IS DISTINCT FROM OLD.execution_id)
       OR (OLD.execution_project_id IS NOT NULL AND NEW.execution_project_id IS DISTINCT FROM OLD.execution_project_id)
       OR (OLD.state IN ('terminal','rejected') AND ROW(NEW.state,NEW.terminal_state) IS DISTINCT FROM ROW(OLD.state,OLD.terminal_state))
    THEN RAISE EXCEPTION 'Evaluation submission is immutable' USING ERRCODE = '23514'; END IF;
    RETURN NEW;
END; $$;
CREATE TRIGGER eval_submissions_protect BEFORE UPDATE ON eval_submissions FOR EACH ROW EXECUTE FUNCTION contractor_eval_protect_submission();
