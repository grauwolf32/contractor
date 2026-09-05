ALTER TABLE projects
    ADD COLUMN lifecycle_state text NOT NULL DEFAULT 'active'
        CHECK (lifecycle_state IN ('active', 'deleting')),
    ADD COLUMN deletion_phase text
        CHECK (deletion_phase IN ('cancelling', 'draining', 'purging_runs', 'purging_artifacts')),
    ADD COLUMN deletion_requested_at timestamptz,
    ADD COLUMN deletion_claim_id text,
    ADD COLUMN deletion_claimed_at timestamptz,
    ADD COLUMN deletion_claim_expires_at timestamptz,
    ADD CONSTRAINT projects_deletion_shape CHECK (
        (lifecycle_state = 'active'
            AND deletion_phase IS NULL
            AND deletion_requested_at IS NULL
            AND deletion_claim_id IS NULL
            AND deletion_claimed_at IS NULL
            AND deletion_claim_expires_at IS NULL)
        OR
        (lifecycle_state = 'deleting'
            AND deletion_phase IS NOT NULL
            AND deletion_requested_at IS NOT NULL
            AND (
                (deletion_claim_id IS NULL
                    AND deletion_claimed_at IS NULL
                    AND deletion_claim_expires_at IS NULL)
                OR
                (deletion_claim_id IS NOT NULL
                    AND deletion_claimed_at IS NOT NULL
                    AND deletion_claim_expires_at > deletion_claimed_at)
            ))
    );

CREATE INDEX projects_deletion_claim_idx
    ON projects (deletion_claim_expires_at, deletion_requested_at, project_id)
    WHERE lifecycle_state = 'deleting';

-- Publicly visible Project changes advance the strong revision. Controller
-- lease bookkeeping is deliberately invisible and may be recovered after a
-- process crash without manufacturing a new representation.
CREATE OR REPLACE FUNCTION contractor_protect_project()
RETURNS trigger
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

-- Taking a row lock in the mutating statement gives Project deletion one
-- serialization boundary with new Runs and ProjectScope writes. If the write
-- wins, cleanup sees it; if deletion wins, the write receives SQLSTATE 55000.
CREATE OR REPLACE FUNCTION contractor_require_active_project(
    checked_project_id text,
    checked_owner_id text DEFAULT NULL
)
RETURNS void
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

CREATE OR REPLACE FUNCTION contractor_guard_workflow_run_project_admission()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.project_id IS NOT NULL THEN
        PERFORM contractor_require_active_project(NEW.project_id, NEW.owner_id);
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER workflow_runs_project_admission
BEFORE INSERT ON workflow_runs
FOR EACH ROW EXECUTE FUNCTION contractor_guard_workflow_run_project_admission();

CREATE OR REPLACE FUNCTION contractor_guard_project_artifact_admission()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.scope_kind = 'project' THEN
        PERFORM contractor_require_active_project(NEW.scope_id);
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER artifact_scopes_project_admission
BEFORE INSERT ON artifact_scopes
FOR EACH ROW EXECUTE FUNCTION contractor_guard_project_artifact_admission();

CREATE TRIGGER artifact_bindings_project_admission
BEFORE INSERT OR UPDATE ON artifact_bindings
FOR EACH ROW EXECUTE FUNCTION contractor_guard_project_artifact_admission();
