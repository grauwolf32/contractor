-- Project approval policy is owner-selected; new and existing Projects default
-- to automatic execution. Started Audits preserve their previous approval mode.
ALTER TABLE projects ADD COLUMN active_checks text NOT NULL DEFAULT 'automatic'
  CONSTRAINT projects_active_checks_policy CHECK (active_checks IN ('automatic', 'approval-required', 'prohibited'));

CREATE OR REPLACE FUNCTION contractor_protect_project() RETURNS trigger
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

-- Upgrade immutable baseline shape once at the migration boundary, preserving
-- existing item approvals and every other baseline member. Runtime readers have
-- no legacy fallback. Drafts have no baseline yet.
ALTER TABLE audits DISABLE TRIGGER audits_protect_preparation_baseline;
UPDATE audits
SET baseline_snapshot = baseline_snapshot || jsonb_build_object('activeChecks',
  CASE WHEN profile_snapshot #>> '{interaction,activeChecks}' = 'prohibited'
       THEN 'prohibited' ELSE 'approval-required' END)
WHERE baseline_snapshot IS NOT NULL;
ALTER TABLE audits ENABLE TRIGGER audits_protect_preparation_baseline;
