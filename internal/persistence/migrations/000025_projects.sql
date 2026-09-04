CREATE TABLE projects (
    project_id text PRIMARY KEY CHECK (
        project_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'
    ),
    owner_id text NOT NULL CHECK (
        btrim(owner_id) <> '' AND octet_length(owner_id) <= 256
    ),
    kind text NOT NULL CHECK (kind IN ('project', 'evaluation')),
    name text NOT NULL CHECK (
        btrim(name) <> '' AND octet_length(name) <= 160
    ),
    description text NOT NULL DEFAULT '' CHECK (
        octet_length(description) <= 4096
    ),
    revision bigint NOT NULL DEFAULT 1 CHECK (revision > 0),
    request_idempotency_key text NOT NULL CHECK (
        request_idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
    ),
    request_digest text NOT NULL CHECK (
        request_digest ~ '^sha256:[0-9a-f]{64}$'
    ),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (owner_id, request_idempotency_key)
);

CREATE INDEX projects_owner_kind_created_idx
    ON projects (owner_id, kind, created_at DESC, project_id DESC);

CREATE OR REPLACE FUNCTION contractor_protect_project()
RETURNS trigger
LANGUAGE plpgsql
AS $$
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
    IF NEW.revision <> OLD.revision + 1 OR NEW.updated_at <= OLD.updated_at THEN
        RAISE EXCEPTION 'Project update must advance revision and time' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER projects_protect_identity
BEFORE UPDATE ON projects
FOR EACH ROW EXECUTE FUNCTION contractor_protect_project();
