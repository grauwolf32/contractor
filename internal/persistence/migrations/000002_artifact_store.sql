CREATE TABLE artifact_scopes (
    scope_kind text NOT NULL CHECK (scope_kind IN ('user', 'run')),
    scope_id text NOT NULL CHECK (btrim(scope_id) <> ''),
    run_id text GENERATED ALWAYS AS (
        CASE WHEN scope_kind = 'run' THEN scope_id ELSE NULL END
    ) STORED,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (scope_kind, scope_id),
    FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id)
);

CREATE TABLE artifact_blobs (
    sha256 bytea PRIMARY KEY CHECK (octet_length(sha256) = 32),
    payload bytea NOT NULL,
    size_bytes bigint NOT NULL CHECK (
        size_bytes = octet_length(payload)
        AND size_bytes >= 0
        AND size_bytes <= 16777216
    ),
    CHECK (sha256 = pg_catalog.sha256(payload)),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE artifact_versions (
    version_id text PRIMARY KEY CHECK (btrim(version_id) <> ''),
    blob_sha256 bytea NOT NULL REFERENCES artifact_blobs(sha256),
    media_type text NOT NULL CHECK (
        media_type = lower(media_type)
        AND media_type ~ '^[a-z0-9][a-z0-9!#$%&''+.^_`|~-]*/[a-z0-9][a-z0-9!#$%&''+.^_`|~-]*$'
    ),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE artifact_binding_revisions (
    scope_kind text NOT NULL,
    scope_id text NOT NULL,
    namespace text NOT NULL CHECK (btrim(namespace) <> '' AND position('/' in namespace) = 0),
    name text NOT NULL CHECK (btrim(name) <> '' AND position('/' in name) = 0),
    revision text NOT NULL CHECK (btrim(revision) <> ''),
    version_id text NOT NULL REFERENCES artifact_versions(version_id),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (scope_kind, scope_id, namespace, name, revision),
    FOREIGN KEY (scope_kind, scope_id)
        REFERENCES artifact_scopes(scope_kind, scope_id)
);

CREATE TABLE artifact_bindings (
    scope_kind text NOT NULL,
    scope_id text NOT NULL,
    namespace text NOT NULL CHECK (btrim(namespace) <> '' AND position('/' in namespace) = 0),
    name text NOT NULL CHECK (btrim(name) <> '' AND position('/' in name) = 0),
    current_revision text NOT NULL CHECK (btrim(current_revision) <> ''),
    frozen boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (scope_kind, scope_id, namespace, name),
    FOREIGN KEY (scope_kind, scope_id)
        REFERENCES artifact_scopes(scope_kind, scope_id),
    FOREIGN KEY (scope_kind, scope_id, namespace, name, current_revision)
        REFERENCES artifact_binding_revisions(scope_kind, scope_id, namespace, name, revision)
        DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX artifact_bindings_list_idx
    ON artifact_bindings (scope_kind, scope_id, namespace, name);

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
    lineage_kind text NOT NULL CHECK (lineage_kind IN ('input_fork', 'output_bind')),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (
        target_scope_kind, target_scope_id, target_namespace, target_name, target_revision
    ),
    FOREIGN KEY (
        target_scope_kind, target_scope_id, target_namespace, target_name, target_revision
    ) REFERENCES artifact_binding_revisions(scope_kind, scope_id, namespace, name, revision),
    FOREIGN KEY (
        source_scope_kind, source_scope_id, source_namespace, source_name, source_revision
    ) REFERENCES artifact_binding_revisions(scope_kind, scope_id, namespace, name, revision)
);

CREATE TABLE artifact_pins (
    pin_kind text NOT NULL CHECK (pin_kind IN (
        'run_input', 'stage_context', 'stage_result', 'run_output'
    )),
    pin_id text NOT NULL CHECK (btrim(pin_id) <> ''),
    scope_kind text NOT NULL,
    scope_id text NOT NULL,
    namespace text NOT NULL,
    name text NOT NULL,
    revision text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (pin_kind, pin_id, scope_kind, scope_id, namespace, name, revision),
    FOREIGN KEY (scope_kind, scope_id, namespace, name, revision)
        REFERENCES artifact_binding_revisions(scope_kind, scope_id, namespace, name, revision)
);

CREATE OR REPLACE FUNCTION contractor_protect_artifact_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'immutable ArtifactStore row cannot be changed' USING ERRCODE = '23514';
END;
$$;

CREATE TRIGGER artifact_blobs_immutable
BEFORE UPDATE OR DELETE ON artifact_blobs
FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_immutable();

CREATE TRIGGER artifact_versions_immutable
BEFORE UPDATE OR DELETE ON artifact_versions
FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_immutable();

CREATE TRIGGER artifact_binding_revisions_immutable
BEFORE UPDATE OR DELETE ON artifact_binding_revisions
FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_immutable();

CREATE TRIGGER artifact_lineage_immutable
BEFORE UPDATE OR DELETE ON artifact_lineage
FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_immutable();

CREATE TRIGGER artifact_pins_immutable
BEFORE UPDATE OR DELETE ON artifact_pins
FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_immutable();

CREATE OR REPLACE FUNCTION contractor_protect_artifact_binding()
RETURNS trigger
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
        RAISE EXCEPTION 'frozen Artifact binding cannot be changed' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER artifact_bindings_protect_identity
BEFORE UPDATE ON artifact_bindings
FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_binding();
