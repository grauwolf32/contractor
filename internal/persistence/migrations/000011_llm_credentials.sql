CREATE TABLE llm_credential_identities (
    credential_id text PRIMARY KEY
        CHECK (credential_id ~ '^[a-z][a-z0-9_-]*$' AND octet_length(credential_id) <= 128),
    reserved_at timestamptz NOT NULL
);

CREATE TABLE llm_credentials (
    credential_id text PRIMARY KEY
        REFERENCES llm_credential_identities(credential_id) ON DELETE RESTRICT,
    llm_gateway_id text NOT NULL
        CHECK (llm_gateway_id ~ '^[a-z][a-z0-9_-]*$' AND octet_length(llm_gateway_id) <= 128),
    llm_gateway_version text NOT NULL
        CHECK (llm_gateway_version ~ '^[A-Za-z0-9][A-Za-z0-9._+-]*$' AND octet_length(llm_gateway_version) <= 64),
    llm_gateway_digest text NOT NULL
        CHECK (llm_gateway_digest ~ '^sha256:[0-9a-f]{64}$'),
    remote_key_id text NOT NULL
        CHECK (btrim(remote_key_id) <> '' AND octet_length(remote_key_id) <= 512),
    label text CHECK (label IS NULL OR (btrim(label) <> '' AND octet_length(label) <= 1024)),
    gateway_policy jsonb NOT NULL
        CHECK (jsonb_typeof(gateway_policy) = 'object' AND octet_length(gateway_policy::text) <= 65536),
    encryption_schema_version text NOT NULL
        CHECK (encryption_schema_version = 'contractor.credentials/v1'),
    key_id text NOT NULL CHECK (key_id ~ '^sha256:[0-9a-f]{64}$'),
    nonce bytea NOT NULL CHECK (octet_length(nonce) = 12),
    ciphertext bytea NOT NULL CHECK (octet_length(ciphertext) BETWEEN 17 AND 16400),
    created_at timestamptz NOT NULL,
    UNIQUE (llm_gateway_digest, remote_key_id)
);

CREATE TABLE llm_credential_tombstones (
    credential_id text PRIMARY KEY
        REFERENCES llm_credential_identities(credential_id) ON DELETE RESTRICT,
    actor_id text NOT NULL CHECK (btrim(actor_id) <> '' AND octet_length(actor_id) <= 256),
    deleted_at timestamptz NOT NULL
);

CREATE TABLE credential_operations (
    operation_id text PRIMARY KEY
        CHECK (operation_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]*$' AND octet_length(operation_id) <= 128),
    idempotency_key text NOT NULL
        CHECK (idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    request_hash text NOT NULL CHECK (request_hash ~ '^sha256:[0-9a-f]{64}$'),
    credential_id text NOT NULL
        REFERENCES llm_credential_identities(credential_id) ON DELETE RESTRICT,
    operation_kind text NOT NULL CHECK (operation_kind IN ('create', 'delete')),
    phase text NOT NULL CHECK (phase IN ('prepared', 'completed')),
    request_schema_version text NOT NULL
        CHECK (request_schema_version = 'contractor.credentials/v1'),
    request jsonb NOT NULL
        CHECK (jsonb_typeof(request) = 'object' AND octet_length(request::text) <= 65536),
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL CHECK (updated_at >= created_at),
    UNIQUE (operation_kind, idempotency_key)
);

CREATE INDEX llm_credentials_created_idx
    ON llm_credentials (created_at, credential_id);
CREATE INDEX credential_operations_prepared_idx
    ON credential_operations (created_at, operation_id)
    WHERE phase = 'prepared';

CREATE OR REPLACE FUNCTION contractor_check_llm_credential_exclusive_state()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    PERFORM 1 FROM llm_credential_identities
    WHERE credential_id = NEW.credential_id FOR UPDATE;
    IF TG_TABLE_NAME = 'llm_credentials'
        AND EXISTS (SELECT 1 FROM llm_credential_tombstones WHERE credential_id = NEW.credential_id)
    THEN
        RAISE EXCEPTION 'credential ID has a tombstone' USING ERRCODE = '23505';
    END IF;
    IF TG_TABLE_NAME = 'llm_credential_tombstones'
        AND EXISTS (SELECT 1 FROM llm_credentials WHERE credential_id = NEW.credential_id)
    THEN
        RAISE EXCEPTION 'credential ID is still active' USING ERRCODE = '23505';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER llm_credentials_exclusive_state
BEFORE INSERT ON llm_credentials
FOR EACH ROW EXECUTE FUNCTION contractor_check_llm_credential_exclusive_state();

CREATE TRIGGER llm_credential_tombstones_exclusive_state
BEFORE INSERT ON llm_credential_tombstones
FOR EACH ROW EXECUTE FUNCTION contractor_check_llm_credential_exclusive_state();

CREATE OR REPLACE FUNCTION contractor_protect_immutable_credential_row()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'credential identity, active record, and tombstone rows are immutable'
        USING ERRCODE = '23514';
END;
$$;

CREATE TRIGGER llm_credential_identities_protect_immutable
BEFORE UPDATE OR DELETE ON llm_credential_identities
FOR EACH ROW EXECUTE FUNCTION contractor_protect_immutable_credential_row();

CREATE TRIGGER llm_credentials_protect_update
BEFORE UPDATE ON llm_credentials
FOR EACH ROW EXECUTE FUNCTION contractor_protect_immutable_credential_row();

CREATE TRIGGER llm_credential_tombstones_protect_immutable
BEFORE UPDATE OR DELETE ON llm_credential_tombstones
FOR EACH ROW EXECUTE FUNCTION contractor_protect_immutable_credential_row();

CREATE OR REPLACE FUNCTION contractor_protect_credential_operation()
RETURNS trigger
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
        OR NEW.phase <> 'completed'
        OR NEW.updated_at < OLD.updated_at
    THEN
        RAISE EXCEPTION 'credential operation mutation is invalid' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER credential_operations_protect_transition
BEFORE UPDATE ON credential_operations
FOR EACH ROW EXECUTE FUNCTION contractor_protect_credential_operation();

CREATE TRIGGER credential_operations_protect_delete
BEFORE DELETE ON credential_operations
FOR EACH ROW EXECUTE FUNCTION contractor_protect_immutable_credential_row();
