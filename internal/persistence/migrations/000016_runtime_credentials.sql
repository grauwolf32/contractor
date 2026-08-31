CREATE TABLE runtime_credentials (
    credential_id text PRIMARY KEY
        CHECK (credential_id ~ '^[a-z][a-z0-9_-]*$' AND octet_length(credential_id) <= 128),
    credential_kind text NOT NULL
        CHECK (credential_kind IN ('otlp-headers@1', 'http-proxy-basic@1', 'http-proxy-bearer@1')),
    encryption_schema_version text NOT NULL
        CHECK (encryption_schema_version = 'contractor.runtime-credentials/v1'),
    key_id text NOT NULL CHECK (key_id ~ '^sha256:[0-9a-f]{64}$'),
    nonce bytea NOT NULL CHECK (octet_length(nonce) = 12),
    ciphertext bytea NOT NULL CHECK (octet_length(ciphertext) BETWEEN 17 AND 32784),
    created_by text NOT NULL CHECK (btrim(created_by) <> '' AND octet_length(created_by) <= 256),
    created_at timestamptz NOT NULL,
    UNIQUE (credential_id, credential_kind)
);

CREATE INDEX runtime_credentials_created_idx
    ON runtime_credentials (created_at, credential_id);

CREATE TABLE runtime_credential_tombstones (
    credential_id text PRIMARY KEY
        REFERENCES runtime_credentials (credential_id) ON DELETE RESTRICT,
    actor_id text NOT NULL CHECK (btrim(actor_id) <> '' AND octet_length(actor_id) <= 256),
    deleted_at timestamptz NOT NULL
);

CREATE TABLE runtime_credential_creations (
    idempotency_key_digest text PRIMARY KEY
        CHECK (idempotency_key_digest ~ '^sha256:[0-9a-f]{64}$'),
    request_mac bytea NOT NULL CHECK (octet_length(request_mac) = 32),
    credential_id text NOT NULL UNIQUE,
    credential_kind text NOT NULL
        CHECK (credential_kind IN ('otlp-headers@1', 'http-proxy-basic@1', 'http-proxy-bearer@1')),
    actor_id text NOT NULL CHECK (btrim(actor_id) <> '' AND octet_length(actor_id) <= 256),
    created_at timestamptz NOT NULL,
    FOREIGN KEY (credential_id, credential_kind)
        REFERENCES runtime_credentials (credential_id, credential_kind) ON DELETE RESTRICT
);

CREATE INDEX runtime_credential_creations_created_idx
    ON runtime_credential_creations (created_at, credential_id);

CREATE OR REPLACE FUNCTION contractor_protect_runtime_credential_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'Runtime credential rows are immutable'
        USING ERRCODE = '23514';
END;
$$;

CREATE TRIGGER runtime_credentials_protect_mutation
BEFORE UPDATE OR DELETE ON runtime_credentials
FOR EACH ROW EXECUTE FUNCTION contractor_protect_runtime_credential_immutable();

CREATE TRIGGER runtime_credential_tombstones_protect_mutation
BEFORE UPDATE OR DELETE ON runtime_credential_tombstones
FOR EACH ROW EXECUTE FUNCTION contractor_protect_runtime_credential_immutable();

CREATE TRIGGER runtime_credential_creations_protect_mutation
BEFORE UPDATE OR DELETE ON runtime_credential_creations
FOR EACH ROW EXECUTE FUNCTION contractor_protect_runtime_credential_immutable();
