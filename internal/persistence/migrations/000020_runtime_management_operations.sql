CREATE TABLE runtime_management_operations (
    idempotency_key_digest text PRIMARY KEY
        CHECK (idempotency_key_digest ~ '^sha256:[0-9a-f]{64}$'),
    request_digest text NOT NULL
        CHECK (request_digest ~ '^sha256:[0-9a-f]{64}$'),
    operation_kind text NOT NULL
        CHECK (operation_kind ~ '^[a-z][a-z0-9.-]{0,63}$'),
    resource_id text NOT NULL
        CHECK (resource_id ~ '^[a-z][a-z0-9_-]{0,62}$'),
    result jsonb NOT NULL CHECK (
        jsonb_typeof(result) = 'object'
        AND octet_length(result::text) <= 16384
    ),
    actor_id text NOT NULL
        CHECK (btrim(actor_id) <> '' AND length(actor_id) <= 256),
    performed_at timestamptz NOT NULL
);

CREATE INDEX runtime_management_operations_performed_at_idx
    ON runtime_management_operations (performed_at, idempotency_key_digest);

CREATE TRIGGER runtime_management_operations_protect_immutable
BEFORE UPDATE OR DELETE ON runtime_management_operations
FOR EACH ROW EXECUTE FUNCTION contractor_protect_runtime_config_immutable();
