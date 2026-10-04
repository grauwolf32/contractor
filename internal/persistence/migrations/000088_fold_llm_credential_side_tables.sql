-- The create operation already reserves an LLM credential ID forever, and a
-- completed delete operation already records who deleted it and when, in the
-- same transaction as the tombstone. The identity and tombstone tables go;
-- credential_operations keeps both guarantees.
CREATE UNIQUE INDEX credential_operations_create_identity
    ON credential_operations (credential_id) WHERE operation_kind = 'create';

ALTER TABLE credential_operations
    DROP CONSTRAINT credential_operations_credential_id_fkey;
ALTER TABLE llm_credentials
    DROP CONSTRAINT llm_credentials_credential_id_fkey;

DROP TABLE llm_credential_tombstones;
DROP TABLE llm_credential_identities;

-- An active record needs its reserved create operation and no completed
-- delete. Locking the create operation serializes activation and deletion of
-- one ID, as the identity row lock did.
CREATE OR REPLACE FUNCTION contractor_check_llm_credential_exclusive_state()
RETURNS trigger
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

-- Completing a delete is the tombstone, so it requires the record to be gone.
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
