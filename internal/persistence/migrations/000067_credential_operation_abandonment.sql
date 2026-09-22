-- A prepared create whose remote alias is confirmed absent but whose replay
-- can no longer validate (for example, a removed exact ModelPolicy) becomes a
-- terminal failed outcome instead of blocking every future startup.
ALTER TABLE credential_operations
    DROP CONSTRAINT credential_operations_phase_check;

ALTER TABLE credential_operations
    ADD CONSTRAINT credential_operations_phase_check
        CHECK (phase IN ('prepared', 'completed', 'abandoned'));

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
    RETURN NEW;
END;
$$;
