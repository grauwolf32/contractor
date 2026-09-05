CREATE TABLE owner_queue_controls (
    owner_id text PRIMARY KEY CHECK (
        btrim(owner_id) <> '' AND octet_length(owner_id) <= 256
    ),
    paused boolean NOT NULL DEFAULT false,
    revision bigint NOT NULL DEFAULT 1 CHECK (revision > 0),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE OR REPLACE FUNCTION contractor_protect_owner_queue_control()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.owner_id IS DISTINCT FROM OLD.owner_id THEN
        RAISE EXCEPTION 'Owner Queue control identity cannot be changed' USING ERRCODE = '23514';
    END IF;
    IF NEW.paused IS NOT DISTINCT FROM OLD.paused
        OR NEW.revision <> OLD.revision + 1
        OR NEW.updated_at <= OLD.updated_at
    THEN
        RAISE EXCEPTION 'Owner Queue control update must change state, revision and time' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER owner_queue_controls_protect_mutation
BEFORE UPDATE ON owner_queue_controls
FOR EACH ROW EXECUTE FUNCTION contractor_protect_owner_queue_control();
