CREATE TABLE scheduler_settings (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    max_concurrent_runs integer NOT NULL
        CHECK (max_concurrent_runs BETWEEN 1 AND 32),
    revision numeric(20, 0) NOT NULL
        CHECK (revision >= 1 AND revision <= 18446744073709551615),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

INSERT INTO scheduler_settings (singleton, max_concurrent_runs, revision)
VALUES (true, 1, 1);

CREATE OR REPLACE FUNCTION contractor_protect_scheduler_settings()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'Scheduler settings singleton cannot be deleted'
            USING ERRCODE = '23514';
    END IF;
    IF NEW.singleton IS DISTINCT FROM OLD.singleton
        OR NEW.max_concurrent_runs IS NOT DISTINCT FROM OLD.max_concurrent_runs
        OR NEW.revision IS DISTINCT FROM OLD.revision + 1
        OR NEW.updated_at <= OLD.updated_at
    THEN
        RAISE EXCEPTION 'Scheduler settings update must be one semantic revision'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER scheduler_settings_protect_mutation
BEFORE UPDATE OR DELETE ON scheduler_settings
FOR EACH ROW EXECUTE FUNCTION contractor_protect_scheduler_settings();
