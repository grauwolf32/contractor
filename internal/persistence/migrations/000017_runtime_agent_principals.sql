CREATE OR REPLACE FUNCTION contractor_valid_runtime_agent_labels(candidate text[])
RETURNS boolean
LANGUAGE sql
IMMUTABLE
STRICT
AS $$
    SELECT cardinality(candidate) <= 32
       AND candidate = ARRAY(
           SELECT value
           FROM unnest(candidate) AS value
           ORDER BY value COLLATE "C"
       )
       AND NOT EXISTS (
           SELECT 1
           FROM unnest(candidate) AS value
           WHERE value !~ '^[a-z][a-z0-9_-]*$'
              OR length(value) > 63
              OR value = 'default'
       )
       AND cardinality(candidate) = (
           SELECT count(DISTINCT value) FROM unnest(candidate) AS value
       )
$$;

CREATE TABLE runtime_agent_principals (
    runtime_agent_id text PRIMARY KEY CHECK (runtime_agent_id ~ '^[0-9a-f]{64}$'),
    labels text[] NOT NULL CHECK (contractor_valid_runtime_agent_labels(labels)),
    label_revision numeric(20, 0) NOT NULL
        CHECK (label_revision >= 1 AND label_revision <= 18446744073709551615),
    created_by text NOT NULL CHECK (btrim(created_by) <> '' AND length(created_by) <= 256),
    created_at timestamptz NOT NULL,
    updated_by text NOT NULL CHECK (btrim(updated_by) <> '' AND length(updated_by) <= 256),
    updated_at timestamptz NOT NULL
);

CREATE INDEX runtime_agent_principals_labels_idx
    ON runtime_agent_principals USING gin (labels);

CREATE INDEX runtime_agent_principals_updated_idx
    ON runtime_agent_principals (updated_at, runtime_agent_id);

CREATE OR REPLACE FUNCTION contractor_protect_runtime_agent_principal()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.runtime_agent_id IS DISTINCT FROM OLD.runtime_agent_id
       OR NEW.created_by IS DISTINCT FROM OLD.created_by
       OR NEW.created_at IS DISTINCT FROM OLD.created_at
       OR NEW.label_revision IS DISTINCT FROM OLD.label_revision + 1
       OR NEW.labels IS NOT DISTINCT FROM OLD.labels
    THEN
        RAISE EXCEPTION 'Runtime Agent principal update must be one semantic label revision'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER runtime_agent_principals_protect_mutation
BEFORE UPDATE ON runtime_agent_principals
FOR EACH ROW EXECUTE FUNCTION contractor_protect_runtime_agent_principal();
