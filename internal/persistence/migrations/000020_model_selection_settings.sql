-- Durable defaults for new executions. Model capabilities and immutable
-- model-specific policy refs remain in the operator catalog.
CREATE FUNCTION contractor_valid_model_selection(default_model text, roles jsonb)
RETURNS boolean LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT CASE WHEN jsonb_typeof(roles) IS DISTINCT FROM 'object' THEN false ELSE
        default_model ~ '^[a-z0-9][a-z0-9.-]{0,74}$'
        AND NOT EXISTS (
            SELECT 1 FROM jsonb_each(roles) AS r(key, value)
            WHERE key NOT IN ('planner', 'worker', 'summarizer', 'audit_completion_worker')
               OR jsonb_typeof(value) IS DISTINCT FROM 'string'
               OR (value #>> '{}') !~ '^[a-z0-9][a-z0-9.-]{0,74}$'
        ) END
$$;

CREATE TABLE model_selection_settings (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    default_model text NOT NULL,
    role_models jsonb NOT NULL,
    revision numeric(20,0) NOT NULL DEFAULT 1
        CHECK (revision BETWEEN 1 AND 18446744073709551615),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT model_selection_settings_value CHECK (
        contractor_valid_model_selection(default_model, role_models)
    )
);

CREATE FUNCTION contractor_protect_model_selection_settings() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'Model selection singleton cannot be deleted' USING ERRCODE = '23514';
    END IF;
    IF NEW.singleton IS DISTINCT FROM OLD.singleton
       OR (NEW.default_model IS NOT DISTINCT FROM OLD.default_model
           AND NEW.role_models IS NOT DISTINCT FROM OLD.role_models)
       OR NEW.revision IS DISTINCT FROM OLD.revision + 1
       OR NEW.updated_at <= OLD.updated_at THEN
        RAISE EXCEPTION 'Model selection update must be one semantic revision' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER model_selection_settings_protect_mutation
BEFORE DELETE OR UPDATE ON model_selection_settings
FOR EACH ROW EXECUTE FUNCTION contractor_protect_model_selection_settings();
