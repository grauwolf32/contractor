-- Storage scope creation is a retained diagnostic fact. Scope revision_count
-- remains mutable; only writes naming created_at invoke this guard.
CREATE FUNCTION contractor_protect_artifact_scope_created_at() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.created_at IS DISTINCT FROM OLD.created_at THEN
        RAISE EXCEPTION 'Artifact scope creation time cannot be changed' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER artifact_scopes_protect_creation
    BEFORE UPDATE OF created_at ON artifact_scopes
    FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_scope_created_at();
