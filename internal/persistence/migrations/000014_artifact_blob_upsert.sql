CREATE OR REPLACE FUNCTION contractor_protect_artifact_blob_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF ROW(NEW.sha256, NEW.payload, NEW.size_bytes, NEW.created_at)
       IS DISTINCT FROM
       ROW(OLD.sha256, OLD.payload, OLD.size_bytes, OLD.created_at)
    THEN
        RAISE EXCEPTION 'immutable Artifact blob cannot be changed' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER artifact_blobs_immutable ON artifact_blobs;

CREATE TRIGGER artifact_blobs_immutable
BEFORE UPDATE OR DELETE ON artifact_blobs
FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_blob_immutable();
