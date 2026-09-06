CREATE TABLE artifact_blob_settings (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    backend text CHECK (backend IN ('postgresql', 'filesystem'))
);
INSERT INTO artifact_blob_settings (singleton, backend)
VALUES (true, CASE WHEN EXISTS (SELECT 1 FROM artifact_blobs) THEN 'postgresql' END);

ALTER TABLE artifact_blobs
    ALTER COLUMN payload DROP NOT NULL,
    ADD COLUMN backend text NOT NULL DEFAULT 'postgresql',
    ADD COLUMN object_key text,
    DROP CONSTRAINT artifact_blobs_size_bytes_check,
    DROP CONSTRAINT artifact_blobs_check1,
    ADD CONSTRAINT artifact_blobs_size_check CHECK (size_bytes BETWEEN 0 AND 67108864),
    ADD CONSTRAINT artifact_blobs_storage_check CHECK (
        (backend = 'postgresql' AND object_key IS NULL AND payload IS NOT NULL
         AND size_bytes = octet_length(payload) AND sha256 = pg_catalog.sha256(payload))
        OR
        (backend = 'filesystem' AND payload IS NULL AND object_key IS NOT NULL
         AND object_key ~ '^[0-9a-f]{2}/[0-9a-f]{32}$')
    );
CREATE UNIQUE INDEX artifact_blobs_object_key_idx ON artifact_blobs(object_key)
    WHERE object_key IS NOT NULL;

CREATE FUNCTION artifact_blob_backend_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    -- Reads of already selected settings need no global serialization.
    IF NOT EXISTS (SELECT 1 FROM artifact_blob_settings WHERE backend = NEW.backend) THEN
        UPDATE artifact_blob_settings SET backend = NEW.backend
        WHERE singleton AND backend IS NULL;
        IF NOT EXISTS (SELECT 1 FROM artifact_blob_settings WHERE backend = NEW.backend) THEN
            RAISE EXCEPTION 'artifact blob backend mismatch' USING ERRCODE = '23514';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER artifact_blob_backend_guard BEFORE INSERT OR UPDATE ON artifact_blobs
FOR EACH ROW EXECUTE FUNCTION artifact_blob_backend_guard();
