-- Policy bytes remain ordinary exact input Artifacts. Only the bounded
-- credential dependency set is denormalized for existing usage/deletion reads.
CREATE FUNCTION contractor_valid_engagement_credential_ids(ids text[]) RETURNS boolean
LANGUAGE sql IMMUTABLE STRICT
BEGIN ATOMIC
    SELECT cardinality(ids) <= 8
       AND (cardinality(ids) = 0 OR (array_ndims(ids) = 1 AND array_lower(ids, 1) = 1))
       AND NOT EXISTS (
           SELECT 1 FROM (
               SELECT id, lag(id, 1, '') OVER (ORDER BY position) AS previous
               FROM unnest(ids) WITH ORDINALITY AS item(id, position)
           ) ordered
           WHERE id IS NULL OR id COLLATE "C" !~ '^[a-z][a-z0-9_-]*$'
              OR octet_length(id) > 128 OR id COLLATE "C" <= previous COLLATE "C"
       );
END;

ALTER TABLE workflow_runs ADD COLUMN engagement_credential_ids text[] NOT NULL DEFAULT '{}',
    ADD CONSTRAINT workflow_runs_engagement_credentials_valid
    CHECK (contractor_valid_engagement_credential_ids(engagement_credential_ids));

CREATE TRIGGER workflow_runs_protect_engagement_credentials
BEFORE UPDATE OF engagement_credential_ids ON workflow_runs
FOR EACH ROW WHEN (OLD.engagement_credential_ids IS DISTINCT FROM NEW.engagement_credential_ids)
EXECUTE FUNCTION contractor_protect_artifact_immutable();

CREATE INDEX workflow_runs_engagement_credentials_idx ON workflow_runs
USING gin (engagement_credential_ids)
WHERE state IN ('initializing', 'pending', 'running', 'waiting', 'cancelling');
