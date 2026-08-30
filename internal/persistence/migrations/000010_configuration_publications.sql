CREATE TABLE configuration_publications (
    kind text NOT NULL CHECK (kind IN ('model-policies', 'llm-gateways')),
    name text NOT NULL CHECK (name ~ '^[a-z][a-z0-9_-]*$'),
    version text NOT NULL CHECK (version ~ '^[A-Za-z0-9][A-Za-z0-9._+-]*$'),
    digest text NOT NULL CHECK (digest ~ '^sha256:[0-9a-f]{64}$'),
    request_digest text NOT NULL CHECK (request_digest ~ '^sha256:[0-9a-f]{64}$'),
    idempotency_key_digest text NOT NULL
        CHECK (idempotency_key_digest ~ '^sha256:[0-9a-f]{64}$'),
    actor_id text NOT NULL CHECK (btrim(actor_id) <> ''),
    published_at timestamptz NOT NULL,
    PRIMARY KEY (kind, name, version)
);

CREATE INDEX configuration_publications_published_at_idx
    ON configuration_publications (published_at, kind, name, version);

CREATE OR REPLACE FUNCTION contractor_protect_configuration_publication_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'configuration publication audit rows are immutable'
        USING ERRCODE = '23514';
END;
$$;

CREATE TRIGGER configuration_publications_protect_immutable
BEFORE UPDATE OR DELETE ON configuration_publications
FOR EACH ROW EXECUTE FUNCTION contractor_protect_configuration_publication_immutable();
