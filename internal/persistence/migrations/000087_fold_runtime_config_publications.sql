-- A RuntimeConfig publish receipt was a 1:1 side table of the version it
-- created. The version row now carries the receipt; the built-in version has
-- none.
ALTER TABLE runtime_config_versions
    ADD COLUMN idempotency_key_digest text UNIQUE
        CHECK (idempotency_key_digest ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN request_digest text CHECK (request_digest ~ '^sha256:[0-9a-f]{64}$'),
    ADD CONSTRAINT runtime_config_versions_publication_receipt
        CHECK ((idempotency_key_digest IS NULL) = (request_digest IS NULL));

ALTER TABLE runtime_config_versions DISABLE TRIGGER runtime_config_versions_protect_immutable;
UPDATE runtime_config_versions AS version
   SET idempotency_key_digest = publication.idempotency_key_digest,
       request_digest = publication.request_digest
  FROM runtime_config_publications AS publication
 WHERE publication.config_name = version.name
   AND publication.config_version = version.version
   AND publication.config_digest = version.digest;
ALTER TABLE runtime_config_versions ENABLE TRIGGER runtime_config_versions_protect_immutable;

DROP TABLE runtime_config_publications;
