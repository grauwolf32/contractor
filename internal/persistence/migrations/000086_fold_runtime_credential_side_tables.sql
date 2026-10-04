-- A Runtime credential's create receipt and deletion marker were 1:1 side
-- tables only because the credential row rejected every UPDATE. They become
-- columns: the receipt is written with the row, and the deletion marker is
-- the one change the row still accepts.
ALTER TABLE runtime_credentials
    ADD COLUMN idempotency_key_digest text UNIQUE
        CHECK (idempotency_key_digest ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN request_mac bytea CHECK (octet_length(request_mac) = 32),
    ADD COLUMN deleted_by text
        CHECK (btrim(deleted_by) <> '' AND octet_length(deleted_by) <= 256),
    ADD COLUMN deleted_at timestamptz,
    ADD CONSTRAINT runtime_credentials_creation_receipt
        CHECK ((idempotency_key_digest IS NULL) = (request_mac IS NULL)),
    ADD CONSTRAINT runtime_credentials_deletion_marker
        CHECK ((deleted_by IS NULL) = (deleted_at IS NULL));

ALTER TABLE runtime_credentials DISABLE TRIGGER runtime_credentials_protect_mutation;
UPDATE runtime_credentials AS credential
   SET idempotency_key_digest = creation.idempotency_key_digest,
       request_mac = creation.request_mac
  FROM runtime_credential_creations AS creation
 WHERE creation.credential_id = credential.credential_id;
UPDATE runtime_credentials AS credential
   SET deleted_by = tombstone.actor_id,
       deleted_at = tombstone.deleted_at
  FROM runtime_credential_tombstones AS tombstone
 WHERE tombstone.credential_id = credential.credential_id;
ALTER TABLE runtime_credentials ENABLE TRIGGER runtime_credentials_protect_mutation;

DROP TABLE runtime_credential_creations;
DROP TABLE runtime_credential_tombstones;

-- Every field is immutable except the deletion marker, which may be set once.
CREATE OR REPLACE FUNCTION contractor_protect_runtime_credential_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'UPDATE' AND OLD.deleted_at IS NULL AND NEW.deleted_at IS NOT NULL
        AND ROW(NEW.credential_id, NEW.credential_kind, NEW.encryption_schema_version, NEW.key_id,
                NEW.nonce, NEW.ciphertext, NEW.created_by, NEW.created_at,
                NEW.idempotency_key_digest, NEW.request_mac)
            IS NOT DISTINCT FROM
            ROW(OLD.credential_id, OLD.credential_kind, OLD.encryption_schema_version, OLD.key_id,
                OLD.nonce, OLD.ciphertext, OLD.created_by, OLD.created_at,
                OLD.idempotency_key_digest, OLD.request_mac)
    THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'Runtime credential rows are immutable'
        USING ERRCODE = '23514';
END;
$$;
