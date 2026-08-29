ALTER TABLE workflow_runs
    ADD COLUMN request_idempotency_key text,
    ADD COLUMN request_digest text,
    ADD CONSTRAINT workflow_runs_request_idempotency_shape CHECK (
        (request_idempotency_key IS NULL AND request_digest IS NULL)
        OR (
            request_idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
            AND request_digest ~ '^sha256:[0-9a-f]{64}$'
        )
    );

CREATE UNIQUE INDEX workflow_runs_owner_idempotency_key
    ON workflow_runs (owner_id, request_idempotency_key)
    WHERE request_idempotency_key IS NOT NULL;
