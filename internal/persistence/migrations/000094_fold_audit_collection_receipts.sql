-- An AuditExecution has at most one collection receipt, and the receipt's Run
-- and terminal observation were copies of the execution's own columns. The
-- collection outcome now lives on the execution; its columns are set once,
-- together with the move to 'collected', and never change afterwards.
ALTER TABLE audit_executions
    ADD COLUMN collection_receipt_id text UNIQUE
        CHECK (collection_receipt_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'),
    ADD COLUMN collection_disposition text CHECK (collection_disposition IN (
        'accepted-result', 'missing-output', 'invalid-result',
        'execution-failed', 'execution-cancelled', 'collection-contract-invalid'
    )),
    ADD COLUMN collection_source_output_ref jsonb CHECK (
        jsonb_typeof(collection_source_output_ref) = 'object'
        AND octet_length(collection_source_output_ref::text) <= 4096
    ),
    ADD COLUMN collection_source_output_digest text
        CHECK (collection_source_output_digest ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN collection_retained_refs jsonb CHECK (
        jsonb_typeof(collection_retained_refs) = 'array'
        AND octet_length(collection_retained_refs::text) <= 8388608
    ),
    ADD COLUMN collection_error_code text CHECK (
        btrim(collection_error_code) <> '' AND octet_length(collection_error_code) <= 128
    ),
    ADD COLUMN collection_request_digest text
        CHECK (collection_request_digest ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN collected_at timestamptz;

-- The receipt copied these columns from its execution when it was written,
-- and they are fixed once the execution starts collecting. Refuse to drop a
-- receipt that disagrees rather than lose its observation.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
          FROM audit_collection_receipts AS receipt
          JOIN audit_executions AS execution USING (execution_id)
         WHERE ROW(receipt.audit_id, receipt.run_id, receipt.terminal_outcome,
                   receipt.terminal_run_generation, receipt.terminal_run_sequence)
               IS DISTINCT FROM
               ROW(execution.audit_id, execution.run_id, execution.terminal_outcome,
                   execution.terminal_run_generation, execution.terminal_run_sequence)
    ) THEN
        RAISE EXCEPTION 'Audit collection receipt disagrees with its execution';
    END IF;
END;
$$;

ALTER TABLE audit_executions DISABLE TRIGGER USER;
UPDATE audit_executions AS execution
   SET collection_receipt_id = receipt.receipt_id,
       collection_disposition = receipt.disposition,
       collection_source_output_ref = receipt.source_output_ref,
       collection_source_output_digest = receipt.source_output_digest,
       collection_retained_refs = receipt.retained_refs,
       collection_error_code = receipt.error_code,
       collection_request_digest = receipt.request_digest,
       collected_at = receipt.created_at
  FROM audit_collection_receipts AS receipt
 WHERE receipt.execution_id = execution.execution_id;
ALTER TABLE audit_executions ENABLE TRIGGER USER;

ALTER TABLE audit_executions
    ADD CONSTRAINT audit_executions_collection_shape CHECK (
        (collection_receipt_id IS NULL) = (collection_disposition IS NULL)
        AND (collection_receipt_id IS NULL) = (collection_retained_refs IS NULL)
        AND (collection_receipt_id IS NULL) = (collection_request_digest IS NULL)
        AND (collection_receipt_id IS NULL) = (collected_at IS NULL)
        AND (collection_receipt_id IS NOT NULL OR
             (collection_error_code IS NULL AND collection_source_output_ref IS NULL))
    ),
    ADD CONSTRAINT audit_executions_collection_source_shape CHECK (
        (collection_source_output_ref IS NULL) = (collection_source_output_digest IS NULL)
        AND (collection_receipt_id IS NULL OR
             (collection_disposition IN ('accepted-result', 'invalid-result')) = (collection_source_output_ref IS NOT NULL))
    ),
    ADD CONSTRAINT audit_executions_collection_observation_shape CHECK (
        collection_receipt_id IS NULL
        OR (run_id IS NOT NULL AND terminal_outcome IS NOT NULL AND terminal_run_generation IS NOT NULL
            AND btrim(terminal_run_generation) <> '' AND terminal_run_sequence >= 0)
        OR (run_id IS NULL AND terminal_outcome = 'submission-failed'
            AND terminal_run_generation IS NULL AND terminal_run_sequence IS NULL)
    );

ALTER TABLE audit_finding_assessments
    DROP CONSTRAINT audit_finding_assessments_collection_fkey,
    ADD CONSTRAINT audit_finding_assessments_collection_fkey
        FOREIGN KEY (collection_receipt_id)
        REFERENCES audit_executions(collection_receipt_id) ON DELETE RESTRICT;

DROP TABLE audit_collection_receipts;
DROP FUNCTION contractor_protect_audit_receipt();

-- A collected execution keeps its receipt and the observation it records.
CREATE FUNCTION contractor_protect_audit_execution_collection()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.collection_receipt_id IS NOT NULL
        AND ROW(NEW.execution_id, NEW.audit_id, NEW.run_id, NEW.terminal_outcome,
                NEW.terminal_run_generation, NEW.terminal_run_sequence,
                NEW.collection_receipt_id, NEW.collection_disposition,
                NEW.collection_source_output_ref, NEW.collection_source_output_digest,
                NEW.collection_retained_refs, NEW.collection_error_code,
                NEW.collection_request_digest, NEW.collected_at)
            IS DISTINCT FROM
            ROW(OLD.execution_id, OLD.audit_id, OLD.run_id, OLD.terminal_outcome,
                OLD.terminal_run_generation, OLD.terminal_run_sequence,
                OLD.collection_receipt_id, OLD.collection_disposition,
                OLD.collection_source_output_ref, OLD.collection_source_output_digest,
                OLD.collection_retained_refs, OLD.collection_error_code,
                OLD.collection_request_digest, OLD.collected_at)
    THEN
        RAISE EXCEPTION 'Audit collection receipts are immutable' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER audit_executions_protect_collection
BEFORE UPDATE ON audit_executions
FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_execution_collection();
