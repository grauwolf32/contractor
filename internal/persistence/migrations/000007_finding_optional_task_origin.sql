-- Run-level incidental observations retain Audit membership without inventing a Task.
ALTER TABLE finding_proposal_receipts
    DROP CONSTRAINT finding_proposal_receipts_audit_shape;
ALTER TABLE finding_proposal_receipts
    ADD CONSTRAINT finding_proposal_receipts_audit_shape CHECK (
        (audit_execution_id IS NULL AND audit_id IS NULL AND audit_task_id IS NULL)
        OR (audit_execution_id IS NOT NULL AND audit_id IS NOT NULL)
    );
ALTER TABLE finding_proposal_receipts
    ADD CONSTRAINT finding_proposal_receipts_hypothesis_task CHECK (
        audit_hypothesis IS NULL OR audit_task_id IS NOT NULL
    );
