-- A receipt's retention state was a 1:1 side table only because the receipt
-- row rejected every UPDATE. The receipt carries it now; its trigger keeps
-- every other column immutable and still forbids DELETE.
ALTER TABLE finding_proposal_receipts
    ADD COLUMN retention_state text NOT NULL DEFAULT 'source-held'
        CHECK (retention_state IN ('source-held', 'audit-held', 'discarded')),
    ADD COLUMN source_run_deleted_at timestamptz,
    ADD COLUMN discarded_at timestamptz,
    ADD COLUMN retention_updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    ADD CONSTRAINT finding_proposal_receipts_retention_shape CHECK (
        (retention_state = 'source-held' AND source_run_deleted_at IS NULL AND discarded_at IS NULL)
        OR (retention_state = 'audit-held' AND discarded_at IS NULL)
        OR (retention_state = 'discarded' AND source_run_deleted_at IS NOT NULL AND discarded_at IS NOT NULL)
    );

ALTER TABLE finding_proposal_receipts DISABLE TRIGGER finding_proposal_receipts_protect_immutable;
UPDATE finding_proposal_receipts AS receipt
   SET retention_state = retention.state,
       source_run_deleted_at = retention.source_run_deleted_at,
       discarded_at = retention.discarded_at,
       retention_updated_at = retention.updated_at
  FROM finding_proposal_retention AS retention
 WHERE retention.receipt_id = receipt.receipt_id;
ALTER TABLE finding_proposal_receipts ENABLE TRIGGER finding_proposal_receipts_protect_immutable;

DROP TABLE finding_proposal_retention;

CREATE FUNCTION contractor_protect_finding_proposal_receipt_retention()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'UPDATE'
        AND to_jsonb(NEW) - ARRAY['retention_state', 'source_run_deleted_at', 'discarded_at', 'retention_updated_at']
            = to_jsonb(OLD) - ARRAY['retention_state', 'source_run_deleted_at', 'discarded_at', 'retention_updated_at']
    THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'finding proposal receipts are immutable' USING ERRCODE = '23514';
END;
$$;

DROP TRIGGER finding_proposal_receipts_protect_immutable ON finding_proposal_receipts;
CREATE TRIGGER finding_proposal_receipts_protect_immutable
BEFORE UPDATE OR DELETE ON finding_proposal_receipts
FOR EACH ROW EXECUTE FUNCTION contractor_protect_finding_proposal_receipt_retention();

CREATE OR REPLACE FUNCTION contractor_reconcile_finding_retention()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    target_receipt text;
BEGIN
    IF TG_OP = 'DELETE' THEN
        target_receipt := OLD.receipt_id;
    ELSE
        target_receipt := NEW.receipt_id;
    END IF;
    IF EXISTS (
        SELECT 1 FROM finding_proposal_audit_holds WHERE receipt_id = target_receipt
    ) THEN
        UPDATE finding_proposal_receipts
           SET retention_state = 'audit-held', discarded_at = NULL, retention_updated_at = clock_timestamp()
         WHERE receipt_id = target_receipt;
    ELSE
        UPDATE finding_proposal_receipts
           SET retention_state = CASE WHEN source_run_deleted_at IS NULL THEN 'source-held' ELSE 'discarded' END,
               discarded_at = CASE WHEN source_run_deleted_at IS NULL THEN NULL ELSE COALESCE(discarded_at, clock_timestamp()) END,
               retention_updated_at = clock_timestamp()
         WHERE receipt_id = target_receipt;
    END IF;
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END;
$$;
