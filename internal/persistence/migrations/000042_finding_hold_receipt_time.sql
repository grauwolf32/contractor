-- Order an Audit's finding proposal holds by their receipt's creation time
-- (V414-019, FND-R186).
--
-- The Audit inbox and the next-Round builder list the receipts an Audit holds
-- in receipt (created_at, receipt_id) order. The hold index carried the hold's
-- own time, so every page probed the receipt of each of the Audit's holds and
-- sorted them before its LIMIT: a page read every hold of the Audit, and a
-- next-Round pass of 51 pages read them 51 times. Each hold now keeps an
-- immutable copy of its receipt's creation time, and the Audit's hold index
-- is ordered by it, so a page walks the index after its cursor and stops at
-- the page.
ALTER TABLE finding_proposal_audit_holds ADD COLUMN receipt_created_at timestamp with time zone;

-- Holds are immutable; the backfill is the only update they get. Every hold's
-- receipt exists (the receipt foreign key cascades), so no hold stays NULL and
-- the NOT NULL below fails rather than leave one.
ALTER TABLE finding_proposal_audit_holds DISABLE TRIGGER finding_proposal_audit_holds_protect_update;
UPDATE finding_proposal_audit_holds AS hold
   SET receipt_created_at = receipt.created_at
  FROM finding_proposal_receipts AS receipt
 WHERE receipt.receipt_id = hold.receipt_id;
ALTER TABLE finding_proposal_audit_holds ENABLE TRIGGER finding_proposal_audit_holds_protect_update;

ALTER TABLE finding_proposal_audit_holds ALTER COLUMN receipt_created_at SET NOT NULL;

-- contractor_copy_finding_hold_receipt_time copies the receipt's creation time
-- into a new hold, so no writer can store another one: a hold whose receipt is
-- missing, or that names a different time, is rejected. A receipt's creation
-- time never changes (contractor_protect_finding_proposal_receipt_retention),
-- and finding_proposal_audit_holds_protect_update rejects every hold update,
-- so the copy stays equal to the receipt's.
CREATE FUNCTION contractor_copy_finding_hold_receipt_time() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    receipt_time timestamp with time zone;
BEGIN
    SELECT receipt.created_at INTO receipt_time
      FROM finding_proposal_receipts AS receipt
     WHERE receipt.receipt_id = NEW.receipt_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'finding proposal Audit hold has no receipt' USING ERRCODE = '23503';
    END IF;
    IF NEW.receipt_created_at IS NOT NULL AND NEW.receipt_created_at <> receipt_time THEN
        RAISE EXCEPTION 'finding proposal Audit hold names another receipt time' USING ERRCODE = '23514';
    END IF;
    NEW.receipt_created_at := receipt_time;
    RETURN NEW;
END;
$$;

CREATE TRIGGER finding_proposal_audit_holds_copy_receipt_time
    BEFORE INSERT ON finding_proposal_audit_holds
    FOR EACH ROW EXECUTE FUNCTION contractor_copy_finding_hold_receipt_time();

-- No statement reads an Audit's holds in hold-time order; the purge batch and
-- the other Audit-scoped reads need only the leading audit_id, which the new
-- order keeps.
DROP INDEX finding_proposal_audit_holds_audit_idx;
CREATE INDEX finding_proposal_audit_holds_audit_receipt_idx
    ON finding_proposal_audit_holds USING btree (audit_id, receipt_created_at, receipt_id);
