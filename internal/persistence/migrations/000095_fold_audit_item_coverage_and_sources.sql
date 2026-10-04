-- Every Audit item gets exactly one coverage row in the statement that
-- creates it, and a later-round item has at most one admitted proposal check
-- as its source. Both now live on the item. Coverage stays mutable; the
-- proposal source is set once and then immutable provenance, and its
-- uniqueness is still the consume-once fence used by next-Round acceptance.
ALTER TABLE audit_items
    ADD COLUMN coverage_status text CHECK (coverage_status IN (
        'not-tested', 'inconclusive', 'satisfied', 'violated',
        'not-applicable', 'blocked', 'excluded',
        'traced-complete', 'traced-partial', 'unmapped'
    )),
    ADD COLUMN coverage_requested jsonb CHECK (
        jsonb_typeof(coverage_requested) = 'array' AND octet_length(coverage_requested::text) <= 1048576
    ),
    ADD COLUMN coverage_completed jsonb CHECK (
        jsonb_typeof(coverage_completed) = 'array' AND octet_length(coverage_completed::text) <= 1048576
    ),
    ADD COLUMN coverage_gaps jsonb CHECK (
        jsonb_typeof(coverage_gaps) = 'array' AND octet_length(coverage_gaps::text) <= 1048576
    ),
    ADD COLUMN coverage_rationale text NOT NULL DEFAULT '' CHECK (octet_length(coverage_rationale) <= 4096),
    ADD COLUMN coverage_result_ref jsonb,
    ADD COLUMN coverage_result_digest text,
    ADD COLUMN coverage_updated_at timestamptz,
    ADD COLUMN proposal_receipt_id text,
    ADD COLUMN proposal_check_ordinal integer CHECK (
        proposal_check_ordinal >= 0 AND proposal_check_ordinal < 512
    ),
    ADD COLUMN proposal_ref jsonb CHECK (
        jsonb_typeof(proposal_ref) = 'object' AND octet_length(proposal_ref::text) <= 4096
    ),
    ADD COLUMN proposal_digest text CHECK (proposal_digest ~ '^sha256:[0-9a-f]{64}$');

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM audit_items AS item
         WHERE NOT EXISTS (SELECT 1 FROM audit_coverage_rows AS coverage WHERE coverage.item_id = item.item_id)
    ) THEN
        RAISE EXCEPTION 'Audit item has no coverage row';
    END IF;
END;
$$;

UPDATE audit_items AS item
   SET coverage_status = coverage.status,
       coverage_requested = coverage.requested,
       coverage_completed = coverage.completed,
       coverage_gaps = coverage.gaps,
       coverage_rationale = coverage.rationale,
       coverage_result_ref = coverage.result_ref,
       coverage_result_digest = coverage.result_digest,
       coverage_updated_at = coverage.updated_at
  FROM audit_coverage_rows AS coverage
 WHERE coverage.item_id = item.item_id;

UPDATE audit_items AS item
   SET proposal_receipt_id = source.receipt_id,
       proposal_check_ordinal = source.proposed_check_ordinal,
       proposal_ref = source.proposal_ref,
       proposal_digest = source.proposal_digest
  FROM audit_proposal_items AS source
 WHERE source.item_id = item.item_id;

ALTER TABLE audit_items
    ALTER COLUMN coverage_status SET NOT NULL,
    ALTER COLUMN coverage_requested SET NOT NULL,
    ALTER COLUMN coverage_completed SET NOT NULL,
    ALTER COLUMN coverage_gaps SET NOT NULL,
    ALTER COLUMN coverage_updated_at SET NOT NULL,
    ALTER COLUMN coverage_updated_at SET DEFAULT clock_timestamp(),
    ADD CONSTRAINT audit_items_coverage_result_shape CHECK (
        (coverage_result_ref IS NULL) = (coverage_result_digest IS NULL)
        AND (coverage_result_ref IS NULL OR (
            jsonb_typeof(coverage_result_ref) = 'object'
            AND octet_length(coverage_result_ref::text) <= 4096
            AND coverage_result_digest ~ '^sha256:[0-9a-f]{64}$'
        ))
    ),
    ADD CONSTRAINT audit_items_proposal_shape CHECK (
        (proposal_receipt_id IS NULL) = (proposal_check_ordinal IS NULL)
        AND (proposal_receipt_id IS NULL) = (proposal_ref IS NULL)
        AND (proposal_receipt_id IS NULL) = (proposal_digest IS NULL)
    ),
    -- Consume-once fence: an admitted proposal check yields at most one item.
    ADD CONSTRAINT audit_items_proposal_check_once
        UNIQUE (audit_id, proposal_receipt_id, proposal_check_ordinal),
    -- Holds go only with their Audit, whose purge cascades to its items, as
    -- the old source rows did.
    ADD CONSTRAINT audit_items_proposal_hold_fkey
        FOREIGN KEY (proposal_receipt_id, audit_id)
        REFERENCES finding_proposal_audit_holds(receipt_id, audit_id) ON DELETE CASCADE;

DROP TABLE audit_coverage_rows;
DROP TABLE audit_proposal_items;
-- Only the coverage foreign key used this key.
ALTER TABLE audit_items DROP CONSTRAINT audit_items_coverage_identity;

CREATE FUNCTION contractor_protect_audit_item_source()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.proposal_receipt_id IS NOT NULL
        AND ROW(NEW.proposal_receipt_id, NEW.proposal_check_ordinal, NEW.proposal_ref, NEW.proposal_digest)
        IS DISTINCT FROM
        ROW(OLD.proposal_receipt_id, OLD.proposal_check_ordinal, OLD.proposal_ref, OLD.proposal_digest)
    THEN
        RAISE EXCEPTION 'Audit finding history is immutable' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER audit_items_protect_source
BEFORE UPDATE ON audit_items
FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_item_source();
