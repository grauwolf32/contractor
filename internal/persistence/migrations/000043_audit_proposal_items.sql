-- Bind each later-round item to the exact admitted proposal check that caused
-- it. The relation is immutable provenance and the uniqueness key is also the
-- durable consume-once fence used by next-Round acceptance.

ALTER TABLE audit_rounds
    ADD COLUMN acceptance_digest text CHECK (
        acceptance_digest IS NULL
        OR acceptance_digest ~ '^sha256:[0-9a-f]{64}$'
    );

CREATE TABLE audit_proposal_items (
    audit_id text NOT NULL,
    receipt_id text NOT NULL,
    proposed_check_ordinal integer NOT NULL CHECK (
        proposed_check_ordinal >= 0 AND proposed_check_ordinal < 512
    ),
    round_id text NOT NULL,
    item_id text NOT NULL,
    proposal_ref jsonb NOT NULL CHECK (
        jsonb_typeof(proposal_ref) = 'object'
        AND octet_length(proposal_ref::text) <= 4096
    ),
    proposal_digest text NOT NULL CHECK (
        proposal_digest ~ '^sha256:[0-9a-f]{64}$'
    ),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (audit_id, receipt_id, proposed_check_ordinal),
    UNIQUE (item_id),
    CONSTRAINT audit_proposal_items_hold_fkey
        FOREIGN KEY (receipt_id, audit_id)
        REFERENCES finding_proposal_audit_holds(receipt_id, audit_id)
        ON DELETE CASCADE,
    CONSTRAINT audit_proposal_items_item_fkey
        FOREIGN KEY (item_id, audit_id, round_id)
        REFERENCES audit_items(item_id, audit_id, round_id)
        ON DELETE CASCADE
);

CREATE INDEX audit_proposal_items_receipt_idx
    ON audit_proposal_items (audit_id, receipt_id, proposed_check_ordinal);

CREATE TRIGGER audit_proposal_items_protect_update
BEFORE UPDATE ON audit_proposal_items
FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_finding_history();
