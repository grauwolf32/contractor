-- Extend the existing Audit review authority beyond finding triage. Action
-- approvals share the request/decision ledger so there is one exact-subject,
-- CAS/idempotency boundary for all human authority.

ALTER TABLE audit_review_decisions
    DROP CONSTRAINT audit_review_decisions_request_fkey;

ALTER TABLE audit_review_requests
    DROP CONSTRAINT audit_review_requests_request_id_audit_id_finding_id_key;

DROP INDEX audit_review_requests_pending_subject_unique;

ALTER TABLE audit_review_requests
    ALTER COLUMN finding_id DROP NOT NULL,
    ADD COLUMN subject_kind text,
    ADD COLUMN subject_id text;

UPDATE audit_review_requests
   SET subject_kind = 'finding', subject_id = finding_id;

ALTER TABLE audit_review_requests
    ALTER COLUMN subject_kind SET NOT NULL,
    ALTER COLUMN subject_id SET NOT NULL,
    ADD CONSTRAINT audit_review_requests_subject_kind_check CHECK (
        subject_kind IN ('finding', 'audit-item-action', 'audit-report')
    ),
    ADD CONSTRAINT audit_review_requests_subject_id_check CHECK (
        subject_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'
    ),
    ADD CONSTRAINT audit_review_requests_subject_shape CHECK (
        (subject_kind = 'finding' AND finding_id = subject_id)
        OR (subject_kind <> 'finding' AND finding_id IS NULL)
    ),
    ADD CONSTRAINT audit_review_requests_request_audit_unique
        UNIQUE (request_id, audit_id);

CREATE UNIQUE INDEX audit_review_requests_pending_subject_unique
    ON audit_review_requests (
        audit_id, subject_kind, subject_id, kind,
        subject_revision, subject_digest
    ) WHERE state = 'pending';

ALTER TABLE audit_review_decisions
    ALTER COLUMN finding_id DROP NOT NULL,
    ALTER COLUMN verdict DROP NOT NULL,
    ADD COLUMN action text;

UPDATE audit_review_decisions SET action = verdict;

ALTER TABLE audit_review_decisions
    ALTER COLUMN action SET NOT NULL,
    ADD CONSTRAINT audit_review_decisions_action_check CHECK (
        action IN (
            'true_positive', 'false_positive', 'duplicate', 'reopen',
            'needs_evidence', 'approve', 'reject'
        )
    ),
    DROP CONSTRAINT audit_review_decisions_shape,
    ADD CONSTRAINT audit_review_decisions_shape CHECK (
        (
            finding_id IS NOT NULL
            AND verdict IS NOT NULL
            AND action = verdict
            AND (verdict = 'true_positive') = (severity IS NOT NULL)
            AND (verdict = 'duplicate') = (duplicate_target_id IS NOT NULL)
        ) OR (
            finding_id IS NULL
            AND verdict IS NULL
            AND severity IS NULL
            AND duplicate_target_id IS NULL
            AND action IN ('approve', 'reject')
        )
    ),
    ADD CONSTRAINT audit_review_decisions_request_fkey
        FOREIGN KEY (request_id, audit_id)
        REFERENCES audit_review_requests(request_id, audit_id) ON DELETE RESTRICT;

ALTER TABLE audit_items
    ADD COLUMN approval_kind text NOT NULL DEFAULT 'none' CHECK (
        approval_kind IN (
            'none', 'active-check-approval', 'requirement-applicability'
        )
    ),
    ADD COLUMN approval_subject_digest text CHECK (
        approval_subject_digest IS NULL
        OR approval_subject_digest ~ '^sha256:[0-9a-f]{64}$'
    ),
    ADD CONSTRAINT audit_items_approval_shape CHECK (
        (approval_kind = 'none') = (approval_subject_digest IS NULL)
    );

CREATE INDEX audit_items_pending_approval_idx
    ON audit_items (audit_id, state, approval_kind, item_id)
    WHERE state = 'awaiting_review';

-- Proposed report bytes remain immutable while the human decision is pending.
-- Acceptance promotes their exact descriptors into the ordinary report links;
-- rejection retains the proposal and decision as terminal provenance.
CREATE TABLE audit_report_candidates (
    audit_id text PRIMARY KEY REFERENCES audits(audit_id) ON DELETE CASCADE,
    request_id text NOT NULL,
    round_id text NOT NULL,
    subject_revision bigint NOT NULL CHECK (subject_revision > 0),
    subject_digest text NOT NULL CHECK (subject_digest ~ '^sha256:[0-9a-f]{64}$'),
    machine_link jsonb NOT NULL CHECK (
        jsonb_typeof(machine_link) = 'object'
        AND octet_length(machine_link::text) <= 16384
    ),
    summary_link jsonb NOT NULL CHECK (
        jsonb_typeof(summary_link) = 'object'
        AND octet_length(summary_link::text) <= 16384
    ),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT audit_report_candidates_round_fkey
        FOREIGN KEY (round_id, audit_id)
        REFERENCES audit_rounds(round_id, audit_id) ON DELETE CASCADE,
    CONSTRAINT audit_report_candidates_request_fkey
        FOREIGN KEY (request_id, audit_id)
        REFERENCES audit_review_requests(request_id, audit_id) ON DELETE RESTRICT,
    UNIQUE (request_id),
    UNIQUE (audit_id, subject_digest)
);
