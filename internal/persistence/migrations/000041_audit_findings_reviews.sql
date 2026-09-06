-- Human finding review is an owner-authenticated control-plane concern.  The
-- tables below retain exact proposal/result provenance while keeping every
-- analyst decision immutable and bound to one finding revision.

CREATE TABLE audit_findings (
    finding_id text PRIMARY KEY CHECK (
        finding_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'
    ),
    audit_id text NOT NULL REFERENCES audits(audit_id) ON DELETE CASCADE,
    first_receipt_id text NOT NULL REFERENCES finding_proposal_receipts(receipt_id) ON DELETE RESTRICT,
    first_proposal_ref jsonb NOT NULL CHECK (
        jsonb_typeof(first_proposal_ref) = 'object'
        AND octet_length(first_proposal_ref::text) <= 4096
    ),
    state text NOT NULL DEFAULT 'proposed' CHECK (state IN (
        'proposed', 'confirmed', 'rejected', 'duplicate', 'needs-evidence'
    )),
    rejection_reason text CHECK (
        rejection_reason IS NULL OR rejection_reason IN ('false-positive', 'policy', 'out-of-scope')
    ),
    duplicate_target_id text,
    current_assessment_id text,
    current_decision_id text,
    revision bigint NOT NULL DEFAULT 1 CHECK (revision > 0),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (finding_id, audit_id),
    UNIQUE (audit_id, first_receipt_id),
    CONSTRAINT audit_findings_duplicate_shape CHECK (
        (state = 'duplicate') = (duplicate_target_id IS NOT NULL)
    ),
    CONSTRAINT audit_findings_rejection_shape CHECK (
        (state = 'rejected') = (rejection_reason IS NOT NULL)
    ),
    CONSTRAINT audit_findings_effective_decision_shape CHECK (
        (state IN ('confirmed', 'rejected')) = (current_decision_id IS NOT NULL)
    ),
    CONSTRAINT audit_findings_duplicate_target_fkey
        FOREIGN KEY (duplicate_target_id, audit_id)
        REFERENCES audit_findings(finding_id, audit_id) DEFERRABLE INITIALLY DEFERRED
);

CREATE INDEX audit_findings_list_idx
    ON audit_findings (audit_id, created_at, finding_id);
CREATE INDEX audit_findings_state_idx
    ON audit_findings (audit_id, state, created_at, finding_id);

CREATE TABLE audit_finding_contributions (
    finding_id text NOT NULL,
    audit_id text NOT NULL,
    receipt_id text NOT NULL REFERENCES finding_proposal_receipts(receipt_id) ON DELETE RESTRICT,
    relation text NOT NULL CHECK (relation IN ('first', 'contributing')),
    proposal_ref jsonb NOT NULL CHECK (
        jsonb_typeof(proposal_ref) = 'object' AND octet_length(proposal_ref::text) <= 4096
    ),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (finding_id, receipt_id),
    UNIQUE (audit_id, receipt_id),
    CONSTRAINT audit_finding_contributions_finding_fkey
        FOREIGN KEY (finding_id, audit_id)
        REFERENCES audit_findings(finding_id, audit_id) ON DELETE CASCADE
);

CREATE TABLE audit_finding_assessments (
    assessment_id text PRIMARY KEY CHECK (
        assessment_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'
    ),
    finding_id text NOT NULL,
    audit_id text NOT NULL,
    receipt_id text NOT NULL REFERENCES finding_proposal_receipts(receipt_id) ON DELETE RESTRICT,
    item_id text,
    execution_item_id text,
    collection_receipt_id text,
    semantic_assessment text NOT NULL CHECK (semantic_assessment IN (
        'supported', 'refuted', 'inconclusive', 'blocked', 'satisfied',
        'violated', 'not-tested'
    )),
    result_ref jsonb NOT NULL CHECK (
        jsonb_typeof(result_ref) = 'object' AND octet_length(result_ref::text) <= 4096
    ),
    result_digest text NOT NULL CHECK (result_digest ~ '^sha256:[0-9a-f]{64}$'),
    direct_verification boolean NOT NULL DEFAULT false,
    contract_ref jsonb,
    contract_digest text,
    accepted_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT audit_finding_assessments_finding_fkey
        FOREIGN KEY (finding_id, audit_id)
        REFERENCES audit_findings(finding_id, audit_id) ON DELETE CASCADE,
    CONSTRAINT audit_finding_assessments_item_fkey
        FOREIGN KEY (execution_item_id, item_id)
        REFERENCES audit_execution_items(execution_item_id, item_id) ON DELETE RESTRICT,
    CONSTRAINT audit_finding_assessments_collection_fkey
        FOREIGN KEY (collection_receipt_id)
        REFERENCES audit_collection_receipts(receipt_id) ON DELETE RESTRICT,
    CONSTRAINT audit_finding_assessments_item_shape CHECK (
        (item_id IS NULL) = (execution_item_id IS NULL)
        AND (direct_verification OR item_id IS NOT NULL)
    ),
    CONSTRAINT audit_finding_assessments_contract_shape CHECK (
        (contract_ref IS NULL) = (contract_digest IS NULL)
        AND (NOT direct_verification OR contract_ref IS NOT NULL)
        AND (contract_ref IS NULL OR (
            jsonb_typeof(contract_ref) = 'object'
            AND octet_length(contract_ref::text) <= 4096
            AND contract_digest ~ '^sha256:[0-9a-f]{64}$'
        ))
    ),
    UNIQUE (receipt_id, execution_item_id, result_digest)
);

ALTER TABLE audit_findings
    ADD CONSTRAINT audit_findings_current_assessment_fkey
    FOREIGN KEY (current_assessment_id)
    REFERENCES audit_finding_assessments(assessment_id)
    DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE audit_review_requests (
    request_id text PRIMARY KEY CHECK (
        request_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'
    ),
    audit_id text NOT NULL REFERENCES audits(audit_id) ON DELETE CASCADE,
    finding_id text NOT NULL,
    kind text NOT NULL CHECK (kind IN (
        'plan-approval', 'active-check-approval', 'provide-evidence',
        'finding-triage', 'requirement-applicability', 'report-acceptance'
    )),
    subject_revision bigint NOT NULL CHECK (subject_revision > 0),
    subject_digest text NOT NULL CHECK (subject_digest ~ '^sha256:[0-9a-f]{64}$'),
    requested_actions jsonb NOT NULL CHECK (
        jsonb_typeof(requested_actions) = 'array'
        AND octet_length(requested_actions::text) <= 65536
    ),
    state text NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'decided', 'expired')),
    expires_at timestamptz,
    revision bigint NOT NULL DEFAULT 1 CHECK (revision > 0),
    idempotency_key text NOT NULL CHECK (
        idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
    ),
    request_digest text NOT NULL CHECK (request_digest ~ '^sha256:[0-9a-f]{64}$'),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT audit_review_requests_finding_fkey
        FOREIGN KEY (finding_id, audit_id)
        REFERENCES audit_findings(finding_id, audit_id) ON DELETE CASCADE,
    UNIQUE (audit_id, idempotency_key),
    UNIQUE (request_id, audit_id, finding_id)
);

CREATE UNIQUE INDEX audit_review_requests_pending_subject_unique
    ON audit_review_requests (audit_id, finding_id, kind, subject_revision, subject_digest)
    WHERE state = 'pending';
CREATE INDEX audit_review_requests_list_idx
    ON audit_review_requests (audit_id, created_at, request_id);

CREATE TABLE audit_review_decisions (
    decision_id text PRIMARY KEY CHECK (
        decision_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$'
    ),
    request_id text NOT NULL,
    audit_id text NOT NULL,
    finding_id text NOT NULL,
    actor_id text NOT NULL CHECK (btrim(actor_id) <> '' AND octet_length(actor_id) <= 256),
    verdict text NOT NULL CHECK (verdict IN (
        'true_positive', 'false_positive', 'duplicate', 'reopen', 'needs_evidence'
    )),
    severity text CHECK (severity IS NULL OR severity IN (
        'informational', 'low', 'medium', 'high', 'critical'
    )),
    rationale text NOT NULL CHECK (btrim(rationale) <> '' AND octet_length(rationale) <= 65536),
    duplicate_target_id text,
    subject_revision bigint NOT NULL CHECK (subject_revision > 0),
    subject_digest text NOT NULL CHECK (subject_digest ~ '^sha256:[0-9a-f]{64}$'),
    idempotency_key text NOT NULL CHECK (
        idempotency_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'
    ),
    request_digest text NOT NULL CHECK (request_digest ~ '^sha256:[0-9a-f]{64}$'),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT audit_review_decisions_request_fkey
        FOREIGN KEY (request_id, audit_id, finding_id)
        REFERENCES audit_review_requests(request_id, audit_id, finding_id) ON DELETE RESTRICT,
    CONSTRAINT audit_review_decisions_duplicate_fkey
        FOREIGN KEY (duplicate_target_id, audit_id)
        REFERENCES audit_findings(finding_id, audit_id) DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT audit_review_decisions_shape CHECK (
        (verdict = 'true_positive') = (severity IS NOT NULL)
        AND (verdict = 'duplicate') = (duplicate_target_id IS NOT NULL)
    ),
    UNIQUE (request_id),
    UNIQUE (audit_id, idempotency_key)
);

ALTER TABLE audit_findings
    ADD CONSTRAINT audit_findings_current_decision_fkey
    FOREIGN KEY (current_decision_id)
    REFERENCES audit_review_decisions(decision_id)
    DEFERRABLE INITIALLY DEFERRED;

CREATE OR REPLACE FUNCTION contractor_admit_audit_finding()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    admitted_finding_id text;
    event_sequence bigint;
    finding_revision bigint;
BEGIN
    admitted_finding_id := 'finding-' || NEW.receipt_id;
    INSERT INTO audit_findings (
        finding_id, audit_id, first_receipt_id, first_proposal_ref
    ) VALUES (
        admitted_finding_id, NEW.audit_id, NEW.receipt_id, NEW.proposal_ref
    )
    ON CONFLICT (audit_id, first_receipt_id) DO NOTHING
    RETURNING revision INTO finding_revision;

    IF finding_revision IS NULL THEN
        RETURN NEW;
    END IF;

    INSERT INTO audit_finding_contributions (
        finding_id, audit_id, receipt_id, relation, proposal_ref
    ) VALUES (
        admitted_finding_id, NEW.audit_id, NEW.receipt_id, 'first', NEW.proposal_ref
    );

    UPDATE audits
       SET revision = revision + 1,
           next_event_sequence = next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
     WHERE audit_id = NEW.audit_id
    RETURNING next_event_sequence - 1 INTO event_sequence;

    INSERT INTO audit_events (
        audit_id, sequence_number, kind, entity_id, entity_revision, summary
    ) VALUES (
        NEW.audit_id, event_sequence, 'finding.proposed', admitted_finding_id,
        finding_revision, jsonb_build_object('state', 'proposed')
    );
    RETURN NEW;
END;
$$;

CREATE TRIGGER finding_proposal_audit_holds_admit_finding
AFTER INSERT ON finding_proposal_audit_holds
FOR EACH ROW EXECUTE FUNCTION contractor_admit_audit_finding();

CREATE OR REPLACE FUNCTION contractor_protect_audit_finding_history()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'Audit finding history is immutable' USING ERRCODE = '23514';
END;
$$;

CREATE TRIGGER audit_finding_contributions_protect_update
BEFORE UPDATE ON audit_finding_contributions
FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_finding_history();
CREATE TRIGGER audit_finding_assessments_protect_update
BEFORE UPDATE ON audit_finding_assessments
FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_finding_history();
CREATE TRIGGER audit_review_decisions_protect_update
BEFORE UPDATE ON audit_review_decisions
FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_finding_history();
