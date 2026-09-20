-- Replace only future admission identity; existing findings and history stay intact.
CREATE OR REPLACE FUNCTION contractor_admit_audit_finding()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    admitted_finding_id text;
    event_sequence bigint;
    finding_revision bigint;
BEGIN
    -- Findings belong to an Audit; the same immutable receipt may be imported
    -- into several compatible Audits. Preserve all previously admitted IDs.
    admitted_finding_id := 'finding-' || encode(sha256(convert_to(
        jsonb_build_array(NEW.audit_id, NEW.receipt_id)::text, 'UTF8'
    )), 'hex');
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

