-- One allocator for Audit events (V370-003).
--
-- Every Audit event used to be numbered by hand in the statement that wrote
-- it: about thirty INSERTs advanced audits.next_event_sequence by k and then
-- wrote k rows at next_event_sequence - k ... next_event_sequence - 1. A
-- statement that miscounted left a gap or reused a number silently.
--
-- contractor_append_audit_events is now the only statement that inserts into
-- audit_events. Its caller reserves the events first: in the same statement
-- (or, in a trigger, the preceding command) it advances next_event_sequence
-- of the locked audits row by the number of events. The function then
-- numbers the events in array order directly after the Audit's last event.
-- It rejects a reservation that does not continue the sequence exactly, so
-- the events of every Audit stay numbered 1..N with N = next_event_sequence
-- - 1 (AUD-R223). The check reads one audits row and the last event through
-- the primary key.
--
-- Each element of events is an object with kind, entityId, an optional
-- entityRevision and a summary object; the audit_events checks validate
-- them. A misused allocator raises assert_failure (SQLSTATE P0004), which no
-- caller classifies, so the statement fails as an internal error.
CREATE FUNCTION contractor_append_audit_events(target_audit_id text, events jsonb) RETURNS bigint
    LANGUAGE plpgsql
    AS $$
DECLARE
    event_count bigint;
    reserved_next bigint;
    last_sequence bigint;
BEGIN
    IF events IS NULL OR jsonb_typeof(events) <> 'array' THEN
        RAISE EXCEPTION 'Audit events must be a JSON array' USING ERRCODE = 'assert_failure';
    END IF;
    event_count := jsonb_array_length(events);

    SELECT audit.next_event_sequence INTO reserved_next
      FROM audits AS audit
     WHERE audit.audit_id = target_audit_id;
    IF reserved_next IS NULL THEN
        RAISE EXCEPTION 'Audit events have no Audit' USING ERRCODE = 'assert_failure';
    END IF;

    SELECT event.sequence_number INTO last_sequence
      FROM audit_events AS event
     WHERE event.audit_id = target_audit_id
     ORDER BY event.sequence_number DESC
     LIMIT 1;
    IF COALESCE(last_sequence, 0) + event_count <> reserved_next - 1 THEN
        RAISE EXCEPTION 'Audit event reservation does not continue the event sequence'
            USING ERRCODE = 'assert_failure',
                  DETAIL = format('last event %s, %s events, next sequence %s',
                                  COALESCE(last_sequence, 0), event_count, reserved_next);
    END IF;

    INSERT INTO audit_events (
        audit_id, sequence_number, kind, entity_id, entity_revision, summary
    )
    SELECT target_audit_id, reserved_next - event_count + event.position - 1,
           event.value->>'kind', event.value->>'entityId',
           (event.value->>'entityRevision')::bigint, event.value->'summary'
      FROM jsonb_array_elements(events) WITH ORDINALITY AS event(value, position);
    RETURN reserved_next - 1;
END;
$$;

-- Admitting a finding records finding.proposed through the allocator.
CREATE OR REPLACE FUNCTION contractor_admit_audit_finding() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    admitted_finding_id text;
    finding_revision bigint;
    admitted_events jsonb;
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

    admitted_events := jsonb_build_array(jsonb_build_object(
        'kind', 'finding.proposed', 'entityId', admitted_finding_id,
        'entityRevision', finding_revision,
        'summary', jsonb_build_object('state', 'proposed')
    ));
    UPDATE audits
       SET revision = revision + jsonb_array_length(admitted_events),
           next_event_sequence = next_event_sequence + jsonb_array_length(admitted_events),
           updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
     WHERE audit_id = NEW.audit_id;
    PERFORM contractor_append_audit_events(NEW.audit_id, admitted_events);
    RETURN NEW;
END;
$$;
