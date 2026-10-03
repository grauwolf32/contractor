-- Collection retries skip already rejected proposal receipts by exact Audit
-- and receipt identity. Keep this lookup bounded as an Audit accumulates events.
CREATE INDEX audit_events_finding_proposal_rejected_idx
    ON audit_events (audit_id, entity_id)
    WHERE kind = 'finding.proposal_rejected';
