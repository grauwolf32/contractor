-- Audit purge deletes the audits row and relies on cascades. PostgreSQL runs
-- cascaded deletes breadth-first and checks a RESTRICT key one cascade level
-- after it deleted the referenced row, so a referencing Audit-owned row must
-- be deleted at the same or an earlier level. Review decisions had no cascade
-- path at all, and finding assessments sat one level below the collection
-- receipts they reference, so purge failed for every Audit holding either.
-- Both now cascade directly from their Audit. Their RESTRICT keys still
-- protect decided requests, receipts and execution items outside a purge, and
-- the deferred current-decision and current-assessment keys are checked at
-- commit, after the findings that reference them are gone.
ALTER TABLE audit_review_decisions
    ADD CONSTRAINT audit_review_decisions_audit_fkey
    FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;

CREATE INDEX audit_finding_assessments_audit_idx
    ON audit_finding_assessments (audit_id);
ALTER TABLE audit_finding_assessments
    ADD CONSTRAINT audit_finding_assessments_audit_fkey
    FOREIGN KEY (audit_id) REFERENCES audits(audit_id) ON DELETE CASCADE;
