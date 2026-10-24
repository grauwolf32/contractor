-- Index the item-action review requests by their item (V409-004).
--
-- Every reconciliation of an Audit with an open Round looks for item
-- approvals whose exact review expired without a live replacement
-- (expiredItemReviewsSQL), and the controller claim asks the same of a
-- waiting_review Audit. Both join an item to its review requests on
-- (audit_id, subject_kind, subject_id) and probe the live-authority view on
-- the same key. Only pending requests had an index on that key
-- (audit_review_requests_pending_subject_unique), so finding the expired or
-- decided ones read every review request of the Audit, finding triage
-- included, or of the whole table, on every reconciliation. This partial
-- index holds the item-action requests only. Decisions and expiry update a
-- request's state and revision, never its Audit, subject kind or subject, so
-- the index costs one entry per item-action request inserted.

CREATE INDEX audit_review_requests_item_subject_idx ON audit_review_requests USING btree (audit_id, subject_id)
    WHERE (subject_kind = 'audit-item-action'::text);
