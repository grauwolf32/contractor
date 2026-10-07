-- Cross-project keysets must not sort the entire finding/review history.
CREATE INDEX audit_findings_global_created_idx
    ON audit_findings (created_at DESC, finding_id DESC);
CREATE INDEX audit_review_requests_global_created_idx
    ON audit_review_requests (created_at DESC, request_id DESC);
CREATE INDEX audit_review_requests_global_state_created_idx
    ON audit_review_requests (state, created_at DESC, request_id DESC);
