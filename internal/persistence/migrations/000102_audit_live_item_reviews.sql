-- Shared exact-subject authority for claiming, renewal and execution admission.
-- Pending requests suppress renewal; only an approved decision admits work.
CREATE VIEW audit_live_item_reviews AS
SELECT item.audit_id, item.item_id, request.request_id,
       request.state = 'decided' AS approved
  FROM audit_items AS item
  JOIN audit_review_requests AS request
    ON request.audit_id = item.audit_id
   AND request.subject_kind = 'audit-item-action'
   AND request.subject_id = item.item_id
   AND request.kind = item.approval_kind
   AND request.subject_revision = 1
   AND request.subject_digest = item.approval_subject_digest
 WHERE (request.expires_at IS NULL OR request.expires_at > clock_timestamp())
   AND (request.state = 'pending' OR (request.state = 'decided' AND EXISTS (
       SELECT 1 FROM audit_review_decisions AS decision
        WHERE decision.request_id = request.request_id
          AND decision.audit_id = request.audit_id
          AND decision.action = 'approve'
          AND decision.subject_revision = request.subject_revision
          AND decision.subject_digest = request.subject_digest
   )));
