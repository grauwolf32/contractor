package auditstore

// SQL statements for item_review_renewal.go.

// expiredItemReviewsSQL finds ready or awaiting_review items of Audit $1 whose
// exact item-action review is expired, or pending or approved past expires_at,
// with no live pending or approved replacement. It locks those requests FOR
// UPDATE and orders each item's rows pending, decided, expired, newest first;
// $2 caps the rows and NULL means no cap. Used by renewExpiredItemReviews.
// Keep the live-authority lookup correlated to one item: pulling this volatile
// view into an anti join can repeatedly scan every pending review before the
// expired-review filter has eliminated the outer rows.
var expiredItemReviewsSQL = `
SELECT item.item_id, item.approval_kind, item.approval_subject_digest,
       review.request_id, review.state
  FROM audit_items AS item
  JOIN audit_review_requests AS review
    ON review.audit_id=item.audit_id
   AND review.subject_kind='audit-item-action'
   AND review.subject_id=item.item_id
   AND review.kind=item.approval_kind
   AND review.subject_revision=1
   AND review.subject_digest=item.approval_subject_digest
 WHERE item.audit_id=$1 AND item.state IN ('ready','awaiting_review')
   AND item.approval_kind <> 'none'
   AND (review.state='expired' OR (
       review.state IN ('pending','decided')
       AND review.expires_at <= clock_timestamp()
   ))
   AND (review.state <> 'decided' OR EXISTS (
       SELECT 1 FROM audit_review_decisions AS decision
        WHERE decision.request_id=review.request_id AND decision.action='approve'
   ))
   AND NOT EXISTS (
       SELECT 1 FROM audit_live_item_reviews AS live
        WHERE live.audit_id=item.audit_id AND live.item_id=item.item_id
       OFFSET 0
   )
 ORDER BY item.ordinal, item.item_id,
          CASE review.state WHEN 'pending' THEN 0 WHEN 'decided' THEN 1 ELSE 2 END,
          review.created_at DESC, review.request_id DESC
 LIMIT $2
 FOR UPDATE OF review`

// insertRenewedItemReviewsSQL inserts one fresh pending audit-item-action
// review request per renewal in Audit $1, expiring at $2, from the arrays
// $3-$7 (request ID, item ID, kind, subject digest, actions). The request ID
// also serves as idempotency key and the subject digest as request digest.
// Used by renewExpiredItemReviews.
var insertRenewedItemReviewsSQL = `
INSERT INTO audit_review_requests (
    request_id,audit_id,finding_id,subject_kind,subject_id,kind,
    subject_revision,subject_digest,requested_actions,state,expires_at,
    idempotency_key,request_digest
)
SELECT renewal.request_id,$1,NULL,'audit-item-action',renewal.item_id,renewal.kind,
       1,renewal.digest,renewal.actions::jsonb,'pending',$2,
       renewal.request_id,renewal.digest
  FROM unnest($3::text[],$4::text[],$5::text[],$6::text[],$7::text[])
       WITH ORDINALITY AS renewal(request_id,item_id,kind,digest,actions,position)
 ORDER BY renewal.position`
