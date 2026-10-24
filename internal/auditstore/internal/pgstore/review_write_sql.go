package pgstore

// insertFindingReviewRequestSQL persists the exact finding subject and actions
// ($1-$6). Explicit expiry $7 or the database-clock TTL $10 bounds authority;
// $8-$9 identify replay. Used inside the service's locked transaction.
var insertFindingReviewRequestSQL = `
INSERT INTO audit_review_requests (
    request_id, audit_id, finding_id, subject_kind, subject_id,
    kind, subject_revision, subject_digest,
    requested_actions, expires_at, idempotency_key, request_digest
) VALUES ($1, $2, $3, 'finding', $3, 'finding-triage', $4, $5, $6,
          COALESCE($7::timestamptz, clock_timestamp() + $10::bigint * interval '1 second'),
          $8, $9)`

// recordReviewDecisionSQL marks request $2 of Audit $3 decided while it is
// still pending and appends the immutable decision ($1-$14) for exactly the
// request it changed, in one statement. The INSERT reads only the UPDATE's
// RETURNING rows, so the statement inserts one row when it decides the
// request and none when the request is absent, belongs to another Audit or
// is no longer pending; the caller treats zero rows as a failed precondition.
// The UPDATE takes the request's row lock, which the caller already holds
// with the Audit after its own validation; the surrounding transaction
// includes projection and event. Used by ReviewStore.RecordReviewDecision.
var recordReviewDecisionSQL = `
WITH decided AS (
    UPDATE audit_review_requests AS request
       SET state='decided', revision=request.revision+1,
           updated_at=GREATEST(clock_timestamp(), request.updated_at + interval '1 microsecond')
     WHERE request.request_id=$2 AND request.audit_id=$3 AND request.state='pending'
    RETURNING request.request_id, request.audit_id
)
INSERT INTO audit_review_decisions (
    decision_id, request_id, audit_id, finding_id, actor_id, action, verdict,
    severity, rationale, duplicate_target_id, subject_revision, subject_digest,
    idempotency_key, request_digest
)
SELECT $1, decided.request_id, decided.audit_id, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
  FROM decided`
