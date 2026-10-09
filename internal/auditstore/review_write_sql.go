package auditstore

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

// recordReviewDecisionSQL appends an immutable decision ($1-$14) and marks
// its locked request decided in one statement. Service validation and row
// locks precede it; the surrounding transaction includes projection and event.
var recordReviewDecisionSQL = `
WITH inserted AS (
    INSERT INTO audit_review_decisions (
        decision_id, request_id, audit_id, finding_id, actor_id, action, verdict,
        severity, rationale, duplicate_target_id, subject_revision, subject_digest,
        idempotency_key, request_digest
    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
    RETURNING request_id
)
UPDATE audit_review_requests AS request
   SET state='decided', revision=revision+1,
       updated_at=GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
  FROM inserted WHERE request.request_id=inserted.request_id AND request.state='pending'`
