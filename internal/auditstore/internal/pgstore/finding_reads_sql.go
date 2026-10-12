package pgstore

import "github.com/grauwolf32/contractor/internal/projectstore"

// SQL statements for finding_reads.go and review_reads.go.

// findingRecordColumns selects a finding aliased as finding in
// scanFindingRecord order.
const findingRecordColumns = `
finding.finding_id, finding.audit_id, finding.first_receipt_id, finding.state,
finding.rejection_reason, finding.duplicate_target_id,
finding.current_assessment_id, finding.current_decision_id,
finding.revision, finding.created_at, finding.updated_at`

// findingFilterSQL is the FROM and WHERE of a finding page of owner $1: the
// Audit $2 or, when NULL, every Audit of the owner; state $3; the current
// decision's verdict $4; only undecided findings when $5; the decision's
// severity $6. Used by QueryStore.ListFindings, ListOwnerFindings and
// FindingPageBasis.
const findingFilterSQL = `
  FROM audit_findings AS finding
  JOIN audits AS audit USING (audit_id)
  LEFT JOIN audit_review_decisions AS decision
    ON decision.decision_id = finding.current_decision_id
 WHERE audit.owner_id = $1 AND ($2::text IS NULL OR finding.audit_id = $2)
   AND ($3::text IS NULL OR finding.state = $3)
   AND ($4::text IS NULL OR decision.verdict = $4)
   AND (NOT $5::boolean OR finding.current_decision_id IS NULL)
   AND ($6::text IS NULL OR decision.severity = $6)`

// listFindingsSQL is a finding page of owner $1's Audit $2 in creation order:
// the findingFilterSQL filters, after the cursor ($7, $8), at most $9 rows.
// Used by QueryStore.ListFindings.
const listFindingsSQL = `
SELECT ` + findingRecordColumns + `
` + findingFilterSQL + `
   AND ($7::timestamptz IS NULL OR
        (finding.created_at, finding.finding_id) > ($7, $8))
 ORDER BY finding.created_at, finding.finding_id
 LIMIT $9`

// findingPageBasisSQL reads the revision of owner $1's Audit $2, the
// statement time and the number of its findings that match the
// findingFilterSQL filters $3 to $6. Used by QueryStore.FindingPageBasis.
const findingPageBasisSQL = `SELECT audit.revision, statement_timestamp(),
 (SELECT count(*) ` + findingFilterSQL + `) FROM audits audit WHERE audit.owner_id=$1 AND audit.audit_id=$2`

// reviewPageBasisSQL reads the revision of owner $1's Audit $2, the statement
// time and the number of its review requests for finding $3 (any when NULL)
// in state $4 (any when NULL). Used by QueryStore.ReviewPageBasis.
const reviewPageBasisSQL = `SELECT audit.revision, statement_timestamp(),
 (SELECT count(*) FROM audit_review_requests request WHERE request.audit_id=audit.audit_id
 AND ($3::text IS NULL OR request.finding_id=$3) AND ($4::text IS NULL OR request.state=$4))
 FROM audits audit WHERE audit.owner_id=$1 AND audit.audit_id=$2`

// listReviewsSQL is a review page of owner $1's Audit $2 in creation order:
// requests for finding $3 (any when NULL) in state $4 (any when NULL), after
// the cursor ($5, $6), at most $7 rows. Used by QueryStore.ListReviews.
const listReviewsSQL = reviewSelect + `
 WHERE audit.owner_id = $1 AND request.audit_id = $2
   AND ($3::text IS NULL OR request.finding_id = $3)
   AND ($4::text IS NULL OR request.state = $4)
   AND ($5::timestamptz IS NULL OR
        (request.created_at, request.request_id) > ($5, $6))
 ORDER BY request.created_at, request.request_id LIMIT $7`

// visibleAuditProjectSQL keeps an owner-wide list to the Audits of Projects
// listed to their owner, by the rule the owner-wide Audit list applies.
var visibleAuditProjectSQL = `
 AND ` + projectstore.OwnerListedProjectSQL("audit.project_id", "audit.owner_id")

// listOwnerFindingsSQL is an owner-wide finding page, newest first: the
// findingFilterSQL filters over every Audit ($2 is NULL), Audits of listed
// Projects only, before the cursor ($7, $8), in the Audit states $10, at most
// $9 rows. Used by QueryStore.ListOwnerFindings.
var listOwnerFindingsSQL = `SELECT ` + findingRecordColumns + findingFilterSQL + visibleAuditProjectSQL + `
   AND ($7::timestamptz IS NULL OR (finding.created_at, finding.finding_id) < ($7, $8))
   AND ($10::text[] IS NULL OR audit.state = ANY($10))
 ORDER BY finding.created_at DESC, finding.finding_id DESC LIMIT $9`

// decisionSelect selects analyst decisions in scanDecision order. Callers
// append the joins and WHERE clause, which filters on audit.owner_id.
const decisionSelect = `
SELECT decision.decision_id, decision.request_id, decision.audit_id,
       decision.finding_id, decision.actor_id, decision.action, decision.verdict,
       decision.severity, decision.rationale, decision.duplicate_target_id,
       decision.subject_revision, decision.subject_digest, decision.created_at
  FROM audit_review_decisions AS decision`

// assessmentSelect selects finding assessments in scanAssessment order.
// Callers append the joins and WHERE clause, which filters on
// audit.owner_id.
const assessmentSelect = `
SELECT assessment.assessment_id, assessment.semantic_assessment,
       assessment.result_ref, assessment.result_digest, assessment.receipt_id,
       assessment.item_id, assessment.execution_item_id,
       assessment.collection_receipt_id, assessment.direct_verification,
       assessment.contract_ref, assessment.contract_digest, assessment.accepted_at
  FROM audit_finding_assessments AS assessment`

// reviewSelect selects review requests with their Audit and optional decision
// in scanReviewRequest column order. Callers append the WHERE clause, which
// filters on audit.owner_id for authorization.
// Used by QueryStore.GetReview, ListReviews and ListOwnerReviews.
const reviewSelect = `
SELECT request.request_id, request.audit_id, request.finding_id,
       request.subject_kind, request.subject_id, request.kind,
       request.subject_revision, request.subject_digest, request.requested_actions,
       request.state, request.expires_at, request.revision,
       request.created_at, request.updated_at,
       decision.decision_id, decision.actor_id, decision.action,
       decision.verdict, decision.severity,
       decision.rationale, decision.duplicate_target_id, decision.created_at
  FROM audit_review_requests AS request
  JOIN audits AS audit USING (audit_id)
  LEFT JOIN audit_review_decisions AS decision
    ON decision.request_id = request.request_id
   AND decision.audit_id = request.audit_id`

// reviewUnexpiredSQL holds when the review request aliased "request" is not
// past its expiry at the statement's PostgreSQL time (REV-R43). Together
// with state 'pending' it is the one rule for a request that can still be
// decided, before anything has stored its expiry: the workspace's pending
// count and the owner-wide pending review list (ListOwnerReviews) use it.
const reviewUnexpiredSQL = `(request.expires_at IS NULL OR request.expires_at > statement_timestamp())`

// listOwnerReviewsSQL is an owner-wide review page, newest first: requests
// of owner $1 in Audits of listed Projects, in state $2 (a pending request
// only while it can still be decided), before the cursor ($3, $4), in the
// Audit states $6, at most $5 rows. Used by QueryStore.ListOwnerReviews.
var listOwnerReviewsSQL = reviewSelect + `
 WHERE audit.owner_id = $1` + visibleAuditProjectSQL + `
   AND ($2::text IS NULL OR request.state = $2)
   AND ($2::text IS DISTINCT FROM 'pending' OR ` + reviewUnexpiredSQL + `)
   AND ($3::timestamptz IS NULL OR (request.created_at, request.request_id) < ($3, $4))
   AND ($6::text[] IS NULL OR audit.state = ANY($6))
 ORDER BY request.created_at DESC, request.request_id DESC LIMIT $5`

// lockFindingReviewSQL locks the finding triage request $3 of owner $1's
// Audit $2 with its finding and Audit, and reads the Audit state, the
// request and the finding in scanFindingRecord order.
// Used by ReviewTxStore.LockFindingReview.
const lockFindingReviewSQL = `
SELECT audit.state,
       request.request_id, request.audit_id, request.finding_id, request.kind,
       request.subject_revision, request.subject_digest,
       request.state, request.expires_at, request.revision,
       request.created_at, request.updated_at,
       ` + findingRecordColumns + `
  FROM audit_review_requests AS request
  JOIN audit_findings AS finding
    ON finding.finding_id = request.finding_id AND finding.audit_id = request.audit_id
  JOIN audits AS audit ON audit.audit_id = request.audit_id
 WHERE audit.owner_id = $1 AND request.audit_id = $2 AND request.request_id = $3
 FOR UPDATE OF audit, request, finding`

// lockActionReviewSQL locks the review request $3 of owner $1's Audit $2 with
// its Audit and reads the Audit state and revision and the request.
// Used by ReviewTxStore.LockActionReview.
const lockActionReviewSQL = `
SELECT audit.state, audit.revision, request.subject_kind, request.subject_id, request.kind,
       request.subject_revision, request.subject_digest,
       request.requested_actions, request.state, request.expires_at,
       request.revision
  FROM audit_review_requests AS request
  JOIN audits AS audit USING (audit_id)
 WHERE audit.owner_id = $1 AND request.audit_id = $2
   AND request.request_id = $3
 FOR UPDATE OF audit, request`

// duplicateTargetCycleSQL reports whether the proposed duplicate target
// exists and whether following duplicate_target_id from it leads back to
// the finding being decided, which would close a cycle.
// Used by ReviewTxStore.DuplicateTarget.
const duplicateTargetCycleSQL = `
WITH RECURSIVE chain(finding_id, duplicate_target_id) AS (
    SELECT finding_id, duplicate_target_id
      FROM audit_findings WHERE audit_id = $1 AND finding_id = $2
    UNION
    SELECT next.finding_id, next.duplicate_target_id
      FROM chain JOIN audit_findings AS next
        ON next.audit_id = $1 AND next.finding_id = chain.duplicate_target_id
     WHERE chain.duplicate_target_id IS NOT NULL
)
SELECT EXISTS (SELECT 1 FROM audit_findings WHERE audit_id = $1 AND finding_id = $2),
       EXISTS (SELECT 1 FROM chain WHERE finding_id = $3)`
