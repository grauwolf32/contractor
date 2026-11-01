package pgstore

import "github.com/grauwolf32/contractor/internal/projectstore"

const projectFindingSummaryColumns = `finding.finding_id,finding.project_id,left(finding.title,256),length(finding.title)>256,
finding.state,finding.analyst_verdict,finding.analyst_severity,finding.duplicate_target_id,
finding.revision,finding.created_at,finding.updated_at`

const findingPredicatesSQL = `
 AND ($3::text IS NULL OR finding.state=$3)
 AND ($4::text IS NULL OR finding.analyst_verdict=$4)
 AND (NOT $5::boolean OR finding.analyst_verdict IS NULL)
 AND ($6::text IS NULL OR finding.analyst_severity=$6)`

var listFindingsSQL = `SELECT ` + projectFindingSummaryColumns + auditFindingMembershipSQL + `
 WHERE audit.owner_id=$1 AND membership.audit_id=$2` + findingPredicatesSQL + `
 AND ($7::timestamptz IS NULL OR (membership.finding_created_at,membership.finding_id)>($7,$8))
 ORDER BY membership.finding_created_at,membership.finding_id LIMIT $9`

const findingPageBasisSQL = `SELECT audit.revision,fence.revision,statement_timestamp()
 FROM audits AS audit JOIN finding_correlation_fences AS fence USING(project_id)
 WHERE audit.owner_id=$1 AND audit.audit_id=$2`

var listOwnerFindingsSQL = `SELECT ` + projectFindingSummaryColumns + `
 FROM project_findings AS finding JOIN projects AS project USING(project_id)
 WHERE project.owner_id=$1 AND $2::text IS NULL AND ` + projectstore.OwnerListedProjectSQL("project.project_id", "project.owner_id") + findingPredicatesSQL + `
 AND ($7::timestamptz IS NULL OR (finding.created_at,finding.finding_id)<($7,$8))
 AND ($10::text[] IS NULL OR EXISTS(SELECT 1 FROM project_finding_audits AS membership
 JOIN audits AS audit USING(audit_id) WHERE membership.finding_id=finding.finding_id AND audit.state=ANY($10)))
 ORDER BY finding.created_at DESC,finding.finding_id DESC LIMIT $9`

var listProjectFindingsSQL = `SELECT ` + projectFindingSummaryColumns + `
 FROM project_findings AS finding JOIN projects AS project USING(project_id)
 WHERE project.owner_id=$1 AND finding.project_id=$2 AND ` + projectstore.OwnerListedProjectSQL("project.project_id", "project.owner_id") + findingPredicatesSQL + `
 AND ($7::timestamptz IS NULL OR (finding.created_at,finding.finding_id)<($7,$8))
 AND $10::text[] IS NULL ORDER BY finding.created_at DESC,finding.finding_id DESC LIMIT $9`

var visibleAuditProjectSQL = ` AND ` + projectstore.OwnerListedProjectSQL("audit.project_id", "audit.owner_id")

// reviewPageBasisSQL reads the revision of owner $1's Audit $2, the statement
// time and the number of its review requests for finding $3 (any when NULL)
// in state $4 (any when NULL). Used by QueryStore.ReviewPageBasis.
const reviewPageBasisSQL = `SELECT audit.revision, statement_timestamp(),
 (SELECT count(*) FROM audit_review_requests request WHERE request.audit_id=audit.audit_id
 AND ` + visibleItemReviewSQL + `
 AND ($3::text IS NULL OR request.state=$3))
 FROM audits audit WHERE audit.owner_id=$1 AND audit.audit_id=$2`

// listReviewsSQL is a review page of owner $1's Audit $2 in creation order:
// requests for finding $3 (any when NULL) in state $4 (any when NULL), after
// the cursor ($5, $6), at most $7 rows. Used by QueryStore.ListReviews.
const listReviewsSQL = reviewSelect + `
 WHERE audit.owner_id = $1 AND request.audit_id = $2
   AND ($3::text IS NULL OR request.state = $3)
   AND ($4::timestamptz IS NULL OR
        (request.created_at, request.request_id) > ($4, $5))
 ORDER BY request.created_at, request.request_id LIMIT $6`

// reviewSelect selects review requests with their Audit and optional decision
// in scanReviewRequest column order. Callers append the WHERE clause, which
// filters on audit.owner_id for authorization.
// Used by QueryStore.GetReview, ListReviews and ListOwnerReviews.
const reviewSelect = `
SELECT request.request_id, request.audit_id,
       request.subject_kind, request.subject_id, request.kind,
       request.subject_revision, request.subject_digest, request.requested_actions,
       request.state, request.expires_at, request.revision,
       request.created_at, request.updated_at,
       decision.decision_id, decision.actor_id, decision.action,
       decision.rationale, decision.created_at
  FROM audit_review_requests AS request
  JOIN audits AS audit USING (audit_id)
  JOIN LATERAL (SELECT 1 WHERE ` + visibleItemReviewSQL + `) AS visible_review ON true
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
   AND ` + visibleItemReviewSQL + `
 FOR UPDATE OF audit, request`
