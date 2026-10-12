package auditservice

// SQL statements for workspace.go.

// reviewUnexpiredSQL holds when the review request aliased "request" is not
// past its expiry at the statement's PostgreSQL time (REV-R43). Together
// with state 'pending' it is the one rule for a request that can still be
// decided, before anything has stored its expiry: the workspace's pending
// count and the owner-wide pending review list (ListOwnerReviews) use it.
const reviewUnexpiredSQL = `(request.expires_at IS NULL OR request.expires_at > statement_timestamp())`

// getWorkspaceSQL reads the workspace summary of owner $1's Audit $2:
// revision, statement time, current round, state, outstanding Runs,
// current-round coverage counts (total, then the completed $3, issue $4, gap
// $5 and unchecked $6 coverage statuses of auditdomain's one bucket table, in
// one pass over the round's items), finding and unreviewed-finding counts,
// and the review requests that can still be decided: pending and not past
// their expiry at the statement time the snapshot reports as asOf.
// Used by Service.GetWorkspace.
var getWorkspaceSQL = `
SELECT audit.audit_id, audit.revision, statement_timestamp(), audit.current_round_id,
       audit.state, audit.outstanding_run_count,
       coverage.total, coverage.completed, coverage.issues, coverage.gaps, coverage.unchecked,
       (SELECT count(*) FROM audit_findings WHERE audit_id = audit.audit_id),
       (SELECT count(*) FROM audit_findings WHERE audit_id = audit.audit_id AND current_decision_id IS NULL),
       (SELECT count(*) FROM audit_review_requests AS request
         WHERE request.audit_id = audit.audit_id AND request.state = 'pending'
           AND ` + reviewUnexpiredSQL + `)
  FROM audits AS audit
 CROSS JOIN LATERAL (
       SELECT count(*) AS total,
              count(*) FILTER (WHERE item.coverage_status = ANY($3::text[])) AS completed,
              count(*) FILTER (WHERE item.coverage_status = ANY($4::text[])) AS issues,
              count(*) FILTER (WHERE item.coverage_status = ANY($5::text[])) AS gaps,
              count(*) FILTER (WHERE item.coverage_status = ANY($6::text[])) AS unchecked
         FROM audit_items AS item
        WHERE item.audit_id = audit.audit_id AND item.round_id = audit.current_round_id
       ) AS coverage
 WHERE audit.owner_id = $1 AND audit.audit_id = $2`
