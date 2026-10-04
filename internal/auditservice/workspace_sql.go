package auditservice

// SQL statements for workspace.go.

// getWorkspaceSQL reads the workspace summary of owner $1's Audit $2:
// revision, statement time, current round, state, outstanding Runs,
// current-round coverage counts by status bucket (total, completed, issues,
// gaps, unchecked), finding and unreviewed-finding counts, and pending review
// requests. Used by Service.GetWorkspace.
var getWorkspaceSQL = `
SELECT audit.audit_id, audit.revision, statement_timestamp(), audit.current_round_id,
       audit.state, audit.outstanding_run_count,
       (SELECT count(*) FROM audit_coverage_rows WHERE audit_id = audit.audit_id AND round_id = audit.current_round_id),
       (SELECT count(*) FROM audit_coverage_rows WHERE audit_id = audit.audit_id AND round_id = audit.current_round_id AND status IN ('satisfied', 'violated', 'traced-complete', 'not-applicable', 'excluded')),
       (SELECT count(*) FROM audit_coverage_rows WHERE audit_id = audit.audit_id AND round_id = audit.current_round_id AND status = 'violated'),
       (SELECT count(*) FROM audit_coverage_rows WHERE audit_id = audit.audit_id AND round_id = audit.current_round_id AND status IN ('inconclusive', 'blocked', 'traced-partial', 'unmapped')),
       (SELECT count(*) FROM audit_coverage_rows WHERE audit_id = audit.audit_id AND round_id = audit.current_round_id AND status = 'not-tested'),
       (SELECT count(*) FROM audit_findings WHERE audit_id = audit.audit_id),
       (SELECT count(*) FROM audit_findings WHERE audit_id = audit.audit_id AND current_decision_id IS NULL),
       (SELECT count(*) FROM audit_review_requests WHERE audit_id = audit.audit_id AND state = 'pending')
FROM audits audit WHERE owner_id = $1 AND audit_id = $2`
