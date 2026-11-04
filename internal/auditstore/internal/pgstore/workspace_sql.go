package pgstore

// SQL statements for workspace.go.

// getWorkspaceSQL reads the workspace summary of owner $1's Audit $2:
// revision, statement time, current round, state, outstanding Runs,
// current-round coverage counts (total, then the completed $3, issue $4, gap
// $5 and unchecked $6 coverage statuses of auditdomain's one bucket table, in
// one pass over the round's items), finding and unreviewed-finding counts (in
// one pass over the Audit's findings). The statement time is reported as
// asOf. Every count is exact, so each pass reads the
// rows it counts. Used by QueryStore.GetWorkspace.
var getWorkspaceSQL = `
SELECT audit.audit_id, audit.revision, statement_timestamp(), audit.current_round_id,
       audit.state, audit.outstanding_run_count,
       coverage.total, coverage.completed, coverage.issues, coverage.gaps, coverage.unchecked,
       findings.total, findings.unreviewed
  FROM audits AS audit
 CROSS JOIN LATERAL (
       SELECT count(*) AS total,
              count(*) FILTER (WHERE item.coverage_status = ANY($3::text[])) AS completed,
              count(*) FILTER (WHERE item.coverage_status = ANY($4::text[])) AS issues,
              count(*) FILTER (WHERE item.coverage_status = ANY($5::text[])) AS gaps,
              count(*) FILTER (WHERE item.coverage_status = ANY($6::text[])) AS unchecked
         FROM audit_items AS item
        WHERE item.audit_id = audit.audit_id AND item.round_id = audit.current_round_id
          AND ` + acceptedItemSQL("item") + `
       ) AS coverage
 CROSS JOIN LATERAL (
       SELECT count(*) AS total,
              count(*) FILTER (WHERE finding.analyst_verdict IS NULL) AS unreviewed
         FROM project_finding_audits AS membership
         JOIN LATERAL (SELECT analyst_verdict FROM project_findings AS current
              WHERE current.project_id=membership.project_id AND current.finding_id=membership.finding_id OFFSET 0) AS finding ON true
        WHERE membership.audit_id = audit.audit_id
       ) AS findings
 WHERE audit.owner_id = $1 AND audit.audit_id = $2`

// listEventsBeforeSQL reads up to $4 events of Audit $1 numbered at most $2
// and, when $3 is set, below $3, newest first.
// Used by QueryStore.ListEventsPage.
const listEventsBeforeSQL = `
SELECT audit_id, sequence_number, kind, entity_id, entity_revision, summary, created_at
 FROM audit_events WHERE audit_id=$1 AND sequence_number <= $2
 AND ($3::bigint IS NULL OR sequence_number < $3)
 ORDER BY sequence_number DESC LIMIT $4`
