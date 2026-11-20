package pgstore

// SQL statements for workspace.go.

// getWorkspaceSQL reads one owner-authorized snapshot. The Task aggregate is
// bounded by the Audit's max_items_total (at most 100000); Findings use the
// existing indexed Audit membership. Technical settlement is not a verdict.
var getWorkspaceSQL = `
SELECT audit.audit_id, audit.revision, statement_timestamp(), audit.task_iteration,
       audit.state, audit.outstanding_run_count,
       tasks.total, tasks.completed, tasks.gaps, tasks.pending,
       findings.total, findings.needs_review
  FROM audits AS audit
 CROSS JOIN LATERAL (
       SELECT count(*) AS total,
              count(*) FILTER (WHERE task.state='settled' AND task.gap_code IS NULL) AS completed,
              count(*) FILTER (WHERE task.state='settled' AND task.gap_code IS NOT NULL) AS gaps,
              count(*) FILTER (WHERE task.state<>'settled') AS pending
         FROM audit_tasks AS task WHERE task.audit_id=audit.audit_id
       ) AS tasks
 CROSS JOIN LATERAL (
       SELECT count(*) AS total,
              count(*) FILTER (WHERE (finding.analyst_verdict IS NULL OR finding.state='proposed') AND coalesce(finding.analyst_severity,finding.severity_suggestion,'')<>'informational') AS needs_review
         FROM project_finding_audits AS membership
         JOIN LATERAL (SELECT analyst_verdict,analyst_severity,severity_suggestion,state FROM project_findings AS current
              WHERE current.project_id=membership.project_id AND current.finding_id=membership.finding_id OFFSET 0) AS finding ON true
        WHERE membership.audit_id=audit.audit_id
       ) AS findings
 WHERE audit.owner_id=$1 AND audit.audit_id=$2`

// listEventsBeforeSQL reads up to $4 events of Audit $1 numbered at most $2
// and, when $3 is set, below $3, newest first.
// Used by QueryStore.ListEventsPage.
const listEventsBeforeSQL = `
SELECT audit_id, sequence_number, kind, entity_id, entity_revision, summary, created_at
 FROM audit_events WHERE audit_id=$1 AND sequence_number <= $2
 AND ($3::bigint IS NULL OR sequence_number < $3)
 ORDER BY sequence_number DESC LIMIT $4`
