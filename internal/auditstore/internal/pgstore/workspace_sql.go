package pgstore

// SQL statements for workspace.go.

// getWorkspaceSQL reads only the owner's coordination row. Results and findings
// have their own paginated reads; no current-Round or history aggregation occurs.
const getWorkspaceSQL = `
SELECT audit_id,revision,statement_timestamp(),state,outstanding_run_count,
       submitted_run_count,task_count
 FROM audits WHERE owner_id=$1 AND audit_id=$2 AND state<>'deleting'`

// listEventsBeforeSQL reads up to $4 events of Audit $1 numbered at most $2
// and, when $3 is set, below $3, newest first.
// Used by QueryStore.ListEventsPage.
const listEventsBeforeSQL = `
SELECT audit_id, sequence_number, kind, entity_id, entity_revision, summary, created_at
 FROM audit_events WHERE audit_id=$1 AND sequence_number <= $2
 AND ($3::bigint IS NULL OR sequence_number < $3)
 ORDER BY sequence_number DESC LIMIT $4`
