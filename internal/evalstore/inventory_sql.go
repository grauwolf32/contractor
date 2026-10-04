package evalstore

// SQL statements for inventory.go.

// inventoryMemberSQL reads a member's projection revision, execution kind and
// submitted execution ID, plus the owner's Run or Audit behind it: its state,
// whether it still exists, whether Audit dispatch is closed (true for Runs)
// and its Project ID. Used by Store.Inventory.
var inventoryMemberSQL = `
SELECT p.revision, m.execution_kind, sub.execution_id, COALESCE(r.state, a.state),
    r.run_id IS NOT NULL OR a.audit_id IS NOT NULL,
    COALESCE(a.dispatch_state = 'closed', TRUE), COALESCE(r.project_id, a.project_id)
FROM eval_members m
JOIN eval_experiments e USING (experiment_id)
JOIN eval_member_projections p USING (experiment_id, member_id)
LEFT JOIN eval_submissions sub USING (experiment_id, member_id)
LEFT JOIN workflow_runs r ON m.execution_kind = 'run'
    AND r.run_id = sub.execution_id AND r.owner_id = e.owner_id
LEFT JOIN audits a ON m.execution_kind = 'audit'
    AND a.audit_id = sub.execution_id AND a.owner_id = e.owner_id
WHERE e.owner_id = $1 AND e.experiment_id = $2 AND m.member_id = $3
`

// inventoryAuditExecutionsSQL lists the child executions of the owner's Audit
// $1 in creation order, limited to $3: intent ID, child Run ID, role, round
// ordinal, collection state, terminal outcome, and the child Run's state,
// existence and Project ID. Used by Store.Inventory.
var inventoryAuditExecutionsSQL = `
SELECT x.execution_id,x.run_id,x.role,round.ordinal,x.state,x.terminal_outcome,r.state,r.run_id IS NOT NULL,r.project_id
FROM audit_executions x
JOIN audits a USING(audit_id)
LEFT JOIN audit_rounds round ON round.audit_id = x.audit_id
AND round.round_id = x.round_id
LEFT JOIN workflow_runs r ON r.run_id = x.run_id
AND r.owner_id = a.owner_id
WHERE x.audit_id = $1
    AND a.owner_id = $2
ORDER BY x.created_at,x.execution_id
LIMIT $3
`
