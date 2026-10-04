package evalstore

// SQL statements for inventory_pages.go.

// inventoryPageMemberSQL reads a member's projection revision and its submitted
// Run or Audit (kind, ID, state, existence, closed dispatch), plus counts of
// Audit child executions that are uncollected or lack a Run without a
// 'submission-failed' outcome, and of child Runs the owner no longer has.
// Used by Store.InventoryPage.
var inventoryPageMemberSQL = `
SELECT p.revision, m.execution_kind, sub.execution_id, COALESCE(r.state, a.state),
    r.run_id IS NOT NULL OR a.audit_id IS NOT NULL,
    COALESCE(a.dispatch_state = 'closed', TRUE),
    (SELECT count(*) FROM audit_executions x
        WHERE m.execution_kind = 'audit' AND x.audit_id = sub.execution_id
            AND (x.state <> 'collected'
                OR (x.run_id IS NULL AND x.terminal_outcome IS DISTINCT FROM 'submission-failed'))),
    (SELECT count(*) FROM audit_executions x
        WHERE m.execution_kind = 'audit' AND x.audit_id = sub.execution_id
            AND x.run_id IS NOT NULL
            AND NOT EXISTS (SELECT 1 FROM workflow_runs child
                WHERE child.run_id = x.run_id AND child.owner_id = e.owner_id))
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

// inventoryPageAuditExecutionsSQL returns a keyset page of the owner's Audit $2
// child executions ordered by execution_id, limited to $4. The cursor $3 is the
// page key '1' || execution_id (the parent entry's key is "0"). Rows carry the
// child Run ID, role, round ordinal, Run state and existence, and terminal
// outcome. Used by Store.InventoryPage.
var inventoryPageAuditExecutionsSQL = `
SELECT x.execution_id,x.run_id,x.role,round.ordinal,r.state,r.run_id IS NOT NULL,x.terminal_outcome
FROM audit_executions x
JOIN audits a USING(audit_id)
LEFT JOIN audit_rounds round ON round.audit_id = x.audit_id
AND round.round_id = x.round_id
LEFT JOIN workflow_runs r ON r.run_id = x.run_id
AND r.owner_id = a.owner_id
WHERE a.owner_id = $1
    AND x.audit_id = $2
    AND '1' || x.execution_id > $3
ORDER BY x.execution_id
LIMIT $4
`
