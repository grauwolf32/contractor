package evalstore

// SQL statements for execution_view.go.

// executionObservationsSQL lists members of the owner's experiment (only member
// $3 when non-empty) in ordinal order, limited to $4. Each row joins the frozen
// member fields with the submission state and execution ID, the owner's Run or
// Audit state and timestamps, and whether an execution tombstone exists.
// Used by Store.executionObservations.
var executionObservationsSQL = `
SELECT m.member_id,m.pair_id,m.ordinal,m.suite_id,m.case_id,m.sample,m.variant_id,m.eligibility,
       m.execution_kind,m.case_sha256,m.binding_sha256, sub.state,sub.execution_id,COALESCE(run.state,
                                                                                       audit.state),
       COALESCE(run.created_at, audit.created_at),COALESCE(run.finished_at,
                                                      audit.finished_at),t.execution_id IS NOT NULL,
                                                                                           m.eligibility_reason
FROM eval_members m
JOIN eval_experiments e USING(experiment_id)
LEFT JOIN eval_submissions sub USING(experiment_id,member_id)
LEFT JOIN workflow_runs run ON m.execution_kind = 'run'
AND sub.execution_id = run.run_id
AND run.owner_id = e.owner_id
LEFT JOIN audits AUDIT ON m.execution_kind = 'audit'
AND sub.execution_id = audit.audit_id
AND audit.owner_id = e.owner_id
LEFT JOIN eval_execution_tombstones t USING(experiment_id,member_id)
WHERE e.owner_id = $1
    AND e.experiment_id = $2
    AND ($3 = ''
         OR m.member_id = $3)
ORDER BY m.ordinal
LIMIT $4
`
