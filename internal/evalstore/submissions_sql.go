package evalstore

// SQL statements for submissions.go.

// registerExecutionProjectSQL checks that Project $2 is the owner's active
// 'project' workspace created with idempotency key $3 and the request digest
// of the member's 'project-create' suboperation, and locks that row FOR SHARE.
// Returns true, or no row when it does not match.
// Used by Store.RegisterExecutionProject.
var registerExecutionProjectSQL = `
SELECT true
FROM projects
WHERE owner_id=$1
    AND project_id=$2
    AND kind='project'
    AND lifecycle_state='active'
    AND request_idempotency_key=$3
    AND request_digest=(SELECT request_sha256
    FROM eval_suboperations
    WHERE experiment_id=$4
        AND member_id=$5
        AND kind='project-create')
FOR SHARE
`

// settleSubmissionSQL decides whether a member's unaccepted intent can settle
// as rejected: true when a never-started execution tombstone exists, or when
// no run/audit-create suboperation is pending or succeeded and either the
// experiment is cancelling ($3) or a member suboperation was rejected.
// Used by Store.Settle.
var settleSubmissionSQL = `
SELECT EXISTS(SELECT 1
    FROM eval_execution_tombstones
    WHERE experiment_id=$1
        AND member_id=$2
        AND never_started) OR (NOT EXISTS(SELECT 1
        FROM eval_suboperations
        WHERE experiment_id=$1
            AND member_id=$2
            AND kind IN ('run-create', 'audit-create')
            AND state IN ('intent', 'succeeded'))
        AND ($3 OR EXISTS(SELECT 1
            FROM eval_suboperations
            WHERE experiment_id=$1
                AND member_id=$2
                AND state='rejected')))
`
