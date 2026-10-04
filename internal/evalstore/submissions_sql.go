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

// bindExecutionSQL reports whether Audit $1 belongs to owner $2, lives in the
// workspace Project registered on the member's submission,
// and was created by an 'audit.create' audit_idempotency entry with the
// member's suboperation key $5 and request digest $6.
// Used by Store.BindExecution.
var bindExecutionSQL = `
SELECT EXISTS(SELECT 1
    FROM audits a
    JOIN audit_idempotency i USING(audit_id)
    JOIN eval_submissions s ON s.execution_project_id=a.project_id
    WHERE a.audit_id=$1
        AND a.owner_id=$2
        AND s.experiment_id=$3
        AND s.member_id=$4
        AND i.owner_id=$2
        AND i.operation='audit.create'
        AND i.idempotency_key=$5
        AND i.request_digest=$6)
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
