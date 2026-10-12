package evalstore

// SQL statements for evidence.go.

// authorizeEvidenceSQL reports whether an artifact reference is evidence for a
// member of the owner's experiment. A 'run' scope must name the member's
// submitted Run or a child Run of its submitted Audit; a 'project' scope must
// be that Audit's Project and match one of its audit_artifact_links by exact
// namespace, name and revision. Used by Store.AuthorizeEvidence.
var authorizeEvidenceSQL = `
SELECT EXISTS (
    SELECT 1 FROM eval_submissions s
    JOIN eval_experiments e USING (experiment_id)
    WHERE e.owner_id = $1 AND s.experiment_id = $2 AND s.member_id = $3
        AND (
            ($4 = 'run' AND (
                (s.execution_kind = 'run' AND s.execution_id = $5)
                OR EXISTS (SELECT 1 FROM audit_executions x
                    WHERE s.execution_kind = 'audit'
                        AND x.audit_id = s.execution_id AND x.run_id = $5)
            ))
            OR ($4 = 'project' AND EXISTS (
                SELECT 1 FROM audits a
                JOIN audit_artifact_links l USING (audit_id)
                WHERE s.execution_kind = 'audit' AND a.audit_id = s.execution_id
                    AND a.owner_id = e.owner_id AND a.project_id = $5
                    AND l.artifact_ref ->> 'namespace' = $6
                    AND l.artifact_ref ->> 'name' = $7
                    AND l.artifact_ref ->> 'revision' = $8
            ))
        )
)
`
