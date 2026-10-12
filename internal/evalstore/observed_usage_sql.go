package evalstore

// SQL statements for observed_usage.go.

// ownedRunForUsage selects Run $1 when owner $2 owns it.
const ownedRunForUsage = `
SELECT run_id
FROM workflow_runs
WHERE run_id = $1 AND owner_id = $2
`

// ownedAuditRunsForUsage selects the distinct child Runs of Audit $1 when
// owner $2 owns it, at most $3 of them in Run ID order.
const ownedAuditRunsForUsage = `
SELECT DISTINCT execution.run_id
FROM audit_executions execution
JOIN audits audit USING (audit_id)
WHERE execution.audit_id = $1 AND audit.owner_id = $2
    AND execution.run_id IS NOT NULL
ORDER BY execution.run_id
LIMIT $3
`

// knownRunTokensSQL and knownAuditTokensSQL sum the known tokens of the Runs
// that ownedRunForUsage and ownedAuditRunsForUsage select, as knownTokensSQL
// describes. Used by Store.KnownTokens.
var (
	knownRunTokensSQL   = knownTokensSQL(ownedRunForUsage)
	knownAuditTokensSQL = knownTokensSQL(ownedAuditRunsForUsage)
)

// knownTokensSQL sums the totalTokens summary of at most $3 stage metric rows,
// in stage execution order, of the Runs that the statement owned selects;
// a row without a total counts zero.
func knownTokensSQL(owned string) string {
	return `WITH owned AS (` + owned + `)
 SELECT COALESCE(sum(known_tokens), 0)
 FROM (
   SELECT COALESCE((metrics.summary->>'totalTokens')::bigint, 0) AS known_tokens
   FROM stage_metrics metrics
   JOIN stage_executions stage USING (stage_execution_id)
   JOIN owned ON owned.run_id = stage.run_id
   ORDER BY metrics.stage_execution_id
   LIMIT $3
 ) observed`
}
