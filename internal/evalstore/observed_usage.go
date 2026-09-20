package evalstore

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

const maxObservedUsageRows = 10_000

const ownedRunForUsage = `
SELECT run_id
FROM workflow_runs
WHERE run_id = $1 AND owner_id = $2
`

const ownedAuditRunsForUsage = `
SELECT DISTINCT execution.run_id
FROM audit_executions execution
JOIN audits audit USING (audit_id)
WHERE execution.audit_id = $1 AND audit.owner_id = $2
    AND execution.run_id IS NOT NULL
ORDER BY execution.run_id
LIMIT $3
`

// KnownTokens is a bounded lower bound for budget enforcement. Each stage row
// contributes once across all owned Audit roles; no parent aggregate is added.
// Complete public usage and provenance belong to the collection read model.
func (s *Store) KnownTokens(ctx context.Context, owner, kind string, executionID *string) (int64, error) {
	if executionID == nil {
		return 0, nil
	}
	owned := ownedRunForUsage
	switch kind {
	case "run":
	case "audit":
		owned = ownedAuditRunsForUsage
	default:
		return 0, evaldomain.Failure("eval_invalid")
	}
	query := `WITH owned AS (` + owned + `)
 SELECT COALESCE(sum(known_tokens), 0)
 FROM (
   SELECT COALESCE((metrics.summary->>'totalTokens')::bigint, 0) AS known_tokens
   FROM stage_metrics metrics
   JOIN stage_executions stage USING (stage_execution_id)
   JOIN owned ON owned.run_id = stage.run_id
   ORDER BY metrics.stage_execution_id
   LIMIT $3
 ) observed`
	var total int64
	if err := s.db.QueryRow(ctx, query, *executionID, owner, maxObservedUsageRows).Scan(&total); err != nil {
		return 0, fmt.Errorf("read eval observed usage: %w", err)
	}
	return total, nil
}
