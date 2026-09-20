package auditimport

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

// Scanner execution coverage is independent from a verified security verdict.
// Even a completed empty report cannot satisfy or refute a security check, and
// raw scanner observations do not on their own establish a verified violation.
func scanCoverage(task auditdomain.ItemTask, result auditdomain.CheckResult, evidence map[string]validatedEvidence, coverage auditstore.Coverage) (auditstore.Coverage, error) {
	invalid := func() (auditstore.Coverage, error) {
		return auditstore.Coverage{}, fmt.Errorf("%s", auditdomain.CodeResultSetInvalid)
	}
	if len(coverage.Completed) > 0 {
		if !task.Scan.Runnable || !equalStrings(coverage.Completed, coverage.Requested) {
			return invalid()
		}
		found := false
		for _, id := range result.EvidenceIDs {
			item, exists := evidence[id]
			if !exists {
				return invalid()
			}
			found = found || item.value.Kind == "scanner-report"
		}
		if !found {
			return invalid()
		}
	}
	switch result.Assessment {
	case "not-tested":
		if len(coverage.Completed) != 0 {
			return invalid()
		}
		coverage.Status = auditstore.CoverageNotTested
	case "blocked":
		if len(coverage.Completed) != 0 || len(coverage.Gaps) == 0 {
			return invalid()
		}
		coverage.Status = auditstore.CoverageBlocked
	case "inconclusive":
		coverage.Status = auditstore.CoverageInconclusive
	default:
		return invalid()
	}
	return coverage, nil
}
