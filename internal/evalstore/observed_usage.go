package evalstore

import (
	"context"
	"fmt"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

const maxObservedUsageRows = 10_000

// KnownTokens is a bounded lower bound for budget enforcement. Each stage row
// contributes once across all owned Audit roles; no parent aggregate is added.
// Complete public usage and provenance belong to the collection read model.
func (s *Store) KnownTokens(ctx context.Context, owner, kind string, executionID *string) (int64, error) {
	if executionID == nil {
		return 0, nil
	}
	query := knownRunTokensSQL
	switch kind {
	case "run":
	case "audit":
		query = knownAuditTokensSQL
	default:
		return 0, evaldomain.Failure("eval_invalid")
	}
	var total int64
	if err := s.db.QueryRow(ctx, query, *executionID, owner, maxObservedUsageRows).Scan(&total); err != nil {
		return 0, fmt.Errorf("read eval observed usage: %w", err)
	}
	return total, nil
}
