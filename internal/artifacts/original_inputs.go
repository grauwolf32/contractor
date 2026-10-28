package artifacts

import (
	"context"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// DescribeOriginalInputs describes immutable Run-creation input forks in the
// requested name order, omitting absent names. It never substitutes an editable
// current input binding. The selection is bounded by MaxExactReadBatchSize and
// reads only metadata through the caller's repository/transaction.
func (s ScopedStore) DescribeOriginalInputs(ctx context.Context, names []string) ([]contracts.ExactArtifact, error) {
	if err := validateOriginalInputs(s.scope, names); err != nil {
		return nil, err
	}
	if len(names) == 0 {
		return []contracts.ExactArtifact{}, nil
	}
	return s.service.repository.DescribeOriginalInputs(ctx, s.scope, names)
}

func validateOriginalInputs(scope Scope, names []string) error {
	if scope.kind != ScopeRun || validateScope(scope) != nil {
		return ErrInvalidScope
	}
	if len(names) > MaxExactReadBatchSize {
		return ErrPayloadTooLarge
	}
	seen := make(map[string]bool, len(names))
	for _, name := range names {
		if validateComponent(name) != nil || seen[name] {
			return ErrInvalidName
		}
		seen[name] = true
	}
	return nil
}
