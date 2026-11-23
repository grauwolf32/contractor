package artifacts

import (
	"context"
)

// ReadPrefixBatch returns at most limit current bindings and their payloads in
// name order. It lets callers detect quota overflow by requesting one extra
// binding without performing a separate listing and one read per artifact.
func (s ScopedStore) ReadPrefixBatch(ctx context.Context, namespace, prefix string, limit int) ([]ReadResult, error) {
	if err := validatePrefixList(namespace, prefix, limit); err != nil {
		return nil, err
	}
	return s.service.repository.ReadPrefixBatch(ctx, s.scope, namespace, prefix, limit)
}
