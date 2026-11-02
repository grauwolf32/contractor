package postgres

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/blob"
)

// ListProjectMetadataSQL lets external tests explain the generic Project
// listing query.
var ListProjectMetadataSQL = listProjectMetadataSQL

// PurgeBatchSize reports the purge batch bound (LIM-DEL-2) that external tests
// measure purge statements against.
func PurgeBatchSize() int { return purgeBatchSize }

// SetPurgeBatchSize shrinks the purge batch for one test, so that a few
// artifacts span several batches. Tests that call it must not run in parallel.
func SetPurgeBatchSize(t testing.TB, size int) {
	t.Helper()
	previous := purgeBatchSize
	purgeBatchSize = size
	t.Cleanup(func() { purgeBatchSize = previous })
}

// NewTestHistoryPolicy returns a retained-history policy below the
// configurable floors, so that a test fills a binding's history with a few
// small revisions.
func NewTestHistoryPolicy(maxRevisions, maxBytes int64) HistoryPolicy {
	return HistoryPolicy{maxRevisions: maxRevisions, maxBytes: maxBytes}
}

// RuntimeForTest lets external registry tests recompose a process policy with the same physical adapter.
func RuntimeForTest(s *Storage) *blob.Runtime { return s.runtime }
