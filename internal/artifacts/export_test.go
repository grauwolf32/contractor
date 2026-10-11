package artifacts

import "testing"

// ErrBlobRuntimeMissing lets external tests identify the internal error that a
// filesystem payload operation returns without an installed blob runtime.
var ErrBlobRuntimeMissing = errBlobRuntimeMissing

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
