// Package artifactpgtest constructs explicit Artifact dependencies for tests.
package artifactpgtest

import (
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	inline "github.com/grauwolf32/contractor/internal/artifacts/blob/backends/postgres"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
)

// NewRuntime provides independent inline transfer admission for a test process.
func NewRuntime() *blob.Runtime { return blob.NewRuntime(inline.Store{}, nil) }

// NewStorage uses inline bytes and the documented history limits.
func NewStorage() *artifactpostgres.Storage {
	return NewConfigured(NewRuntime(), artifactpostgres.DefaultHistoryPolicy())
}

// NewConfigured preserves the test's physical store, admission and retention.
func NewConfigured(runtime *blob.Runtime, history artifactpostgres.HistoryPolicy) *artifactpostgres.Storage {
	storage, err := artifactpostgres.NewStorage(runtime, history)
	if err != nil {
		panic(err)
	}
	return storage
}
