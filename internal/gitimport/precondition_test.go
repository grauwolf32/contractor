package gitimport

import (
	"context"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

type preconditionStore struct {
	artifacts.Repository
	artifacts.QueryRepository
	metadata artifacts.Metadata
	err      error
}

func (s preconditionStore) Metadata(context.Context, artifacts.Scope, artifacts.ArtifactRef) (artifacts.Metadata, error) {
	return s.metadata, s.err
}

func TestGitImportImplicitCreatePreservesUpdateCAS(t *testing.T) {
	current, stale := "revision-current", "revision-stale"
	target := artifacts.ArtifactRef{Namespace: "source", Name: "repository"}
	for _, tc := range []struct {
		name     string
		absent   bool
		expected *string
		conflict bool
	}{
		{"absent-without-precondition", true, nil, false},
		{"existing-without-precondition", false, nil, true},
		{"absent-with-update-precondition", true, &current, true},
		{"existing-stale-update", false, &stale, true},
		{"existing-exact-update", false, &current, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			store := preconditionStore{metadata: artifacts.Metadata{Ref: artifacts.ArtifactRef{Namespace: target.Namespace, Name: target.Name, Revision: &current}}}
			if tc.absent {
				store.err = artifacts.ErrArtifactNotFound
			}
			scoped, err := artifacts.NewService(store).User("user-1")
			if err != nil {
				t.Fatal(err)
			}
			err = checkImportPrecondition(t.Context(), scoped, target, tc.expected)
			var conflict *artifacts.ConflictError
			if errors.As(err, &conflict) != tc.conflict || (!tc.conflict && err != nil) {
				t.Fatalf("precondition error = %v, want conflict=%v", err, tc.conflict)
			}
		})
	}
}
