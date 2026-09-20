package evalservice

import (
	"context"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	pg "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

func verifyArtifact(ctx context.Context, db pg.DBTX, service *artifacts.Service, owner string, ref evaldomain.Artifact) error {
	var store artifacts.ScopedStore
	var err error
	switch ref.Scope {
	case "user":
		if ref.ScopeID != owner {
			return evaldomain.Failure("eval_not_found")
		}
		store, err = service.User(owner)
	case "project":
		if _, err = projectstore.NewPostgresStore(db).Get(ctx, owner, ref.ScopeID); err != nil {
			return preflightDependencyError(err)
		}
		store, err = service.Project(ref.ScopeID)
	default:
		return evaldomain.Failure("eval_invalid")
	}
	if err != nil {
		return err
	}
	metadata, err := store.Metadata(ctx, contracts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name, Revision: &ref.Revision})
	if err != nil {
		return preflightDependencyError(err)
	}
	if metadata.Digest != ref.SHA256 || metadata.MediaType != ref.MediaType || metadata.Size != ref.SizeBytes {
		return evaldomain.Failure("eval_pin_mismatch")
	}
	return nil
}

// artifactScope follows a preceding owner check; it must not be exposed as an
// authorization shortcut to a public request handler.
func artifactScope(service *artifacts.Service, ref evaldomain.Artifact) (artifacts.ScopedStore, error) {
	if ref.Scope == "user" {
		return service.User(ref.ScopeID)
	}
	if ref.Scope == "project" {
		return service.Project(ref.ScopeID)
	}
	return artifacts.ScopedStore{}, evaldomain.Failure("eval_invalid")
}
