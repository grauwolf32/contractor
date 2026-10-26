package auditservice

import (
	"context"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

// dependencySourceReader is implemented by both the ordinary query store and
// the typed admission session's transaction-scoped reader. It exposes only the
// accepted source lookups needed to construct an exact dependency closure.
type dependencySourceReader interface {
	GetPreparationOutput(context.Context, string, string, string) (auditstore.AcceptedPreparationOutput, error)
	GetArtifactLink(context.Context, string, string) (auditstore.ArtifactLink, error)
	DependencyResultSource(context.Context, string, string, string) (auditstore.DependencyResultSource, error)
}

func (w auditArtifactWriter) withArtifactService(ctx context.Context, namespace string, operation func(*artifacts.Service) error) error {
	if w.claim != nil {
		return w.writes.WithArtifactWrite(ctx, *w.claim, w.projectID, namespace, operation)
	}
	return operation(w.service)
}
