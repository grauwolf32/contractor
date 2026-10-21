package auditservice

import (
	"context"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

// ReadArtifact resolves one committed receipt link in an owned Audit. The
// requested revision is mandatory: later report acceptance cannot change the
// bytes beneath a saved link. Staged Project artifacts have no receipt link.
func (s *Service) ReadArtifact(ctx context.Context, ownerID, auditID, key, revision string) (artifacts.ReadResult, error) {
	audit, err := s.reads().Get(ctx, ownerID, auditID)
	if err != nil {
		return artifacts.ReadResult{}, err
	}
	if audit.State == auditstore.AuditDeleting {
		return artifacts.ReadResult{}, auditstore.ErrNotFound
	}
	link, err := s.reads().GetArtifactLink(ctx, auditID, key)
	if err != nil {
		return artifacts.ReadResult{}, err
	}
	if link.Artifact.Ref.Revision == nil || *link.Artifact.Ref.Revision != revision {
		return artifacts.ReadResult{}, auditstore.ErrNotFound
	}
	project, err := artifacts.NewService(artifacts.NewPostgresRepository(s.pool)).Project(audit.ProjectID)
	if err != nil {
		return artifacts.ReadResult{}, err
	}
	read, err := project.ReadExact(ctx, link.Artifact)
	if err != nil {
		return artifacts.ReadResult{}, err
	}
	if err := s.checkPageRevision(ctx, ownerID, auditID, audit.Revision); err != nil {
		return artifacts.ReadResult{}, err
	}
	return read, nil
}
