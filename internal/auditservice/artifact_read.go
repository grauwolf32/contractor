package auditservice

import (
	"context"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"

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
	var link auditstore.ArtifactLink
	// A Task result is already an accepted exact Artifact. This read names
	// that owner projection directly; it creates no extra retention or receipt.
	if taskID, found := strings.CutPrefix(key, "task/"); found {
		taskID, found = strings.CutSuffix(taskID, "/result")
		if !found || !contracts.ValidResourceID(taskID) {
			return artifacts.ReadResult{}, auditstore.ErrNotFound
		}
		task, readErr := s.reads().GetTask(ctx, ownerID, auditID, taskID)
		if readErr != nil {
			return artifacts.ReadResult{}, readErr
		}
		if task.Result == nil || task.Result.ScopeKind != "audit" || task.Result.ScopeID != auditID {
			return artifacts.ReadResult{}, auditstore.ErrNotFound
		}
		result := task.Result
		link = auditstore.ArtifactLink{Scope: artifacts.ScopeAudit, Artifact: contracts.ExactArtifact{Ref: result.Ref, Digest: result.Digest, MediaType: result.MediaType, SizeBytes: result.SizeBytes}}
	} else {
		link, err = s.reads().GetArtifactLink(ctx, auditID, key)
		if err != nil {
			return artifacts.ReadResult{}, err
		}
	}
	if link.Artifact.Ref.Revision == nil || *link.Artifact.Ref.Revision != revision {
		return artifacts.ReadResult{}, auditstore.ErrNotFound
	}

	scope, err := artifacts.ProjectScope(audit.ProjectID)
	if link.Scope == artifacts.ScopeAudit {
		scope, err = artifacts.AuditScope(auditID)
	}
	if err != nil {
		return artifacts.ReadResult{}, err
	}
	project, err := artifacts.NewService(s.storage.Repository(s.pool)).Scoped(scope)
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
