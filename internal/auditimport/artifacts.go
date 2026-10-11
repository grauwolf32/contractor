package auditimport

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
)

type ArtifactAccess interface {
	ReadProjectExact(context.Context, string, auditstore.ExactArtifact) ([]byte, error)
	ReadRunBinding(context.Context, string, contracts.ArtifactRef) (auditstore.ExactArtifact, []byte, bool, error)
	ReadRunExact(context.Context, string, contracts.ArtifactRef) (auditstore.ExactArtifact, []byte, error)
	RetainRunExact(context.Context, string, auditstore.ExactArtifact, string, contracts.ArtifactRef) (auditstore.ExactArtifact, error)
	PutImmutableProject(context.Context, string, contracts.ArtifactRef, artifacts.Payload) (auditstore.ExactArtifact, error)
}

type ServiceArtifactAccess struct{ service *artifacts.Service }

func NewArtifactAccess(service *artifacts.Service) (*ServiceArtifactAccess, error) {
	if service == nil {
		return nil, errors.New("Audit import Artifact service is required")
	}
	return &ServiceArtifactAccess{service: service}, nil
}

// ReadProjectExact reads expected through the one exact read
// (artifacts.ScopedStore.ReadExact). Digest-only descriptors of rounds and
// executions pin the bytes; complete ones also pin media type and size.
func (a *ServiceArtifactAccess) ReadProjectExact(
	ctx context.Context, projectID string, expected auditstore.ExactArtifact,
) ([]byte, error) {
	store, err := a.service.Project(projectID)
	if err != nil {
		return nil, err
	}
	read, err := store.ReadExact(ctx, expected)
	if err != nil {
		return nil, err
	}
	return append([]byte(nil), read.Payload.Data...), nil
}

func (a *ServiceArtifactAccess) ReadRunBinding(
	ctx context.Context, runID string, ref contracts.ArtifactRef,
) (auditstore.ExactArtifact, []byte, bool, error) {
	store, err := a.service.Run(runID)
	if err != nil {
		return auditstore.ExactArtifact{}, nil, false, err
	}
	metadata, err := store.Metadata(ctx, ref)
	if err != nil {
		return auditstore.ExactArtifact{}, nil, false, err
	}
	read, err := store.ReadExact(ctx, metadata.Exact())
	if err != nil {
		return auditstore.ExactArtifact{}, nil, false, err
	}
	return metadata.Exact(), append([]byte(nil), read.Payload.Data...), metadata.Frozen, nil
}

func (a *ServiceArtifactAccess) ReadRunExact(
	ctx context.Context, runID string, ref contracts.ArtifactRef,
) (auditstore.ExactArtifact, []byte, error) {
	if err := ref.ValidateExact(); err != nil {
		return auditstore.ExactArtifact{}, nil, artifacts.ErrExactRevisionRequired
	}
	store, err := a.service.Run(runID)
	if err != nil {
		return auditstore.ExactArtifact{}, nil, err
	}
	metadata, err := store.Metadata(ctx, ref)
	if err != nil {
		return auditstore.ExactArtifact{}, nil, err
	}
	read, err := store.ReadExact(ctx, metadata.Exact())
	if err != nil {
		return auditstore.ExactArtifact{}, nil, err
	}
	return metadata.Exact(), append([]byte(nil), read.Payload.Data...), nil
}

func (a *ServiceArtifactAccess) RetainRunExact(
	ctx context.Context,
	runID string,
	source auditstore.ExactArtifact,
	projectID string,
	target contracts.ArtifactRef,
) (auditstore.ExactArtifact, error) {
	result, err := a.service.ImportAuditArtifact(ctx, runID, source.Ref, projectID, target)
	if err == nil {
		retained := auditstore.ExactArtifact{
			Ref: result.TargetRef, Digest: source.Digest,
			MediaType: result.MediaType, SizeBytes: result.Size,
		}
		if retained.MediaType != source.MediaType || retained.SizeBytes != source.SizeBytes {
			return auditstore.ExactArtifact{}, artifacts.ErrArtifactIntegrity
		}
		return retained, nil
	}
	if !errors.Is(err, artifacts.ErrArtifactConflict) {
		return auditstore.ExactArtifact{}, err
	}
	store, scopeErr := a.service.Project(projectID)
	if scopeErr != nil {
		return auditstore.ExactArtifact{}, scopeErr
	}
	metadata, metadataErr := store.Metadata(ctx, target)
	if metadataErr != nil {
		return auditstore.ExactArtifact{}, metadataErr
	}
	// Crash replay: the retained binding must hold the source's content.
	expected := source
	expected.Ref = metadata.Ref
	if err := expected.Verify(metadata.Exact()); err != nil {
		return auditstore.ExactArtifact{}, fmt.Errorf("%w: Audit retained binding collision", artifacts.ErrArtifactIntegrity)
	}
	lineage, lineageErr := store.ListLineage(ctx, metadata.Ref, artifacts.LineagePageQuery{Limit: 2})
	if lineageErr != nil {
		return auditstore.ExactArtifact{}, lineageErr
	}
	if len(lineage) != 1 || lineage[0].Kind != artifacts.LineageAuditImport ||
		lineage[0].SourceScope != artifacts.ScopeRun || lineage[0].SourceScopeID != runID ||
		lineage[0].TargetScope != artifacts.ScopeProject || lineage[0].TargetScopeID != projectID ||
		!lineage[0].Source.SameExact(source.Ref) || !lineage[0].Target.SameExact(metadata.Ref) {
		return auditstore.ExactArtifact{}, fmt.Errorf("%w: Audit retained lineage collision", artifacts.ErrArtifactIntegrity)
	}
	return metadata.Exact(), nil
}

// PutImmutableProject writes a content-named report or retained scan package
// as a frozen Audit artifact. WriteAuditArtifact recovers a lost
// acknowledgement; another payload under the name is an integrity failure.
func (a *ServiceArtifactAccess) PutImmutableProject(
	ctx context.Context,
	projectID string,
	target contracts.ArtifactRef,
	payload artifacts.Payload,
) (auditstore.ExactArtifact, error) {
	written, err := a.service.WriteAuditArtifact(ctx, projectID, target, payload)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	return contracts.ExactContent(written.Ref, written.MediaType, payload.Data), nil
}
