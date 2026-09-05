package auditimport

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
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

func (a *ServiceArtifactAccess) ReadProjectExact(
	ctx context.Context, projectID string, expected auditstore.ExactArtifact,
) ([]byte, error) {
	store, err := a.service.Project(projectID)
	if err != nil {
		return nil, err
	}
	read, err := store.Read(ctx, expected.Ref)
	if err != nil {
		return nil, err
	}
	if read.Ref.Revision == nil || expected.Ref.Revision == nil ||
		*read.Ref.Revision != *expected.Ref.Revision || digestBytes(read.Payload.Data) != expected.Digest ||
		expected.MediaType != "" && (read.Payload.MediaType != expected.MediaType || int64(len(read.Payload.Data)) != expected.SizeBytes) {
		return nil, artifacts.ErrArtifactIntegrity
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
	read, err := store.Read(ctx, metadata.Ref)
	if err != nil {
		return auditstore.ExactArtifact{}, nil, false, err
	}
	if digestBytes(read.Payload.Data) != metadata.Digest || read.Payload.MediaType != metadata.MediaType {
		return auditstore.ExactArtifact{}, nil, false, artifacts.ErrArtifactIntegrity
	}
	return exactFromMetadata(metadata), append([]byte(nil), read.Payload.Data...), metadata.Frozen, nil
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
	read, err := store.Read(ctx, metadata.Ref)
	if err != nil {
		return auditstore.ExactArtifact{}, nil, err
	}
	if digestBytes(read.Payload.Data) != metadata.Digest || read.Payload.MediaType != metadata.MediaType {
		return auditstore.ExactArtifact{}, nil, artifacts.ErrArtifactIntegrity
	}
	return exactFromMetadata(metadata), append([]byte(nil), read.Payload.Data...), nil
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
	if metadata.Digest != source.Digest || metadata.MediaType != source.MediaType || metadata.Size != source.SizeBytes {
		return auditstore.ExactArtifact{}, fmt.Errorf("%w: Audit retained binding collision", artifacts.ErrArtifactIntegrity)
	}
	lineage, lineageErr := store.ListLineage(ctx, metadata.Ref, artifacts.LineagePageQuery{Limit: 2})
	if lineageErr != nil {
		return auditstore.ExactArtifact{}, lineageErr
	}
	if len(lineage) != 1 || lineage[0].Kind != artifacts.LineageAuditImport ||
		lineage[0].SourceScope != artifacts.ScopeRun || lineage[0].SourceScopeID != runID ||
		lineage[0].TargetScope != artifacts.ScopeProject || lineage[0].TargetScopeID != projectID ||
		!sameArtifactRef(lineage[0].Source, source.Ref) || !sameArtifactRef(lineage[0].Target, metadata.Ref) {
		return auditstore.ExactArtifact{}, fmt.Errorf("%w: Audit retained lineage collision", artifacts.ErrArtifactIntegrity)
	}
	return exactFromMetadata(metadata), nil
}

func (a *ServiceArtifactAccess) PutImmutableProject(
	ctx context.Context,
	projectID string,
	target contracts.ArtifactRef,
	payload artifacts.Payload,
) (auditstore.ExactArtifact, error) {
	written, err := a.service.WriteAuditArtifact(ctx, projectID, target, payload)
	if err == nil {
		return auditstore.ExactArtifact{
			Ref: written.Ref, Digest: digestBytes(payload.Data),
			MediaType: written.MediaType, SizeBytes: written.Size,
		}, nil
	}
	if !errors.Is(err, artifacts.ErrArtifactConflict) {
		return auditstore.ExactArtifact{}, err
	}
	store, err := a.service.Project(projectID)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	read, readErr := store.Read(ctx, target)
	if readErr != nil {
		return auditstore.ExactArtifact{}, readErr
	}
	if read.Payload.MediaType != payload.MediaType || !bytes.Equal(read.Payload.Data, payload.Data) {
		return auditstore.ExactArtifact{}, fmt.Errorf("%w: Audit immutable binding collision", artifacts.ErrArtifactIntegrity)
	}
	return auditstore.ExactArtifact{
		Ref: read.Ref, Digest: digestBytes(read.Payload.Data),
		MediaType: read.Payload.MediaType, SizeBytes: int64(len(read.Payload.Data)),
	}, nil
}

func exactFromMetadata(value artifacts.Metadata) auditstore.ExactArtifact {
	return auditstore.ExactArtifact{
		Ref: value.Ref, Digest: value.Digest,
		MediaType: value.MediaType, SizeBytes: value.Size,
	}
}

func digestBytes(value []byte) string {
	digest := sha256.Sum256(value)
	return "sha256:" + hex.EncodeToString(digest[:])
}

func sameArtifactRef(left, right contracts.ArtifactRef) bool {
	return left.Namespace == right.Namespace && left.Name == right.Name &&
		left.Revision != nil && right.Revision != nil && *left.Revision == *right.Revision
}
