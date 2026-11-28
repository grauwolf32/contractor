package privateartifacts

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
)

func (h *handler) writeArtifactPayload(ctx context.Context, allocationID string, identity authenticatedRuntime, ref contracts.ArtifactRef, mediaType string, payload []byte, expectedRevision *string) (artifacts.WriteResult, error) {
	prepared, err := h.dependencies.Artifacts.PreparePayload(ctx, artifacts.Payload{MediaType: mediaType, Data: payload})
	if err != nil {
		return artifacts.WriteResult{}, err
	}
	var result artifacts.WriteResult
	err = h.dependencies.Registry.WithWriteGrant(allocationID, func(grant controlplane.AllocationGrant) error {
		store, err := h.runStoreForGrant(allocationID, grant, identity, true)
		if err != nil {
			return err
		}
		if ref.Namespace == artifactpolicy.HTTPExchangeNamespace {
			envelope, err := auditdomain.DecodeHTTPExchangeArtifact(payload)
			if !grant.HTTPResultEvidence {
				return errArtifactAccessDenied
			}
			if expectedRevision != nil || mediaType != auditdomain.HTTPExchangeMediaType || err != nil || envelope.Origin.RunID != grant.RunID || envelope.Capture.Source != "http" || ref.Name != auditdomain.HTTPExchangeArtifactName(envelope.Origin.InvocationID, envelope.Capture.Source, envelope.Capture.RequestID) {
				return errInvalidRequest
			}
			scope, err := artifacts.RunScope(grant.RunID)
			if err != nil {
				return err
			}
			result, err = h.dependencies.Artifacts.CreateFrozen(ctx, scope, ref, prepared)
			if errors.Is(err, artifacts.ErrAuditArtifactCollision) {
				return artifacts.ErrArtifactFrozen
			}
			return err
		}
		prepared.ExecutionOrigin = &artifacts.ExecutionOrigin{
			RunID: grant.RunID, StageExecutionID: grant.StageExecutionID,
			AllocationID: grant.AllocationID, LogicalAgentName: grant.LogicalAgentName,
		}
		result, err = store.Write(ctx, ref, prepared, expectedRevision)
		return err
	})
	return result, err
}
