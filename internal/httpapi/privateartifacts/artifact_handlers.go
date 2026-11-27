package privateartifacts

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
	"github.com/grauwolf32/contractor/internal/httpapi/httpx"
)

const (
	bindingCreatedAtHeader  = "X-Contractor-Binding-Created-At"
	revisionCreatedAtHeader = "X-Contractor-Revision-Created-At"
)

// getArtifact takes a transfer slot only after the request and the allocation
// grant are accepted, and holds it while the payload is read and written.
func (h *handler) getArtifact(w http.ResponseWriter, r *http.Request) {
	err := h.withReadTarget(r, false, func(ctx context.Context, store artifacts.ScopedStore, ref contracts.ArtifactRef) error {
		admitted, transfer, err := artifacttransfer.Acquire(w, r.WithContext(ctx), h.dependencies.Transfers)
		if err != nil {
			return err
		}
		defer transfer.Close()
		ctx, cancel := artifacttransfer.StorageContext(admitted.Context())
		defer cancel()
		result, err := store.Read(ctx, ref)
		if err != nil {
			transfer.ReleaseBeforeWrite(0)
			return err
		}
		transfer.BoundWrite(int64(len(result.Payload.Data)))
		w.Header().Set("Content-Type", result.Payload.MediaType)
		w.Header().Set("ETag", httpx.QuotedETag(result.Ref.Revision))
		setArtifactTimestampHeaders(w.Header(), result.BindingCreatedAt, result.RevisionCreatedAt)
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(result.Payload.Data)))
		w.Header().Set("Cache-Control", "no-store")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(result.Payload.Data)
		return nil
	})
	if err != nil {
		h.handleError(w, err)
	}
}

// putArtifact takes a transfer slot after the request and allocation checks,
// before buffering the body, and releases it before writing the small result.
func (h *handler) putArtifact(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	expectedRevision, create, err := artifactWritePrecondition(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	// The names are request syntax too: a malformed one is rejected before the
	// allocation lookup and before a payload is buffered or prepared.
	ref := contracts.ArtifactRef{Namespace: r.PathValue("namespace"), Name: r.PathValue("name")}
	if contracts.ValidateArtifactName(ref.Namespace) != nil || contracts.ValidateArtifactName(ref.Name) != nil {
		h.handleError(w, fmt.Errorf("%w: invalid Artifact namespace or name", errInvalidRequest))
		return
	}
	// Reject unknown/fenced allocations before accepting a large payload. The
	// second lookup below is the write authorization linearization point and
	// catches a fence established while the request body was arriving.
	if _, err := h.runStore(r, true); err != nil {
		h.handleError(w, err)
		return
	}
	allocationID := r.PathValue("allocationID")
	identity, ok := r.Context().Value(authenticatedRuntimeContextKey{}).(authenticatedRuntime)
	if !ok {
		h.handleError(w, errArtifactAccessDenied)
		return
	}
	r, transfer, err := artifacttransfer.Acquire(w, r, h.dependencies.Transfers)
	if err != nil {
		h.handleError(w, err)
		return
	}
	defer transfer.Close()
	payload, err := readArtifactBody(r, transfer)
	if err != nil {
		transfer.ReleaseBeforeWrite(0)
		h.handleError(w, err)
		return
	}
	ctx, cancel := artifacttransfer.StorageContext(r.Context())
	defer cancel()
	preparedPayload, err := h.dependencies.Artifacts.PreparePayload(ctx, artifacts.Payload{MediaType: mediaType, Data: payload})
	if err != nil {
		transfer.ReleaseBeforeWrite(0)
		h.handleError(w, err)
		return
	}
	var result artifacts.WriteResult
	err = h.dependencies.Registry.WithWriteGrant(
		allocationID,
		func(grant controlplane.AllocationGrant) error {
			store, storeErr := h.runStoreForGrant(allocationID, grant, identity, true)
			if storeErr != nil {
				return storeErr
			}
			if ref.Namespace == artifactpolicy.HTTPExchangeNamespace {
				envelope, decodeErr := auditdomain.DecodeHTTPExchangeArtifact(payload)
				if !grant.HTTPResultEvidence {
					return errArtifactAccessDenied
				}
				if !create || mediaType != auditdomain.HTTPExchangeMediaType || decodeErr != nil || envelope.Origin.RunID != grant.RunID || envelope.Capture.Source != "http" || ref.Name != auditdomain.HTTPExchangeArtifactName(envelope.Origin.InvocationID, envelope.Capture.Source, envelope.Capture.RequestID) {
					return errInvalidRequest
				}
				scope, err := artifacts.RunScope(grant.RunID)
				if err != nil {
					return err
				}
				result, storeErr = h.dependencies.Artifacts.CreateFrozen(ctx, scope, ref, preparedPayload)
				if errors.Is(storeErr, artifacts.ErrAuditArtifactCollision) {
					return artifacts.ErrArtifactFrozen
				}
				return storeErr
			}
			preparedPayload.ExecutionOrigin = &artifacts.ExecutionOrigin{
				RunID: grant.RunID, StageExecutionID: grant.StageExecutionID,
				AllocationID: grant.AllocationID, LogicalAgentName: grant.LogicalAgentName,
			}
			result, storeErr = store.Write(ctx, ref, preparedPayload, expectedRevision)
			return storeErr
		},
	)
	transfer.ReleaseBeforeWrite(0)
	if err != nil {
		h.handleError(w, err)
		return
	}
	status := http.StatusOK
	if create {
		status = http.StatusCreated
	}
	w.Header().Set("ETag", httpx.QuotedETag(result.Ref.Revision))
	setArtifactTimestampHeaders(w.Header(), result.BindingCreatedAt, result.RevisionCreatedAt)
	writeJSON(w, status, contracts.ArtifactWriteResult{
		APIVersion: contracts.APIVersion, Artifact: result.Ref,
		MediaType: result.MediaType, Size: result.Size,
	})
}

func setArtifactTimestampHeaders(header http.Header, bindingCreatedAt, revisionCreatedAt time.Time) {
	header.Set(bindingCreatedAtHeader, bindingCreatedAt.UTC().Format(time.RFC3339Nano))
	header.Set(revisionCreatedAtHeader, revisionCreatedAt.UTC().Format(time.RFC3339Nano))
}
