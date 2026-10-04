package privateartifacts

import (
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
	"github.com/grauwolf32/contractor/internal/httpapi/httpx"
)

const (
	bindingCreatedAtHeader  = "X-Contractor-Binding-Created-At"
	revisionCreatedAtHeader = "X-Contractor-Revision-Created-At"
)

func (h *handler) listArtifacts(w http.ResponseWriter, r *http.Request) {
	query, err := exactQuery(r.URL.RawQuery, "namespace", "namePrefix", "limit")
	if err != nil {
		h.handleError(w, err)
		return
	}
	var namespace *string
	if values, present := query["namespace"]; present {
		if values[0] == "" {
			h.handleError(w, fmt.Errorf("%w: namespace must not be empty", errInvalidRequest))
			return
		}
		namespace = &values[0]
	}
	store, err := h.runStore(r, false)
	if err != nil {
		h.handleError(w, err)
		return
	}
	var refs []artifacts.ArtifactRef
	prefixValues, prefixPresent := query["namePrefix"]
	limitValues, limitPresent := query["limit"]
	if prefixPresent || limitPresent {
		if !prefixPresent || !limitPresent || namespace == nil {
			h.handleError(w, errInvalidRequest)
			return
		}
		limit, parseErr := strconv.Atoi(limitValues[0])
		if parseErr != nil {
			h.handleError(w, errInvalidRequest)
			return
		}
		refs, err = store.ListPrefix(r.Context(), *namespace, prefixValues[0], limit)
	} else {
		refs, err = store.List(r.Context(), namespace)
	}
	if err != nil {
		h.handleError(w, err)
		return
	}
	visible := refs[:0]
	for _, ref := range refs {
		if !artifactpolicy.IsRunSystemNamespace(ref.Namespace) {
			visible = append(visible, ref)
		}
	}
	refs = visible
	response := contracts.ArtifactListResult{APIVersion: contracts.APIVersion, Artifacts: refs}
	if err := response.Validate(); err != nil {
		h.handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, response)
}

// getArtifact takes a transfer slot only after the request and the allocation
// grant are accepted, and holds it while the payload is read and written.
func (h *handler) getArtifact(w http.ResponseWriter, r *http.Request) {
	query, err := exactQuery(r.URL.RawQuery, "revision")
	if err != nil {
		h.handleError(w, err)
		return
	}
	ref := contracts.ArtifactRef{Namespace: r.PathValue("namespace"), Name: r.PathValue("name")}
	if values, present := query["revision"]; present {
		if values[0] == "" {
			h.handleError(w, fmt.Errorf("%w: revision must not be empty", errInvalidRequest))
			return
		}
		ref.Revision = &values[0]
	}
	store, err := h.runStore(r, false)
	if err != nil {
		h.handleError(w, err)
		return
	}
	if artifactpolicy.IsRunSystemNamespace(ref.Namespace) {
		h.handleError(w, artifacts.ErrReservedNamespace)
		return
	}
	r, transfer, err := artifacttransfer.Acquire(w, r, artifacts.MaxPayloadSize)
	if err != nil {
		h.handleError(w, err)
		return
	}
	defer transfer.Close()
	result, err := store.Read(r.Context(), ref)
	if err != nil {
		transfer.ReleaseBeforeWrite(0)
		h.handleError(w, err)
		return
	}
	transfer.BoundWrite(int64(len(result.Payload.Data)))
	w.Header().Set("Content-Type", result.Payload.MediaType)
	w.Header().Set("ETag", httpx.QuotedETag(result.Ref.Revision))
	setArtifactTimestampHeaders(w.Header(), result.BindingCreatedAt, result.RevisionCreatedAt)
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(result.Payload.Data)))
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(result.Payload.Data)
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
	// Reject unknown/fenced allocations before accepting a large payload. The
	// second lookup below is the write authorization linearization point and
	// catches a fence established while the request body was arriving.
	if _, err := h.runStore(r, true); err != nil {
		h.handleError(w, err)
		return
	}
	if artifactpolicy.IsRunSystemNamespace(r.PathValue("namespace")) {
		h.handleError(w, artifacts.ErrReservedNamespace)
		return
	}
	allocationID := r.PathValue("allocationID")
	identity, ok := r.Context().Value(authenticatedRuntimeContextKey{}).(authenticatedRuntime)
	if !ok {
		h.handleError(w, errArtifactAccessDenied)
		return
	}
	r, transfer, err := artifacttransfer.Acquire(w, r, r.ContentLength)
	if err != nil {
		h.handleError(w, err)
		return
	}
	defer transfer.Close()
	payload, err := readArtifactBody(w, r)
	if err != nil {
		transfer.ReleaseBeforeWrite(0)
		h.handleError(w, err)
		return
	}
	preparedPayload, err := artifacts.PreparePayload(r.Context(), artifacts.Payload{MediaType: mediaType, Data: payload})
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
			result, storeErr = store.Write(
				r.Context(),
				contracts.ArtifactRef{Namespace: r.PathValue("namespace"), Name: r.PathValue("name")},
				preparedPayload,
				expectedRevision,
			)
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
