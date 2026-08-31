package privateartifacts

import (
	"fmt"
	"net/http"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
)

func (h *handler) listArtifacts(w http.ResponseWriter, r *http.Request) {
	query, err := exactQuery(r.URL.RawQuery, "namespace")
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
	refs, err := store.List(r.Context(), namespace)
	if err != nil {
		h.handleError(w, err)
		return
	}
	response := contracts.ArtifactListResult{APIVersion: contracts.APIVersion, Artifacts: refs}
	if err := response.Validate(); err != nil {
		h.handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, response)
}

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
	result, err := store.Read(r.Context(), ref)
	if err != nil {
		h.handleError(w, err)
		return
	}
	w.Header().Set("Content-Type", result.Payload.MediaType)
	w.Header().Set("ETag", quotedETag(result.Ref.Revision))
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(result.Payload.Data)))
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(result.Payload.Data)
}

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
	payload, err := readArtifactBody(w, r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	allocationID := r.PathValue("allocationID")
	var result artifacts.WriteResult
	err = h.dependencies.Registry.WithWriteGrant(
		allocationID,
		func(grant controlplane.AllocationGrant) error {
			store, storeErr := h.runStoreForGrant(allocationID, grant, true)
			if storeErr != nil {
				return storeErr
			}
			result, storeErr = store.Write(
				r.Context(),
				contracts.ArtifactRef{Namespace: r.PathValue("namespace"), Name: r.PathValue("name")},
				artifacts.Payload{MediaType: mediaType, Data: payload},
				expectedRevision,
			)
			return storeErr
		},
	)
	if err != nil {
		h.handleError(w, err)
		return
	}
	status := http.StatusOK
	if create {
		status = http.StatusCreated
	}
	w.Header().Set("ETag", quotedETag(result.Ref.Revision))
	writeJSON(w, status, contracts.ArtifactWriteResult{
		APIVersion: contracts.APIVersion, Artifact: result.Ref,
		MediaType: result.MediaType, Size: result.Size,
	})
}
