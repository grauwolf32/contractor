package public

import (
	"fmt"
	"net/http"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
)

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
	expectedRevision, err := artifactWritePrecondition(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	payload, err := readArtifactBody(w, r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	store, err := h.dependencies.Artifacts.User(h.dependencies.UserID)
	if err != nil {
		h.handleError(w, err)
		return
	}
	result, err := store.Write(r.Context(), contracts.ArtifactRef{
		Namespace: r.PathValue("namespace"),
		Name:      r.PathValue("name"),
	}, artifacts.Payload{MediaType: mediaType, Data: payload}, expectedRevision)
	if err != nil {
		h.handleError(w, err)
		return
	}

	status := http.StatusCreated
	if expectedRevision != nil {
		status = http.StatusOK
	}
	w.Header().Set("ETag", quotedETag(result.Ref.Revision))
	writeJSON(w, status, artifactWriteResponse{
		Artifact:  result.Ref,
		MediaType: result.MediaType,
		Size:      result.Size,
	})
}

func (h *handler) getArtifact(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodHead {
		h.methodNotAllowed(w, r)
		return
	}
	query, err := exactQuery(r.URL.RawQuery, "revision")
	if err != nil {
		h.handleError(w, err)
		return
	}
	ref := contracts.ArtifactRef{
		Namespace: r.PathValue("namespace"),
		Name:      r.PathValue("name"),
	}
	if revision, present := query["revision"]; present {
		if revision[0] == "" {
			h.handleError(w, fmt.Errorf("%w: revision must not be empty", errInvalidRequest))
			return
		}
		ref.Revision = &revision[0]
	}
	store, err := h.dependencies.Artifacts.User(h.dependencies.UserID)
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
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(result.Payload.Data)
}
