package public

import (
	"context"
	"net/http"

	"github.com/grauwolf32/contractor/internal/findingintake"
)

type FindingCollectionManagement interface {
	PublishCollection(context.Context, findingintake.PublishCollectionParams) (findingintake.PublishedCollection, error)
}

func (h *handler) publishFindingCollection(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.dependencies.FindingCollections == nil {
		h.handleError(w, findingintake.ErrNotFound)
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		h.handleError(w, errInvalidRequest)
		return
	}
	var request findingintake.PublishCollectionRequest
	if err := decodeJSON(w, r, &request); err != nil {
		h.handleError(w, err)
		return
	}
	result, err := h.dependencies.FindingCollections.PublishCollection(r.Context(), findingintake.PublishCollectionParams{
		OwnerID: principalUserID(r.Context()), Request: request,
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	status := http.StatusCreated
	if result.Replayed {
		status = http.StatusOK
	}
	writeJSON(w, status, result)
}
