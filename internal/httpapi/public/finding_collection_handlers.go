package public

import (
	"context"
	"net/http"

	"github.com/grauwolf32/contractor/internal/findingintake"
)

type FindingCollectionManagement interface {
	PublishCollection(context.Context, findingintake.PublishCollectionParams) (findingintake.PublishedCollection, error)
}

func (h *findingHandlers) publishFindingCollection(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		handleError(w, errInvalidRequest)
		return
	}
	var request findingintake.PublishCollectionRequest
	if err := decodeJSON(w, r, &request); err != nil {
		handleError(w, err)
		return
	}
	result, err := h.FindingCollections.PublishCollection(r.Context(), findingintake.PublishCollectionParams{
		OwnerID: principalUserID(r.Context()), Request: request,
	})
	if err != nil {
		handleError(w, err)
		return
	}
	status := http.StatusCreated
	if result.Replayed {
		status = http.StatusOK
	}
	writeJSON(w, status, result)
}
