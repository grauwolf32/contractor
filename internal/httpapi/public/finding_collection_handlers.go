package public

import (
	"context"
	"net/http"

	"github.com/grauwolf32/contractor/internal/findingintake"
)

type FindingCollectionManagement interface {
	PublishCollection(context.Context, findingintake.PublishCollectionParams) (findingintake.PublishedCollection, error)
}

func (h *findingHandlers) publishFindingCollection(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	if err := requireJSONMediaType(r); err != nil {
		return err
	}
	var request findingintake.PublishCollectionRequest
	if err := decodeJSON(w, r, &request); err != nil {
		return err
	}
	result, err := h.FindingCollections.PublishCollection(r.Context(), findingintake.PublishCollectionParams{
		OwnerID: principalUserID(r.Context()), Request: request,
	})
	if err != nil {
		return err
	}
	status := http.StatusCreated
	if result.Replayed {
		status = http.StatusOK
	}
	writeJSON(w, status, result)
	return nil
}
