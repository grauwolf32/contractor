package public

import (
	"context"
	"net/http"

	"github.com/grauwolf32/contractor/internal/projectsitemap"
)

// ProjectSitemap keeps the public surface owner-bound; allocation authority is
// established separately by the private adapter, never by these JSON fields.
type ProjectSitemap interface {
	Mutate(context.Context, projectsitemap.Authority, string, projectsitemap.Mutation) (projectsitemap.Result, error)
}

func (h *projectHandlers) mutateSitemap(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	if err := requireJSONMediaType(r); err != nil {
		handleError(w, err)
		return
	}
	key, err := requireIdempotencyKey(r)
	if err != nil {
		handleError(w, err)
		return
	}
	var command projectsitemap.EditorMutation
	if err := decodeJSONBounded(w, r, &command, projectsitemap.MaxBatchBytes); err != nil {
		handleError(w, err)
		return
	}
	if command.Operation == "observe" {
		handleError(w, errInvalidRequest)
		return
	}
	sourceID := projectsitemap.OwnerSourceID
	authority := projectsitemap.Authority{OwnerID: principalUserID(r.Context()), ProjectID: r.PathValue("projectId"),
		SourceID: sourceID, Kind: "manual"}
	result, err := h.Sitemap.Mutate(r.Context(), authority, key, command.WithSource(sourceID))
	if err != nil {
		handleError(w, err)
		return
	}
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	writeJSON(w, http.StatusOK, result)
}
