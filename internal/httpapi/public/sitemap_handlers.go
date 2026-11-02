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

func (h *projectHandlers) mutateSitemap(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	if err := requireJSONMediaType(r); err != nil {
		return err
	}
	key, err := requireIdempotencyKey(r)
	if err != nil {
		return err
	}
	var command projectsitemap.EditorMutation
	if err := decodeJSONBounded(w, r, &command, projectsitemap.MaxBatchBytes); err != nil {
		return err
	}
	if command.Operation == "observe" {
		return errInvalidRequest
	}
	sourceID := projectsitemap.OwnerSourceID
	authority := projectsitemap.Authority{OwnerID: principalUserID(r.Context()), ProjectID: r.PathValue("projectId"),
		SourceID: sourceID, Kind: "manual"}
	result, err := h.Sitemap.Mutate(r.Context(), authority, key, command.WithSource(sourceID))
	if err != nil {
		return err
	}
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	writeJSON(w, http.StatusOK, result)
	return nil
}
