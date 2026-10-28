package public

import (
	"context"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
	"net/http"

	"github.com/grauwolf32/contractor/internal/projectsitemap"
)

// ProjectSitemap keeps the public surface owner-bound; allocation authority is
// established separately by the private adapter, never by these JSON fields.
type ProjectSitemap interface {
	Read(context.Context, string, string, string) (projectsitemap.View, error)
	Mutate(context.Context, projectsitemap.Authority, string, projectsitemap.Mutation) (projectsitemap.Result, error)
}

func (h *projectHandlers) getSitemap(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	query, err := exactQuery(r.URL.RawQuery, "revision")
	if err != nil {
		handleError(w, err)
		return
	}
	if _, err := h.Projects.Get(r.Context(), principalUserID(r.Context()), r.PathValue("projectId")); err != nil {
		handleError(w, err)
		return
	}
	r, transfer, err := artifacttransfer.Acquire(w, r)
	if err != nil {
		handleError(w, err)
		return
	}
	defer transfer.Close()
	view, err := h.Sitemap.Read(r.Context(), principalUserID(r.Context()), r.PathValue("projectId"), query.Get("revision"))
	if err != nil {
		transfer.ReleaseBeforeWrite(0)
		handleError(w, err)
		return
	}
	transfer.BoundWrite(projectsitemap.MaxDocumentBytes + 1024)
	writeJSON(w, http.StatusOK, view)
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
	var mutation projectsitemap.Mutation
	if err := decodeJSONBounded(w, r, &mutation, projectsitemap.MaxBatchBytes); err != nil {
		handleError(w, err)
		return
	}
	if mutation.Operation == "observe" {
		handleError(w, errInvalidRequest)
		return
	}
	sourceID := projectsitemap.OwnerSourceID
	if mutation.Source != nil && mutation.Source.SourceID != sourceID {
		handleError(w, errInvalidRequest)
		return
	}
	authority := projectsitemap.Authority{OwnerID: principalUserID(r.Context()), ProjectID: r.PathValue("projectId"),
		SourceID: sourceID, Kind: "manual"}
	result, err := h.Sitemap.Mutate(r.Context(), authority, key, mutation)
	if err != nil {
		handleError(w, err)
		return
	}
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	writeJSON(w, http.StatusOK, result)
}
