package privateartifacts

import (
	"context"
	"net/http"
	"slices"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
)

// Automatic is composition-owned transport state, never an import tool argument.
// It selects the collection grant and confines report notifications to the
// publishing allocation namespace. Explicit imports use ordinary Run read access.
type sitemapImportRequest struct {
	APIVersion string                         `json:"apiVersion" strictjson:"required"`
	Selection  projectsitemap.ImportSelection `json:"selection" strictjson:"required"`
	Automatic  bool                           `json:"automatic" strictjson:"required"`
}

func (h *handler) importSitemapSource(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	media, err := requestMediaType(r)
	if err != nil || media != "application/json" {
		h.handleError(w, errInvalidRequest)
		return
	}
	keys := r.Header.Values("Idempotency-Key")
	if len(keys) != 1 || !contracts.ValidIdempotencyKey(keys[0]) {
		h.handleError(w, errInvalidRequest)
		return
	}
	var input sitemapImportRequest
	if err := decodeBoundedJSON(w, r, projectsitemap.MaxBatchBytes, &input); err != nil {
		h.handleError(w, err)
		return
	}
	if input.APIVersion != contracts.APIVersion || input.Selection.Validate() != nil {
		h.handleError(w, errInvalidRequest)
		return
	}
	err = h.withSitemapGrant(r, func(grant controlplane.AllocationGrant) bool {
		if input.Automatic {
			return sitemapCollector(grant) && input.Selection.Artifact.Namespace == grant.Namespace
		}
		return grant.WritePolicy == controlplane.WriteInputsAndIntermediates && slices.Contains(grant.SitemapTools, contracts.SitemapImportTool)
	}, func(ctx context.Context, access projectsitemap.AllocationAccess) error {
		result, err := h.dependencies.Sitemap.ImportAllocation(ctx, access, keys[0], input.Selection)
		if err != nil {
			return err
		}
		writeJSON(w, http.StatusOK, struct {
			APIVersion string                `json:"apiVersion"`
			Result     projectsitemap.Result `json:"result"`
		}{contracts.APIVersion, result})
		return nil
	})
	if err != nil {
		h.handleError(w, err)
	}
}
