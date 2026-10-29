package privateartifacts

import (
	"context"
	"net/http"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
)

type sitemapCollectionRequest struct {
	APIVersion string                           `json:"apiVersion" strictjson:"required"`
	Input      projectsitemap.CollectionIngress `json:"input" strictjson:"required"`
}

func sitemapCollector(grant controlplane.AllocationGrant) bool {
	return grant.SitemapPolicy.CollectObservations && grant.WritePolicy == controlplane.WriteInputsAndIntermediates
}

func (h *handler) collectSitemapObservations(w http.ResponseWriter, r *http.Request) {
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
	var input sitemapCollectionRequest
	if err := decodeBoundedJSON(w, r, projectsitemap.MaxBatchBytes, &input); err != nil {
		h.handleError(w, err)
		return
	}
	if input.APIVersion != contracts.APIVersion {
		h.handleError(w, errInvalidRequest)
		return
	}
	err = h.withSitemapGrant(r, sitemapCollector, func(ctx context.Context, access projectsitemap.AllocationAccess) error {
		result, err := h.dependencies.Sitemap.CollectAllocation(ctx, access, keys[0], input.Input)
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
