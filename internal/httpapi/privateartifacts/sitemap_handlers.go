package privateartifacts

import (
	"context"
	"errors"
	"net/http"
	"slices"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
)

var (
	errSitemapAccessDenied = errors.New("allocation does not permit this sitemap operation")
	errSitemapUnavailable  = errors.New("sitemap service is unavailable")
)

type sitemapMutationRequest struct {
	APIVersion string                        `json:"apiVersion" strictjson:"required"`
	Mutation   projectsitemap.EditorMutation `json:"mutation" strictjson:"required"`
}
type sitemapSnapshotRequest struct {
	APIVersion string                           `json:"apiVersion" strictjson:"required"`
	Selection  projectsitemap.SnapshotSelection `json:"selection" strictjson:"required"`
}

func (h *handler) withSitemapGrant(r *http.Request, allowed func(controlplane.AllocationGrant) bool, operation func(context.Context, projectsitemap.AllocationAccess) error) error {
	_, identity, err := h.allocationGrant(r)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(r.Context(), projectsitemap.OperationTimeout)
	defer cancel()
	return h.dependencies.Registry.WithWriteGrant(r.PathValue("allocationID"), func(grant controlplane.AllocationGrant) error {
		if grant.RuntimeAgentID != identity.principalID || grant.RuntimeInstanceID != identity.instanceID || grant.AllocationID != r.PathValue("allocationID") {
			return controlplane.ErrAllocationNotFound
		}
		if grant.WriteFenced || grant.Lost {
			return errAllocationWriteFenced
		}
		if grant.ReadPolicy != controlplane.ReadCurrentRun || grant.SitemapPolicy == nil || grant.SitemapPolicy.Validate() != nil || !allowed(grant) {
			return errSitemapAccessDenied
		}
		if h.dependencies.Sitemap == nil {
			return errSitemapUnavailable
		}
		access, err := h.dependencies.Sitemap.AllocationAccess(ctx, grant.RunID, grant.Namespace, *grant.SitemapPolicy)
		if err != nil {
			return err
		}
		return operation(ctx, access)
	})
}

func (h *handler) mutateSitemap(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		h.handleError(w, errInvalidRequest)
		return
	}
	keys := r.Header.Values("Idempotency-Key")
	if len(keys) != 1 || !contracts.ValidIdempotencyKey(keys[0]) {
		h.handleError(w, errInvalidRequest)
		return
	}
	var input sitemapMutationRequest
	if err := decodeBoundedJSON(w, r, projectsitemap.MaxBatchBytes, &input); err != nil {
		h.handleError(w, err)
		return
	}
	if input.APIVersion != contracts.APIVersion {
		h.handleError(w, errInvalidRequest)
		return
	}
	command := input.Mutation
	tool := ""
	switch command.Operation {
	case "declare", "annotate":
		tool = contracts.SitemapUpsertTool
	case "remove":
		tool = contracts.SitemapRemoveTool
	case "restore":
		tool = contracts.SitemapRestoreTool
	case "replace-source":
		tool = contracts.SitemapGroupingTool
	default:
		h.handleError(w, errInvalidRequest)
		return
	}
	var result projectsitemap.Result
	err = h.withSitemapGrant(r, func(grant controlplane.AllocationGrant) bool {
		return grant.WritePolicy == controlplane.WriteInputsAndIntermediates && slices.Contains(grant.SitemapTools, tool)
	}, func(ctx context.Context, access projectsitemap.AllocationAccess) error {
		var mutationErr error
		result, mutationErr = h.dependencies.Sitemap.MutateAllocation(ctx, access, keys[0], command)
		return mutationErr
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, struct {
		APIVersion string                `json:"apiVersion"`
		Result     projectsitemap.Result `json:"result"`
	}{contracts.APIVersion, result})
}

func (h *handler) snapshotSitemap(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		h.handleError(w, errInvalidRequest)
		return
	}
	keys := r.Header.Values("Idempotency-Key")
	if len(keys) != 1 || !contracts.ValidIdempotencyKey(keys[0]) {
		h.handleError(w, errInvalidRequest)
		return
	}
	var input sitemapSnapshotRequest
	if err := decodeBoundedJSON(w, r, projectsitemap.MaxBatchBytes, &input); err != nil {
		h.handleError(w, err)
		return
	}
	if input.APIVersion != contracts.APIVersion {
		h.handleError(w, errInvalidRequest)
		return
	}
	selection := input.Selection
	if err := selection.Validate(); err != nil {
		h.handleError(w, err)
		return
	}
	var result projectsitemap.SnapshotResult
	err = h.withSitemapGrant(r, func(grant controlplane.AllocationGrant) bool {
		return grant.WritePolicy == controlplane.WriteInputsAndIntermediates && slices.Contains(grant.SitemapTools, contracts.SitemapSnapshotTool)
	}, func(ctx context.Context, access projectsitemap.AllocationAccess) error {
		var mutationErr error
		result, mutationErr = h.dependencies.Sitemap.SnapshotAllocation(ctx, access, keys[0], selection)
		return mutationErr
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, struct {
		APIVersion string                        `json:"apiVersion"`
		Result     projectsitemap.SnapshotResult `json:"result"`
	}{contracts.APIVersion, result})
}
