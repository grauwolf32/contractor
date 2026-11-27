package privateartifacts

import (
	"context"
	"errors"
	"net/http"
	"slices"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// withReadTarget keeps Project read authority stable through the entire read,
// using the same gate as allocation fence/release. Ordinary Run reads preserve
// their existing grant and namespace policy.
func (h *handler) withReadTarget(r *http.Request, accessOnly bool, read func(context.Context, artifacts.ScopedStore, contracts.ArtifactRef) error) error {
	allowed := []string{"scope", "operation"}
	if !accessOnly {
		allowed = append(allowed, "revision")
	}
	query, err := exactQuery(r.URL.RawQuery, allowed...)
	if err != nil {
		return err
	}
	ref := contracts.ArtifactRef{Namespace: r.PathValue("namespace"), Name: r.PathValue("name")}
	if contracts.ValidateArtifactName(ref.Namespace) != nil || contracts.ValidateArtifactName(ref.Name) != nil {
		return errInvalidRequest
	}
	if values, present := query["revision"]; present {
		if !contracts.ValidResourceID(values[0]) {
			return errInvalidRequest
		}
		ref.Revision = &values[0]
	}
	var scope, operation *string
	if values, present := query["scope"]; present {
		scope = &values[0]
	}
	if values, present := query["operation"]; present {
		operation = &values[0]
	}
	_, identity, err := h.allocationGrant(r)
	if err != nil {
		return err
	}
	return h.withArtifactTarget(r.Context(), r.PathValue("allocationID"), identity, ref, scope, operation, accessOnly, read)
}

func (h *handler) withArtifactTarget(ctx context.Context, allocationID string, identity authenticatedRuntime, ref contracts.ArtifactRef, scope, operation *string, accessOnly bool, read func(context.Context, artifacts.ScopedStore, contracts.ArtifactRef) error) error {
	if ref.Validate() != nil {
		return errInvalidRequest
	}
	if scope == nil {
		if operation != nil || accessOnly {
			return errInvalidRequest
		}
		grant, err := h.dependencies.Registry.GetGrant(allocationID)
		if err != nil {
			return err
		}
		store, err := h.runStoreForGrant(allocationID, grant, identity, false)
		if err != nil {
			return err
		}
		return read(ctx, store, ref)
	}
	if *scope != "project" || operation == nil || !contracts.ValidName(*operation) || accessOnly && ref.Revision != nil {
		return errInvalidRequest
	}
	ctx, cancel := artifacttransfer.StorageContext(ctx)
	defer cancel()
	return h.dependencies.Registry.WithWriteGrant(allocationID, func(grant controlplane.AllocationGrant) error {
		if grant.AllocationID != allocationID || grant.RuntimeAgentID != identity.principalID || grant.RuntimeInstanceID != identity.instanceID {
			return controlplane.ErrAllocationNotFound
		}
		if grant.WriteFenced || grant.Lost {
			return errAllocationWriteFenced
		}
		if grant.ReadPolicy != controlplane.ReadCurrentRun || !permitsProjectRead(grant.ProjectArtifactReads, ref, *operation) || h.dependencies.RunProjects == nil || h.dependencies.Projects == nil {
			return errArtifactAccessDenied
		}
		owning, err := h.dependencies.RunProjects.ProjectContext(ctx, grant.RunID)
		if errors.Is(err, runstore.ErrNotFound) {
			return errArtifactAccessDenied
		}
		if err != nil {
			return err
		}
		project, err := h.dependencies.Projects.Get(ctx, owning.OwnerID, owning.ProjectID)
		if errors.Is(err, projectstore.ErrNotFound) {
			return errArtifactAccessDenied
		}
		if err != nil {
			return err
		}
		if project.Lifecycle != projectstore.LifecycleActive {
			return projectstore.ErrDeleting
		}
		store, err := h.dependencies.Artifacts.Project(owning.ProjectID)
		if err != nil {
			return err
		}
		return read(ctx, store, ref)
	})
}

func permitsProjectRead(targets []controlplane.ProjectArtifactReadTarget, ref contracts.ArtifactRef, operation string) bool {
	// Targets derive from the bounded pinned toolset configuration, never from
	// a request. Reuse its collection bounds and reject malformed grants closed.
	if len(targets) > control.MaxRuntimeCapabilityRefs {
		return false
	}
	for _, target := range targets {
		if len(target.Operations) > control.MaxToolsetCapabilityTools {
			return false
		}
		if target.Namespace == ref.Namespace && target.Name == ref.Name && slices.Contains(target.Operations, operation) {
			return true
		}
	}
	return false
}
