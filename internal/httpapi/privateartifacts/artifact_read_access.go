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
	_, scoped := query["scope"]
	_, operated := query["operation"]
	if !scoped {
		if operated || accessOnly {
			return errInvalidRequest
		}
		store, err := h.runStore(r, false)
		if err != nil {
			return err
		}
		return read(r.Context(), store, ref)
	}
	operation := query.Get("operation")
	if query.Get("scope") != "project" || !operated || !contracts.ValidName(operation) {
		return errInvalidRequest
	}
	_, identity, err := h.allocationGrant(r)
	if err != nil {
		return err
	}
	ctx, cancel := artifacttransfer.StorageContext(r.Context())
	defer cancel()
	return h.dependencies.Registry.WithWriteGrant(r.PathValue("allocationID"), func(grant controlplane.AllocationGrant) error {
		if grant.AllocationID != r.PathValue("allocationID") || grant.RuntimeAgentID != identity.principalID || grant.RuntimeInstanceID != identity.instanceID {
			return controlplane.ErrAllocationNotFound
		}
		if grant.WriteFenced || grant.Lost {
			return errAllocationWriteFenced
		}
		if grant.ReadPolicy != controlplane.ReadCurrentRun || !permitsProjectRead(grant.ProjectArtifactReads, ref, operation) || h.dependencies.RunProjects == nil || h.dependencies.Projects == nil {
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

// getArtifactAccess verifies only current authority. A previously downloaded
// immutable view remains usable after the source binding/revision is evicted;
// this check must not read Artifact metadata, bytes or lineage, or acquire a hold.
func (h *handler) getArtifactAccess(w http.ResponseWriter, r *http.Request) {
	err := h.withReadTarget(r, true, func(context.Context, artifacts.ScopedStore, contracts.ArtifactRef) error {
		w.Header().Set("Cache-Control", "no-store")
		w.WriteHeader(http.StatusNoContent)
		return nil
	})
	if err != nil {
		h.handleError(w, err)
	}
}
