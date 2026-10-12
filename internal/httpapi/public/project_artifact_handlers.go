package public

import (
	"net/http"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

func (h *artifactHandlers) listProjectArtifacts(w http.ResponseWriter, r *http.Request) {
	if rejectHead(w, r) {
		return
	}
	store, projectID, err := h.ownedProjectArtifactStore(r)
	if err != nil {
		handleError(w, err)
		return
	}
	h.listArtifactBindings(
		w, r, store, "project-artifacts:"+projectID, false, "",
		artifactpolicy.AuditManagedProjectNamespacePrefix,
	)
}

func (h *artifactHandlers) getProjectArtifact(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodHead {
		methodNotAllowed(w, r)
		return
	}
	store, _, err := h.ownedProjectArtifactStore(r)
	if err != nil {
		handleError(w, err)
		return
	}
	if h.rejectAuditManagedProjectArtifact(w, r) {
		return
	}
	ref, err := artifactRouteRef(r)
	if err != nil {
		handleError(w, err)
		return
	}
	h.writeArtifactRead(w, r, store, ref)
}

func (h *artifactHandlers) putProjectArtifact(w http.ResponseWriter, r *http.Request) {
	store, _, err := h.ownedActiveProjectArtifactStore(r)
	if err != nil {
		handleError(w, err)
		return
	}
	if h.rejectAuditManagedProjectArtifact(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	if err := validateArtifactRouteNames(r); err != nil {
		handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil {
		handleError(w, err)
		return
	}
	expectedRevision, err := artifactWritePrecondition(r)
	if err != nil {
		handleError(w, err)
		return
	}
	h.writeArtifactUpload(w, r, store, mediaType, expectedRevision)
}

func (h *artifactHandlers) ownedActiveProjectArtifactStore(
	r *http.Request,
) (artifacts.ScopedStore, string, error) {
	projectID := r.PathValue("projectId")
	project, err := h.Projects.Get(
		r.Context(), principalUserID(r.Context()), projectID,
	)
	if err != nil {
		return artifacts.ScopedStore{}, "", err
	}
	if project.Lifecycle == projectstore.LifecycleDeleting {
		return artifacts.ScopedStore{}, "", projectstore.ErrDeleting
	}
	store, err := h.Artifacts.Project(project.ProjectID)
	return store, project.ProjectID, err
}

func (h *artifactHandlers) getProjectArtifactMetadata(w http.ResponseWriter, r *http.Request) {
	if rejectHead(w, r) {
		return
	}
	store, _, err := h.ownedProjectArtifactStore(r)
	if err != nil {
		handleError(w, err)
		return
	}
	if h.rejectAuditManagedProjectArtifact(w, r) {
		return
	}
	h.getArtifactMetadataFromStore(w, r, store)
}

func (h *artifactHandlers) listProjectArtifactVersions(w http.ResponseWriter, r *http.Request) {
	if rejectHead(w, r) {
		return
	}
	store, projectID, err := h.ownedProjectArtifactStore(r)
	if err != nil {
		handleError(w, err)
		return
	}
	if h.rejectAuditManagedProjectArtifact(w, r) {
		return
	}
	h.listArtifactVersionsFromStore(w, r, store, "project-artifact-versions:"+projectID)
}

func (h *artifactHandlers) listProjectArtifactLineage(w http.ResponseWriter, r *http.Request) {
	if rejectHead(w, r) {
		return
	}
	store, projectID, err := h.ownedProjectArtifactStore(r)
	if err != nil {
		handleError(w, err)
		return
	}
	if h.rejectAuditManagedProjectArtifact(w, r) {
		return
	}
	h.listArtifactLineageFromStore(w, r, store, "project-artifact-lineage:"+projectID)
}

func (h *artifactHandlers) ownedProjectArtifactStore(
	r *http.Request,
) (artifacts.ScopedStore, string, error) {
	projectID := r.PathValue("projectId")
	project, err := h.Projects.Get(
		r.Context(), principalUserID(r.Context()), projectID,
	)
	if err != nil {
		return artifacts.ScopedStore{}, "", err
	}
	store, err := h.Artifacts.Project(project.ProjectID)
	return store, project.ProjectID, err
}

func (h *artifactHandlers) rejectAuditManagedProjectArtifact(w http.ResponseWriter, r *http.Request) bool {
	if !artifactpolicy.IsAuditManagedProjectNamespace(r.PathValue("namespace")) {
		return false
	}
	// The binding may be staged but not yet accepted. Exposing even its
	// existence would violate the Audit receipt boundary.
	handleError(w, artifacts.ErrArtifactNotFound)
	return true
}
