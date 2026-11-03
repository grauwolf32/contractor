package public

import (
	"net/http"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

func (h *artifactHandlers) listProjectArtifacts(w http.ResponseWriter, r *http.Request) error {
	store, project, err := h.ownedProjectArtifactStore(r)
	if err != nil {
		return err
	}
	return h.listArtifactBindings(
		w, r, store, artifactListOptions{
			uploadsAllowed:         project.Lifecycle != projectstore.LifecycleDeleting,
			cursorPrefix:           "project-artifacts:" + project.ProjectID,
			excludeNamespacePrefix: artifactpolicy.AuditManagedProjectNamespacePrefix,
		},
	)
}

func (h *artifactHandlers) getProjectArtifact(w http.ResponseWriter, r *http.Request) error {
	store, _, err := h.ownedProjectArtifactStore(r)
	if err != nil {
		return err
	}
	if h.rejectAuditManagedProjectArtifact(w, r) {
		return nil
	}
	ref, err := artifactRouteRef(r)
	if err != nil {
		return err
	}
	return h.writeArtifactRead(w, r, store, ref)
}

// putProjectArtifact checks the request metadata before it reads the Project,
// so a malformed request answers the same whichever Project it names; then
// Project ownership and lifecycle, then the Audit-managed namespace, and only
// then takes a transfer slot for the body (API-R112).
func (h *artifactHandlers) putProjectArtifact(w http.ResponseWriter, r *http.Request) error {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	if err := validateArtifactRouteNames(r); err != nil {
		return err
	}
	mediaType, err := requestMediaType(r)
	if err != nil {
		return err
	}
	expectedRevision, err := artifactWritePrecondition(r)
	if err != nil {
		return err
	}
	store, _, err := h.ownedActiveProjectArtifactStore(r)
	if err != nil {
		return err
	}
	if h.rejectAuditManagedProjectArtifact(w, r) {
		return nil
	}
	return h.writeArtifactUpload(w, r, store, mediaType, expectedRevision)
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

func (h *artifactHandlers) getProjectArtifactMetadata(w http.ResponseWriter, r *http.Request) error {
	store, project, err := h.ownedProjectArtifactStore(r)
	if err != nil {
		return err
	}
	if h.rejectAuditManagedProjectArtifact(w, r) {
		return nil
	}
	return h.getArtifactMetadataFromStore(w, r, store, project.Lifecycle != projectstore.LifecycleDeleting)
}

func (h *artifactHandlers) listProjectArtifactVersions(w http.ResponseWriter, r *http.Request) error {
	store, project, err := h.ownedProjectArtifactStore(r)
	if err != nil {
		return err
	}
	if h.rejectAuditManagedProjectArtifact(w, r) {
		return nil
	}
	return h.listArtifactVersionsFromStore(w, r, store, project.Lifecycle != projectstore.LifecycleDeleting, "project-artifact-versions:"+project.ProjectID)
}

func (h *artifactHandlers) listProjectArtifactLineage(w http.ResponseWriter, r *http.Request) error {
	store, project, err := h.ownedProjectArtifactStore(r)
	if err != nil {
		return err
	}
	if h.rejectAuditManagedProjectArtifact(w, r) {
		return nil
	}
	return h.listArtifactLineageFromStore(w, r, store, "project-artifact-lineage:"+project.ProjectID)
}

func (h *artifactHandlers) ownedProjectArtifactStore(
	r *http.Request,
) (artifacts.ScopedStore, projectstore.Project, error) {
	projectID := r.PathValue("projectId")
	project, err := h.Projects.Get(
		r.Context(), principalUserID(r.Context()), projectID,
	)
	if err != nil {
		return artifacts.ScopedStore{}, projectstore.Project{}, err
	}
	store, err := h.Artifacts.Project(project.ProjectID)
	return store, project, err
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
