package public

import (
	"net/http"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

func (h *handler) listProjectArtifacts(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	store, projectID, err := h.ownedProjectArtifactStore(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	h.listArtifactBindings(w, r, store, "project-artifacts:"+projectID, false)
}

func (h *handler) getProjectArtifact(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodHead {
		h.methodNotAllowed(w, r)
		return
	}
	store, _, err := h.ownedProjectArtifactStore(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	ref, err := artifactRouteRef(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	result, err := store.Read(r.Context(), ref)
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeArtifactBytes(w, result)
}

func (h *handler) putProjectArtifact(w http.ResponseWriter, r *http.Request) {
	store, _, err := h.ownedActiveProjectArtifactStore(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	if err := validateArtifactRouteNames(r); err != nil {
		h.handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	expectedRevision, err := artifactWritePrecondition(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	payload, err := readArtifactBody(w, r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	result, err := store.Write(r.Context(), contracts.ArtifactRef{
		Namespace: r.PathValue("namespace"), Name: r.PathValue("name"),
	}, artifacts.Payload{MediaType: mediaType, Data: payload}, expectedRevision)
	if err != nil {
		h.handleError(w, err)
		return
	}
	status := http.StatusCreated
	if expectedRevision != nil {
		status = http.StatusOK
	}
	w.Header().Set("ETag", quotedETag(result.Ref.Revision))
	writeJSON(w, status, artifactWriteResponse{
		Artifact: result.Ref, MediaType: result.MediaType, Size: result.Size,
	})
}

func (h *handler) ownedActiveProjectArtifactStore(
	r *http.Request,
) (artifacts.ScopedStore, string, error) {
	projectID := r.PathValue("projectId")
	project, err := h.dependencies.Projects.Get(
		r.Context(), principalUserID(r.Context()), projectID,
	)
	if err != nil {
		return artifacts.ScopedStore{}, "", err
	}
	if project.Lifecycle == projectstore.LifecycleDeleting {
		return artifacts.ScopedStore{}, "", projectstore.ErrDeleting
	}
	store, err := h.dependencies.Artifacts.Project(project.ProjectID)
	return store, project.ProjectID, err
}

func (h *handler) getProjectArtifactMetadata(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	store, _, err := h.ownedProjectArtifactStore(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	h.getArtifactMetadataFromStore(w, r, store)
}

func (h *handler) listProjectArtifactVersions(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	store, projectID, err := h.ownedProjectArtifactStore(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	h.listArtifactVersionsFromStore(w, r, store, "project-artifact-versions:"+projectID)
}

func (h *handler) listProjectArtifactLineage(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	store, projectID, err := h.ownedProjectArtifactStore(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	h.listArtifactLineageFromStore(w, r, store, "project-artifact-lineage:"+projectID)
}

func (h *handler) ownedProjectArtifactStore(
	r *http.Request,
) (artifacts.ScopedStore, string, error) {
	projectID := r.PathValue("projectId")
	project, err := h.dependencies.Projects.Get(
		r.Context(), principalUserID(r.Context()), projectID,
	)
	if err != nil {
		return artifacts.ScopedStore{}, "", err
	}
	store, err := h.dependencies.Artifacts.Project(project.ProjectID)
	return store, project.ProjectID, err
}
