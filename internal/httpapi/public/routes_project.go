package public

import "net/http"

func (h *handler) registerProjectRoutes(mux *http.ServeMux) {
	mux.HandleFunc("POST /v1/projects", h.createProject)
	mux.HandleFunc("GET /v1/projects", h.listProjects)
	mux.HandleFunc("GET /v1/projects/{projectId}", h.getProject)
	mux.HandleFunc("PATCH /v1/projects/{projectId}", h.updateProject)
	mux.HandleFunc("DELETE /v1/projects/{projectId}", h.deleteProject)
	mux.HandleFunc("GET /v1/projects/{projectId}/artifacts", h.listProjectArtifacts)
	mux.HandleFunc("PUT /v1/projects/{projectId}/artifacts/{namespace}/{name}", h.putProjectArtifact)
	mux.HandleFunc("POST /v1/projects/{projectId}/artifacts/{namespace}/{name}/git-import", h.importGitArtifact)
	mux.HandleFunc("GET /v1/projects/{projectId}/artifacts/{namespace}/{name}", h.getProjectArtifact)
	mux.HandleFunc("GET /v1/projects/{projectId}/artifacts/{namespace}/{name}/metadata", h.getProjectArtifactMetadata)
	mux.HandleFunc("GET /v1/projects/{projectId}/artifacts/{namespace}/{name}/versions", h.listProjectArtifactVersions)
	mux.HandleFunc("GET /v1/projects/{projectId}/artifacts/{namespace}/{name}/lineage", h.listProjectArtifactLineage)
	mux.HandleFunc("POST /v1/projects/{projectId}/runs", h.createProjectRun)
	mux.HandleFunc("GET /v1/projects/{projectId}/runs", h.listProjectRuns)
	mux.HandleFunc("/v1/projects/{projectId}", h.methodNotAllowed)
	mux.HandleFunc("/v1/projects", h.methodNotAllowed)
	mux.HandleFunc("/v1/projects/{projectId}/artifacts/{namespace}/{name}/metadata", h.methodNotAllowed)
	mux.HandleFunc("/v1/projects/{projectId}/artifacts/{namespace}/{name}/versions", h.methodNotAllowed)
	mux.HandleFunc("/v1/projects/{projectId}/artifacts/{namespace}/{name}/lineage", h.methodNotAllowed)
	mux.HandleFunc("/v1/projects/{projectId}/artifacts/{namespace}/{name}", h.methodNotAllowed)
	mux.HandleFunc("/v1/projects/{projectId}/artifacts", h.methodNotAllowed)
	mux.HandleFunc("/v1/projects/{projectId}/runs", h.methodNotAllowed)
}
