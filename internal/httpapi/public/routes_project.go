package public

import "net/http"

func (c contextHandlers) registerProjectRoutes(mux *http.ServeMux) {
	c.artifacts.registerArchiveRoutes(mux, "/v1/projects/{projectId}/artifacts/{namespace}/{name}", c.artifacts.projectArchiveStore)
	mux.HandleFunc("POST /v1/projects", c.projects.createProject)
	mux.HandleFunc("GET /v1/projects", c.projects.listProjects)
	mux.HandleFunc("GET /v1/projects/{projectId}", c.projects.getProject)
	mux.HandleFunc("PATCH /v1/projects/{projectId}", c.projects.updateProject)
	mux.HandleFunc("DELETE /v1/projects/{projectId}", c.projects.deleteProject)
	mux.HandleFunc("GET /v1/projects/{projectId}/artifacts", c.artifacts.listProjectArtifacts)
	mux.HandleFunc("PUT /v1/projects/{projectId}/artifacts/{namespace}/{name}", c.artifacts.putProjectArtifact)
	mux.HandleFunc("POST /v1/projects/{projectId}/artifacts/{namespace}/{name}/git-import", c.artifacts.importGitArtifact)
	mux.HandleFunc("GET /v1/projects/{projectId}/artifacts/{namespace}/{name}", c.artifacts.getProjectArtifact)
	mux.HandleFunc("GET /v1/projects/{projectId}/artifacts/{namespace}/{name}/metadata", c.artifacts.getProjectArtifactMetadata)
	mux.HandleFunc("GET /v1/projects/{projectId}/artifacts/{namespace}/{name}/versions", c.artifacts.listProjectArtifactVersions)
	mux.HandleFunc("GET /v1/projects/{projectId}/artifacts/{namespace}/{name}/lineage", c.artifacts.listProjectArtifactLineage)
	mux.HandleFunc("POST /v1/projects/{projectId}/runs", c.runs.createProjectRun)
	mux.HandleFunc("GET /v1/projects/{projectId}/runs", c.runs.listProjectRuns)
	mux.HandleFunc("/v1/projects/{projectId}", methodNotAllowed)
	mux.HandleFunc("/v1/projects", methodNotAllowed)
	mux.HandleFunc("/v1/projects/{projectId}/artifacts/{namespace}/{name}/metadata", methodNotAllowed)
	mux.HandleFunc("/v1/projects/{projectId}/artifacts/{namespace}/{name}/versions", methodNotAllowed)
	mux.HandleFunc("/v1/projects/{projectId}/artifacts/{namespace}/{name}/lineage", methodNotAllowed)
	mux.HandleFunc("/v1/projects/{projectId}/artifacts/{namespace}/{name}", methodNotAllowed)
	mux.HandleFunc("/v1/projects/{projectId}/artifacts", methodNotAllowed)
	mux.HandleFunc("/v1/projects/{projectId}/runs", methodNotAllowed)
}
