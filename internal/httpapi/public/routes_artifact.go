package public

import "net/http"

func (c contextHandlers) registerArtifactRoutes(mux *http.ServeMux) {
	c.artifacts.registerArchiveRoutes(mux, "/v1/artifacts/{namespace}/{name}", c.artifacts.userArchiveStore)
	mux.HandleFunc("GET /v1/artifacts", c.artifacts.listArtifacts)
	mux.HandleFunc("GET /v1/artifacts/{namespace}/{name}/metadata", c.artifacts.getArtifactMetadata)
	mux.HandleFunc("GET /v1/artifacts/{namespace}/{name}/versions", c.artifacts.listArtifactVersions)
	mux.HandleFunc("GET /v1/artifacts/{namespace}/{name}/lineage", c.artifacts.listArtifactLineage)
	mux.HandleFunc("PUT /v1/artifacts/{namespace}/{name}", c.artifacts.putArtifact)
	mux.HandleFunc("POST /v1/artifacts/{namespace}/{name}/git-import", c.artifacts.importGitArtifact)
	mux.HandleFunc("GET /v1/artifacts/{namespace}/{name}", c.artifacts.getArtifact)
	mux.HandleFunc("/v1/artifacts/{namespace}/{name}/metadata", methodNotAllowed)
	mux.HandleFunc("/v1/artifacts/{namespace}/{name}/versions", methodNotAllowed)
	mux.HandleFunc("/v1/artifacts/{namespace}/{name}/lineage", methodNotAllowed)
	mux.HandleFunc("/v1/artifacts/{namespace}/{name}", methodNotAllowed)
	mux.HandleFunc("/v1/artifacts", methodNotAllowed)
}
