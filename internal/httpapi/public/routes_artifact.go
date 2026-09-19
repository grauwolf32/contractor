package public

import "net/http"

func (h *handler) registerArtifactRoutes(mux *http.ServeMux) {
	h.registerArchiveRoutes(mux, "/v1/artifacts/{namespace}/{name}", h.userArchiveStore)
	mux.HandleFunc("GET /v1/artifacts", h.listArtifacts)
	mux.HandleFunc("GET /v1/artifacts/{namespace}/{name}/metadata", h.getArtifactMetadata)
	mux.HandleFunc("GET /v1/artifacts/{namespace}/{name}/versions", h.listArtifactVersions)
	mux.HandleFunc("GET /v1/artifacts/{namespace}/{name}/lineage", h.listArtifactLineage)
	mux.HandleFunc("PUT /v1/artifacts/{namespace}/{name}", h.putArtifact)
	mux.HandleFunc("POST /v1/artifacts/{namespace}/{name}/git-import", h.importGitArtifact)
	mux.HandleFunc("GET /v1/artifacts/{namespace}/{name}", h.getArtifact)
	mux.HandleFunc("/v1/artifacts/{namespace}/{name}/metadata", h.methodNotAllowed)
	mux.HandleFunc("/v1/artifacts/{namespace}/{name}/versions", h.methodNotAllowed)
	mux.HandleFunc("/v1/artifacts/{namespace}/{name}/lineage", h.methodNotAllowed)
	mux.HandleFunc("/v1/artifacts/{namespace}/{name}", h.methodNotAllowed)
	mux.HandleFunc("/v1/artifacts", h.methodNotAllowed)
}
