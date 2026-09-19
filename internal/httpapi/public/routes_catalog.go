package public

import "net/http"

func (h *handler) registerCatalogRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /v1/workflows", h.listWorkflows)
	mux.HandleFunc("GET /v1/workflows/{name}/versions/{version}", h.getWorkflow)
	mux.HandleFunc("GET /v1/configurations/{kind}", h.listConfigurations)
	mux.HandleFunc("POST /v1/configurations/{kind}", h.publishConfiguration)
	mux.HandleFunc("GET /v1/configurations/{kind}/{name}/versions/{version}", h.getConfiguration)
	mux.HandleFunc("GET /v1/configurations/agent-templates/{name}/versions/{version}/instructions", h.getAgentInstructions)
	mux.HandleFunc("GET /v1/configurations/agent-templates/{name}/versions/{version}/workflow-bindings", h.listAgentTemplateWorkflowBindings)
	mux.HandleFunc("/v1/configurations/{kind}/{name}/versions/{version}", h.methodNotAllowed)
	mux.HandleFunc("/v1/configurations/agent-templates/{name}/versions/{version}/instructions", h.methodNotAllowed)
	mux.HandleFunc("/v1/configurations/agent-templates/{name}/versions/{version}/workflow-bindings", h.methodNotAllowed)
	mux.HandleFunc("/v1/configurations/{kind}", h.methodNotAllowed)
	mux.HandleFunc("/v1/workflows/{name}/versions/{version}", h.methodNotAllowed)
	mux.HandleFunc("/v1/workflows", h.methodNotAllowed)
}
