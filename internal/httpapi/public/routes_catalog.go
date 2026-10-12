package public

import "net/http"

func (c contextHandlers) registerCatalogRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /v1/workflows", c.catalog.listWorkflows)
	mux.HandleFunc("GET /v1/workflows/{name}/versions/{version}", c.catalog.getWorkflow)
	mux.HandleFunc("GET /v1/configurations/{kind}", c.catalog.listConfigurations)
	mux.Handle("POST /v1/configurations/{kind}", withOperationsCapability(http.HandlerFunc(c.catalog.publishConfiguration)))
	mux.HandleFunc("GET /v1/configurations/{kind}/{name}/versions/{version}", c.catalog.getConfiguration)
	mux.HandleFunc("GET /v1/configurations/agent-templates/{name}/versions/{version}/instructions", c.catalog.getAgentInstructions)
	mux.HandleFunc("GET /v1/configurations/agent-templates/{name}/versions/{version}/workflow-bindings", c.catalog.listAgentTemplateWorkflowBindings)
	mux.HandleFunc("/v1/configurations/{kind}/{name}/versions/{version}", methodNotAllowed)
	mux.HandleFunc("/v1/configurations/agent-templates/{name}/versions/{version}/instructions", methodNotAllowed)
	mux.HandleFunc("/v1/configurations/agent-templates/{name}/versions/{version}/workflow-bindings", methodNotAllowed)
	mux.HandleFunc("/v1/configurations/{kind}", methodNotAllowed)
	mux.HandleFunc("/v1/workflows/{name}/versions/{version}", methodNotAllowed)
	mux.HandleFunc("/v1/workflows", methodNotAllowed)
}
