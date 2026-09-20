package public

import "net/http"

func (h *handler) registerEvalRoutes(mux *http.ServeMux) {
	routes := []struct {
		method, path string
		handler      http.HandlerFunc
	}{
		{"GET", "/v1/eval-capabilities", h.evalCapabilities},
		{"GET", "/v1/projects/{projectId}/eval-datasets", h.listEvalDatasets},
		{"POST", "/v1/projects/{projectId}/eval-datasets", h.importEvalDataset},
		{"GET", "/v1/projects/{projectId}/eval-datasets/{datasetId}/revisions/{revision}/cases", h.listEvalCases},
		{"POST", "/v1/projects/{projectId}/eval-experiments", h.createEvalExperiment},
		{"GET", "/v1/eval-experiments", h.listEvalExperiments},
		{"GET", "/v1/eval-experiments/{id}", h.getEvalExperiment},
		{"PATCH", "/v1/eval-experiments/{id}", h.updateEvalDraft},
		{"DELETE", "/v1/eval-experiments/{id}", h.deleteEvalExperiment},
		{"POST", "/v1/eval-experiments/{id}/commands", h.commandEvalExperiment},
		{"GET", "/v1/eval-experiments/{id}/commands/{commandId}", h.getEvalCommand},
		{"GET", "/v1/eval-experiments/{id}/members", h.listEvalMembers},
		{"POST", "/v1/eval-experiments/{id}/members/{memberId}/submissions", h.submitEvalMember},
	}
	paths := map[string]bool{}
	for _, route := range routes {
		mux.HandleFunc(route.method+" "+route.path, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Cache-Control", "no-store")
			if h.rejectHead(w, r) {
				return
			}
			if h.dependencies.Evals == nil {
				h.writeError(w, http.StatusServiceUnavailable, "eval_unavailable", "Evaluation service is unavailable", true)
				return
			}
			route.handler(w, r)
		})
		paths[route.path] = true
	}
	for path := range paths {
		mux.HandleFunc(path, h.methodNotAllowed)
	}
}
