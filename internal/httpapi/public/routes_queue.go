package public

import "net/http"

func (h *handler) registerQueueRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /v1/queue", h.listRunQueue)
	mux.HandleFunc("GET /v1/queue/control", h.getOwnerQueueControl)
	mux.HandleFunc("PUT /v1/queue/control", h.putOwnerQueueControl)
	mux.HandleFunc("/v1/queue", h.methodNotAllowed)
	mux.HandleFunc("/v1/queue/control", h.methodNotAllowed)
}
