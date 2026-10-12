package public

import "net/http"

func (c contextHandlers) registerQueueRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /v1/queue", c.queue.listRunQueue)
	mux.HandleFunc("GET /v1/queue/control", c.queue.getOwnerQueueControl)
	mux.HandleFunc("PUT /v1/queue/control", c.queue.putOwnerQueueControl)
	mux.HandleFunc("/v1/queue", methodNotAllowed)
	mux.HandleFunc("/v1/queue/control", methodNotAllowed)
}
