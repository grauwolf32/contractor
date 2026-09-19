package public

import "net/http"

func (h *handler) registerSessionRoutes(mux *http.ServeMux) {
	mux.HandleFunc("POST /v1/auth/login", h.login)
	mux.HandleFunc("GET /v1/auth/session", h.getSession)
	mux.HandleFunc("POST /v1/auth/logout", h.logout)
	mux.HandleFunc("GET /v1/events/ws", h.connectEventsWebSocket)
	mux.HandleFunc("/v1/auth/login", h.methodNotAllowed)
	mux.HandleFunc("/v1/auth/session", h.methodNotAllowed)
	mux.HandleFunc("/v1/auth/logout", h.methodNotAllowed)
	mux.HandleFunc("/v1/events/ws", h.methodNotAllowed)
}
