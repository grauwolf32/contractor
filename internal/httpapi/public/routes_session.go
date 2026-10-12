package public

import "net/http"

func (c contextHandlers) registerSessionRoutes(mux *http.ServeMux) {
	mux.HandleFunc("POST /v1/auth/login", c.session.login)
	mux.HandleFunc("GET /v1/auth/session", c.session.getSession)
	mux.HandleFunc("POST /v1/auth/logout", c.session.logout)
	mux.HandleFunc("GET /v1/events/ws", c.session.connectEventsWebSocket)
	mux.HandleFunc("/v1/auth/login", methodNotAllowed)
	mux.HandleFunc("/v1/auth/session", methodNotAllowed)
	mux.HandleFunc("/v1/auth/logout", methodNotAllowed)
	mux.HandleFunc("/v1/events/ws", methodNotAllowed)
}
