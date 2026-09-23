package public

import "net/http"

func (h *handler) registerSettingsRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /v1/settings/git-key", h.gitKeySettings)
	mux.HandleFunc("PUT /v1/settings/git-key", h.gitKeySettings)
	mux.HandleFunc("DELETE /v1/settings/git-key", h.gitKeySettings)
	mux.HandleFunc("/v1/settings/git-key", h.methodNotAllowed)
}
