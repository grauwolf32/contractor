package public

import "net/http"

func (c contextHandlers) registerSettingsRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /v1/settings/git-key", c.gitKeys.gitKeySettings)
	mux.HandleFunc("PUT /v1/settings/git-key", c.gitKeys.gitKeySettings)
	mux.HandleFunc("DELETE /v1/settings/git-key", c.gitKeys.gitKeySettings)
	mux.HandleFunc("/v1/settings/git-key", methodNotAllowed)
}
