package public

import (
	"context"
	"errors"
	"net/http"

	"github.com/grauwolf32/contractor/internal/credentials"
)

type GitKeySettings interface {
	Metadata(context.Context, string) (credentials.GitKeyMetadata, error)
	Replace(context.Context, string, []byte) (credentials.GitKeyMetadata, error)
	Delete(context.Context, string) error
}

func (h *handler) gitKeySettings(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.gitKeyError(w, err)
		return
	}
	if h.dependencies.GitKeys == nil {
		h.gitKeyError(w, credentials.ErrKeyUnavailable)
		return
	}
	owner := principalUserID(r.Context())
	switch r.Method {
	case http.MethodGet:
		m, err := h.dependencies.GitKeys.Metadata(r.Context(), owner)
		if err != nil {
			h.gitKeyError(w, err)
			return
		}
		writeJSON(w, http.StatusOK, m)
	case http.MethodPut:
		mt, err := requestMediaType(r)
		if err != nil || mt != "application/json" {
			h.handleError(w, errInvalidRequest)
			return
		}
		var body struct {
			PrivateKey string `json:"privateKey"`
		}
		if err := decodeJSONBounded(w, r, &body, 64<<10); err != nil {
			h.handleError(w, credentials.ErrGitKeyInvalid)
			return
		}
		key := []byte(body.PrivateKey)
		body.PrivateKey = ""
		defer clear(key)
		m, err := h.dependencies.GitKeys.Replace(r.Context(), owner, key)
		if err != nil {
			h.gitKeyError(w, err)
			return
		}
		writeJSON(w, http.StatusOK, m)
	case http.MethodDelete:
		if err := h.dependencies.GitKeys.Delete(r.Context(), owner); err != nil {
			h.gitKeyError(w, err)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}
}

func (h *handler) gitKeyError(w http.ResponseWriter, err error) {
	if errors.Is(err, credentials.ErrKeyUnavailable) {
		h.writeError(w, http.StatusServiceUnavailable, "git_key_unavailable", "Git key encryption is not configured", false)
		return
	}
	h.handleError(w, err)
}
