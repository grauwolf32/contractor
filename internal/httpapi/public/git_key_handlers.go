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

// GitKeyDependencies serve the owner's Git SSH key.
type GitKeyDependencies struct {
	GitKeys GitKeySettings
}

type gitKeyHandlers struct {
	GitKeyDependencies
}

func (h *gitKeyHandlers) gitKeySettings(w http.ResponseWriter, r *http.Request) {
	if rejectHead(w, r) {
		return
	}
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		gitKeyError(w, err)
		return
	}
	owner := principalUserID(r.Context())
	switch r.Method {
	case http.MethodGet:
		m, err := h.GitKeys.Metadata(r.Context(), owner)
		if err != nil {
			gitKeyError(w, err)
			return
		}
		writeJSON(w, http.StatusOK, m)
	case http.MethodPut:
		mt, err := requestMediaType(r)
		if err != nil || mt != "application/json" {
			handleError(w, errInvalidRequest)
			return
		}
		var body struct {
			PrivateKey string `json:"privateKey"`
		}
		if err := decodeJSONBounded(w, r, &body, 64<<10); err != nil {
			if !errors.Is(err, errRequestTooLarge) {
				err = credentials.ErrGitKeyInvalid
			}
			handleError(w, err)
			return
		}
		key := []byte(body.PrivateKey)
		body.PrivateKey = ""
		defer clear(key)
		m, err := h.GitKeys.Replace(r.Context(), owner, key)
		if err != nil {
			gitKeyError(w, err)
			return
		}
		writeJSON(w, http.StatusOK, m)
	case http.MethodDelete:
		if err := h.GitKeys.Delete(r.Context(), owner); err != nil {
			gitKeyError(w, err)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}
}
