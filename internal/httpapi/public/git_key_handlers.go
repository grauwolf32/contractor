package public

import (
	"context"
	"errors"
	"net/http"

	"github.com/grauwolf32/contractor/internal/credentials"
)

// maximumGitKeyBodyBytes bounds the JSON body that replaces the Git SSH key
// (LIM-GIT-21): room for the escaped form of the largest key input
// (LIM-GIT-3).
const maximumGitKeyBodyBytes int64 = 64 << 10

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

func (h *gitKeyHandlers) getGitKey(w http.ResponseWriter, r *http.Request) error {
	if !acceptGitKeyRequest(w, r) {
		return nil
	}
	m, err := h.GitKeys.Metadata(r.Context(), principalUserID(r.Context()))
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, m)
	return nil
}

func (h *gitKeyHandlers) replaceGitKey(w http.ResponseWriter, r *http.Request) error {
	if !acceptGitKeyRequest(w, r) {
		return nil
	}
	if err := requireJSONMediaType(r); err != nil {
		return err
	}
	var body struct {
		PrivateKey string `json:"privateKey"`
	}
	if err := decodeOpaqueJSONBounded(w, r, &body, maximumGitKeyBodyBytes); err != nil {
		if !errors.Is(err, errRequestTooLarge) {
			err = credentials.ErrGitKeyInvalid
		}
		return err
	}
	key := []byte(body.PrivateKey)
	body.PrivateKey = ""
	defer clear(key)
	m, err := h.GitKeys.Replace(r.Context(), principalUserID(r.Context()), key)
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, m)
	return nil
}

func (h *gitKeyHandlers) deleteGitKey(w http.ResponseWriter, r *http.Request) error {
	if !acceptGitKeyRequest(w, r) {
		return nil
	}
	if err := h.GitKeys.Delete(r.Context(), principalUserID(r.Context())); err != nil {
		return err
	}
	w.WriteHeader(http.StatusNoContent)
	return nil
}

// acceptGitKeyRequest keeps every Git key response out of caches and refuses
// a query string.
func acceptGitKeyRequest(w http.ResponseWriter, r *http.Request) bool {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return false
	}
	return true
}
