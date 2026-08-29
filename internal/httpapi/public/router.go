package public

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"fmt"
	"net/http"
	"strings"
	"time"
)

type handler struct {
	dependencies Dependencies
	tokenDigest  [sha256.Size]byte
}

func NewHandler(dependencies Dependencies) (http.Handler, error) {
	if dependencies.Config == nil || dependencies.Runs == nil ||
		dependencies.Artifacts == nil || dependencies.Transactions == nil {
		return nil, fmt.Errorf("public API dependencies are incomplete")
	}
	if strings.TrimSpace(dependencies.UserID) == "" || dependencies.BearerToken.Reveal() == "" {
		return nil, fmt.Errorf("public API user ID and bearer token are required")
	}
	if dependencies.NewID == nil {
		dependencies.NewID = randomID
	}
	if dependencies.NewRequestID == nil {
		dependencies.NewRequestID = func() (string, error) { return randomID("request_") }
	}
	if dependencies.Now == nil {
		dependencies.Now = time.Now
	}
	current := &handler{
		dependencies: dependencies,
		tokenDigest:  sha256.Sum256([]byte(dependencies.BearerToken.Reveal())),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("PUT /v1/artifacts/{namespace}/{name}", current.putArtifact)
	mux.HandleFunc("GET /v1/artifacts/{namespace}/{name}", current.getArtifact)
	mux.HandleFunc("POST /v1/runs", current.createRun)
	mux.HandleFunc("POST /v1/runs/{runID}/cancel", current.cancelRun)
	mux.HandleFunc("GET /v1/runs/{runID}", current.getRun)
	mux.HandleFunc("GET /v1/runs/{runID}/outputs/{slot}", current.getRunOutput)
	mux.HandleFunc("/v1/artifacts/{namespace}/{name}", current.methodNotAllowed)
	mux.HandleFunc("/v1/runs/{runID}/outputs/{slot}", current.methodNotAllowed)
	mux.HandleFunc("/v1/runs/{runID}/cancel", current.methodNotAllowed)
	mux.HandleFunc("/v1/runs/{runID}", current.methodNotAllowed)
	mux.HandleFunc("/v1/runs", current.methodNotAllowed)
	mux.HandleFunc("/", current.notFound)

	return current.withRequestID(current.authenticate(mux)), nil
}

func (h *handler) authenticate(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		values := r.Header.Values("Authorization")
		if len(values) != 1 || !strings.HasPrefix(values[0], "Bearer ") {
			w.Header().Set("WWW-Authenticate", "Bearer")
			h.writeError(w, http.StatusUnauthorized, "unauthorized", "valid bearer authentication is required", false)
			return
		}
		candidate := strings.TrimPrefix(values[0], "Bearer ")
		candidateDigest := sha256.Sum256([]byte(candidate))
		if candidate == "" || subtle.ConstantTimeCompare(candidateDigest[:], h.tokenDigest[:]) != 1 {
			w.Header().Set("WWW-Authenticate", "Bearer")
			h.writeError(w, http.StatusUnauthorized, "unauthorized", "valid bearer authentication is required", false)
			return
		}
		next.ServeHTTP(w, r)
	})
}

type requestIDContextKey struct{}

func (h *handler) withRequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestID, err := h.dependencies.NewRequestID()
		if err != nil {
			h.writeError(w, http.StatusInternalServerError, "internal_error", "request could not be processed", true)
			return
		}
		w.Header().Set("X-Request-ID", requestID)
		next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), requestIDContextKey{}, requestID)))
	})
}

func randomID(prefix string) (string, error) {
	buffer := make([]byte, 16)
	if _, err := rand.Read(buffer); err != nil {
		return "", fmt.Errorf("generate identifier: %w", err)
	}
	return prefix + hex.EncodeToString(buffer), nil
}

func (h *handler) methodNotAllowed(w http.ResponseWriter, _ *http.Request) {
	h.writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "method is not allowed", false)
}

func (h *handler) notFound(w http.ResponseWriter, _ *http.Request) {
	h.writeError(w, http.StatusNotFound, "not_found", "resource was not found", false)
}
