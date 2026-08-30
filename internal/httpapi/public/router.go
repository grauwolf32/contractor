package public

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/requestid"
)

type handler struct {
	dependencies Dependencies
	tokenDigest  [sha256.Size]byte
}

func NewHandler(dependencies Dependencies) (http.Handler, error) {
	if dependencies.ConfigurationPublisher == nil {
		if publisher, ok := dependencies.Config.(ConfigurationPublisher); ok {
			dependencies.ConfigurationPublisher = publisher
		}
	}
	if dependencies.Config == nil || dependencies.ConfigurationPublisher == nil ||
		dependencies.Credentials == nil || dependencies.ManagedCredentials == nil || dependencies.Runs == nil ||
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
	mux.HandleFunc("GET /v1/workflows", current.listWorkflows)
	mux.HandleFunc("GET /v1/workflows/{name}/versions/{version}", current.getWorkflow)
	mux.HandleFunc("GET /v1/configurations/{kind}", current.listConfigurations)
	mux.HandleFunc("POST /v1/configurations/{kind}", current.publishConfiguration)
	mux.HandleFunc("GET /v1/configurations/{kind}/{name}/versions/{version}", current.getConfiguration)
	mux.HandleFunc("GET /v1/operations/credentials", current.listCredentials)
	mux.HandleFunc("POST /v1/operations/credentials", current.createCredential)
	mux.HandleFunc("GET /v1/operations/credentials/{credentialId}", current.getCredential)
	mux.HandleFunc("DELETE /v1/operations/credentials/{credentialId}", current.deleteCredential)
	mux.HandleFunc("GET /v1/artifacts", current.listArtifacts)
	mux.HandleFunc("GET /v1/artifacts/{namespace}/{name}/metadata", current.getArtifactMetadata)
	mux.HandleFunc("GET /v1/artifacts/{namespace}/{name}/versions", current.listArtifactVersions)
	mux.HandleFunc("GET /v1/artifacts/{namespace}/{name}/lineage", current.listArtifactLineage)
	mux.HandleFunc("PUT /v1/artifacts/{namespace}/{name}", current.putArtifact)
	mux.HandleFunc("GET /v1/artifacts/{namespace}/{name}", current.getArtifact)
	mux.HandleFunc("POST /v1/runs", current.createRun)
	mux.HandleFunc("GET /v1/runs", current.listRuns)
	mux.HandleFunc("POST /v1/runs/{runID}/cancel", current.cancelRun)
	mux.HandleFunc("GET /v1/runs/{runID}", current.getRun)
	mux.HandleFunc("GET /v1/runs/{runID}/outputs/{slot}", current.getRunOutput)
	mux.HandleFunc("GET /v1/runs/{runID}/artifacts", current.listRunArtifacts)
	mux.HandleFunc("GET /v1/runs/{runID}/artifacts/{namespace}/{name}", current.getRunArtifact)
	mux.HandleFunc("GET /v1/runs/{runID}/artifacts/{namespace}/{name}/metadata", current.getRunArtifactMetadata)
	mux.HandleFunc("GET /v1/runs/{runID}/artifacts/{namespace}/{name}/versions", current.listRunArtifactVersions)
	mux.HandleFunc("GET /v1/runs/{runID}/artifacts/{namespace}/{name}/lineage", current.listRunArtifactLineage)
	mux.HandleFunc("/v1/runs/{runID}/artifacts/{namespace}/{name}/metadata", current.methodNotAllowed)
	mux.HandleFunc("/v1/runs/{runID}/artifacts/{namespace}/{name}/versions", current.methodNotAllowed)
	mux.HandleFunc("/v1/runs/{runID}/artifacts/{namespace}/{name}/lineage", current.methodNotAllowed)
	mux.HandleFunc("/v1/runs/{runID}/artifacts/{namespace}/{name}", current.methodNotAllowed)
	mux.HandleFunc("/v1/runs/{runID}/artifacts", current.methodNotAllowed)
	mux.HandleFunc("/v1/artifacts/{namespace}/{name}/metadata", current.methodNotAllowed)
	mux.HandleFunc("/v1/artifacts/{namespace}/{name}/versions", current.methodNotAllowed)
	mux.HandleFunc("/v1/artifacts/{namespace}/{name}/lineage", current.methodNotAllowed)
	mux.HandleFunc("/v1/artifacts/{namespace}/{name}", current.methodNotAllowed)
	mux.HandleFunc("/v1/artifacts", current.methodNotAllowed)
	mux.HandleFunc("/v1/runs/{runID}/outputs/{slot}", current.methodNotAllowed)
	mux.HandleFunc("/v1/runs/{runID}/cancel", current.methodNotAllowed)
	mux.HandleFunc("/v1/runs/{runID}", current.methodNotAllowed)
	mux.HandleFunc("/v1/runs", current.methodNotAllowed)
	mux.HandleFunc("/v1/configurations/{kind}/{name}/versions/{version}", current.methodNotAllowed)
	mux.HandleFunc("/v1/configurations/{kind}", current.methodNotAllowed)
	mux.HandleFunc("/v1/operations/credentials/{credentialId}", current.methodNotAllowed)
	mux.HandleFunc("/v1/operations/credentials", current.methodNotAllowed)
	mux.HandleFunc("/v1/workflows/{name}/versions/{version}", current.methodNotAllowed)
	mux.HandleFunc("/v1/workflows", current.methodNotAllowed)
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

func (h *handler) withRequestID(next http.Handler) http.Handler {
	return requestid.Middleware(next, requestid.Options{
		Generator: h.dependencies.NewRequestID,
		Logger:    h.dependencies.Logger, Boundary: "public-api",
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

func (h *handler) rejectHead(w http.ResponseWriter, r *http.Request) bool {
	if r.Method != http.MethodHead {
		return false
	}
	h.methodNotAllowed(w, r)
	return true
}

func (h *handler) notFound(w http.ResponseWriter, _ *http.Request) {
	h.writeError(w, http.StatusNotFound, "not_found", "resource was not found", false)
}
