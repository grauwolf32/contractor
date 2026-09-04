package public

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/requestid"
)

type handler struct {
	dependencies Dependencies
	tokenDigest  [sha256.Size]byte
	cookieName   string
	secureCookie bool
}

func NewHandler(dependencies Dependencies) (http.Handler, error) {
	if dependencies.ConfigurationPublisher == nil {
		if publisher, ok := dependencies.Config.(ConfigurationPublisher); ok {
			dependencies.ConfigurationPublisher = publisher
		}
	}
	if dependencies.Config == nil || dependencies.ConfigurationPublisher == nil ||
		dependencies.Credentials == nil || dependencies.ManagedCredentials == nil || dependencies.Runs == nil ||
		dependencies.RuntimeConfigs == nil || dependencies.RuntimeCredentials == nil ||
		dependencies.RuntimeAgentPrincipals == nil ||
		dependencies.Projects == nil ||
		dependencies.Artifacts == nil || dependencies.Transactions == nil || dependencies.Operations == nil ||
		dependencies.OperationsInvalidator == nil || dependencies.Events == nil ||
		dependencies.Authentication == nil || len(dependencies.BrowserOrigins.Values()) == 0 {
		return nil, fmt.Errorf("public API dependencies are incomplete")
	}
	bearerToken := dependencies.BearerToken.Reveal()
	if bearerToken == "" || len(bearerToken) > 4096 {
		return nil, fmt.Errorf("public API bearer token must contain 1 through 4096 bytes")
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
	if dependencies.Logger == nil {
		dependencies.Logger = slog.Default()
	}
	tokenDigest := sha256.Sum256([]byte(bearerToken))
	bearerToken = ""
	dependencies.BearerToken = contracts.NewSecretString("")
	current := &handler{
		dependencies: dependencies,
		tokenDigest:  tokenDigest,
		cookieName:   auth.SecureCookieName,
		secureCookie: true,
	}
	if dependencies.InsecureLoopbackCookie {
		current.cookieName = auth.LoopbackCookieName
		current.secureCookie = false
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/auth/login", current.login)
	mux.HandleFunc("GET /v1/auth/session", current.getSession)
	mux.HandleFunc("POST /v1/auth/logout", current.logout)
	mux.HandleFunc("GET /v1/events/ws", current.connectEventsWebSocket)
	mux.HandleFunc("GET /v1/workflows", current.listWorkflows)
	mux.HandleFunc("GET /v1/workflows/{name}/versions/{version}", current.getWorkflow)
	mux.HandleFunc("POST /v1/projects", current.createProject)
	mux.HandleFunc("GET /v1/projects", current.listProjects)
	mux.HandleFunc("GET /v1/projects/{projectId}", current.getProject)
	mux.HandleFunc("PATCH /v1/projects/{projectId}", current.updateProject)
	mux.HandleFunc("GET /v1/projects/{projectId}/artifacts", current.listProjectArtifacts)
	mux.HandleFunc("PUT /v1/projects/{projectId}/artifacts/{namespace}/{name}", current.putProjectArtifact)
	mux.HandleFunc("GET /v1/projects/{projectId}/artifacts/{namespace}/{name}", current.getProjectArtifact)
	mux.HandleFunc("GET /v1/projects/{projectId}/artifacts/{namespace}/{name}/metadata", current.getProjectArtifactMetadata)
	mux.HandleFunc("GET /v1/projects/{projectId}/artifacts/{namespace}/{name}/versions", current.listProjectArtifactVersions)
	mux.HandleFunc("GET /v1/projects/{projectId}/artifacts/{namespace}/{name}/lineage", current.listProjectArtifactLineage)
	mux.HandleFunc("POST /v1/projects/{projectId}/runs", current.createProjectRun)
	mux.HandleFunc("GET /v1/projects/{projectId}/runs", current.listProjectRuns)
	mux.HandleFunc("GET /v1/configurations/{kind}", current.listConfigurations)
	mux.HandleFunc("POST /v1/configurations/{kind}", current.publishConfiguration)
	mux.HandleFunc("GET /v1/configurations/{kind}/{name}/versions/{version}", current.getConfiguration)
	mux.HandleFunc("GET /v1/operations/credentials", current.listCredentials)
	mux.HandleFunc("POST /v1/operations/credentials", current.createCredential)
	mux.HandleFunc("GET /v1/operations/credentials/{credentialId}", current.getCredential)
	mux.HandleFunc("DELETE /v1/operations/credentials/{credentialId}", current.deleteCredential)
	mux.HandleFunc("GET /v1/operations/runtime-configs", current.listRuntimeConfigs)
	mux.HandleFunc("POST /v1/operations/runtime-configs", current.publishRuntimeConfig)
	mux.HandleFunc("GET /v1/operations/runtime-configs/{name}/versions/{version}", current.getRuntimeConfig)
	mux.HandleFunc("GET /v1/operations/runtime-labels", current.listRuntimeLabels)
	mux.HandleFunc("GET /v1/operations/runtime-labels/{label}", current.getRuntimeLabel)
	mux.HandleFunc("PUT /v1/operations/runtime-labels/{label}", current.putRuntimeLabel)
	mux.HandleFunc("DELETE /v1/operations/runtime-labels/{label}", current.deleteRuntimeLabel)
	mux.HandleFunc("GET /v1/operations/runtime-credentials", current.listRuntimeCredentials)
	mux.HandleFunc("POST /v1/operations/runtime-credentials", current.createRuntimeCredential)
	mux.HandleFunc("GET /v1/operations/runtime-credentials/{credentialId}", current.getRuntimeCredential)
	mux.HandleFunc("DELETE /v1/operations/runtime-credentials/{credentialId}", current.deleteRuntimeCredential)
	mux.HandleFunc("GET /v1/operations/runtime-agent-principals", current.listRuntimeAgentPrincipals)
	mux.HandleFunc("GET /v1/operations/runtime-agent-principals/{runtimeAgentId}", current.getRuntimeAgentPrincipal)
	mux.HandleFunc("DELETE /v1/operations/runtime-agent-principals/{runtimeAgentId}", current.deleteRuntimeAgentPrincipal)
	mux.HandleFunc("PUT /v1/operations/runtime-agent-principals/{runtimeAgentId}/labels", current.putRuntimeAgentLabels)
	mux.HandleFunc("GET /v1/operations/snapshot", current.getOperationsSnapshot)
	mux.HandleFunc("GET /v1/operations/runtime-agents", current.listRuntimeAgents)
	mux.HandleFunc("GET /v1/operations/allocations", current.listAllocations)
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
	mux.HandleFunc("/v1/operations/runtime-configs/{name}/versions/{version}", current.methodNotAllowed)
	mux.HandleFunc("/v1/operations/runtime-configs", current.methodNotAllowed)
	mux.HandleFunc("/v1/operations/runtime-labels/{label}", current.methodNotAllowed)
	mux.HandleFunc("/v1/operations/runtime-labels", current.methodNotAllowed)
	mux.HandleFunc("/v1/operations/runtime-credentials/{credentialId}", current.methodNotAllowed)
	mux.HandleFunc("/v1/operations/runtime-credentials", current.methodNotAllowed)
	mux.HandleFunc("/v1/operations/snapshot", current.methodNotAllowed)
	mux.HandleFunc("/v1/operations/runtime-agents", current.methodNotAllowed)
	mux.HandleFunc("/v1/operations/allocations", current.methodNotAllowed)
	mux.HandleFunc("/v1/auth/login", current.methodNotAllowed)
	mux.HandleFunc("/v1/auth/session", current.methodNotAllowed)
	mux.HandleFunc("/v1/auth/logout", current.methodNotAllowed)
	mux.HandleFunc("/v1/events/ws", current.methodNotAllowed)
	mux.HandleFunc("/v1/workflows/{name}/versions/{version}", current.methodNotAllowed)
	mux.HandleFunc("/v1/workflows", current.methodNotAllowed)
	mux.HandleFunc("/v1/projects/{projectId}", current.methodNotAllowed)
	mux.HandleFunc("/v1/projects", current.methodNotAllowed)
	mux.HandleFunc("/v1/projects/{projectId}/artifacts/{namespace}/{name}/metadata", current.methodNotAllowed)
	mux.HandleFunc("/v1/projects/{projectId}/artifacts/{namespace}/{name}/versions", current.methodNotAllowed)
	mux.HandleFunc("/v1/projects/{projectId}/artifacts/{namespace}/{name}/lineage", current.methodNotAllowed)
	mux.HandleFunc("/v1/projects/{projectId}/artifacts/{namespace}/{name}", current.methodNotAllowed)
	mux.HandleFunc("/v1/projects/{projectId}/artifacts", current.methodNotAllowed)
	mux.HandleFunc("/v1/projects/{projectId}/runs", current.methodNotAllowed)
	mux.HandleFunc("/", current.notFound)

	return withAPIVersion(current.withRequestID(current.cors(current.authenticate(mux), mux))), nil
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
