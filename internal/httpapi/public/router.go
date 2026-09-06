package public

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/requestid"
	"github.com/grauwolf32/contractor/internal/runservice"
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
		dependencies.Projects == nil || dependencies.Audits == nil ||
		dependencies.Artifacts == nil || dependencies.Transactions == nil || dependencies.Operations == nil ||
		dependencies.OperationsInvalidator == nil || dependencies.SchedulerSettings == nil || dependencies.Events == nil ||
		dependencies.Authentication == nil || len(dependencies.BrowserOrigins.Values()) == 0 {
		return nil, fmt.Errorf("public API dependencies are incomplete")
	}
	if dependencies.RunCreator == nil {
		creator, err := runservice.New(runservice.Options{
			Runs: dependencies.Runs, Workflows: dependencies.Config,
			LLMCredentials: dependencies.Credentials, CredentialGuard: dependencies.ManagedCredentials,
			RuntimeCredentials: dependencies.RuntimeCredentials, Projects: dependencies.Projects,
			SkillInitializationAvailable: dependencies.RunSkills != nil,
			PublicTransaction: func(
				ctx context.Context,
				fn func(runservice.PublicRunWriter, *artifacts.Service) error,
			) error {
				return dependencies.Transactions.Do(ctx, func(runs RunWriter, service *artifacts.Service) error {
					return fn(runs, service)
				})
			},
		})
		if err != nil {
			return nil, fmt.Errorf("configure public Run creation: %w", err)
		}
		dependencies.RunCreator = creator
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
	mux.HandleFunc("GET /v1/settings/git-key", current.gitKeySettings)
	mux.HandleFunc("PUT /v1/settings/git-key", current.gitKeySettings)
	mux.HandleFunc("DELETE /v1/settings/git-key", current.gitKeySettings)
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
	mux.HandleFunc("DELETE /v1/projects/{projectId}", current.deleteProject)
	mux.HandleFunc("GET /v1/projects/{projectId}/artifacts", current.listProjectArtifacts)
	mux.HandleFunc("PUT /v1/projects/{projectId}/artifacts/{namespace}/{name}", current.putProjectArtifact)
	mux.HandleFunc("POST /v1/projects/{projectId}/artifacts/{namespace}/{name}/git-import", current.importGitArtifact)
	mux.HandleFunc("GET /v1/projects/{projectId}/artifacts/{namespace}/{name}", current.getProjectArtifact)
	mux.HandleFunc("GET /v1/projects/{projectId}/artifacts/{namespace}/{name}/metadata", current.getProjectArtifactMetadata)
	mux.HandleFunc("GET /v1/projects/{projectId}/artifacts/{namespace}/{name}/versions", current.listProjectArtifactVersions)
	mux.HandleFunc("GET /v1/projects/{projectId}/artifacts/{namespace}/{name}/lineage", current.listProjectArtifactLineage)
	mux.HandleFunc("POST /v1/projects/{projectId}/runs", current.createProjectRun)
	mux.HandleFunc("GET /v1/projects/{projectId}/runs", current.listProjectRuns)
	mux.HandleFunc("GET /v1/audit-profiles", current.listAuditProfiles)
	mux.HandleFunc("GET /v1/audit-profiles/{name}/versions/{version}", current.getAuditProfile)
	mux.HandleFunc("GET /v1/audit-standards", current.listAuditStandards)
	mux.HandleFunc("GET /v1/audit-standards/{scheme}/versions/{version}", current.getAuditStandard)
	mux.HandleFunc("POST /v1/projects/{projectId}/audits", current.createAudit)
	mux.HandleFunc("GET /v1/projects/{projectId}/audits", current.listProjectAudits)
	mux.HandleFunc("GET /v1/audits/{auditId}", current.getAudit)
	mux.HandleFunc("POST /v1/audits/{auditId}/start", current.startAudit)
	mux.HandleFunc("POST /v1/audits/{auditId}/pause", current.pauseAudit)
	mux.HandleFunc("POST /v1/audits/{auditId}/resume", current.resumeAudit)
	mux.HandleFunc("POST /v1/audits/{auditId}/cancel", current.cancelAudit)
	mux.HandleFunc("DELETE /v1/audits/{auditId}", current.deleteAudit)
	mux.HandleFunc("GET /v1/audits/{auditId}/items", current.listAuditItems)
	mux.HandleFunc("GET /v1/audits/{auditId}/coverage", current.listAuditCoverage)
	mux.HandleFunc("GET /v1/audits/{auditId}/report", current.getAuditReport)
	mux.HandleFunc("GET /v1/audits/{auditId}/finding-proposals", current.listAuditFindingProposals)
	mux.HandleFunc("POST /v1/audits/{auditId}/finding-proposal-imports", current.importAuditFindingProposal)
	mux.HandleFunc("GET /v1/audits/{auditId}/findings", current.listAuditFindings)
	mux.HandleFunc("GET /v1/audits/{auditId}/findings/{findingId}", current.getAuditFinding)
	mux.HandleFunc("GET /v1/audits/{auditId}/findings/{findingId}/provenance", current.listAuditFindingProvenance)
	mux.HandleFunc("POST /v1/audits/{auditId}/findings/{findingId}/reviews", current.createAuditFindingReview)
	mux.HandleFunc("GET /v1/audits/{auditId}/reviews", current.listAuditReviews)
	mux.HandleFunc("POST /v1/audits/{auditId}/reviews/{requestId}/decisions", current.decideAuditReview)
	mux.HandleFunc("GET /v1/configurations/{kind}", current.listConfigurations)
	mux.HandleFunc("POST /v1/configurations/{kind}", current.publishConfiguration)
	mux.HandleFunc("GET /v1/configurations/{kind}/{name}/versions/{version}", current.getConfiguration)
	mux.HandleFunc("GET /v1/configurations/agent-templates/{name}/versions/{version}/instructions", current.getAgentInstructions)
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
	mux.HandleFunc("GET /v1/operations/settings/scheduler", current.getSchedulerSettings)
	mux.HandleFunc("PUT /v1/operations/settings/scheduler", current.putSchedulerSettings)
	mux.HandleFunc("GET /v1/artifacts", current.listArtifacts)
	mux.HandleFunc("GET /v1/artifacts/{namespace}/{name}/metadata", current.getArtifactMetadata)
	mux.HandleFunc("GET /v1/artifacts/{namespace}/{name}/versions", current.listArtifactVersions)
	mux.HandleFunc("GET /v1/artifacts/{namespace}/{name}/lineage", current.listArtifactLineage)
	mux.HandleFunc("PUT /v1/artifacts/{namespace}/{name}", current.putArtifact)
	mux.HandleFunc("POST /v1/artifacts/{namespace}/{name}/git-import", current.importGitArtifact)
	mux.HandleFunc("GET /v1/artifacts/{namespace}/{name}", current.getArtifact)
	mux.HandleFunc("POST /v1/runs", current.createRun)
	mux.HandleFunc("GET /v1/runs", current.listRuns)
	mux.HandleFunc("GET /v1/queue", current.listRunQueue)
	mux.HandleFunc("GET /v1/queue/control", current.getOwnerQueueControl)
	mux.HandleFunc("PUT /v1/queue/control", current.putOwnerQueueControl)
	mux.HandleFunc("POST /v1/runs/{runID}/cancel", current.cancelRun)
	mux.HandleFunc("DELETE /v1/runs/{runID}", current.deleteRun)
	mux.HandleFunc("GET /v1/runs/{runID}", current.getRun)
	mux.HandleFunc("GET /v1/runs/{runID}/outputs/{slot}", current.getRunOutput)
	mux.HandleFunc("GET /v1/runs/{runID}/artifacts", current.listRunArtifacts)
	mux.HandleFunc("GET /v1/runs/{runID}/finding-proposals", current.listRunFindingProposals)
	mux.HandleFunc("GET /v1/runs/{runID}/artifacts/{namespace}/{name}", current.getRunArtifact)
	mux.HandleFunc("GET /v1/runs/{runID}/artifacts/{namespace}/{name}/metadata", current.getRunArtifactMetadata)
	mux.HandleFunc("GET /v1/runs/{runID}/artifacts/{namespace}/{name}/versions", current.listRunArtifactVersions)
	mux.HandleFunc("GET /v1/runs/{runID}/artifacts/{namespace}/{name}/lineage", current.listRunArtifactLineage)
	mux.HandleFunc("/v1/runs/{runID}/artifacts/{namespace}/{name}/metadata", current.methodNotAllowed)
	mux.HandleFunc("/v1/runs/{runID}/artifacts/{namespace}/{name}/versions", current.methodNotAllowed)
	mux.HandleFunc("/v1/runs/{runID}/artifacts/{namespace}/{name}/lineage", current.methodNotAllowed)
	mux.HandleFunc("/v1/runs/{runID}/artifacts/{namespace}/{name}", current.methodNotAllowed)
	mux.HandleFunc("/v1/runs/{runID}/artifacts", current.methodNotAllowed)
	mux.HandleFunc("/v1/runs/{runID}/finding-proposals", current.methodNotAllowed)
	mux.HandleFunc("/v1/artifacts/{namespace}/{name}/metadata", current.methodNotAllowed)
	mux.HandleFunc("/v1/artifacts/{namespace}/{name}/versions", current.methodNotAllowed)
	mux.HandleFunc("/v1/artifacts/{namespace}/{name}/lineage", current.methodNotAllowed)
	mux.HandleFunc("/v1/artifacts/{namespace}/{name}", current.methodNotAllowed)
	mux.HandleFunc("/v1/artifacts", current.methodNotAllowed)
	mux.HandleFunc("/v1/runs/{runID}/outputs/{slot}", current.methodNotAllowed)
	mux.HandleFunc("/v1/runs/{runID}/cancel", current.methodNotAllowed)
	mux.HandleFunc("/v1/runs/{runID}", current.methodNotAllowed)
	mux.HandleFunc("/v1/runs", current.methodNotAllowed)
	mux.HandleFunc("/v1/queue", current.methodNotAllowed)
	mux.HandleFunc("/v1/queue/control", current.methodNotAllowed)
	mux.HandleFunc("/v1/configurations/{kind}/{name}/versions/{version}", current.methodNotAllowed)
	mux.HandleFunc("/v1/configurations/agent-templates/{name}/versions/{version}/instructions", current.methodNotAllowed)
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
	mux.HandleFunc("/v1/operations/settings/scheduler", current.methodNotAllowed)
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
	mux.HandleFunc("/v1/audit-profiles/{name}/versions/{version}", current.methodNotAllowed)
	mux.HandleFunc("/v1/audit-profiles", current.methodNotAllowed)
	mux.HandleFunc("/v1/projects/{projectId}/audits", current.methodNotAllowed)
	mux.HandleFunc("/v1/audits/{auditId}/start", current.methodNotAllowed)
	mux.HandleFunc("/v1/audits/{auditId}/items", current.methodNotAllowed)
	mux.HandleFunc("/v1/audits/{auditId}/coverage", current.methodNotAllowed)
	mux.HandleFunc("/v1/audits/{auditId}/report", current.methodNotAllowed)
	mux.HandleFunc("/v1/audits/{auditId}/finding-proposals", current.methodNotAllowed)
	mux.HandleFunc("/v1/audits/{auditId}/finding-proposal-imports", current.methodNotAllowed)
	mux.HandleFunc("/v1/audits/{auditId}/findings/{findingId}/provenance", current.methodNotAllowed)
	mux.HandleFunc("/v1/audits/{auditId}/findings/{findingId}/reviews", current.methodNotAllowed)
	mux.HandleFunc("/v1/audits/{auditId}/findings/{findingId}", current.methodNotAllowed)
	mux.HandleFunc("/v1/audits/{auditId}/findings", current.methodNotAllowed)
	mux.HandleFunc("/v1/audits/{auditId}/reviews/{requestId}/decisions", current.methodNotAllowed)
	mux.HandleFunc("/v1/audits/{auditId}/reviews", current.methodNotAllowed)
	mux.HandleFunc("/v1/audits/{auditId}", current.methodNotAllowed)
	mux.HandleFunc("/", current.notFound)

	return withAPIVersion(current.withRequestID(current.cors(current.authenticate(withErrorDiagnostics(mux)), mux))), nil
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
