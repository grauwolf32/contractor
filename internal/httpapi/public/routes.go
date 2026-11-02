package public

import (
	"net/http"

	"github.com/grauwolf32/contractor/internal/httpapi/httpx"
)

// route is one operation of the public contract: its method, its OpenAPI
// path template (also the ServeMux pattern, so handlers read OpenAPI's
// parameter names), its operationId and the handler that serves it.
type route struct {
	method, path, operationID string
	handler                   http.HandlerFunc
}

// routes lists every public operation once. register serves each of them and
// answers any other method of their paths, HEAD included, with 405;
// TestRouteTableMatchesOpenAPI keeps the table equal to the operations of
// the OpenAPI contract, and every operation must pass a validated test
// (TestMain).
func (c contextHandlers) routes() []route {
	return []route{
		{"GET", "/v1/config", "getSharedConfig", getSharedConfig},
		{"POST", "/v1/auth/login", "login", serve(c.session.login)},
		{"GET", "/v1/auth/session", "getSession", serve(c.session.getSession)},
		{"POST", "/v1/auth/logout", "logout", serve(c.session.logout)},
		{"GET", "/v1/events/ws", "connectEventsWebSocket", serve(c.session.connectEventsWebSocket)},

		{"GET", "/v1/settings/git-key", "getGitKey", serve(c.gitKeys.getGitKey)},
		{"PUT", "/v1/settings/git-key", "replaceGitKey", serve(c.gitKeys.replaceGitKey)},
		{"DELETE", "/v1/settings/git-key", "deleteGitKey", serve(c.gitKeys.deleteGitKey)},

		{"GET", "/v1/workflows", "listWorkflows", serve(c.catalog.listWorkflows)},
		{"GET", "/v1/workflows/{name}/versions/{version}", "getWorkflow", serve(c.catalog.getWorkflow)},
		{"GET", "/v1/configurations/{kind}", "listConfigurations", serve(c.catalog.listConfigurations)},
		{"POST", "/v1/configurations/{kind}", "publishConfiguration",
			withOperationsCapability(serve(c.catalog.publishConfiguration)).ServeHTTP},
		{"GET", "/v1/configurations/{kind}/{name}/versions/{version}", "getConfiguration", serve(c.catalog.getConfiguration)},
		{"GET", "/v1/configurations/agent-templates/{name}/versions/{version}/instructions",
			"getAgentInstructions", serve(c.catalog.getAgentInstructions)},
		{"GET", "/v1/configurations/agent-templates/{name}/versions/{version}/workflow-bindings",
			"listAgentTemplateWorkflowBindings", serve(c.catalog.listAgentTemplateWorkflowBindings)},

		{"POST", "/v1/projects", "createProject", serve(c.projects.createProject)},
		{"GET", "/v1/projects", "listProjects", serve(c.projects.listProjects)},
		{"GET", "/v1/projects/{projectId}", "getProject", serve(c.projects.getProject)},
		{"PATCH", "/v1/projects/{projectId}", "updateProject", serve(c.projects.updateProject)},
		{"DELETE", "/v1/projects/{projectId}", "deleteProject", serve(c.projects.deleteProject)},
		{"POST", "/v1/projects/{projectId}/sitemap/mutations", "mutateProjectSitemap", serve(c.projects.mutateSitemap)},

		{"POST", "/v1/runs", "createRun", serve(c.runs.createRun)},
		{"GET", "/v1/runs", "listRuns", serve(c.runs.listRuns)},
		{"POST", "/v1/projects/{projectId}/runs", "createProjectRun", serve(c.runs.createProjectRun)},
		{"GET", "/v1/projects/{projectId}/runs", "listProjectRuns", serve(c.runs.listProjectRuns)},
		{"GET", "/v1/runs/{runId}", "getRun", serve(c.runs.getRun)},
		{"GET", "/v1/runs/{runId}/attempts/{stageExecutionId}/runtime-diagnostics", "getRunRuntimeDiagnostics", serve(c.runs.runtimeDiagnostics)},
		{"GET", "/v1/runs/{runId}/attempts/{stageExecutionId}/tool-calls", "getRunToolCalls", serve(c.runs.toolCalls)},
		{"DELETE", "/v1/runs/{runId}", "deleteRun", serve(c.runs.deleteRun)},
		{"POST", "/v1/runs/{runId}/cancel", "cancelRun", serve(c.runs.cancelRun)},
		{"POST", "/v1/runs/{runId}/resume", "resumeRun", serve(c.runs.resumeRun)},
		{"POST", "/v1/runs/{runId}/retry-gateway", "retryRunGateway", serve(c.runs.retryRunGateway)},
		{"GET", "/v1/runs/{runId}/repeat-draft", "getRunRepeatDraft", serve(c.runRepeat.getRunRepeatDraft)},

		{"GET", "/v1/queue", "listRunQueue", serve(c.queue.listRunQueue)},
		{"GET", "/v1/queue/control", "getOwnerQueueControl", serve(c.queue.getOwnerQueueControl)},
		{"PUT", "/v1/queue/control", "putOwnerQueueControl", serve(c.queue.putOwnerQueueControl)},

		{"GET", "/v1/artifacts", "listArtifacts", serve(c.artifacts.listArtifacts)},
		{"GET", "/v1/artifacts/{namespace}/{name}", "downloadArtifact", serve(c.artifacts.getArtifact)},
		{"PUT", "/v1/artifacts/{namespace}/{name}", "putArtifact", serve(c.artifacts.putArtifact)},
		{"GET", "/v1/artifacts/{namespace}/{name}/metadata", "getArtifactMetadata", serve(c.artifacts.getArtifactMetadata)},
		{"GET", "/v1/artifacts/{namespace}/{name}/versions", "listArtifactVersions", serve(c.artifacts.listArtifactVersions)},
		{"GET", "/v1/artifacts/{namespace}/{name}/lineage", "getArtifactLineage", serve(c.artifacts.listArtifactLineage)},
		{"GET", "/v1/artifacts/{namespace}/{name}/archive", "getArtifactArchive", serve(c.artifacts.getArtifactArchive)},
		{"GET", "/v1/artifacts/{namespace}/{name}/archive/file", "getArtifactArchiveFile", serve(c.artifacts.getArtifactArchiveFile)},
		{"POST", "/v1/artifacts/{namespace}/{name}/git-import", "importGitArtifact", serve(c.artifacts.importGitArtifact)},
		{"GET", "/v1/projects/{projectId}/artifacts", "listProjectArtifacts", serve(c.artifacts.listProjectArtifacts)},
		{"GET", "/v1/projects/{projectId}/artifacts/{namespace}/{name}", "downloadProjectArtifact", serve(c.artifacts.getProjectArtifact)},
		{"PUT", "/v1/projects/{projectId}/artifacts/{namespace}/{name}", "putProjectArtifact", serve(c.artifacts.putProjectArtifact)},
		{"GET", "/v1/projects/{projectId}/artifacts/{namespace}/{name}/metadata",
			"getProjectArtifactMetadata", serve(c.artifacts.getProjectArtifactMetadata)},
		{"GET", "/v1/projects/{projectId}/artifacts/{namespace}/{name}/versions",
			"listProjectArtifactVersions", serve(c.artifacts.listProjectArtifactVersions)},
		{"GET", "/v1/projects/{projectId}/artifacts/{namespace}/{name}/lineage",
			"getProjectArtifactLineage", serve(c.artifacts.listProjectArtifactLineage)},
		{"GET", "/v1/projects/{projectId}/artifacts/{namespace}/{name}/archive",
			"getProjectArtifactArchive", serve(c.artifacts.getProjectArtifactArchive)},
		{"GET", "/v1/projects/{projectId}/artifacts/{namespace}/{name}/archive/file",
			"getProjectArtifactArchiveFile", serve(c.artifacts.getProjectArtifactArchiveFile)},
		{"POST", "/v1/projects/{projectId}/artifacts/{namespace}/{name}/git-import",
			"importProjectGitArtifact", serve(c.artifacts.importGitArtifact)},
		{"GET", "/v1/runs/{runId}/outputs/{slot}", "downloadRunOutput", serve(c.artifacts.getRunOutput)},
		{"GET", "/v1/runs/{runId}/artifacts", "listRunArtifacts", serve(c.artifacts.listRunArtifacts)},
		{"GET", "/v1/runs/{runId}/artifacts/{namespace}/{name}", "downloadRunArtifact", serve(c.artifacts.getRunArtifact)},
		{"GET", "/v1/runs/{runId}/artifacts/{namespace}/{name}/metadata", "getRunArtifactMetadata", serve(c.artifacts.getRunArtifactMetadata)},
		{"GET", "/v1/runs/{runId}/artifacts/{namespace}/{name}/versions", "listRunArtifactVersions", serve(c.artifacts.listRunArtifactVersions)},
		{"GET", "/v1/runs/{runId}/artifacts/{namespace}/{name}/lineage", "getRunArtifactLineage", serve(c.artifacts.listRunArtifactLineage)},
		{"GET", "/v1/runs/{runId}/artifacts/{namespace}/{name}/archive", "getRunArtifactArchive", serve(c.artifacts.getRunArtifactArchive)},
		{"GET", "/v1/runs/{runId}/artifacts/{namespace}/{name}/archive/file",
			"getRunArtifactArchiveFile", serve(c.artifacts.getRunArtifactArchiveFile)},

		{"GET", "/v1/audit-profiles", "listAuditProfiles", serve(c.audits.listAuditProfiles)},
		{"GET", "/v1/audit-profiles/{name}/versions/{version}", "getAuditProfile", serve(c.audits.getAuditProfile)},
		{"GET", "/v1/audit-standards", "listAuditStandards", serve(c.audits.listAuditStandards)},
		{"GET", "/v1/audit-standards/{scheme}/versions/{version}", "getAuditStandard", serve(c.audits.getAuditStandard)},
		{"GET", "/v1/audits", "listOwnerAudits", serve(c.audits.listProjectAudits)},
		{"GET", "/v1/audits/{auditId}/findings/{findingId}/duplicate-candidates", "listFindingDuplicateCandidates", serve(c.audits.listFindingDuplicateCandidates)},
		{"GET", "/v1/projects/{projectId}/finding-comparisons", "listProjectFindingComparisons", serve(c.audits.listProjectFindingComparisons)},
		{"GET", "/v1/findings", "listOwnerFindings", serve(c.audits.listOwnerFindings)},
		{"GET", "/v1/projects/{projectId}/findings", "listProjectFindings", serve(c.audits.listProjectFindings)},
		{"GET", "/v1/projects/{projectId}/findings/{findingId}", "getProjectFinding", serve(c.audits.getProjectFinding)},
		{"PATCH", "/v1/projects/{projectId}/findings/{findingId}", "editProjectFinding", serve(c.audits.editProjectFinding)},
		{"GET", "/v1/projects/{projectId}/finding-sources", "listProjectFindingSources", serve(c.findings.listProjectFindingSources)},
		{"GET", "/v1/projects/{projectId}/finding-sources/{receiptId}", "getProjectFindingSource", serve(c.findings.getProjectFindingSource)},
		{"GET", "/v1/projects/{projectId}/finding-sources/{receiptId}/history", "listProjectFindingSourceHistory", serve(c.findings.listProjectFindingSourceHistory)},
		{"GET", "/v1/reviews", "listOwnerReviews", serve(c.audits.listOwnerReviews)},
		{"POST", "/v1/projects/{projectId}/audits", "createAudit", serve(c.audits.createAudit)},
		{"GET", "/v1/projects/{projectId}/audits", "listProjectAudits", serve(c.audits.listProjectAudits)},
		{"GET", "/v1/audits/{auditId}", "getAudit", serve(c.audits.getAudit)},
		{"DELETE", "/v1/audits/{auditId}", "deleteAudit", serve(c.audits.deleteAudit)},
		{"POST", "/v1/audits/{auditId}/start", "startAudit", serve(c.audits.startAudit)},
		{"POST", "/v1/audits/{auditId}/pause", "pauseAudit", serve(c.audits.pauseAudit)},
		{"POST", "/v1/audits/{auditId}/resume", "resumeAudit", serve(c.audits.resumeAudit)},
		{"POST", "/v1/audits/{auditId}/cancel", "cancelAudit", serve(c.audits.cancelAudit)},
		{"GET", "/v1/audits/{auditId}/events", "listAuditEvents", serve(c.audits.listAuditEvents)},
		{"GET", "/v1/audits/{auditId}/items", "listAuditItems", serve(c.audits.listAuditItems)},
		{"GET", "/v1/audits/{auditId}/coverage", "listAuditCoverage", serve(c.audits.listAuditCoverage)},
		{"GET", "/v1/audits/{auditId}/workspace", "getAuditWorkspace", serve(c.audits.getAuditWorkspace)},
		{"POST", "/v1/projects/{projectId}/audit-material-publications", "publishAuditMaterial", serve(c.audits.publishAuditMaterial)},
		{"GET", "/v1/audits/{auditId}/artifacts", "getAuditArtifact", serve(c.audits.getAuditArtifact)},
		{"GET", "/v1/audits/{auditId}/report", "getAuditReport", serve(c.audits.getAuditReport)},
		{"GET", "/v1/audits/{auditId}/findings", "listAuditFindings", serve(c.audits.listAuditFindings)},
		{"POST", "/v1/audits/{auditId}/findings/exports", "publishAuditFindingExport", serve(c.findings.publishAuditFindingExport)},
		{"GET", "/v1/audits/{auditId}/findings/exports/{exportId}/content", "downloadAuditFindingExport", serve(c.findings.downloadAuditFindingExport)},
		{"GET", "/v1/audits/{auditId}/findings/exports/{exportId}/snapshot", "downloadAuditFindingExportSnapshot", serve(c.findings.downloadAuditFindingExportSnapshot)},
		{"GET", "/v1/audits/{auditId}/findings/exports/{exportId}/diagnostics", "downloadAuditFindingExportDiagnostics", serve(c.findings.downloadAuditFindingExportDiagnostics)},
		{"GET", "/v1/audits/{auditId}/findings/{findingId}", "getAuditFinding", serve(c.audits.getAuditFinding)},
		{"GET", "/v1/audits/{auditId}/reviews", "listAuditReviews", serve(c.audits.listAuditReviews)},
		{"GET", "/v1/audits/{auditId}/reviews/{requestId}", "getAuditReview", serve(c.audits.getAuditReview)},
		{"POST", "/v1/audits/{auditId}/reviews/{requestId}/decisions", "decideAuditReview", serve(c.audits.decideAuditReview)},

		{"GET", "/v1/runs/{runId}/finding-proposals", "listRunFindingProposals", serve(c.findings.listRunFindingProposals)},
		{"GET", "/v1/audits/{auditId}/finding-proposals", "listAuditFindingProposals", serve(c.findings.listAuditFindingProposals)},
		{"POST", "/v1/audits/{auditId}/finding-proposal-imports", "importAuditFindingProposal", serve(c.findings.importAuditFindingProposal)},
		{"POST", "/v1/finding-collections", "publishFindingCollection", serve(c.findings.publishFindingCollection)},

		{"GET", "/v1/eval-capabilities", "getEvalCapabilities", evalRoute(serve(c.evals.evalCapabilities))},
		{"GET", "/v1/projects/{projectId}/eval-datasets", "listEvalDatasets", evalRoute(serve(c.evals.listEvalDatasets))},
		{"POST", "/v1/projects/{projectId}/eval-datasets", "importEvalDataset", evalRoute(serve(c.evals.importEvalDataset))},
		{"GET", "/v1/projects/{projectId}/eval-datasets/{datasetId}/revisions/{revision}/cases",
			"listEvalCases", evalRoute(serve(c.evals.listEvalCases))},
		{"POST", "/v1/projects/{projectId}/eval-experiments", "createEvalExperiment", evalRoute(serve(c.evals.createEvalExperiment))},
		{"GET", "/v1/eval-experiments", "listEvalExperiments", evalRoute(serve(c.evals.listEvalExperiments))},
		{"GET", "/v1/eval-experiments/{id}", "getEvalExperiment", evalRoute(serve(c.evals.getEvalExperiment))},
		{"PATCH", "/v1/eval-experiments/{id}", "updateEvalDraft", evalRoute(serve(c.evals.updateEvalDraft))},
		{"DELETE", "/v1/eval-experiments/{id}", "deleteEvalExperiment", evalRoute(serve(c.evals.deleteEvalExperiment))},
		{"POST", "/v1/eval-experiments/{id}/commands", "commandEvalExperiment", evalRoute(serve(c.evals.commandEvalExperiment))},
		{"GET", "/v1/eval-experiments/{id}/commands/{commandId}", "getEvalCommand", evalRoute(serve(c.evals.getEvalCommand))},
		{"GET", "/v1/eval-experiments/{id}/members", "listEvalMembers", evalRoute(serve(c.evals.listEvalMembers))},
		{"POST", "/v1/eval-experiments/{id}/members/{memberId}/submissions", "submitEvalMember", evalRoute(serve(c.evals.submitEvalMember))},
		{"POST", "/v1/eval-experiments/{id}/members/{memberId}/results", "ingestEvalResult", evalRoute(serve(c.evals.ingestEvalResult))},
		{"POST", "/v1/eval-experiments/{id}/members/{memberId}/assessments", "assessEvalMember", evalRoute(serve(c.evals.assessEvalMember))},
		{"GET", "/v1/eval-experiments/{id}/members/{memberId}/review", "reviewEvalMember", evalRoute(serve(c.evals.reviewEvalMember))},
		{"GET", "/v1/eval-experiments/{id}/members/{memberId}/executions",
			"listEvalMemberExecutions", evalRoute(serve(c.evals.listEvalMemberExecutions))},
		{"POST", "/v1/eval-experiments/{id}/selections", "selectEvalRecords", evalRoute(serve(c.evals.selectEvalRecords))},
		{"GET", "/v1/eval-experiments/{id}/pairs", "listEvalPairs", evalRoute(serve(c.evals.listEvalPairs))},
		{"GET", "/v1/eval-experiments/{id}/pairs/{pairId}", "getEvalPair", evalRoute(serve(c.evals.getEvalPair))},
		{"GET", "/v1/eval-experiments/{id}/charts/{chart}", "getEvalChart", evalRoute(serve(c.evals.getEvalChart))},
		{"GET", "/v1/eval-experiments/{id}/report", "getEvalReport", evalRoute(serve(c.evals.getEvalReport))},

		{"GET", "/v1/operations/snapshot", "getOperationsSnapshot", serve(c.operations.getOperationsSnapshot)},
		{"GET", "/v1/operations/runtime-agents", "listRuntimeAgents", serve(c.operations.listRuntimeAgents)},
		{"GET", "/v1/operations/allocations", "listAllocations", serve(c.operations.listAllocations)},
		{"GET", "/v1/operations/allocation-history", "listAllocationResourceHistory", serve(c.operations.listAllocationResourceHistory)},
		{"GET", "/v1/operations/performance", "getPerformance", serve(c.operations.getPerformance)},
		{"GET", "/v1/operations/performance/history", "getPerformanceHistory", serve(c.operations.getPerformanceHistory)},
		{"GET", "/v1/operations/settings/models", "getModelSelection", serve(c.operations.getModelSelection)},
		{"PUT", "/v1/operations/settings/models", "putModelSelection", serve(c.operations.putModelSelection)},
		{"GET", "/v1/operations/settings/scheduler", "getSchedulerSettings", serve(c.operations.getSchedulerSettings)},
		{"PUT", "/v1/operations/settings/scheduler", "putSchedulerSettings", serve(c.operations.putSchedulerSettings)},

		{"GET", "/v1/operations/credentials", "listCredentials", serve(c.credentials.listCredentials)},
		{"POST", "/v1/operations/credentials", "createCredential", serve(c.credentials.createCredential)},
		{"GET", "/v1/operations/credentials/{credentialId}", "getCredential", serve(c.credentials.getCredential)},
		{"DELETE", "/v1/operations/credentials/{credentialId}", "deleteCredential", serve(c.credentials.deleteCredential)},

		{"GET", "/v1/operations/runtime-configs", "listRuntimeConfigs", serve(c.runtimeConfiguration.listRuntimeConfigs)},
		{"POST", "/v1/operations/runtime-configs", "publishRuntimeConfig", serve(c.runtimeConfiguration.publishRuntimeConfig)},
		{"GET", "/v1/operations/runtime-configs/{name}/versions/{version}", "getRuntimeConfig", serve(c.runtimeConfiguration.getRuntimeConfig)},
		{"GET", "/v1/operations/runtime-labels", "listRuntimeLabels", serve(c.runtimeConfiguration.listRuntimeLabels)},
		{"GET", "/v1/operations/runtime-labels/{label}", "getRuntimeLabel", serve(c.runtimeConfiguration.getRuntimeLabel)},
		{"PUT", "/v1/operations/runtime-labels/{label}", "putRuntimeLabel", serve(c.runtimeConfiguration.putRuntimeLabel)},
		{"DELETE", "/v1/operations/runtime-labels/{label}", "deleteRuntimeLabel", serve(c.runtimeConfiguration.deleteRuntimeLabel)},
		{"GET", "/v1/operations/runtime-credentials", "listRuntimeCredentials", serve(c.runtimeConfiguration.listRuntimeCredentials)},
		{"POST", "/v1/operations/runtime-credentials", "createRuntimeCredential", serve(c.runtimeConfiguration.createRuntimeCredential)},
		{"GET", "/v1/operations/runtime-credentials/{credentialId}", "getRuntimeCredential", serve(c.runtimeConfiguration.getRuntimeCredential)},
		{"DELETE", "/v1/operations/runtime-credentials/{credentialId}", "deleteRuntimeCredential", serve(c.runtimeConfiguration.deleteRuntimeCredential)},
		{"GET", "/v1/operations/runtime-agent-principals", "listRuntimeAgentPrincipals", serve(c.runtimeConfiguration.listRuntimeAgentPrincipals)},
		{"GET", "/v1/operations/runtime-agent-principals/{runtimeAgentId}",
			"getRuntimeAgentPrincipal", serve(c.runtimeConfiguration.getRuntimeAgentPrincipal)},
		{"DELETE", "/v1/operations/runtime-agent-principals/{runtimeAgentId}",
			"deleteRuntimeAgentPrincipal", serve(c.runtimeConfiguration.deleteRuntimeAgentPrincipal)},
		{"PUT", "/v1/operations/runtime-agent-principals/{runtimeAgentId}/labels",
			"replaceRuntimeAgentPrincipalLabels", serve(c.runtimeConfiguration.putRuntimeAgentLabels)},
	}
}

// register serves routes through the routing rule the private API shares:
// another method of a route's path, HEAD included, is 405 before any
// operation handler runs (API-R73, API-R111), and any other path is 404.
// Authentication and authorization wrap the mux, so they still answer first,
// as they do for GET.
func register(mux *http.ServeMux, routes []route) func(*http.Request) bool {
	served := make([]httpx.Route, 0, len(routes))
	for _, route := range routes {
		served = append(served, httpx.Route{Method: route.method, Path: route.path, Handler: route.handler})
	}
	return httpx.ServeRoutes(mux, served, methodNotAllowed, notFound)
}

// evalRoute keeps every Eval response out of caches.
func evalRoute(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		next(w, r)
	}
}
