package public

import "net/http"

// route is one operation of the public contract: its method, its OpenAPI
// path template (also the ServeMux pattern, so handlers read OpenAPI's
// parameter names), its operationId and the handler that serves it.
type route struct {
	method, path, operationID string
	handler                   http.HandlerFunc
}

// routes lists every public operation once. register serves each of them and
// answers any other method of their paths with 405;
// TestRouteTableMatchesOpenAPI keeps the table equal to the operations of
// the OpenAPI contract, and every operation must pass a validated test
// (TestMain).
func (c contextHandlers) routes() []route {
	return []route{
		{"POST", "/v1/auth/login", "login", c.session.login},
		{"GET", "/v1/auth/session", "getSession", c.session.getSession},
		{"POST", "/v1/auth/logout", "logout", c.session.logout},
		{"GET", "/v1/events/ws", "connectEventsWebSocket", c.session.connectEventsWebSocket},

		{"GET", "/v1/settings/git-key", "getGitKey", c.gitKeys.getGitKey},
		{"PUT", "/v1/settings/git-key", "replaceGitKey", c.gitKeys.replaceGitKey},
		{"DELETE", "/v1/settings/git-key", "deleteGitKey", c.gitKeys.deleteGitKey},

		{"GET", "/v1/workflows", "listWorkflows", c.catalog.listWorkflows},
		{"GET", "/v1/workflows/{name}/versions/{version}", "getWorkflow", c.catalog.getWorkflow},
		{"GET", "/v1/configurations/{kind}", "listConfigurations", c.catalog.listConfigurations},
		{"POST", "/v1/configurations/{kind}", "publishConfiguration",
			withOperationsCapability(http.HandlerFunc(c.catalog.publishConfiguration)).ServeHTTP},
		{"GET", "/v1/configurations/{kind}/{name}/versions/{version}", "getConfiguration", c.catalog.getConfiguration},
		{"GET", "/v1/configurations/agent-templates/{name}/versions/{version}/instructions",
			"getAgentInstructions", c.catalog.getAgentInstructions},
		{"GET", "/v1/configurations/agent-templates/{name}/versions/{version}/workflow-bindings",
			"listAgentTemplateWorkflowBindings", c.catalog.listAgentTemplateWorkflowBindings},

		{"POST", "/v1/projects", "createProject", c.projects.createProject},
		{"GET", "/v1/projects", "listProjects", c.projects.listProjects},
		{"GET", "/v1/projects/{projectId}", "getProject", c.projects.getProject},
		{"PATCH", "/v1/projects/{projectId}", "updateProject", c.projects.updateProject},
		{"DELETE", "/v1/projects/{projectId}", "deleteProject", c.projects.deleteProject},

		{"POST", "/v1/runs", "createRun", c.runs.createRun},
		{"GET", "/v1/runs", "listRuns", c.runs.listRuns},
		{"POST", "/v1/projects/{projectId}/runs", "createProjectRun", c.runs.createProjectRun},
		{"GET", "/v1/projects/{projectId}/runs", "listProjectRuns", c.runs.listProjectRuns},
		{"GET", "/v1/runs/{runId}", "getRun", c.runs.getRun},
		{"DELETE", "/v1/runs/{runId}", "deleteRun", c.runs.deleteRun},
		{"POST", "/v1/runs/{runId}/cancel", "cancelRun", c.runs.cancelRun},
		{"POST", "/v1/runs/{runId}/resume", "resumeRun", c.runs.resumeRun},
		{"POST", "/v1/runs/{runId}/retry-gateway", "retryRunGateway", c.runs.retryRunGateway},
		{"GET", "/v1/runs/{runId}/repeat-draft", "getRunRepeatDraft", c.runRepeat.getRunRepeatDraft},

		{"GET", "/v1/queue", "listRunQueue", c.queue.listRunQueue},
		{"GET", "/v1/queue/control", "getOwnerQueueControl", c.queue.getOwnerQueueControl},
		{"PUT", "/v1/queue/control", "putOwnerQueueControl", c.queue.putOwnerQueueControl},

		{"GET", "/v1/artifacts", "listArtifacts", c.artifacts.listArtifacts},
		{"GET", "/v1/artifacts/{namespace}/{name}", "downloadArtifact", c.artifacts.getArtifact},
		{"PUT", "/v1/artifacts/{namespace}/{name}", "putArtifact", c.artifacts.putArtifact},
		{"GET", "/v1/artifacts/{namespace}/{name}/metadata", "getArtifactMetadata", c.artifacts.getArtifactMetadata},
		{"GET", "/v1/artifacts/{namespace}/{name}/versions", "listArtifactVersions", c.artifacts.listArtifactVersions},
		{"GET", "/v1/artifacts/{namespace}/{name}/lineage", "getArtifactLineage", c.artifacts.listArtifactLineage},
		{"GET", "/v1/artifacts/{namespace}/{name}/archive", "getArtifactArchive", c.artifacts.getArtifactArchive},
		{"GET", "/v1/artifacts/{namespace}/{name}/archive/file", "getArtifactArchiveFile", c.artifacts.getArtifactArchiveFile},
		{"POST", "/v1/artifacts/{namespace}/{name}/git-import", "importGitArtifact", c.artifacts.importGitArtifact},
		{"GET", "/v1/projects/{projectId}/artifacts", "listProjectArtifacts", c.artifacts.listProjectArtifacts},
		{"GET", "/v1/projects/{projectId}/artifacts/{namespace}/{name}", "downloadProjectArtifact", c.artifacts.getProjectArtifact},
		{"PUT", "/v1/projects/{projectId}/artifacts/{namespace}/{name}", "putProjectArtifact", c.artifacts.putProjectArtifact},
		{"GET", "/v1/projects/{projectId}/artifacts/{namespace}/{name}/metadata",
			"getProjectArtifactMetadata", c.artifacts.getProjectArtifactMetadata},
		{"GET", "/v1/projects/{projectId}/artifacts/{namespace}/{name}/versions",
			"listProjectArtifactVersions", c.artifacts.listProjectArtifactVersions},
		{"GET", "/v1/projects/{projectId}/artifacts/{namespace}/{name}/lineage",
			"getProjectArtifactLineage", c.artifacts.listProjectArtifactLineage},
		{"GET", "/v1/projects/{projectId}/artifacts/{namespace}/{name}/archive",
			"getProjectArtifactArchive", c.artifacts.getProjectArtifactArchive},
		{"GET", "/v1/projects/{projectId}/artifacts/{namespace}/{name}/archive/file",
			"getProjectArtifactArchiveFile", c.artifacts.getProjectArtifactArchiveFile},
		{"POST", "/v1/projects/{projectId}/artifacts/{namespace}/{name}/git-import",
			"importProjectGitArtifact", c.artifacts.importGitArtifact},
		{"GET", "/v1/runs/{runId}/outputs/{slot}", "downloadRunOutput", c.artifacts.getRunOutput},
		{"GET", "/v1/runs/{runId}/artifacts", "listRunArtifacts", c.artifacts.listRunArtifacts},
		{"GET", "/v1/runs/{runId}/artifacts/{namespace}/{name}", "downloadRunArtifact", c.artifacts.getRunArtifact},
		{"GET", "/v1/runs/{runId}/artifacts/{namespace}/{name}/metadata", "getRunArtifactMetadata", c.artifacts.getRunArtifactMetadata},
		{"GET", "/v1/runs/{runId}/artifacts/{namespace}/{name}/versions", "listRunArtifactVersions", c.artifacts.listRunArtifactVersions},
		{"GET", "/v1/runs/{runId}/artifacts/{namespace}/{name}/lineage", "getRunArtifactLineage", c.artifacts.listRunArtifactLineage},
		{"GET", "/v1/runs/{runId}/artifacts/{namespace}/{name}/archive", "getRunArtifactArchive", c.artifacts.getRunArtifactArchive},
		{"GET", "/v1/runs/{runId}/artifacts/{namespace}/{name}/archive/file",
			"getRunArtifactArchiveFile", c.artifacts.getRunArtifactArchiveFile},

		{"GET", "/v1/audit-profiles", "listAuditProfiles", c.audits.listAuditProfiles},
		{"GET", "/v1/audit-profiles/{name}/versions/{version}", "getAuditProfile", c.audits.getAuditProfile},
		{"GET", "/v1/audit-standards", "listAuditStandards", c.audits.listAuditStandards},
		{"GET", "/v1/audit-standards/{scheme}/versions/{version}", "getAuditStandard", c.audits.getAuditStandard},
		{"GET", "/v1/audits", "listOwnerAudits", c.audits.listProjectAudits},
		{"GET", "/v1/findings", "listOwnerFindings", c.audits.listOwnerFindings},
		{"GET", "/v1/reviews", "listOwnerReviews", c.audits.listOwnerReviews},
		{"POST", "/v1/projects/{projectId}/audits", "createAudit", c.audits.createAudit},
		{"GET", "/v1/projects/{projectId}/audits", "listProjectAudits", c.audits.listProjectAudits},
		{"GET", "/v1/audits/{auditId}", "getAudit", c.audits.getAudit},
		{"DELETE", "/v1/audits/{auditId}", "deleteAudit", c.audits.deleteAudit},
		{"POST", "/v1/audits/{auditId}/start", "startAudit", c.audits.startAudit},
		{"POST", "/v1/audits/{auditId}/pause", "pauseAudit", c.audits.pauseAudit},
		{"POST", "/v1/audits/{auditId}/resume", "resumeAudit", c.audits.resumeAudit},
		{"POST", "/v1/audits/{auditId}/cancel", "cancelAudit", c.audits.cancelAudit},
		{"GET", "/v1/audits/{auditId}/events", "listAuditEvents", c.audits.listAuditEvents},
		{"GET", "/v1/audits/{auditId}/items", "listAuditItems", c.audits.listAuditItems},
		{"GET", "/v1/audits/{auditId}/coverage", "listAuditCoverage", c.audits.listAuditCoverage},
		{"GET", "/v1/audits/{auditId}/workspace", "getAuditWorkspace", c.audits.getAuditWorkspace},
		{"GET", "/v1/audits/{auditId}/report", "getAuditReport", c.audits.getAuditReport},
		{"GET", "/v1/audits/{auditId}/findings", "listAuditFindings", c.audits.listAuditFindings},
		{"GET", "/v1/audits/{auditId}/findings/{findingId}", "getAuditFinding", c.audits.getAuditFinding},
		{"GET", "/v1/audits/{auditId}/findings/{findingId}/provenance", "listAuditFindingProvenance", c.audits.listAuditFindingProvenance},
		{"POST", "/v1/audits/{auditId}/findings/{findingId}/reviews", "createAuditFindingReview", c.audits.createAuditFindingReview},
		{"GET", "/v1/audits/{auditId}/reviews", "listAuditReviews", c.audits.listAuditReviews},
		{"GET", "/v1/audits/{auditId}/reviews/{requestId}", "getAuditReview", c.audits.getAuditReview},
		{"POST", "/v1/audits/{auditId}/reviews/{requestId}/decisions", "decideAuditReview", c.audits.decideAuditReview},

		{"GET", "/v1/runs/{runId}/finding-proposals", "listRunFindingProposals", c.findings.listRunFindingProposals},
		{"GET", "/v1/audits/{auditId}/finding-proposals", "listAuditFindingProposals", c.findings.listAuditFindingProposals},
		{"POST", "/v1/audits/{auditId}/finding-proposal-imports", "importAuditFindingProposal", c.findings.importAuditFindingProposal},
		{"POST", "/v1/finding-collections", "publishFindingCollection", c.findings.publishFindingCollection},

		{"GET", "/v1/eval-capabilities", "getEvalCapabilities", evalRoute(c.evals.evalCapabilities)},
		{"GET", "/v1/projects/{projectId}/eval-datasets", "listEvalDatasets", evalRoute(c.evals.listEvalDatasets)},
		{"POST", "/v1/projects/{projectId}/eval-datasets", "importEvalDataset", evalRoute(c.evals.importEvalDataset)},
		{"GET", "/v1/projects/{projectId}/eval-datasets/{datasetId}/revisions/{revision}/cases",
			"listEvalCases", evalRoute(c.evals.listEvalCases)},
		{"POST", "/v1/projects/{projectId}/eval-experiments", "createEvalExperiment", evalRoute(c.evals.createEvalExperiment)},
		{"GET", "/v1/eval-experiments", "listEvalExperiments", evalRoute(c.evals.listEvalExperiments)},
		{"GET", "/v1/eval-experiments/{id}", "getEvalExperiment", evalRoute(c.evals.getEvalExperiment)},
		{"PATCH", "/v1/eval-experiments/{id}", "updateEvalDraft", evalRoute(c.evals.updateEvalDraft)},
		{"DELETE", "/v1/eval-experiments/{id}", "deleteEvalExperiment", evalRoute(c.evals.deleteEvalExperiment)},
		{"POST", "/v1/eval-experiments/{id}/commands", "commandEvalExperiment", evalRoute(c.evals.commandEvalExperiment)},
		{"GET", "/v1/eval-experiments/{id}/commands/{commandId}", "getEvalCommand", evalRoute(c.evals.getEvalCommand)},
		{"GET", "/v1/eval-experiments/{id}/members", "listEvalMembers", evalRoute(c.evals.listEvalMembers)},
		{"POST", "/v1/eval-experiments/{id}/members/{memberId}/submissions", "submitEvalMember", evalRoute(c.evals.submitEvalMember)},
		{"POST", "/v1/eval-experiments/{id}/members/{memberId}/results", "ingestEvalResult", evalRoute(c.evals.ingestEvalResult)},
		{"POST", "/v1/eval-experiments/{id}/members/{memberId}/assessments", "assessEvalMember", evalRoute(c.evals.assessEvalMember)},
		{"GET", "/v1/eval-experiments/{id}/members/{memberId}/review", "reviewEvalMember", evalRoute(c.evals.reviewEvalMember)},
		{"GET", "/v1/eval-experiments/{id}/members/{memberId}/executions",
			"listEvalMemberExecutions", evalRoute(c.evals.listEvalMemberExecutions)},
		{"POST", "/v1/eval-experiments/{id}/selections", "selectEvalRecords", evalRoute(c.evals.selectEvalRecords)},
		{"GET", "/v1/eval-experiments/{id}/pairs", "listEvalPairs", evalRoute(c.evals.listEvalPairs)},
		{"GET", "/v1/eval-experiments/{id}/pairs/{pairId}", "getEvalPair", evalRoute(c.evals.getEvalPair)},
		{"GET", "/v1/eval-experiments/{id}/charts/{chart}", "getEvalChart", evalRoute(c.evals.getEvalChart)},
		{"GET", "/v1/eval-experiments/{id}/report", "getEvalReport", evalRoute(c.evals.getEvalReport)},

		{"GET", "/v1/operations/snapshot", "getOperationsSnapshot", c.operations.getOperationsSnapshot},
		{"GET", "/v1/operations/runtime-agents", "listRuntimeAgents", c.operations.listRuntimeAgents},
		{"GET", "/v1/operations/allocations", "listAllocations", c.operations.listAllocations},
		{"GET", "/v1/operations/allocation-history", "listAllocationResourceHistory", c.operations.listAllocationResourceHistory},
		{"GET", "/v1/operations/performance", "getPerformance", c.operations.getPerformance},
		{"GET", "/v1/operations/performance/history", "getPerformanceHistory", c.operations.getPerformanceHistory},
		{"GET", "/v1/operations/settings/scheduler", "getSchedulerSettings", c.operations.getSchedulerSettings},
		{"PUT", "/v1/operations/settings/scheduler", "putSchedulerSettings", c.operations.putSchedulerSettings},

		{"GET", "/v1/operations/credentials", "listCredentials", c.credentials.listCredentials},
		{"POST", "/v1/operations/credentials", "createCredential", c.credentials.createCredential},
		{"GET", "/v1/operations/credentials/{credentialId}", "getCredential", c.credentials.getCredential},
		{"DELETE", "/v1/operations/credentials/{credentialId}", "deleteCredential", c.credentials.deleteCredential},

		{"GET", "/v1/operations/runtime-configs", "listRuntimeConfigs", c.runtimeConfiguration.listRuntimeConfigs},
		{"POST", "/v1/operations/runtime-configs", "publishRuntimeConfig", c.runtimeConfiguration.publishRuntimeConfig},
		{"GET", "/v1/operations/runtime-configs/{name}/versions/{version}", "getRuntimeConfig", c.runtimeConfiguration.getRuntimeConfig},
		{"GET", "/v1/operations/runtime-labels", "listRuntimeLabels", c.runtimeConfiguration.listRuntimeLabels},
		{"GET", "/v1/operations/runtime-labels/{label}", "getRuntimeLabel", c.runtimeConfiguration.getRuntimeLabel},
		{"PUT", "/v1/operations/runtime-labels/{label}", "putRuntimeLabel", c.runtimeConfiguration.putRuntimeLabel},
		{"DELETE", "/v1/operations/runtime-labels/{label}", "deleteRuntimeLabel", c.runtimeConfiguration.deleteRuntimeLabel},
		{"GET", "/v1/operations/runtime-credentials", "listRuntimeCredentials", c.runtimeConfiguration.listRuntimeCredentials},
		{"POST", "/v1/operations/runtime-credentials", "createRuntimeCredential", c.runtimeConfiguration.createRuntimeCredential},
		{"GET", "/v1/operations/runtime-credentials/{credentialId}", "getRuntimeCredential", c.runtimeConfiguration.getRuntimeCredential},
		{"DELETE", "/v1/operations/runtime-credentials/{credentialId}", "deleteRuntimeCredential", c.runtimeConfiguration.deleteRuntimeCredential},
		{"GET", "/v1/operations/runtime-agent-principals", "listRuntimeAgentPrincipals", c.runtimeConfiguration.listRuntimeAgentPrincipals},
		{"GET", "/v1/operations/runtime-agent-principals/{runtimeAgentId}",
			"getRuntimeAgentPrincipal", c.runtimeConfiguration.getRuntimeAgentPrincipal},
		{"DELETE", "/v1/operations/runtime-agent-principals/{runtimeAgentId}",
			"deleteRuntimeAgentPrincipal", c.runtimeConfiguration.deleteRuntimeAgentPrincipal},
		{"PUT", "/v1/operations/runtime-agent-principals/{runtimeAgentId}/labels",
			"replaceRuntimeAgentPrincipalLabels", c.runtimeConfiguration.putRuntimeAgentLabels},
	}
}

// register serves every route, answers another method of a route's path with
// 405 and any other path with 404.
func (c contextHandlers) register(mux *http.ServeMux) {
	paths := map[string]bool{}
	for _, route := range c.routes() {
		mux.HandleFunc(route.method+" "+route.path, route.handler)
		paths[route.path] = true
	}
	for path := range paths {
		mux.HandleFunc(path, methodNotAllowed)
	}
	mux.HandleFunc("/", notFound)
}

// evalRoute keeps every Eval response out of caches and refuses HEAD, which
// the Eval documents do not define.
func evalRoute(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		if rejectHead(w, r) {
			return
		}
		next(w, r)
	}
}
