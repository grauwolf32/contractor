package public

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/getkin/kin-openapi/routers"
	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/telemetry"
)

// TestPublicOperationsConformToOpenAPI validates a response of each public
// operation that the other contract tests of this package serve only on
// PostgreSQL or not at all, so the operation coverage that TestMain enforces
// holds without CONTRACTOR_TEST_DATABASE_URL too. Every operation here
// answers with a success response, except the Eval operations, which answer
// from a fixture that serves no Eval; the Eval tests on PostgreSQL validate
// their success responses.
func TestPublicOperationsConformToOpenAPI(t *testing.T) {
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatalf("build contract router: %v", err)
	}
	t.Run("audits and findings", func(t *testing.T) { conformAuditOperations(t, router) })
	t.Run("projects and queue", func(t *testing.T) { conformProjectAndQueueOperations(t, router) })
	t.Run("runs", func(t *testing.T) { conformRunOperations(t, router) })
	t.Run("catalog and operations", func(t *testing.T) { conformCatalogAndOperations(t, router) })
	t.Run("settings and events", func(t *testing.T) { conformSettingsAndEvents(t, router) })
	t.Run("evals", func(t *testing.T) { conformEvalOperations(t, router) })
}

// contractCall serves request, validates the response, and the request too
// when a success is expected, against the public contract, and fails unless
// the response status is want.
func contractCall(
	t *testing.T, router routers.Router, handler http.Handler, request *http.Request, want int,
) *httptest.ResponseRecorder {
	t.Helper()
	response := serveAndValidatePublicContract(t, router, handler, request, want < http.StatusBadRequest)
	if response.Code != want {
		t.Fatalf("%s %s = %d, want %d: %s",
			request.Method, request.URL.RequestURI(), response.Code, want, response.Body.String())
	}
	return response
}

// jsonContractRequest is a bearer-authenticated request with an optional
// JSON body and the given header name and value pairs.
func jsonContractRequest(method, path, body string, headers ...string) *http.Request {
	request := newPublicContractRequest(method, path, []byte(body))
	if body != "" {
		request.Header.Set("Content-Type", "application/json")
	}
	for index := 0; index+1 < len(headers); index += 2 {
		request.Header.Set(headers[index], headers[index+1])
	}
	return request
}

func conformAuditOperations(t *testing.T, router routers.Router) {
	now := time.Date(2026, 8, 29, 11, 0, 0, 0, time.UTC)
	profile, err := loadPublicAuditConfiguration(t).AuditProfile("public-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	revision := "revision-1"
	input := auditstore.ExactArtifact{
		Ref:    contracts.ArtifactRef{Namespace: "inputs", Name: "checklist", Revision: &revision},
		Digest: auditHandlerDigest("checklist"), MediaType: "application/json", SizeBytes: 64,
	}
	selection, err := json.Marshal(auditservice.DraftSelection{
		Schema:        auditservice.DraftSelectionSchema,
		Inputs:        map[string]auditstore.ExactArtifact{"checklist": input},
		RuntimeLabels: []string{}, Scope: auditservice.Scope{Objective: "Review the service"},
	})
	if err != nil {
		t.Fatal(err)
	}
	draft := auditstore.Audit{
		AuditID: "audit-one", OwnerID: "user-1", ProjectID: "project_fixed",
		Profile: auditstore.ProfileIdentity{
			Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest,
		},
		InputSelection: selection, State: auditstore.AuditDraft, Revision: 1,
		Dispatch: auditstore.DispatchOpen, Hold: auditstore.HoldPending,
		Limits: auditstore.Limits{
			MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 10, MaxItemsTotal: 10,
			MaxSubmittedRuns: 20, MaxItemRunAttempts: 2, MaxEvidenceBytes: 1 << 20,
		},
		CreatedAt: now, UpdatedAt: now,
	}
	roundID := "round-one"
	active := draft
	active.State, active.Revision, active.Hold = auditstore.AuditActive, 2, auditstore.HoldHeld
	active.CurrentRoundID, active.StartedAt = &roundID, &now
	item := auditstore.Item{
		ItemID: "item-one", AuditID: active.AuditID, RoundID: roundID, ItemKey: "check-one",
		Kind: "checklist", SubjectKey: "check-one@1", Task: input,
		Origin: auditstore.ItemOrigin{
			Schema: auditstore.ItemOriginSchema, SourceRef: &input.Ref, SourceContentDigest: input.Digest,
			SourceMediaType: input.MediaType, CanonicalInventoryDigest: auditHandlerDigest("inventory"),
			EntryKey: "check-one",
		},
		WorkflowRole: "check", State: auditstore.ItemSubmitted, ApprovalKind: auditstore.ItemApprovalNone,
		CreatedAt: now, UpdatedAt: now,
	}
	runID := "run-one"
	attempt := auditstore.ItemAttempt{
		ExecutionItemID: "execution-item-one", ExecutionID: "execution-one", ItemID: item.ItemID,
		ItemAttempt: 1, Role: auditstore.ExecutionCheck, State: auditstore.ItemSubmitted,
		RunID: &runID, CreatedAt: now,
	}
	receipt := conformanceFindingReceipt(now)
	verdict, severity := auditstore.VerdictTruePositive, auditstore.SeverityHigh
	review := auditstore.ReviewRequest{
		RequestID: "review-one", AuditID: active.AuditID, FindingID: "finding-one",
		SubjectKind: auditstore.ReviewSubjectFinding, SubjectID: "finding-one",
		Kind: auditstore.FindingReviewKind, SubjectRevision: 1,
		SubjectDigest:    auditHandlerDigest("finding-subject"),
		RequestedActions: []auditstore.ReviewRequestedAction{auditstore.ReviewRequestedAction(verdict)},
		State:            auditstore.ReviewPending, Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
	decided := review
	decided.State, decided.Revision = auditstore.ReviewDecided, 2
	decided.Decision = &auditstore.ReviewDecision{
		DecisionID: "decision-one", RequestID: review.RequestID, AuditID: active.AuditID,
		FindingID: "finding-one", ActorID: "user-1", Verdict: verdict, Severity: &severity,
		Rationale: "Confirmed from exact evidence.", SubjectRevision: 1,
		SubjectDigest: review.SubjectDigest, CreatedAt: now,
	}
	standardRevision := "standard-r1"
	management := &fakeAuditManagement{
		profiles: []auditservice.ProfileProjection{{
			Profile: profile, Compatibility: auditservice.ProfileCompatibility(profile),
		}},
		standards: []auditstandards.PackageProjection{{
			Reference: auditstandards.Reference{Scheme: "example", Version: "1"},
			Title:     "Example", Description: "Example standard.",
			Source: auditstandards.Source{Name: "Source", URL: "https://example.invalid/source"},
			License: auditstandards.License{
				ID: "CC-BY-4.0", URL: "https://creativecommons.org/licenses/by/4.0/",
				Attribution: "Example authors", Disclosure: auditstandards.DisclosureFull,
			},
			Digest: auditHandlerDigest("standard"), Artifact: contracts.ArtifactRef{
				Namespace: auditstandards.CatalogNamespace, Name: "std-example", Revision: &standardRevision,
			},
			EntryCount: 1, MappingCount: 1, EvidenceContractCount: 1,
		}},
		audit: draft,
		started: auditservice.StartedAudit{
			Audit: active,
			Round: auditstore.Round{
				RoundID: roundID, AuditID: active.AuditID, Ordinal: 1, Manifest: input,
				State: auditstore.RoundExecuting, ExpectedItemCount: 1, Revision: 1,
				CreatedAt: now, UpdatedAt: now,
			},
			Items: []auditstore.Item{item},
		},
		attempts: map[string][]auditstore.ItemAttempt{item.ItemID: {attempt}},
		coverage: auditservice.CoveragePage{AuditRevision: active.Revision, Rows: []auditstore.CoverageRow{{
			AuditID: active.AuditID, RoundID: roundID, ItemID: item.ItemID, ItemKey: item.ItemKey,
			SubjectKey: item.SubjectKey, Task: input, UpdatedAt: now,
			Coverage: auditstore.Coverage{
				Status: auditstore.CoverageNotTested, Requested: []string{}, Completed: []string{}, Gaps: []string{},
			},
		}}},
		events: auditservice.EventPage{ThroughSequence: 1, Total: 1, Items: []auditservice.Event{{
			AuditID: active.AuditID, Sequence: 1, Kind: "audit.created", EntityID: active.AuditID,
			Summary: map[string]any{"state": "draft"}, CreatedAt: now,
		}}},
		findings: []auditservice.Finding{{
			FindingID: "finding-one", AuditID: active.AuditID, State: auditstore.FindingProposed,
			FirstProposal: receipt, Revision: 1, CreatedAt: now, UpdatedAt: now,
		}},
		reviews: []auditstore.ReviewRequest{review},
		provenance: []auditservice.FindingProvenance{{
			RecordID: "proposal:one", Kind: auditstore.ProvenanceSourceProposal,
			ReceiptID: receipt.ReceiptID, Relation: "first", Proposal: receipt.Proposal,
			Origin: receipt.Origin, CreatedAt: now,
		}},
	}
	proposals := &fakeFindingProposalManagement{receipts: []findingintake.Receipt{receipt}}
	collectionRevision := "collection-r1"
	collections := publishedFindingCollections{published: findingintake.PublishedCollection{
		Artifact: findingintake.ExactArtifact{
			Ref: contracts.ArtifactRef{
				Namespace: "finding-collections", Name: "collection-one", Revision: &collectionRevision,
			},
			Digest: auditHandlerDigest("collection"), MediaType: "application/zip", SizeBytes: 128,
		},
		SnapshotAt: now.Format(time.RFC3339Nano), EntryCount: 1,
	}}
	fixture := newHandlerFixtureWithAuth(
		t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(services *testServices) {
			services.Audits, services.FindingProposals = management, proposals
			services.FindingCollections = collections
		},
	)
	fixture.runs.runs[runID] = runstore.WorkflowRun{
		RunID: runID, OwnerID: "user-1", State: runstore.RunSucceeded,
	}
	call := func(request *http.Request, want int) {
		t.Helper()
		contractCall(t, router, fixture.handler, request, want)
	}

	call(jsonContractRequest(http.MethodPost, "/v1/projects", `{"kind":"project","name":"Audited service"}`,
		"Idempotency-Key", "audited-project"), http.StatusCreated)
	for _, path := range []string{
		"/v1/audit-profiles", "/v1/audit-profiles/public-checklist/versions/1",
		"/v1/audit-standards", "/v1/audit-standards/example/versions/1",
	} {
		call(jsonContractRequest(http.MethodGet, path, ""), http.StatusOK)
	}
	call(jsonContractRequest(http.MethodPost, "/v1/projects/project_fixed/audits",
		`{"profile":{"name":"public-checklist","version":"1"},`+
			`"inputs":{"checklist":{"namespace":"inputs","name":"checklist","revision":"revision-1"}},`+
			`"runtimeLabels":[],"scope":{"objective":"Review the service"}}`,
		"Idempotency-Key", "create-audit"), http.StatusCreated)
	for _, path := range []string{"/v1/projects/project_fixed/audits", "/v1/audits", "/v1/audits/audit-one"} {
		call(jsonContractRequest(http.MethodGet, path, ""), http.StatusOK)
	}
	call(jsonContractRequest(http.MethodPost, "/v1/audits/audit-one/start", "",
		"Idempotency-Key", "start-audit", "If-Match", `"1"`), http.StatusOK)

	management.audit = active
	management.artifactRead = artifacts.ReadResult{Ref: input.Ref, Payload: artifacts.Payload{MediaType: "text/plain", Data: []byte("Accepted output")}}
	call(jsonContractRequest(http.MethodGet, "/v1/audits/audit-one/artifacts?key=prepare%3Aexecution%3Aapi&revision=revision-1", ""), http.StatusOK)
	for _, path := range []string{
		"/v1/audits/audit-one/items", "/v1/audits/audit-one/coverage", "/v1/audits/audit-one/events",
		"/v1/audits/audit-one/workspace", "/v1/audits/audit-one/finding-proposals",
		"/v1/runs/run-one/finding-proposals", "/v1/audits/audit-one/findings",
		"/v1/audits/audit-one/findings/finding-one", "/v1/audits/audit-one/findings/finding-one/provenance",
		"/v1/audits/audit-one/reviews", "/v1/audits/audit-one/reviews/review-one",
		"/v1/findings", "/v1/reviews",
	} {
		call(jsonContractRequest(http.MethodGet, path, ""), http.StatusOK)
	}
	call(jsonContractRequest(http.MethodPost, "/v1/audits/audit-one/finding-proposal-imports",
		`{"runId":"run-one","proposal":{"namespace":"finding-proposals","name":"proposal-one","revision":"proposal-r1"}}`),
		http.StatusCreated)
	call(jsonContractRequest(http.MethodPost, "/v1/finding-collections",
		`{"clientKey":"collection-one","sources":[{"kind":"run","id":"run-one","receiptIds":["receipt-one"],"findings":[]}]}`),
		http.StatusCreated)
	call(jsonContractRequest(http.MethodPost, "/v1/audits/audit-one/findings/finding-one/reviews", `{}`,
		"Idempotency-Key", "review-finding", "If-Match", `"1"`), http.StatusCreated)

	management.reviews = []auditstore.ReviewRequest{decided}
	call(jsonContractRequest(http.MethodPost, "/v1/audits/audit-one/reviews/review-one/decisions",
		`{"verdict":"true_positive","severity":"high","rationale":"Confirmed from exact evidence."}`,
		"Idempotency-Key", "decide-review", "If-Match", `"1"`), http.StatusOK)
	for _, lifecycle := range []struct {
		method, path, key string
		status            int
	}{
		{http.MethodPost, "/v1/audits/audit-one/pause", "pause-audit", http.StatusOK},
		{http.MethodPost, "/v1/audits/audit-one/resume", "resume-audit", http.StatusOK},
		{http.MethodPost, "/v1/audits/audit-one/cancel", "cancel-audit", http.StatusAccepted},
		{http.MethodDelete, "/v1/audits/audit-one", "delete-audit", http.StatusAccepted},
	} {
		call(jsonContractRequest(lifecycle.method, lifecycle.path, "",
			"Idempotency-Key", lifecycle.key, "If-Match", `"2"`), lifecycle.status)
	}
}

// conformanceFindingReceipt is one complete finding proposal receipt of
// run-one.
func conformanceFindingReceipt(createdAt time.Time) findingintake.Receipt {
	revision := "proposal-r1"
	origin := findingintake.Origin{
		RunID: "run-one", StageExecutionID: "stage-one", AllocationID: "allocation-one",
		InvocationID: "invocation-one", LogicalAgentName: "reviewer",
		Workflow: findingintake.WorkflowOrigin{
			Name: "artifact-copy", Version: "1", SchemaVersion: contracts.APIVersion,
			ClosureDigest: auditHandlerDigest("closure"),
		},
	}
	origin.Workflow.ConfigurationRef.Name, origin.Workflow.ConfigurationRef.Version = "artifact-copy", "1"
	return findingintake.Receipt{
		ReceiptID: "receipt-one", ProposalID: "proposal-one", RequestDigest: auditHandlerDigest("request"),
		ClientKey: "candidate-one",
		Proposal: findingintake.ExactArtifact{
			Ref: contracts.ArtifactRef{
				Namespace: "finding-proposals", Name: "proposal-one", Revision: &revision,
			},
			Digest: auditHandlerDigest("proposal"), MediaType: "application/json", SizeBytes: 256,
		},
		Document: auditdomain.FindingProposal{
			Schema: auditdomain.FindingProposalSchema, ClientKey: "candidate-one",
			Title: "Missing authorization", Description: "The handler serves another owner's record.",
			Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{},
			EvidenceIDs: []string{}, ProposedChecks: []auditdomain.ProposedCheck{},
			SeveritySuggestion: "high", Limitations: []string{},
		},
		Evidence: []findingintake.ExactArtifact{}, Origin: origin,
		Retention: findingintake.RetentionSourceHeld, AuditHolds: []findingintake.AuditHold{},
		CreatedAt: createdAt,
	}
}

// publishedFindingCollections publishes every selection as one collection.
type publishedFindingCollections struct {
	published findingintake.PublishedCollection
}

func (c publishedFindingCollections) PublishCollection(
	context.Context, findingintake.PublishCollectionParams,
) (findingintake.PublishedCollection, error) {
	return c.published, nil
}

func conformProjectAndQueueOperations(t *testing.T, router routers.Router) {
	fixture := newHandlerFixture(t)
	call := func(request *http.Request, want int) {
		t.Helper()
		contractCall(t, router, fixture.handler, request, want)
	}
	call(jsonContractRequest(http.MethodPost, "/v1/projects", `{"kind":"project","name":"Retired service"}`,
		"Idempotency-Key", "retired-project"), http.StatusCreated)
	call(jsonContractRequest(http.MethodDelete, "/v1/projects/project_fixed", "", "If-Match", `"1"`),
		http.StatusAccepted)
	call(jsonContractRequest(http.MethodGet, "/v1/queue/control", ""), http.StatusOK)
	call(jsonContractRequest(http.MethodPut, "/v1/queue/control", `{"paused":true}`, "If-Match", `"0"`),
		http.StatusOK)
}

// acceptedGatewayRetry accepts the gateway retry of every Run.
type acceptedGatewayRetry struct{}

func (acceptedGatewayRetry) Retry(context.Context, string, string) error { return nil }

func conformRunOperations(t *testing.T, router routers.Router) {
	fixture := newHandlerFixtureWithAuth(
		t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(services *testServices) { services.GatewayRecovery = acceptedGatewayRetry{} },
	)
	call := func(request *http.Request, want int) {
		t.Helper()
		contractCall(t, router, fixture.handler, request, want)
	}
	user, err := fixture.artifacts.User("user-1")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := user.Write(
		t.Context(), contracts.ArtifactRef{Namespace: "projects", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("source")}, nil,
	); err != nil {
		t.Fatal(err)
	}
	call(jsonContractRequest(http.MethodPost, "/v1/runs",
		`{"workflow":"artifact-copy@1","parameters":{"objective":"copy"},"artifacts":{"source":{"namespace":"projects","name":"source"}}}`,
		"Idempotency-Key", "conformance-run"), http.StatusAccepted)
	call(jsonContractRequest(http.MethodPost, "/v1/runs/run_fixed/retry-gateway", `{}`), http.StatusAccepted)

	finished := fixture.runs.runs["run_fixed"]
	finished.State, finished.FinishedAt = runstore.RunSucceeded, timePointer(finished.CreatedAt.Add(time.Minute))
	fixture.runs.runs[finished.RunID] = finished
	call(jsonContractRequest(http.MethodGet, "/v1/runs/run_fixed/repeat-draft", ""), http.StatusOK)

	fixture.runs.runs["run-resume"] = runstore.WorkflowRun{
		RunID: "run-resume", OwnerID: "user-1", State: runstore.RunFailed,
	}
	fixture.runs.executions["run-resume"] = []runstore.StageExecution{{
		RunID: "run-resume", StageExecutionID: "failed-stage", StageName: "copy",
		State: runstore.StageFailed, Attempt: 1,
	}}
	call(jsonContractRequest(http.MethodPost, "/v1/runs/run-resume/resume", `{"stageExecutionId":"failed-stage"}`),
		http.StatusAccepted)
}

func conformCatalogAndOperations(t *testing.T, router routers.Router) {
	finished := time.Date(2026, 8, 29, 11, 0, 0, 0, time.UTC)
	fixture := newHandlerFixtureWithAuth(
		t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(services *testServices) {
			services.AllocationResources = &fakeAllocationResourceReader{
				byStage: map[string][]telemetry.AllocationResourceSummary{},
				items: []telemetry.AllocationResourceSummary{{
					AllocationID: "allocation-1", RunID: "run-1", StageExecutionID: "stage-1", Stage: "analyze",
					LogicalAgent: "worker", Outcome: "succeeded", FinishedAt: finished,
					CollectionPolicy: reporting.PerformanceCollectionDisabled,
					Status:           telemetry.AllocationResourceDisabled,
				}},
			}
		},
	)
	for _, path := range []string{
		"/v1/configurations/agent-templates/artifact_builder/versions/1/workflow-bindings",
		"/v1/operations/performance",
		"/v1/operations/allocation-history?limit=1",
	} {
		contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodGet, path, ""), http.StatusOK)
	}
	// The contract declares the step parameter as an enum without a type,
	// which the request validator cannot decode; the response is validated.
	history := serveAndValidatePublicContract(t, router, fixture.handler, jsonContractRequest(http.MethodGet,
		"/v1/operations/performance/history?from=2026-08-29T11:30:00Z&to=2026-08-29T12:00:00Z&step=15s", ""), false)
	if history.Code != http.StatusOK {
		t.Fatalf("performance history = %d %s", history.Code, history.Body.String())
	}
}

func conformSettingsAndEvents(t *testing.T, router routers.Router) {
	fixture := newHandlerFixtureWithAuth(
		t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(services *testServices) { services.GitKeys = &requestContractGit{} },
	)
	contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodGet, "/v1/settings/git-key", ""),
		http.StatusOK)
	// The upgrade itself needs a hijackable connection; the event handler
	// tests complete it on a real server.
	upgrade := jsonContractRequest(http.MethodGet, "/v1/events/ws", "",
		"Origin", testBrowserOrigin, "Sec-WebSocket-Protocol", "contractor.events.v1")
	upgrade.Header.Del("Authorization")
	contractCall(t, router, fixture.handler, upgrade, http.StatusUnauthorized)
}

func conformEvalOperations(t *testing.T, router routers.Router) {
	fixture := newHandlerFixture(t)
	contractCall(t, router, fixture.handler, jsonContractRequest(http.MethodGet, "/v1/eval-capabilities", ""),
		http.StatusOK)
	const experiment = "/v1/eval-experiments/experiment-one"
	for _, operation := range []struct{ method, path, body string }{
		{http.MethodGet, "/v1/projects/evaluation/eval-datasets", ""},
		{http.MethodPost, "/v1/projects/evaluation/eval-datasets", `{}`},
		{http.MethodGet, "/v1/projects/evaluation/eval-datasets/dataset-one/revisions/1/cases", ""},
		{http.MethodPost, "/v1/projects/evaluation/eval-experiments", `{}`},
		{http.MethodGet, "/v1/eval-experiments", ""},
		{http.MethodGet, experiment, ""},
		{http.MethodPatch, experiment, `{}`},
		{http.MethodDelete, experiment, `{}`},
		{http.MethodPost, experiment + "/commands", `{}`},
		{http.MethodGet, experiment + "/commands/command-one", ""},
		{http.MethodGet, experiment + "/members", ""},
		{http.MethodPost, experiment + "/members/member-one/submissions", `{}`},
		{http.MethodPost, experiment + "/members/member-one/results", `{}`},
		{http.MethodPost, experiment + "/members/member-one/assessments", `{}`},
		{http.MethodGet, experiment + "/members/member-one/review", ""},
		{http.MethodGet, experiment + "/members/member-one/executions", ""},
		{http.MethodPost, experiment + "/selections", `{}`},
		{http.MethodGet, experiment + "/pairs", ""},
		{http.MethodGet, experiment + "/pairs/pair-one", ""},
		{http.MethodGet, experiment + "/charts/score", ""},
		{http.MethodGet, experiment + "/report", ""},
	} {
		request := jsonContractRequest(operation.method, operation.path, operation.body,
			"Idempotency-Key", "eval-conformance")
		response := serveAndValidatePublicContract(t, router, fixture.handler, request, false)
		if response.Code < http.StatusBadRequest {
			t.Fatalf("%s %s succeeded without an Eval service: %d %s",
				operation.method, operation.path, response.Code, response.Body.String())
		}
	}
}
