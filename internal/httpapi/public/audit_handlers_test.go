package public

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
)

func TestAuditProfileHandlersExposeCompatibilityAndExactDetail(t *testing.T) {
	profiles := &fakeAuditManagement{profiles: []auditservice.ProfileProjection{{
		Profile: config.ResolvedAuditProfile{
			Ref:  config.AuditProfileRef{Name: "checklist", Version: "1", Digest: auditHandlerDigest("profile")},
			Mode: config.AuditModeCustomChecklist, Standards: []config.AuditStandardRef{},
			Inputs: map[string]config.AuditProfileInput{
				"checklist": {Required: true, MediaTypes: []string{"application/json"}},
			},
			Inventory: config.AuditInventory{Implementation: "checklist@1", SourceInput: "checklist", ItemWorkflowRole: "check"},
			Workflows: map[string]config.ResolvedAuditWorkflowBinding{},
			Execution: config.AuditExecutionPolicy{MaxRounds: 1, BatchSize: 1},
			Interaction: config.AuditInteractionPolicy{
				ActiveChecks:        config.AuditActiveChecksProhibited,
				FindingConfirmation: config.AuditFindingDisabled,
				NotApplicable:       config.AuditNotApplicableProfileRule,
				ReportAcceptance:    config.AuditReportAutomatic,
			},
		},
		Compatibility: auditservice.Compatibility{
			ServerCompatible: false, RequiresInputValidation: true,
			Reasons: []auditservice.CompatibilityReason{auditservice.ReasonManualApplicabilityUnsupported},
		},
	}}}
	h := auditTestHandler(profiles)

	list := auditAuthenticatedRequest(http.MethodGet, "/v1/audit-profiles", nil)
	response := httptest.NewRecorder()
	h.listAuditProfiles(response, list)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"serverCompatible":false`) ||
		!strings.Contains(response.Body.String(), `"compatibilityReasons":["manual_applicability_unsupported"]`) ||
		strings.Contains(response.Body.String(), `"workflows"`) {
		t.Fatalf("profile list = %d %s", response.Code, response.Body.String())
	}

	detail := auditAuthenticatedRequest(http.MethodGet, "/v1/audit-profiles/checklist/versions/1", nil)
	detail.SetPathValue("name", "checklist")
	detail.SetPathValue("version", "1")
	detailResponse := httptest.NewRecorder()
	h.getAuditProfile(detailResponse, detail)
	if detailResponse.Code != http.StatusOK || detailResponse.Header().Get("ETag") != `"`+auditHandlerDigest("profile")+`"` ||
		!strings.Contains(detailResponse.Body.String(), `"mode":"custom-checklist"`) {
		t.Fatalf("profile detail = %d headers=%v body=%s", detailResponse.Code, detailResponse.Header(), detailResponse.Body.String())
	}
}

func TestAuditCreateStartQueryAndStableErrors(t *testing.T) {
	now := time.Date(2026, 9, 5, 19, 0, 0, 0, time.UTC)
	revision := "input-r1"
	selection := auditservice.DraftSelection{
		Schema: auditservice.DraftSelectionSchema,
		Inputs: map[string]auditstore.ExactArtifact{
			"checklist": {
				Ref:    contracts.ArtifactRef{Namespace: "inputs", Name: "checklist", Revision: &revision},
				Digest: auditHandlerDigest("input"), MediaType: "application/json", SizeBytes: 10,
			},
		},
		RuntimeLabels: []string{}, Scope: auditservice.Scope{Objective: "Review"},
	}
	encodedSelection, _ := json.Marshal(selection)
	audit := auditstore.Audit{
		AuditID: "audit-fixed", OwnerID: "user-1", ProjectID: "project-one",
		Profile:         auditstore.ProfileIdentity{Name: "checklist", Version: "1", Digest: auditHandlerDigest("profile")},
		ProfileSnapshot: json.RawMessage(`{"ref":{"name":"checklist","version":"1","digest":"` + auditHandlerDigest("profile") + `"}}`),
		InputSelection:  encodedSelection, State: auditstore.AuditDraft, Revision: 1,
		Dispatch: auditstore.DispatchOpen, Hold: auditstore.HoldPending,
		Limits:    auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 10, MaxItemsTotal: 10, MaxSubmittedRuns: 10, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024},
		CreatedAt: now, UpdatedAt: now,
	}
	management := &fakeAuditManagement{audit: audit}
	h := auditTestHandler(management)

	body := []byte(`{"profile":{"name":"checklist","version":"1"},"inputs":{"checklist":{"namespace":"inputs","name":"checklist","revision":"input-r1"}},"runtimeLabels":[],"scope":{"objective":"Review"}}`)
	create := auditAuthenticatedRequest(http.MethodPost, "/v1/projects/project-one/audits", body)
	create.SetPathValue("projectId", "project-one")
	create.Header.Set("Content-Type", "application/json")
	create.Header.Set("Idempotency-Key", "create-audit")
	created := httptest.NewRecorder()
	h.createAudit(created, create)
	if created.Code != http.StatusCreated || created.Header().Get("ETag") != `"1"` ||
		management.created.OwnerID != "user-1" || management.created.ProjectID != "project-one" ||
		management.created.RequestDigest == "" {
		t.Fatalf("create Audit = %d headers=%v body=%s params=%+v", created.Code, created.Header(), created.Body.String(), management.created)
	}

	management.audit.State = auditstore.AuditActive
	management.audit.Revision = 2
	management.audit.Hold = auditstore.HoldHeld
	roundID := "round-one"
	management.audit.CurrentRoundID = &roundID
	management.started = auditservice.StartedAudit{
		Audit: management.audit,
		Round: auditstore.Round{RoundID: roundID, AuditID: management.audit.AuditID, Ordinal: 1,
			Manifest: selection.Inputs["checklist"], State: auditstore.RoundAccepted,
			ExpectedItemCount: 1, Revision: 1, CreatedAt: now, UpdatedAt: now},
		Items: []auditstore.Item{{
			ItemID: "item-one", AuditID: management.audit.AuditID, RoundID: roundID,
			ItemKey: "check-one", Ordinal: 0, Kind: "checklist", SubjectKey: "check-one@1",
			Task: selection.Inputs["checklist"], WorkflowRole: "check", State: auditstore.ItemReady,
			CreatedAt: now, UpdatedAt: now,
		}},
	}
	start := auditAuthenticatedRequest(http.MethodPost, "/v1/audits/audit-fixed/start", nil)
	start.SetPathValue("auditId", "audit-fixed")
	start.Header.Set("Idempotency-Key", "start-audit")
	start.Header.Set("If-Match", `"1"`)
	started := httptest.NewRecorder()
	h.startAudit(started, start)
	if started.Code != http.StatusOK || started.Header().Get("ETag") != `"2"` ||
		management.start.ExpectedRevision != 1 || management.start.RequestDigest == "" ||
		!strings.Contains(started.Body.String(), `"expectedItemCount":1`) {
		t.Fatalf("start Audit = %d headers=%v body=%s params=%+v", started.Code, started.Header(), started.Body.String(), management.start)
	}

	management.err = auditstore.ErrNotFound
	foreign := auditAuthenticatedRequest(http.MethodGet, "/v1/audits/audit-fixed", nil)
	foreign.SetPathValue("auditId", "audit-fixed")
	foreignResponse := httptest.NewRecorder()
	h.getAudit(foreignResponse, foreign)
	if foreignResponse.Code != http.StatusNotFound || !strings.Contains(foreignResponse.Body.String(), `"code":"not_found"`) {
		t.Fatalf("foreign Audit = %d %s", foreignResponse.Code, foreignResponse.Body.String())
	}

	management.err = &auditservice.UnsupportedError{Reasons: []auditservice.CompatibilityReason{
		auditservice.ReasonActiveCheckApprovalUnsupported,
	}}
	unsupported := auditAuthenticatedRequest(http.MethodPost, "/v1/audits/audit-fixed/start", nil)
	unsupported.SetPathValue("auditId", "audit-fixed")
	unsupported.Header.Set("Idempotency-Key", "unsupported-audit")
	unsupported.Header.Set("If-Match", `"1"`)
	unsupportedResponse := httptest.NewRecorder()
	h.startAudit(unsupportedResponse, unsupported)
	if unsupportedResponse.Code != http.StatusUnprocessableEntity ||
		!strings.Contains(unsupportedResponse.Body.String(), `"code":"audit_profile_unsupported"`) ||
		!strings.Contains(unsupportedResponse.Body.String(), `"active_check_approval_unsupported"`) {
		t.Fatalf("unsupported Audit = %d %s", unsupportedResponse.Code, unsupportedResponse.Body.String())
	}
}

func TestAuditReportHandlerReturnsOnlyAcceptedProjection(t *testing.T) {
	revision := "report-r1"
	management := &fakeAuditManagement{report: auditservice.ReportProjection{
		Status: auditservice.ReportReady,
		MachineArtifact: &auditstore.ExactArtifact{
			Ref:    contracts.ArtifactRef{Namespace: "audit-hidden", Name: "report.json", Revision: &revision},
			Digest: auditHandlerDigest("machine"), MediaType: "application/json", SizeBytes: 41,
		},
		SummaryArtifact: &auditstore.ExactArtifact{
			Ref:    contracts.ArtifactRef{Namespace: "audit-hidden", Name: "report.txt", Revision: &revision},
			Digest: auditHandlerDigest("summary"), MediaType: "text/plain", SizeBytes: 16,
		},
		Machine: json.RawMessage(`{"schema":"contractor.audit.report.v1"}`),
		Summary: "bounded summary",
	}}
	h := auditTestHandler(management)
	request := auditAuthenticatedRequest(http.MethodGet, "/v1/audits/audit-fixed/report", nil)
	request.SetPathValue("auditId", "audit-fixed")
	response := httptest.NewRecorder()
	h.getAuditReport(response, request)
	if response.Code != http.StatusOK || response.Header().Get("Cache-Control") != "no-store" ||
		!strings.Contains(response.Body.String(), `"status":"ready"`) ||
		!strings.Contains(response.Body.String(), `"schema":"contractor.audit.report.v1"`) ||
		!strings.Contains(response.Body.String(), `"summary":"bounded summary"`) {
		t.Fatalf("Audit report = %d headers=%v body=%s", response.Code, response.Header(), response.Body.String())
	}

	management.report = auditservice.ReportProjection{Status: auditservice.ReportPending}
	pending := httptest.NewRecorder()
	h.getAuditReport(pending, request)
	if pending.Code != http.StatusOK || pending.Body.String() != "{\"status\":\"pending\"}\n" {
		t.Fatalf("pending Audit report = %d %s", pending.Code, pending.Body.String())
	}
}

func TestAuditFindingProposalHandlersExposeInboxAndExactImport(t *testing.T) {
	revision := "finding-revision"
	findings := &fakeFindingProposalManagement{receipts: []findingintake.Receipt{{
		ReceiptID: "receipt-one", ProposalID: "proposal-one", ClientKey: "candidate-one",
		Proposal: findingintake.ExactArtifact{Ref: contracts.ArtifactRef{
			Namespace: "finding-proposals", Name: "proposal-one", Revision: &revision,
		}},
		Retention:  findingintake.RetentionSourceHeld,
		AuditHolds: []findingintake.AuditHold{}, CreatedAt: time.Now().UTC(),
	}}}
	h := auditTestHandler(&fakeAuditManagement{audit: auditstore.Audit{AuditID: "audit-fixed"}})
	h.dependencies.FindingProposals = findings

	list := auditAuthenticatedRequest(http.MethodGet, "/v1/audits/audit-fixed/finding-proposals?limit=10", nil)
	list.SetPathValue("auditId", "audit-fixed")
	listed := httptest.NewRecorder()
	h.listAuditFindingProposals(listed, list)
	if listed.Code != http.StatusOK || listed.Header().Get("Cache-Control") != "no-store" ||
		!strings.Contains(listed.Body.String(), `"receiptId":"receipt-one"`) ||
		findings.listOwner != "user-1" || findings.listAudit != "audit-fixed" {
		t.Fatalf("Audit finding inbox = %d headers=%v body=%s calls=%+v",
			listed.Code, listed.Header(), listed.Body.String(), findings)
	}

	body := []byte(`{"runId":"run-one","proposal":{"namespace":"finding-proposals","name":"proposal-one","revision":"finding-revision"}}`)
	request := auditAuthenticatedRequest(http.MethodPost, "/v1/audits/audit-fixed/finding-proposal-imports", body)
	request.SetPathValue("auditId", "audit-fixed")
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	h.importAuditFindingProposal(response, request)
	if response.Code != http.StatusCreated || findings.imported.OwnerID != "user-1" ||
		findings.imported.AuditID != "audit-fixed" || findings.imported.RunID != "run-one" ||
		!strings.Contains(response.Body.String(), `"replayed":false`) {
		t.Fatalf("Audit finding import = %d body=%s request=%+v",
			response.Code, response.Body.String(), findings.imported)
	}
}

func TestAuditFindingReviewHandlersBindCASIdempotencyAndProvenanceRevision(t *testing.T) {
	now := time.Date(2026, 9, 6, 4, 0, 0, 0, time.UTC)
	severity := auditservice.SeverityHigh
	verdict := auditservice.VerdictTruePositive
	decision := auditservice.ReviewDecision{
		DecisionID: "decision-one", RequestID: "review-one", AuditID: "audit-fixed",
		FindingID: "finding-one", ActorID: "user-1", Verdict: verdict, Severity: &severity,
		Rationale: "Confirmed from exact evidence.", SubjectRevision: 3,
		SubjectDigest: auditHandlerDigest("finding-subject"), CreatedAt: now.Add(time.Second),
	}
	management := &fakeAuditManagement{
		audit: auditstore.Audit{AuditID: "audit-fixed", OwnerID: "user-1", Revision: 9},
		findings: []auditservice.Finding{
			{FindingID: "finding-one", AuditID: "audit-fixed", State: auditservice.FindingConfirmed,
				AnalystVerdict: &verdict, AnalystSeverity: &severity, Revision: 4,
				CreatedAt: now, UpdatedAt: now.Add(time.Second)},
			{FindingID: "finding-two", AuditID: "audit-fixed", State: auditservice.FindingProposed,
				Revision: 1, CreatedAt: now.Add(2 * time.Second), UpdatedAt: now.Add(2 * time.Second)},
		},
		reviews: []auditservice.ReviewRequest{{
			RequestID: "review-one", AuditID: "audit-fixed", FindingID: "finding-one",
			SubjectKind: auditservice.ReviewSubjectFinding, SubjectID: "finding-one",
			Kind: auditservice.FindingReviewKind, SubjectRevision: 3,
			SubjectDigest: auditHandlerDigest("finding-subject"),
			RequestedActions: []auditservice.ReviewRequestedAction{
				auditservice.ReviewRequestedAction(auditservice.VerdictTruePositive),
			},
			State: auditservice.ReviewPending, Revision: 1, CreatedAt: now, UpdatedAt: now,
		}},
		provenance: []auditservice.FindingProvenance{
			{RecordID: "proposal:one", Kind: auditservice.ProvenanceSourceProposal,
				ReceiptID: "receipt-one", CreatedAt: now},
			{RecordID: "assessment:one", Kind: auditservice.ProvenanceCheckAttempt,
				ReceiptID: "receipt-one", CreatedAt: now.Add(time.Second)},
		},
	}
	h := auditTestHandler(management)

	listRequest := auditAuthenticatedRequest(http.MethodGet,
		"/v1/audits/audit-fixed/findings?limit=1&verdict=true_positive&severity=high", nil)
	listRequest.SetPathValue("auditId", "audit-fixed")
	listResponse := httptest.NewRecorder()
	h.listAuditFindings(listResponse, listRequest)
	var findingPage findingPageResponse
	if err := json.Unmarshal(listResponse.Body.Bytes(), &findingPage); err != nil {
		t.Fatal(err)
	}
	if listResponse.Code != http.StatusOK || len(findingPage.Items) != 1 || !findingPage.Page.HasMore ||
		findingPage.Page.NextCursor == nil || management.findingListParams.OwnerID != "user-1" ||
		management.findingListParams.Limit != 2 || management.findingListParams.Verdict == nil ||
		*management.findingListParams.Verdict != verdict || management.findingListParams.Severity == nil ||
		*management.findingListParams.Severity != severity {
		t.Fatalf("finding page = %d %+v params=%+v body=%s", listResponse.Code, findingPage,
			management.findingListParams, listResponse.Body.String())
	}

	detail := auditAuthenticatedRequest(http.MethodGet, "/v1/audits/audit-fixed/findings/finding-one", nil)
	detail.SetPathValue("auditId", "audit-fixed")
	detail.SetPathValue("findingId", "finding-one")
	detailResponse := httptest.NewRecorder()
	h.getAuditFinding(detailResponse, detail)
	if detailResponse.Code != http.StatusOK || detailResponse.Header().Get("ETag") != `"4"` {
		t.Fatalf("finding detail = %d headers=%v body=%s", detailResponse.Code,
			detailResponse.Header(), detailResponse.Body.String())
	}

	create := auditAuthenticatedRequest(http.MethodPost,
		"/v1/audits/audit-fixed/findings/finding-one/reviews", []byte(`{}`))
	create.SetPathValue("auditId", "audit-fixed")
	create.SetPathValue("findingId", "finding-one")
	create.Header.Set("Content-Type", "application/json")
	create.Header.Set("If-Match", `"4"`)
	create.Header.Set("Idempotency-Key", "create-review-one")
	createResponse := httptest.NewRecorder()
	h.createAuditFindingReview(createResponse, create)
	if createResponse.Code != http.StatusCreated || createResponse.Header().Get("ETag") != `"1"` ||
		management.createReviewParams.OwnerID != "user-1" ||
		management.createReviewParams.ExpectedRevision != 4 ||
		management.createReviewParams.IdempotencyKey != "create-review-one" ||
		management.createReviewParams.RequestDigest == "" {
		t.Fatalf("create review = %d headers=%v params=%+v body=%s", createResponse.Code,
			createResponse.Header(), management.createReviewParams, createResponse.Body.String())
	}

	management.reviews[0].State = auditservice.ReviewDecided
	management.reviews[0].Revision = 2
	management.reviews[0].Decision = &decision
	management.decisionReplayed = true
	decide := auditAuthenticatedRequest(http.MethodPost,
		"/v1/audits/audit-fixed/reviews/review-one/decisions",
		[]byte(`{"verdict":"true_positive","severity":"high","rationale":"Confirmed from exact evidence."}`))
	decide.SetPathValue("auditId", "audit-fixed")
	decide.SetPathValue("requestId", "review-one")
	decide.Header.Set("Content-Type", "application/json")
	decide.Header.Set("If-Match", `"1"`)
	decide.Header.Set("Idempotency-Key", "decide-review-one")
	decideResponse := httptest.NewRecorder()
	h.decideAuditReview(decideResponse, decide)
	if decideResponse.Code != http.StatusOK || decideResponse.Header().Get("ETag") != `"2"` ||
		decideResponse.Header().Get("Idempotency-Replayed") != "true" ||
		management.decideFindingParams.ExpectedRequestRevision != 1 ||
		management.decideFindingParams.Verdict != verdict ||
		management.decideFindingParams.Severity == nil ||
		*management.decideFindingParams.Severity != severity ||
		management.decideFindingParams.RequestDigest == "" {
		t.Fatalf("decide review = %d headers=%v params=%+v body=%s", decideResponse.Code,
			decideResponse.Header(), management.decideFindingParams, decideResponse.Body.String())
	}

	action := auditservice.ReviewApprove
	management.reviews[0] = auditservice.ReviewRequest{
		RequestID: "review-action", AuditID: "audit-fixed",
		SubjectKind: auditservice.ReviewSubjectItemAction, SubjectID: "item-one",
		Kind: auditservice.ActiveCheckReviewKind, SubjectRevision: 1,
		SubjectDigest: auditHandlerDigest("active-action"),
		RequestedActions: []auditservice.ReviewRequestedAction{
			auditservice.ReviewRequestedAction(action),
		},
		State: auditservice.ReviewDecided, Revision: 2, CreatedAt: now, UpdatedAt: now,
		Decision: &auditservice.ReviewDecision{
			DecisionID: "decision-action", RequestID: "review-action", AuditID: "audit-fixed",
			Action: action, ActorID: "user-1", Rationale: "Approved exact active action.",
			SubjectRevision: 1, SubjectDigest: auditHandlerDigest("active-action"), CreatedAt: now,
		},
	}
	actionRequest := auditAuthenticatedRequest(http.MethodPost,
		"/v1/audits/audit-fixed/reviews/review-action/decisions",
		[]byte(`{"action":"approve","rationale":"Approved exact active action."}`))
	actionRequest.SetPathValue("auditId", "audit-fixed")
	actionRequest.SetPathValue("requestId", "review-action")
	actionRequest.Header.Set("Content-Type", "application/json")
	actionRequest.Header.Set("If-Match", `"1"`)
	actionRequest.Header.Set("Idempotency-Key", "decide-action-one")
	actionResponse := httptest.NewRecorder()
	h.decideAuditReview(actionResponse, actionRequest)
	if actionResponse.Code != http.StatusOK || actionResponse.Header().Get("ETag") != `"2"` ||
		management.decideActionParams.Action != auditservice.ReviewApprove ||
		management.decideActionParams.ExpectedRequestRevision != 1 ||
		management.decideActionParams.Rationale != "Approved exact active action." ||
		management.decideActionParams.RequestDigest == "" {
		t.Fatalf("decide action review = %d headers=%v params=%+v body=%s",
			actionResponse.Code, actionResponse.Header(), management.decideActionParams,
			actionResponse.Body.String())
	}

	provenance := auditAuthenticatedRequest(http.MethodGet,
		"/v1/audits/audit-fixed/findings/finding-one/provenance?limit=1", nil)
	provenance.SetPathValue("auditId", "audit-fixed")
	provenance.SetPathValue("findingId", "finding-one")
	provenanceResponse := httptest.NewRecorder()
	h.listAuditFindingProvenance(provenanceResponse, provenance)
	var provenancePage findingProvenancePageResponse
	if err := json.Unmarshal(provenanceResponse.Body.Bytes(), &provenancePage); err != nil {
		t.Fatal(err)
	}
	if provenanceResponse.Code != http.StatusOK || provenancePage.AuditRevision != 9 ||
		provenancePage.FindingRevision != 4 || len(provenancePage.Items) != 1 ||
		provenancePage.Page.NextCursor == nil || management.provenanceParams.Limit != 2 {
		t.Fatalf("provenance page = %d %+v params=%+v body=%s", provenanceResponse.Code,
			provenancePage, management.provenanceParams, provenanceResponse.Body.String())
	}
	management.audit.Revision = 10
	stale := auditAuthenticatedRequest(http.MethodGet,
		"/v1/audits/audit-fixed/findings/finding-one/provenance?cursor="+
			url.QueryEscape(*provenancePage.Page.NextCursor), nil)
	stale.SetPathValue("auditId", "audit-fixed")
	stale.SetPathValue("findingId", "finding-one")
	staleResponse := httptest.NewRecorder()
	h.listAuditFindingProvenance(staleResponse, stale)
	if staleResponse.Code != http.StatusConflict {
		t.Fatalf("stale provenance cursor = %d body=%s", staleResponse.Code, staleResponse.Body.String())
	}

	missingCAS := auditAuthenticatedRequest(http.MethodPost,
		"/v1/audits/audit-fixed/findings/finding-one/reviews", []byte(`{}`))
	missingCAS.SetPathValue("auditId", "audit-fixed")
	missingCAS.SetPathValue("findingId", "finding-one")
	missingCAS.Header.Set("Content-Type", "application/json")
	missingCAS.Header.Set("Idempotency-Key", "missing-cas")
	missingCASResponse := httptest.NewRecorder()
	h.createAuditFindingReview(missingCASResponse, missingCAS)
	if missingCASResponse.Code != http.StatusBadRequest {
		t.Fatalf("missing review CAS = %d body=%s", missingCASResponse.Code, missingCASResponse.Body.String())
	}
}

func TestAuditLifecycleHandlersRequireCASAndIdempotency(t *testing.T) {
	now := time.Date(2026, 9, 6, 1, 0, 0, 0, time.UTC)
	revision := "input-r1"
	selection, err := json.Marshal(auditservice.DraftSelection{
		Schema: auditservice.DraftSelectionSchema,
		Inputs: map[string]auditstore.ExactArtifact{"checklist": {
			Ref:    contracts.ArtifactRef{Namespace: "inputs", Name: "checklist", Revision: &revision},
			Digest: auditHandlerDigest("input"), MediaType: "application/json", SizeBytes: 1,
		}}, RuntimeLabels: []string{}, Scope: auditservice.Scope{},
	})
	if err != nil {
		t.Fatal(err)
	}
	management := &fakeAuditManagement{audit: auditstore.Audit{
		AuditID: "audit-fixed", OwnerID: "user-1", ProjectID: "project-one",
		Profile:        auditstore.ProfileIdentity{Name: "checklist", Version: "1", Digest: auditHandlerDigest("profile")},
		InputSelection: selection, State: auditstore.AuditPaused, Revision: 7,
		Dispatch: auditstore.DispatchOpen, Hold: auditstore.HoldHeld,
		CreatedAt: now, UpdatedAt: now,
	}}
	h := auditTestHandler(management)

	for _, test := range []struct {
		name   string
		method string
		path   string
		status int
		call   func(http.ResponseWriter, *http.Request)
	}{
		{"pause", http.MethodPost, "/v1/audits/audit-fixed/pause", http.StatusOK, h.pauseAudit},
		{"resume", http.MethodPost, "/v1/audits/audit-fixed/resume", http.StatusOK, h.resumeAudit},
		{"cancel", http.MethodPost, "/v1/audits/audit-fixed/cancel", http.StatusAccepted, h.cancelAudit},
		{"delete", http.MethodDelete, "/v1/audits/audit-fixed", http.StatusAccepted, h.deleteAudit},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := auditAuthenticatedRequest(test.method, test.path, nil)
			request.SetPathValue("auditId", "audit-fixed")
			request.Header.Set("Idempotency-Key", "lifecycle-"+test.name)
			request.Header.Set("If-Match", `"7"`)
			response := httptest.NewRecorder()
			test.call(response, request)
			if response.Code != test.status || response.Header().Get("ETag") != `"7"` ||
				management.mutation.AuditID != "audit-fixed" || management.mutation.ExpectedRevision != 7 ||
				management.mutation.IdempotencyKey != "lifecycle-"+test.name || management.mutation.RequestDigest == "" {
				t.Fatalf("response=%d headers=%v mutation=%+v body=%s", response.Code, response.Header(), management.mutation, response.Body.String())
			}
		})
	}

	missingCAS := auditAuthenticatedRequest(http.MethodPost, "/v1/audits/audit-fixed/cancel", nil)
	missingCAS.SetPathValue("auditId", "audit-fixed")
	missingCAS.Header.Set("Idempotency-Key", "missing-cas")
	response := httptest.NewRecorder()
	h.cancelAudit(response, missingCAS)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("missing CAS status = %d body=%s", response.Code, response.Body.String())
	}
}

type fakeAuditManagement struct {
	profiles            []auditservice.ProfileProjection
	audit               auditstore.Audit
	started             auditservice.StartedAudit
	created             auditservice.CreateDraftParams
	start               auditservice.StartParams
	mutation            auditservice.MutationParams
	err                 error
	report              auditservice.ReportProjection
	findings            []auditservice.Finding
	reviews             []auditservice.ReviewRequest
	provenance          []auditservice.FindingProvenance
	findingListParams   auditservice.FindingListParams
	reviewListParams    auditservice.ReviewListParams
	createReviewParams  auditservice.CreateFindingReviewParams
	decideFindingParams auditservice.DecideFindingParams
	decideActionParams  auditservice.DecideActionReviewParams
	provenanceParams    auditservice.ProvenanceListParams
	reviewReplayed      bool
	decisionReplayed    bool
}

type fakeFindingProposalManagement struct {
	receipts  []findingintake.Receipt
	listOwner string
	listAudit string
	imported  findingintake.ImportRequest
}

func (f *fakeFindingProposalManagement) ListRun(
	context.Context, string, string, findingintake.ListQuery,
) ([]findingintake.Receipt, error) {
	return append([]findingintake.Receipt(nil), f.receipts...), nil
}

func (f *fakeFindingProposalManagement) ListAuditInbox(
	_ context.Context, ownerID, auditID string, _ findingintake.ListQuery,
) ([]findingintake.Receipt, error) {
	f.listOwner, f.listAudit = ownerID, auditID
	return append([]findingintake.Receipt(nil), f.receipts...), nil
}

func (f *fakeFindingProposalManagement) ImportIntoAudit(
	_ context.Context, request findingintake.ImportRequest,
) (findingintake.AuditHold, bool, error) {
	f.imported = request
	return findingintake.AuditHold{
		AuditID: request.AuditID, ProjectID: "project-one",
		Proposal: f.receipts[0].Proposal, Evidence: []findingintake.ExactArtifact{},
		CreatedAt: time.Now().UTC(),
	}, false, nil
}

func (f *fakeAuditManagement) Profiles() []auditservice.ProfileProjection {
	return append([]auditservice.ProfileProjection(nil), f.profiles...)
}

func (f *fakeAuditManagement) Profile(selector auditservice.ProfileSelector) (auditservice.ProfileProjection, error) {
	for _, profile := range f.profiles {
		if profile.Profile.Ref.Name == selector.Name && profile.Profile.Ref.Version == selector.Version {
			return profile, nil
		}
	}
	return auditservice.ProfileProjection{}, auditservice.ErrProfileNotFound
}

func (f *fakeAuditManagement) CreateDraft(
	_ context.Context, params auditservice.CreateDraftParams,
) (auditstore.Audit, bool, error) {
	f.created = params
	return f.audit, true, f.err
}

func (f *fakeAuditManagement) Start(
	_ context.Context, params auditservice.StartParams,
) (auditservice.StartedAudit, error) {
	f.start = params
	return f.started, f.err
}

func (f *fakeAuditManagement) Pause(
	_ context.Context, params auditservice.MutationParams,
) (auditservice.MutationResult, error) {
	f.mutation = params
	return auditservice.MutationResult{Audit: f.audit}, f.err
}

func (f *fakeAuditManagement) Resume(
	_ context.Context, params auditservice.MutationParams,
) (auditservice.MutationResult, error) {
	f.mutation = params
	return auditservice.MutationResult{Audit: f.audit}, f.err
}

func (f *fakeAuditManagement) Cancel(
	_ context.Context, params auditservice.MutationParams,
) (auditservice.MutationResult, error) {
	f.mutation = params
	return auditservice.MutationResult{Audit: f.audit}, f.err
}

func (f *fakeAuditManagement) Delete(
	_ context.Context, params auditservice.MutationParams,
) (auditservice.MutationResult, error) {
	f.mutation = params
	return auditservice.MutationResult{Audit: f.audit}, f.err
}

func (f *fakeAuditManagement) Get(context.Context, string, string) (auditstore.Audit, error) {
	return f.audit, f.err
}

func (f *fakeAuditManagement) List(context.Context, auditstore.ListParams) ([]auditstore.Audit, error) {
	if f.err != nil {
		return nil, f.err
	}
	return []auditstore.Audit{f.audit}, nil
}

func (f *fakeAuditManagement) ListItems(context.Context, auditstore.ListItemsParams) ([]auditstore.Item, error) {
	return append([]auditstore.Item(nil), f.started.Items...), f.err
}

func (f *fakeAuditManagement) ListItemAttempts(
	context.Context, string, string, []string,
) (map[string][]auditstore.ItemAttempt, error) {
	return map[string][]auditstore.ItemAttempt{}, f.err
}

func (f *fakeAuditManagement) GetRound(context.Context, string, string, string) (auditstore.Round, error) {
	return f.started.Round, f.err
}

func (f *fakeAuditManagement) ListCoverage(
	context.Context, string, string, string, int, int,
) ([]auditstore.CoverageRow, error) {
	return nil, f.err
}

func (f *fakeAuditManagement) GetReport(
	context.Context, string, string,
) (auditservice.ReportProjection, error) {
	return f.report, f.err
}

func (f *fakeAuditManagement) ListFindings(
	_ context.Context, params auditservice.FindingListParams,
) ([]auditservice.Finding, error) {
	f.findingListParams = params
	return append([]auditservice.Finding(nil), f.findings...), f.err
}

func (f *fakeAuditManagement) GetFinding(
	context.Context, string, string, string,
) (auditservice.Finding, error) {
	if len(f.findings) == 0 {
		return auditservice.Finding{}, f.err
	}
	return f.findings[0], f.err
}

func (f *fakeAuditManagement) CreateFindingReview(
	_ context.Context, params auditservice.CreateFindingReviewParams,
) (auditservice.FindingReviewResult, error) {
	f.createReviewParams = params
	if len(f.reviews) == 0 {
		return auditservice.FindingReviewResult{}, f.err
	}
	return auditservice.FindingReviewResult{Request: f.reviews[0], Replayed: f.reviewReplayed}, f.err
}

func (f *fakeAuditManagement) DecideFinding(
	_ context.Context, params auditservice.DecideFindingParams,
) (auditservice.FindingDecisionResult, error) {
	f.decideFindingParams = params
	if len(f.findings) == 0 || len(f.reviews) == 0 || f.reviews[0].Decision == nil {
		return auditservice.FindingDecisionResult{}, f.err
	}
	return auditservice.FindingDecisionResult{
		Finding: f.findings[0], Request: f.reviews[0], Decision: *f.reviews[0].Decision,
		Replayed: f.decisionReplayed,
	}, f.err
}

func (f *fakeAuditManagement) DecideActionReview(
	_ context.Context, params auditservice.DecideActionReviewParams,
) (auditservice.ActionReviewDecisionResult, error) {
	f.decideActionParams = params
	if len(f.reviews) == 0 || f.reviews[0].Decision == nil {
		return auditservice.ActionReviewDecisionResult{}, f.err
	}
	return auditservice.ActionReviewDecisionResult{
		Request: f.reviews[0], Decision: *f.reviews[0].Decision,
		Replayed: f.decisionReplayed,
	}, f.err
}

func (f *fakeAuditManagement) GetReview(
	context.Context, string, string, string,
) (auditservice.ReviewRequest, error) {
	if len(f.reviews) == 0 {
		return auditservice.ReviewRequest{}, f.err
	}
	return f.reviews[0], f.err
}

func (f *fakeAuditManagement) ListReviews(
	_ context.Context, params auditservice.ReviewListParams,
) ([]auditservice.ReviewRequest, error) {
	f.reviewListParams = params
	return append([]auditservice.ReviewRequest(nil), f.reviews...), f.err
}

func (f *fakeAuditManagement) ListFindingProvenance(
	_ context.Context, params auditservice.ProvenanceListParams,
) ([]auditservice.FindingProvenance, error) {
	f.provenanceParams = params
	return append([]auditservice.FindingProvenance(nil), f.provenance...), f.err
}

func auditTestHandler(audits AuditManagement) *handler {
	return &handler{dependencies: Dependencies{
		Audits: audits,
		NewID:  func(string) (string, error) { return "audit-fixed", nil },
	}}
}

func auditAuthenticatedRequest(method, target string, body []byte) *http.Request {
	request := httptest.NewRequest(method, target, bytes.NewReader(body))
	principal, _ := auth.NewPrincipal("user-1", "user")
	return request.WithContext(auth.WithPrincipal(request.Context(), principal))
}

func auditHandlerDigest(value string) string {
	digest := sha256.Sum256([]byte(value))
	return "sha256:" + hex.EncodeToString(digest[:])
}

var _ AuditManagement = (*fakeAuditManagement)(nil)
var _ FindingProposalManagement = (*fakeFindingProposalManagement)(nil)
