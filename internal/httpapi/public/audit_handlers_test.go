package public

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
)

func TestAuditProfileHandlersExposeCompatibilityAndExactDetail(t *testing.T) {
	profiles := &fakeAuditManagement{profiles: []auditservice.ProfileProjection{{
		Profile: auditprofile.ResolvedAuditProfile{
			Ref:  auditprofile.AuditProfileRef{Name: "checklist", Version: "1", Digest: auditHandlerDigest("profile")},
			Mode: auditprofile.AuditModeCustomChecklist, Standards: []auditprofile.AuditStandardRef{},
			Inputs: map[string]auditprofile.AuditProfileInput{
				"checklist": {Required: true, MediaTypes: []string{"application/json"}},
			},
			Inventory: auditprofile.AuditInventory{Implementation: "checklist@1", Source: &auditprofile.AuditWorkflowInputMapping{Source: auditprofile.AuditInputFromAudit, Name: "checklist"}, ItemWorkflowRole: "check"},
			Workflows: map[string]auditprofile.ResolvedAuditWorkflowBinding{},
			Execution: auditprofile.AuditExecutionPolicy{MaxRounds: 1, BatchSize: 1},
			Interaction: auditprofile.AuditInteractionPolicy{
				ActiveChecks:     auditprofile.AuditActiveChecksProhibited,
				NotApplicable:    auditprofile.AuditNotApplicableProfileRule,
				ReportAcceptance: auditprofile.AuditReportAutomatic,
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

func TestAuditStandardHandlersExposeCatalogProjectionOnly(t *testing.T) {
	revision := "standard-r1"
	management := &fakeAuditManagement{standards: []auditstandards.PackageProjection{{
		Reference: auditstandards.Reference{Scheme: "example", Version: "1"},
		Title:     "Example", Description: "Example standard.",
		Source: auditstandards.Source{Name: "Source", URL: "https://example.invalid/source"},
		License: auditstandards.License{
			ID: "CC-BY-4.0", URL: "https://creativecommons.org/licenses/by/4.0/",
			Attribution: "Example authors", Disclosure: auditstandards.DisclosureFull,
		},
		Digest: auditHandlerDigest("standard"), Artifact: contracts.ArtifactRef{
			Namespace: artifactpolicy.AuditStandardCatalogNamespace, Name: "std-example", Revision: &revision,
		},
		EntryCount: 1, MappingCount: 1, EvidenceContractCount: 1,
	}}}
	h := auditTestHandler(management)

	list := auditAuthenticatedRequest(http.MethodGet, "/v1/audit-standards", nil)
	response := httptest.NewRecorder()
	h.listAuditStandards(response, list)
	if response.Code != http.StatusOK ||
		!strings.Contains(response.Body.String(), `"scheme":"example"`) ||
		!strings.Contains(response.Body.String(), `"entryCount":1`) ||
		strings.Contains(response.Body.String(), `"entries"`) {
		t.Fatalf("standard list = %d %s", response.Code, response.Body.String())
	}

	management.standards[0].Entries = []auditstandards.EntryProjection{{
		ID: "EX-1", Kind: "requirement", Title: "Example requirement",
	}}
	detail := auditAuthenticatedRequest(http.MethodGet, "/v1/audit-standards/example/versions/1", nil)
	detail.SetPathValue("scheme", "example")
	detail.SetPathValue("version", "1")
	detailResponse := httptest.NewRecorder()
	h.getAuditStandard(detailResponse, detail)
	if detailResponse.Code != http.StatusOK ||
		!strings.Contains(detailResponse.Body.String(), `"id":"EX-1"`) {
		t.Fatalf("standard detail = %d %s", detailResponse.Code, detailResponse.Body.String())
	}
}

func TestAuditCreateStartQueryAndStableErrors(t *testing.T) {
	now := time.Date(2026, 9, 5, 19, 0, 0, 0, time.UTC)
	revision := "input-r1"
	selection := auditservice.DraftSelection{Labels: map[string]string{}, RequiredPlacementLabels: []string{},
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
	// A start binds its action like every other lifecycle request.
	startDigest := contentdigest.Bytes([]byte(`{"action":"start","auditId":"audit-fixed","revision":1}`))
	if started.Code != http.StatusOK || started.Header().Get("ETag") != `"2"` ||
		management.start.ExpectedRevision != 1 || management.start.RequestDigest != startDigest ||
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

	for _, reason := range []auditservice.CompatibilityReason{
		auditservice.ReasonActiveCheckApprovalUnsupported,
		auditservice.ReasonEngagementCapabilityUnsupported,
		auditservice.ReasonEngagementSourceClosureUnsupported,
		auditservice.ReasonEngagementActiveChecksUnsupported,
	} {
		management.err = &auditservice.UnsupportedError{Reasons: []auditservice.CompatibilityReason{reason}}
		unsupported := auditAuthenticatedRequest(http.MethodPost, "/v1/audits/audit-fixed/start", nil)
		unsupported.SetPathValue("auditId", "audit-fixed")
		unsupported.Header.Set("Idempotency-Key", "unsupported-audit")
		unsupported.Header.Set("If-Match", `"1"`)
		unsupportedResponse := httptest.NewRecorder()
		h.startAudit(unsupportedResponse, unsupported)
		if unsupportedResponse.Code != http.StatusUnprocessableEntity ||
			!strings.Contains(unsupportedResponse.Body.String(), `"code":"audit_profile_unsupported"`) ||
			!strings.Contains(unsupportedResponse.Body.String(), `"`+string(reason)+`"`) {
			t.Fatalf("unsupported Audit lost reason %s = %d %s", reason, unsupportedResponse.Code, unsupportedResponse.Body.String())
		}
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
			Ref:    contracts.ArtifactRef{Namespace: "audit-hidden", Name: "report.md", Revision: &revision},
			Digest: auditHandlerDigest("summary"), MediaType: "text/markdown", SizeBytes: 16,
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
	h := findingTestHandler(&fakeAuditManagement{audit: auditstore.Audit{AuditID: "audit-fixed"}}, findings)

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

func TestAuditLifecycleHandlersRequireCASAndIdempotency(t *testing.T) {
	now := time.Date(2026, 9, 6, 1, 0, 0, 0, time.UTC)
	revision := "input-r1"
	selection, err := json.Marshal(auditservice.DraftSelection{Labels: map[string]string{}, RequiredPlacementLabels: []string{},
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
	AuditManagement
	events             auditservice.EventPage
	eventParams        auditstore.EventListParams
	ownerFindings      auditstore.OwnerFindingListParams
	ownerReviews       auditstore.OwnerReviewListParams
	profiles           []auditservice.ProfileProjection
	standards          []auditstandards.PackageProjection
	audit              auditstore.Audit
	started            auditservice.StartedAudit
	attempts           map[string][]auditstore.ItemAttempt
	coverage           auditservice.CoveragePage
	preparation        map[string]auditservice.PreparationRoleProjection
	preparationOwner   string
	snapshot           *auditstore.Audit
	detailErr          error
	preparationPages   []int
	preparationErr     error
	artifactRead       artifacts.ReadResult
	created            auditservice.CreateDraftParams
	start              auditservice.StartParams
	mutation           auditservice.MutationParams
	err                error
	report             auditservice.ReportProjection
	findings           []auditstore.ProjectFinding
	reviews            []auditstore.ReviewRequest
	findingListParams  auditstore.FindingListParams
	reviewListParams   auditstore.ReviewListParams
	decideActionParams auditservice.DecideActionReviewParams
	reviewReplayed     bool
	decisionReplayed   bool
}

// Detail returns audit unless prepared selects it. A selected Audit is read
// again, as the service does, in its snapshot: the fake's later state when
// snapshot is set, with the fake's preparation; detailErr fails that read.
func (f *fakeAuditManagement) Detail(
	_ context.Context, ownerID string, audit auditstore.Audit, prepared func(auditstore.Audit) bool,
) (auditservice.AuditDetail, error) {
	if !prepared(audit) {
		return auditservice.AuditDetail{Audit: audit}, nil
	}
	f.preparationOwner = ownerID
	if f.detailErr != nil {
		return auditservice.AuditDetail{}, f.detailErr
	}
	if f.snapshot != nil {
		audit = *f.snapshot
	}
	return auditservice.AuditDetail{Audit: audit, Preparation: f.preparation}, f.err
}

// StartedDetail is Detail for a start response; a selected Audit's Round and
// items are those of the fake's start.
func (f *fakeAuditManagement) StartedDetail(
	ctx context.Context, ownerID string, started auditservice.StartedAudit, prepared func(auditstore.Audit) bool,
) (auditservice.StartedDetail, error) {
	detail, err := f.Detail(ctx, ownerID, started.Audit, prepared)
	if err != nil {
		return auditservice.StartedDetail{}, err
	}
	result := auditservice.StartedDetail{AuditDetail: detail, Items: started.Items}
	if detail.Audit.CurrentRoundID != nil {
		round := started.Round
		result.Round = &round
	}
	return result, nil
}

func (f *fakeAuditManagement) ReadArtifact(_ context.Context, _, _, _, _ string) (artifacts.ReadResult, error) {
	return f.artifactRead, f.err
}

type fakeFindingProposalManagement struct {
	FindingProposalManagement
	receipts  []findingintake.Receipt
	listOwner string
	listAudit string
	imported  findingintake.ImportRequest
}

func (f *fakeFindingProposalManagement) ListRun(
	_ context.Context, _, _ string, query findingintake.ListQuery,
) ([]findingintake.Receipt, error) {
	if err := fakePageFetch(query.Limit, findingintake.ErrInvalid); err != nil {
		return nil, err
	}
	return append([]findingintake.Receipt(nil), f.receipts...), nil
}

func (f *fakeFindingProposalManagement) ListAuditInbox(
	_ context.Context, ownerID, auditID string, query findingintake.ListQuery,
) ([]findingintake.Receipt, error) {
	if err := fakePageFetch(query.Limit, findingintake.ErrInvalid); err != nil {
		return nil, err
	}
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

func (f *fakeAuditManagement) ListEventsPage(_ context.Context, p auditstore.EventListParams) (auditservice.EventPage, error) {
	if err := fakePageFetch(p.Limit, auditstore.ErrInvalid); err != nil {
		return auditservice.EventPage{}, err
	}
	f.eventParams = p
	return f.events, f.err
}

func (f *fakeAuditManagement) Profile(selector auditservice.ProfileSelector) (auditservice.ProfileProjection, error) {
	for _, profile := range f.profiles {
		if profile.Profile.Ref.Name == selector.Name && profile.Profile.Ref.Version == selector.Version {
			return profile, nil
		}
	}
	return auditservice.ProfileProjection{}, auditservice.ErrProfileNotFound
}

func (f *fakeAuditManagement) Standards(
	_ context.Context, _ string,
) ([]auditstandards.PackageProjection, error) {
	return append([]auditstandards.PackageProjection(nil), f.standards...), f.err
}

func (f *fakeAuditManagement) Standard(
	_ context.Context, _ string, selector auditstandards.Reference,
) (auditstandards.PackageProjection, error) {
	for _, standard := range f.standards {
		if standard.Reference == selector {
			return standard, f.err
		}
	}
	return auditstandards.PackageProjection{}, auditstandards.ErrNotFound
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

// ListPage lists the fake Audit, recording how many listed Audits prepared
// selected; preparationErr fails the page read after the selection.
func (f *fakeAuditManagement) ListPage(
	_ context.Context, params auditstore.ListParams, prepared func(auditstore.Audit) bool,
) ([]auditstore.Audit, map[string]map[string]auditservice.PreparationRoleProjection, error) {
	if err := fakePageFetch(params.Limit, auditstore.ErrInvalid); err != nil {
		return nil, nil, err
	}
	if f.err != nil {
		return nil, nil, f.err
	}
	audits := []auditstore.Audit{f.audit}
	result := map[string]map[string]auditservice.PreparationRoleProjection{}
	for _, audit := range audits {
		if prepared(audit) {
			result[audit.AuditID] = f.preparation
		}
	}
	f.preparationPages = append(f.preparationPages, len(result))
	if f.preparationErr != nil {
		return nil, nil, f.preparationErr
	}
	return audits, result, nil
}

func (f *fakeAuditManagement) ListItems(_ context.Context, params auditstore.ListItemsParams) ([]auditstore.Item, error) {
	if err := fakePageFetch(params.Limit, auditstore.ErrInvalid); err != nil {
		return nil, err
	}
	return append([]auditstore.Item(nil), f.started.Items...), f.err
}

func (f *fakeAuditManagement) ListItemAttempts(
	_ context.Context, _, _ string, itemIDs []string,
) (map[string][]auditstore.ItemAttempt, error) {
	if len(itemIDs) == 0 || len(itemIDs) > pagelimit.MaxRows {
		return nil, fmt.Errorf("%w: fake attempt selection of %d items", auditstore.ErrInvalid, len(itemIDs))
	}
	// Like the store, the result answers every requested item, with an empty
	// list for an item without attempts.
	attempts := make(map[string][]auditstore.ItemAttempt, len(itemIDs))
	for _, itemID := range itemIDs {
		attempts[itemID] = append([]auditstore.ItemAttempt{}, f.attempts[itemID]...)
	}
	return attempts, f.err
}

func (f *fakeAuditManagement) GetRound(context.Context, string, string, string) (auditstore.Round, error) {
	return f.started.Round, f.err
}

func (f *fakeAuditManagement) ListCoverage(
	_ context.Context, _, _, _ string, _ int, limit int, _ *uint64,
) (auditservice.CoveragePage, error) {
	if err := fakePageFetch(limit, auditstore.ErrInvalid); err != nil {
		return auditservice.CoveragePage{}, err
	}
	return f.coverage, f.err
}

func (f *fakeAuditManagement) GetReport(
	context.Context, string, string,
) (auditservice.ReportProjection, error) {
	return f.report, f.err
}

func (f *fakeAuditManagement) ListFindings(
	_ context.Context, params auditstore.FindingListParams,
) ([]auditstore.ProjectFindingSummary, error) {
	if err := fakePageFetch(params.Limit, auditstore.ErrInvalid); err != nil {
		return nil, err
	}
	f.findingListParams = params
	result := make([]auditstore.ProjectFindingSummary, 0, len(f.findings))
	for _, finding := range f.findings {
		result = append(result, auditstore.ProjectFindingSummary{FindingID: finding.FindingID, ProjectID: finding.ProjectID, Title: finding.Title, State: finding.State, AnalystVerdict: finding.AnalystVerdict, AnalystSeverity: finding.AnalystSeverity, DuplicateTargetID: finding.DuplicateTargetID, Revision: finding.Revision, CreatedAt: finding.CreatedAt, UpdatedAt: finding.UpdatedAt})
	}
	return result, f.err
}

func (f *fakeAuditManagement) GetFinding(
	context.Context, string, string, string,
) (auditstore.ProjectFinding, error) {
	if len(f.findings) == 0 {
		return auditstore.ProjectFinding{}, f.err
	}
	return f.findings[0], f.err
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
) (auditstore.ReviewRequest, error) {
	if len(f.reviews) == 0 {
		return auditstore.ReviewRequest{}, f.err
	}
	return f.reviews[0], f.err
}

func (f *fakeAuditManagement) ListReviews(
	_ context.Context, params auditstore.ReviewListParams,
) ([]auditstore.ReviewRequest, error) {
	if err := fakePageFetch(params.Limit, auditstore.ErrInvalid); err != nil {
		return nil, err
	}
	f.reviewListParams = params
	return append([]auditstore.ReviewRequest(nil), f.reviews...), f.err
}

func auditTestHandler(audits AuditManagement) *auditHandlers {
	return &auditHandlers{AuditDependencies: AuditDependencies{
		Audits: audits, Projects: newFakeProjectStore(),
		NewID: func(string) (string, error) { return "audit-fixed", nil },
	}}
}

// findingTestHandler serves the finding proposals of the Audits of audits.
func findingTestHandler(audits AuditReader, proposals FindingProposalManagement) *findingHandlers {
	return &findingHandlers{FindingDependencies: FindingDependencies{
		FindingProposals: proposals, FindingCollections: missingFindingCollections{}, FindingExports: missingFindingExports{},
		Audits: audits, Runs: newFakeRunStore(),
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

func (f *fakeAuditManagement) GetWorkspace(_ context.Context, _, _ string) (auditstore.WorkspaceSummary, error) {
	return auditstore.WorkspaceSummary{AuditID: f.audit.AuditID, AuditRevision: f.audit.Revision, AsOf: time.Now(), ExecutionState: string(f.audit.State)}, f.err
}
func (f *fakeAuditManagement) ListFindingsPage(ctx context.Context, params auditstore.FindingListParams) (auditservice.FindingPage, error) {
	if params.AuditRevision != nil && *params.AuditRevision != f.audit.Revision {
		return auditservice.FindingPage{}, auditstore.ErrConflict
	}
	items, err := f.ListFindings(ctx, params)
	return auditservice.FindingPage{Items: items, FindingPageBasis: auditstore.FindingPageBasis{AuditRevision: f.audit.Revision, ProjectRevision: 1, AsOf: time.Now()}}, err
}
func (f *fakeAuditManagement) ListReviewsPage(ctx context.Context, params auditstore.ReviewListParams) (auditservice.ReviewPage, error) {
	if params.AuditRevision != nil && *params.AuditRevision != f.audit.Revision {
		return auditservice.ReviewPage{}, auditstore.ErrConflict
	}
	items, err := f.ListReviews(ctx, params)
	return auditservice.ReviewPage{Items: items, PageBasis: auditstore.PageBasis{AuditRevision: f.audit.Revision, AsOf: time.Now(), Total: len(items)}}, err
}

func (f *fakeAuditManagement) ListOwnerFindings(ctx context.Context, params auditstore.OwnerFindingListParams) (auditstore.FindingWindow, error) {
	if err := fakePageFetch(params.Limit, auditstore.ErrInvalid); err != nil {
		return auditstore.FindingWindow{}, err
	}
	f.ownerFindings = params
	items, err := f.ListFindings(ctx, auditstore.FindingListParams{OwnerID: params.OwnerID, AuditID: "owner-list", Limit: params.Limit})
	window := auditstore.FindingWindow{Items: items}
	if len(items) == params.Limit && params.Limit > 1 {
		window.Items = items[:params.Limit-1]
		window.ContinueBefore = &items[params.Limit-2]
	}
	return window, err
}
func (f *fakeAuditManagement) ListOwnerReviews(_ context.Context, params auditstore.OwnerReviewListParams) ([]auditstore.ReviewRequest, error) {
	if err := fakePageFetch(params.Limit, auditstore.ErrInvalid); err != nil {
		return nil, err
	}
	f.ownerReviews = params
	return f.reviews, f.err
}

func (f *fakeAuditManagement) FindingCandidates(context.Context, auditservice.FindingCandidatesParams) (auditservice.FindingCandidatesPage, error) {
	return auditservice.FindingCandidatesPage{}, f.err
}
func (f *fakeAuditManagement) CompareFindings(context.Context, auditservice.FindingComparisonParams) (auditservice.FindingComparisonPage, error) {
	return auditservice.FindingComparisonPage{}, f.err
}

func (f *fakeAuditManagement) GetProjectFinding(ctx context.Context, _, _, _ string) (auditstore.ProjectFinding, error) {
	if len(f.findings) == 0 {
		return auditstore.ProjectFinding{}, f.err
	}
	return f.findings[0], f.err
}
func (f *fakeAuditManagement) GetAuditProjectFinding(ctx context.Context, owner, audit, id string) (auditstore.ProjectFinding, error) {
	return f.GetFinding(ctx, owner, audit, id)
}
func (f *fakeAuditManagement) ListProjectFindings(ctx context.Context, p auditservice.ProjectFindingListParams) (auditservice.ProjectFindingPage, error) {
	items, err := f.ListOwnerFindings(ctx, p.OwnerFindingListParams)
	return auditservice.ProjectFindingPage{Items: items.Items, ContinueBefore: items.ContinueBefore, ProjectRevision: 1}, err
}
func (f *fakeAuditManagement) EditProjectFinding(_ context.Context, p auditservice.ProjectFindingEditParams) (auditservice.ProjectFindingEditResult, error) {
	if f.err != nil {
		return auditservice.ProjectFindingEditResult{}, f.err
	}
	if len(f.findings) == 0 {
		return auditservice.ProjectFindingEditResult{}, auditstore.ErrNotFound
	}
	value := f.findings[0]
	value.Revision++
	value.AnalystVerdict = &p.Verdict
	value.AnalystSeverity = p.Severity
	value.DuplicateTargetID = p.DuplicateTargetID
	value.Rationale = &p.Rationale
	value.ActorID = &p.OwnerID
	now := time.Now()
	value.AssessedAt = &now
	states := map[auditstore.AnalystVerdict]auditstore.FindingState{auditstore.VerdictTruePositive: auditstore.FindingConfirmed, auditstore.VerdictFalsePositive: auditstore.FindingRejected, auditstore.VerdictDuplicate: auditstore.FindingDuplicate, auditstore.VerdictNeedsEvidence: auditstore.FindingNeedsEvidence, auditstore.VerdictReopen: auditstore.FindingProposed}
	state := states[p.Verdict]
	value.State = &state
	f.findings[0] = value
	return auditservice.ProjectFindingEditResult{Finding: value, Replayed: f.decisionReplayed}, nil
}

func (f *fakeFindingProposalManagement) GetProjectReceipt(_ context.Context, _, project, id string) (findingintake.Receipt, error) {
	for _, source := range f.receipts {
		if source.ProjectID != nil && *source.ProjectID == project && source.ReceiptID == id && source.Retention == findingintake.RetentionProjectHeld && source.ProjectProposal != nil {
			return source, nil
		}
	}
	return findingintake.Receipt{}, findingintake.ErrNotFound
}
func (f *fakeFindingProposalManagement) ListProjectSources(_ context.Context, _, project, id string, q findingintake.ProjectSourceQuery) (findingintake.ProjectSourcePage, error) {
	if q.Limit < 1 || q.Limit > findingintake.MaxProjectSourcePage+1 {
		return findingintake.ProjectSourcePage{}, findingintake.ErrInvalid
	}
	rows := []findingintake.ProjectSourcePreview{}
	for _, s := range f.receipts {
		if s.ProjectID == nil || *s.ProjectID != project || s.ProjectProposal == nil || s.Retention != findingintake.RetentionProjectHeld || (id != "" && (s.FindingID == nil || *s.FindingID != id)) {
			continue
		}
		rows = append(rows, findingintake.ProjectSourcePreview{ReceiptID: s.ReceiptID, ProposalID: s.ProposalID, ProjectID: project, FindingID: s.FindingID, Proposal: *s.ProjectProposal, Origin: s.Origin, EvidenceCount: len(s.ProjectEvidence), PublicationRequested: s.PublicationRequested, CreatedAt: s.CreatedAt})
	}
	return findingintake.ProjectSourcePage{Items: rows, ProjectRevision: 1}, nil
}
func (f *fakeFindingProposalManagement) ListProjectSourceHistory(ctx context.Context, owner, project, id, _ string, _ int, _ *uint64) (findingintake.ProjectSourceHistoryPage, error) {
	if _, err := f.GetProjectReceipt(ctx, owner, project, id); err != nil {
		return findingintake.ProjectSourceHistoryPage{}, err
	}
	return findingintake.ProjectSourceHistoryPage{Items: []artifacts.Metadata{}, ProjectRevision: 1}, nil
}
