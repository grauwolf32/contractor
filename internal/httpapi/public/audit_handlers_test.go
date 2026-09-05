package public

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
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

type fakeAuditManagement struct {
	profiles []auditservice.ProfileProjection
	audit    auditstore.Audit
	started  auditservice.StartedAudit
	created  auditservice.CreateDraftParams
	start    auditservice.StartParams
	err      error
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

func (f *fakeAuditManagement) GetRound(context.Context, string, string, string) (auditstore.Round, error) {
	return f.started.Round, f.err
}

func (f *fakeAuditManagement) ListCoverage(
	context.Context, string, string, string, int, int,
) ([]auditstore.CoverageRow, error) {
	return nil, f.err
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
