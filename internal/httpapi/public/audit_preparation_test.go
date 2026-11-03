package public

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5/pgconn"
)

// preparedHandlerAudit returns a preparing Audit of the prepared OpenAPI scan
// profile, which pins the generate-api prepare role, and that role's
// accepted preparation.
func preparedHandlerAudit(t *testing.T) (auditstore.Audit, map[string]auditservice.PreparationRoleProjection) {
	t.Helper()
	root := configtest.CopyWithPolicies(t, "../../auditservice/testdata/catalog")
	profileYAML := contractstest.ReadFile(t, "api", "testdata", "audit-composition", "prepared-openapi-scan.yaml")
	if err := os.WriteFile(filepath.Join(root, "audit-profiles/prepared.yaml"), profileYAML, 0o600); err != nil {
		t.Fatal(err)
	}
	catalog, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	profile, err := catalog.AuditProfile("prepared-openapi-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(profile)
	inputRevision := "source-r1"
	selection, _ := json.Marshal(auditservice.DraftSelection{Labels: map[string]string{}, RequiredPlacementLabels: []string{}, Schema: auditservice.DraftSelectionSchema,
		Inputs: map[string]auditstore.ExactArtifact{"source": {Ref: contracts.ArtifactRef{Namespace: "inputs", Name: "source", Revision: &inputRevision}, Digest: auditHandlerDigest("source"), MediaType: "application/zip", SizeBytes: 1}}, RuntimeLabels: []string{}})
	selected, _ := auditservice.DecodeDraftSelection(selection)
	baseline, err := auditservice.EncodeBaseline(auditservice.BaselineSnapshot{Labels: map[string]string{}, RequiredPlacementLabels: []string{}, ActiveChecks: contracts.ActiveChecksAutomatic, Schema: auditservice.BaselineSchema, Inputs: selected.Inputs,
		RuntimeLabels: []string{}, RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(), Skills: []agentskills.RunSkillSnapshot{},
		LLMCredentialIDs: []string{}, RuntimeCredentialIDs: []string{}, Standards: []auditstandards.PinnedPackage{}})
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	audit := auditstore.Audit{AuditID: "prepared-audit", OwnerID: "user-1", ProjectID: "project-one",
		Profile:         auditstore.ProfileIdentity{Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest},
		ProfileSnapshot: raw, InputSelection: selection, BaselineSnapshot: baseline, State: auditstore.AuditActive, Phase: auditdomain.AuditPhasePreparing,
		Revision: 2, Dispatch: auditstore.DispatchOpen, Hold: auditstore.HoldHeld, CreatedAt: now, UpdatedAt: now,
		Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 16, MaxItemsTotal: 16,
			MaxSubmittedRuns: 50, MaxItemRunAttempts: 3, MaxEvidenceBytes: 64 << 20}}
	revision, executionID, runID := "retained-output-r1", "prepare-execution", "prepare-run"
	return audit, map[string]auditservice.PreparationRoleProjection{"generate-api": {
		PreparationRole: auditstore.PreparationRole{WorkflowRole: "generate-api", Status: auditdomain.PreparationAccepted, Attempts: 1, MaxRunAttempts: 2, ExecutionID: &executionID},
		RunID:           &runID, Outputs: map[string]auditstore.AcceptedPreparationOutput{"api": {ExecutionID: executionID, RunID: runID,
			Output: auditstore.PreparationOutput{WorkflowOutput: "openapi", Retained: auditstore.ExactArtifact{
				Ref: contracts.ArtifactRef{Namespace: "audit-managed", Name: "api", Revision: &revision}, Digest: auditHandlerDigest("api"), MediaType: "application/yaml", SizeBytes: 0}}}},
	}}
}

func TestSharedAuditPreparationHandlersWithoutRound(t *testing.T) {
	audit, preparation := preparedHandlerAudit(t)
	management := &fakeAuditManagement{audit: audit, started: auditservice.StartedAudit{Audit: audit, Items: []auditstore.Item{}},
		preparation: preparation}
	h := auditTestHandler(management)
	start := auditAuthenticatedRequest(http.MethodPost, "/v1/audits/prepared-audit/start", nil)
	start.SetPathValue("auditId", audit.AuditID)
	start.Header.Set("Idempotency-Key", "start-prepared")
	start.Header.Set("If-Match", `"1"`)
	response := httptest.NewRecorder()
	serve(h.startAudit)(response, start)
	var body map[string]json.RawMessage
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &body) != nil || body["round"] != nil || string(body["items"]) != "[]" || !strings.Contains(string(body["audit"]), `"phase":"preparing"`) {
		t.Fatalf("preparation start invented a Round: %d %s", response.Code, response.Body.String())
	}
	if !strings.Contains(response.Body.String(), `"revision":"retained-output-r1"`) || !strings.Contains(response.Body.String(), `"sizeBytes":0`) || management.preparationOwner != "user-1" {
		t.Fatalf("preparation projection lost exact output or owner: %s", response.Body.String())
	}
	for _, phase := range []auditdomain.AuditPhase{auditdomain.AuditPhasePreparing, auditdomain.AuditPhaseInventory} {
		management.audit.Phase = phase
		request := auditAuthenticatedRequest(http.MethodGet, "/v1/audits/prepared-audit", nil)
		request.SetPathValue("auditId", audit.AuditID)
		response := httptest.NewRecorder()
		serve(h.getAudit)(response, request)
		if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"phase":"`+string(phase)+`"`) || strings.Contains(response.Body.String(), `"currentRoundId"`) {
			t.Fatalf("preparation detail: %d %s", response.Code, response.Body.String())
		}
		list := auditAuthenticatedRequest(http.MethodGet, "/v1/audits", nil)
		listed := httptest.NewRecorder()
		serve(h.listProjectAudits)(listed, list)
		if listed.Code != http.StatusOK || !strings.Contains(listed.Body.String(), `"preparation":{"roles"`) ||
			len(management.preparationPages) == 0 || management.preparationPages[len(management.preparationPages)-1] != 1 {
			t.Fatalf("preparation list omitted required role status: %d %s", listed.Code, listed.Body.String())
		}
	}
	// A failed page read fails the list with its own class; it never returns
	// the page without preparation.
	for _, failure := range []struct {
		err    error
		status int
		code   string
	}{
		{fmt.Errorf("list Audit page preparation: %w", &pgconn.PgError{Code: persistencepostgres.SQLStateSerializationFailure}),
			http.StatusServiceUnavailable, "storage_transaction_conflict"},
		{errors.New("storage unavailable"), http.StatusInternalServerError, "internal_error"},
	} {
		management.preparationErr = failure.err
		listed := httptest.NewRecorder()
		serve(h.listProjectAudits)(listed, auditAuthenticatedRequest(http.MethodGet, "/v1/audits", nil))
		var body errorResponse
		if listed.Code != failure.status || json.Unmarshal(listed.Body.Bytes(), &body) != nil || body.Code != failure.code {
			t.Fatalf("list with hydration error %v = %d %s", failure.err, listed.Code, listed.Body.String())
		}
	}
	management.preparationErr = nil
	management.audit.Phase, management.audit.State = auditdomain.AuditPhaseNotStarted, auditstore.AuditDraft
	drafts := httptest.NewRecorder()
	serve(h.listProjectAudits)(drafts, auditAuthenticatedRequest(http.MethodGet, "/v1/audits", nil))
	if drafts.Code != http.StatusOK || strings.Contains(drafts.Body.String(), `"preparation"`) ||
		management.preparationPages[len(management.preparationPages)-1] != 0 {
		t.Fatalf("draft list hydrated preparation: %d %s", drafts.Code, drafts.Body.String())
	}
	management.audit.Phase, management.audit.State = auditdomain.AuditPhaseInventory, auditstore.AuditActive
	for _, action := range []struct {
		name   string
		handle func(http.ResponseWriter, *http.Request)
		status int
	}{{"pause", serve(h.pauseAudit), http.StatusOK}, {"resume", serve(h.resumeAudit), http.StatusOK}, {"cancel", serve(h.cancelAudit), http.StatusAccepted}, {"delete", serve(h.deleteAudit), http.StatusAccepted}} {
		request := auditAuthenticatedRequest(http.MethodPost, "/v1/audits/prepared-audit/"+action.name, nil)
		if action.name == "delete" {
			request.Method = http.MethodDelete
		}
		request.SetPathValue("auditId", audit.AuditID)
		request.Header.Set("Idempotency-Key", "prepared-"+action.name)
		request.Header.Set("If-Match", `"2"`)
		response := httptest.NewRecorder()
		action.handle(response, request)
		if response.Code != action.status || !strings.Contains(response.Body.String(), `"phase":"inventory"`) || strings.Contains(response.Body.String(), `"currentRoundId"`) {
			t.Fatalf("preparation %s: %d %s", action.name, response.Code, response.Body.String())
		}
	}
	management.audit.Phase = auditdomain.AuditPhaseNotStarted
	management.audit.State = auditstore.AuditDraft
	management.audit.BaselineSnapshot = nil
	draft := auditAuthenticatedRequest(http.MethodGet, "/v1/audits/prepared-audit", nil)
	draft.SetPathValue("auditId", audit.AuditID)
	drafted := httptest.NewRecorder()
	serve(h.getAudit)(drafted, draft)
	if drafted.Code != http.StatusOK || strings.Contains(drafted.Body.String(), `"preparation"`) {
		t.Fatalf("draft falsely entered preparation: %d %s", drafted.Code, drafted.Body.String())
	}
}

// TestSharedPublicAuditDetailSnapshotContracts serves the
// detail and every mutation response built from it for a preparing Audit
// whose detail snapshot is a later revision than the read or the mutation
// returned, as when the Audit Controller commits in between, and validates
// each answer against the public contract: one revision, ETag and
// projection, never an undeclared or misleading conflict (ASC-R149,
// V414-018).
func TestSharedPublicAuditDetailSnapshotContracts(t *testing.T) {
	audit, preparation := preparedHandlerAudit(t)
	later := audit
	later.Revision, later.UpdatedAt = audit.Revision+1, audit.UpdatedAt.Add(time.Second)
	management := &fakeAuditManagement{audit: audit, started: auditservice.StartedAudit{Audit: audit, Items: []auditstore.Item{}},
		preparation: preparation, snapshot: &later}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(dependencies *testServices) { dependencies.Audits = management })
	router := publicContractRouter(t)
	for _, operation := range []struct {
		name    string
		request *http.Request
		status  int
	}{
		{"getAudit", jsonContractRequest(http.MethodGet, "/v1/audits/prepared-audit", ""), http.StatusOK},
		{"startAudit", jsonContractRequest(http.MethodPost, "/v1/audits/prepared-audit/start", "",
			"Idempotency-Key", "start-later", "If-Match", `"1"`), http.StatusOK},
		{"pauseAudit", jsonContractRequest(http.MethodPost, "/v1/audits/prepared-audit/pause", "",
			"Idempotency-Key", "pause-later", "If-Match", `"2"`), http.StatusOK},
		{"resumeAudit", jsonContractRequest(http.MethodPost, "/v1/audits/prepared-audit/resume", "",
			"Idempotency-Key", "resume-later", "If-Match", `"2"`), http.StatusOK},
		{"cancelAudit", jsonContractRequest(http.MethodPost, "/v1/audits/prepared-audit/cancel", "",
			"Idempotency-Key", "cancel-later", "If-Match", `"2"`), http.StatusAccepted},
		{"deleteAudit", jsonContractRequest(http.MethodDelete, "/v1/audits/prepared-audit", "",
			"Idempotency-Key", "delete-later", "If-Match", `"2"`), http.StatusAccepted},
	} {
		t.Run(operation.name, func(t *testing.T) {
			response := serveAndValidatePublicContract(t, router, fixture.handler, operation.request, true)
			body := response.Body.Bytes()
			if operation.name == "startAudit" {
				var started struct {
					Audit json.RawMessage `json:"audit"`
				}
				if err := json.Unmarshal(body, &started); err != nil {
					t.Fatal(err)
				}
				body = started.Audit
			}
			var read struct {
				Revision    uint64                    `json:"revision"`
				Preparation *auditPreparationResponse `json:"preparation"`
			}
			if response.Code != operation.status || json.Unmarshal(body, &read) != nil || read.Revision != later.Revision ||
				response.Header().Get("ETag") != `"3"` || read.Preparation == nil ||
				read.Preparation.Roles["generate-api"].Status != auditdomain.PreparationAccepted {
				t.Fatalf("%s = %d ETag %s: %s", operation.name, response.Code, response.Header().Get("ETag"), response.Body.String())
			}
		})
	}
	// A failed detail read keeps its class: a transaction conflict is the
	// declared retryable 503, never the 409 of a lost race.
	management.detailErr = fmt.Errorf("read Audit detail: %w", &pgconn.PgError{Code: persistencepostgres.SQLStateSerializationFailure})
	for _, request := range []*http.Request{
		jsonContractRequest(http.MethodGet, "/v1/audits/prepared-audit", ""),
		jsonContractRequest(http.MethodPost, "/v1/audits/prepared-audit/pause", "", "Idempotency-Key", "pause-conflict", "If-Match", `"2"`),
	} {
		response := serveAndValidatePublicContract(t, router, fixture.handler, request, true)
		var failure errorResponse
		if response.Code != http.StatusServiceUnavailable || json.Unmarshal(response.Body.Bytes(), &failure) != nil ||
			failure.Code != "storage_transaction_conflict" || !failure.Retryable {
			t.Fatalf("%s %s with a detail transaction conflict = %d %s", request.Method, request.URL.Path, response.Code, response.Body.String())
		}
	}
}
