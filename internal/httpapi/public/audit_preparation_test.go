package public

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func TestAuditPreparationHandlersWithoutRound(t *testing.T) {
	root := configtest.CopyWithPolicies(t, "../../auditservice/testdata/catalog")
	profileYAML, err := os.ReadFile("../../../api/testdata/audit-composition/prepared-openapi-scan.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "audit-profiles/prepared.yaml"), profileYAML, 0o600); err != nil {
		t.Fatal(err)
	}
	catalog, err := config.Load(root, config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	profile, err := catalog.AuditProfile("prepared-openapi-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(profile)
	inputRevision := "source-r1"
	selection, _ := json.Marshal(auditservice.DraftSelection{Schema: auditservice.DraftSelectionSchema,
		Inputs: map[string]auditstore.ExactArtifact{"source": {Ref: contracts.ArtifactRef{Namespace: "inputs", Name: "source", Revision: &inputRevision}, Digest: auditHandlerDigest("source"), MediaType: "application/zip", SizeBytes: 1}}, RuntimeLabels: []string{}})
	selected, _ := auditservice.DecodeDraftSelection(selection)
	baseline, err := auditservice.EncodeBaseline(auditservice.BaselineSnapshot{Schema: auditservice.BaselineSchema, Inputs: selected.Inputs,
		RuntimeLabels: []string{}, RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(), Skills: []contracts.RunSkillSnapshot{},
		LLMCredentialIDs: []string{}, RuntimeCredentialIDs: []string{}, Standards: []auditstandards.PinnedPackage{}})
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	audit := auditstore.Audit{AuditID: "prepared-audit", OwnerID: "user-1", ProjectID: "project-one",
		Profile:         auditstore.ProfileIdentity{Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest},
		ProfileSnapshot: raw, InputSelection: selection, BaselineSnapshot: baseline, State: auditstore.AuditActive, Phase: auditdomain.AuditPhasePreparing,
		Revision: 2, Dispatch: auditstore.DispatchOpen, Hold: auditstore.HoldHeld, CreatedAt: now, UpdatedAt: now}
	revision, executionID, runID := "retained-output-r1", "prepare-execution", "prepare-run"
	management := &fakeAuditManagement{audit: audit, started: auditservice.StartedAudit{Audit: audit, Items: []auditstore.Item{}},
		preparation: map[string]auditservice.PreparationRoleProjection{"generate-api": {
			PreparationRole: auditstore.PreparationRole{WorkflowRole: "generate-api", Status: auditdomain.PreparationAccepted, Attempts: 1, MaxRunAttempts: 2, ExecutionID: &executionID},
			RunID:           &runID, Outputs: map[string]auditstore.AcceptedPreparationOutput{"api": {ExecutionID: executionID, RunID: runID,
				Output: auditstore.PreparationOutput{WorkflowOutput: "openapi", Retained: auditstore.ExactArtifact{
					Ref: contracts.ArtifactRef{Namespace: "audit-managed", Name: "api", Revision: &revision}, Digest: auditHandlerDigest("api"), MediaType: "application/yaml", SizeBytes: 0}}}},
		}}}
	h := auditTestHandler(management)
	start := auditAuthenticatedRequest(http.MethodPost, "/v1/audits/prepared-audit/start", nil)
	start.SetPathValue("auditId", audit.AuditID)
	start.Header.Set("Idempotency-Key", "start-prepared")
	start.Header.Set("If-Match", `"1"`)
	response := httptest.NewRecorder()
	h.startAudit(response, start)
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
		h.getAudit(response, request)
		if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"phase":"`+string(phase)+`"`) || strings.Contains(response.Body.String(), `"currentRoundId"`) {
			t.Fatalf("preparation detail: %d %s", response.Code, response.Body.String())
		}
		list := auditAuthenticatedRequest(http.MethodGet, "/v1/audits", nil)
		listed := httptest.NewRecorder()
		h.listProjectAudits(listed, list)
		if listed.Code != http.StatusOK || !strings.Contains(listed.Body.String(), `"preparation":{"roles"`) {
			t.Fatalf("preparation list omitted required role status: %d %s", listed.Code, listed.Body.String())
		}
	}
	for _, action := range []struct {
		name   string
		handle func(http.ResponseWriter, *http.Request)
		status int
	}{{"pause", h.pauseAudit, http.StatusOK}, {"resume", h.resumeAudit, http.StatusOK}, {"cancel", h.cancelAudit, http.StatusAccepted}, {"delete", h.deleteAudit, http.StatusAccepted}} {
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
	h.getAudit(drafted, draft)
	if drafted.Code != http.StatusOK || strings.Contains(drafted.Body.String(), `"preparation"`) {
		t.Fatalf("draft falsely entered preparation: %d %s", drafted.Code, drafted.Body.String())
	}
}
