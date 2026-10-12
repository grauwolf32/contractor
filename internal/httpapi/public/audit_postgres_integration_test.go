package public

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

func TestAuditPostgresPublicCreateStartAndQuery(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	configuration := loadPublicAuditConfiguration(t)
	gateway, err := configuration.LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	managedCredentials := newFakeManagedCredentials()
	managedCredentials.lookups["development-worker"] = resolved.CredentialMetadata{
		Ref:        llmgateway.LLMCredentialRef{CredentialID: "development-worker"},
		LLMGateway: gateway.Ref, Unrestricted: true,
	}
	audits, err := auditservice.New(auditservice.Options{
		Pool: pool, Profiles: configuration,
		TransactionLLMCredentials: runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(
			func(pgx.Tx) (resolved.CredentialLookup, error) { return managedCredentials, nil },
		),
		CredentialGuard: managedCredentials,
		Now:             func() time.Time { return time.Date(2026, 9, 5, 20, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatal(err)
	}

	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-public-audit", OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: "Public Audit", IdempotencyKey: "create-public-audit-project",
		RequestDigest: auditHandlerDigest("public-audit-project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	source, err := projectArtifacts.Write(ctx, contracts.ArtifactRef{
		Namespace: "inputs", Name: "source",
	}, artifacts.Payload{MediaType: "application/zip", Data: []byte("fixture source archive")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	checklist, err := projectArtifacts.Write(ctx, contracts.ArtifactRef{
		Namespace: "inputs", Name: "checklist",
	}, artifacts.Payload{MediaType: "application/json", Data: []byte(
		`{"schema":"contractor.audit.checklist.v1","items":[{"key":"check-auth","version":"1","statement":"Verify authentication boundaries.","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"automatic"}]}`,
	)}, nil)
	if err != nil {
		t.Fatal(err)
	}

	h := auditTestHandler(audits)
	h.dependencies.Projects = projects
	requestBody, err := json.Marshal(createAuditRequest{
		Profile: auditProfileSelectorRequest{Name: "public-checklist", Version: "1"},
		Inputs: map[string]contracts.ArtifactRef{
			"source": source.Ref, "checklist": checklist.Ref,
		},
		RuntimeLabels: []string{}, Scope: auditservice.Scope{Objective: "Review authentication"},
	})
	if err != nil {
		t.Fatal(err)
	}
	create := auditAuthenticatedRequest(http.MethodPost, "/v1/projects/"+project.ProjectID+"/audits", requestBody)
	create.SetPathValue("projectId", project.ProjectID)
	create.Header.Set("Content-Type", "application/json")
	create.Header.Set("Idempotency-Key", "create-public-audit")
	created := httptest.NewRecorder()
	h.createAudit(created, create)
	if created.Code != http.StatusCreated || created.Header().Get("ETag") != `"1"` {
		t.Fatalf("create Audit = %d headers=%v body=%s", created.Code, created.Header(), created.Body.String())
	}
	// The same key with another request is permanent; it never creates a
	// second draft or changes the first.
	changedBody, err := json.Marshal(createAuditRequest{
		Profile: auditProfileSelectorRequest{Name: "public-checklist", Version: "1"},
		Inputs: map[string]contracts.ArtifactRef{
			"source": source.Ref, "checklist": checklist.Ref,
		},
		RuntimeLabels: []string{}, Scope: auditservice.Scope{Objective: "Review authorization"},
	})
	if err != nil {
		t.Fatal(err)
	}
	reusedCreate := auditAuthenticatedRequest(http.MethodPost, "/v1/projects/"+project.ProjectID+"/audits", changedBody)
	reusedCreate.SetPathValue("projectId", project.ProjectID)
	reusedCreate.Header.Set("Content-Type", "application/json")
	reusedCreate.Header.Set("Idempotency-Key", "create-public-audit")
	reusedCreated := httptest.NewRecorder()
	h.createAudit(reusedCreated, reusedCreate)
	assertIdempotencyConflict(t, reusedCreated)
	var draft auditResponse
	if err := json.Unmarshal(created.Body.Bytes(), &draft); err != nil || draft.State != auditstore.AuditDraft ||
		draft.Inputs["source"].Ref.Revision == nil || draft.Baseline != nil {
		t.Fatalf("draft response = (%+v, %v)", draft, err)
	}

	start := auditAuthenticatedRequest(http.MethodPost, "/v1/audits/"+draft.AuditID+"/start", nil)
	start.SetPathValue("auditId", draft.AuditID)
	start.Header.Set("Idempotency-Key", "start-public-audit")
	start.Header.Set("If-Match", `"1"`)
	startedResponse := httptest.NewRecorder()
	h.startAudit(startedResponse, start)
	if startedResponse.Code != http.StatusOK || startedResponse.Header().Get("ETag") != `"2"` {
		t.Fatalf("start Audit = %d headers=%v body=%s", startedResponse.Code, startedResponse.Header(), startedResponse.Body.String())
	}
	var started auditStartResponse
	if err := json.Unmarshal(startedResponse.Body.Bytes(), &started); err != nil ||
		started.Audit.State != auditstore.AuditActive || started.Audit.HoldState != auditstore.HoldHeld ||
		started.Audit.Baseline == nil || len(started.Items) != 1 || started.Round.ExpectedItemCount != 1 {
		t.Fatalf("started response = (%+v, %v)", started, err)
	}

	replay := auditAuthenticatedRequest(http.MethodPost, "/v1/audits/"+draft.AuditID+"/start", nil)
	replay.SetPathValue("auditId", draft.AuditID)
	replay.Header.Set("Idempotency-Key", "start-public-audit")
	replay.Header.Set("If-Match", `"1"`)
	replayed := httptest.NewRecorder()
	h.startAudit(replayed, replay)
	if replayed.Code != http.StatusOK || replayed.Header().Get("Idempotency-Replayed") != "true" ||
		replayed.Body.String() != startedResponse.Body.String() {
		t.Fatalf("start replay = %d headers=%v body=%s", replayed.Code, replayed.Header(), replayed.Body.String())
	}
	reusedStart := auditAuthenticatedRequest(http.MethodPost, "/v1/audits/"+draft.AuditID+"/start", nil)
	reusedStart.SetPathValue("auditId", draft.AuditID)
	reusedStart.Header.Set("Idempotency-Key", "start-public-audit")
	reusedStart.Header.Set("If-Match", `"2"`)
	reusedStarted := httptest.NewRecorder()
	h.startAudit(reusedStarted, reusedStart)
	assertIdempotencyConflict(t, reusedStarted)

	coverage := auditAuthenticatedRequest(http.MethodGet, "/v1/audits/"+draft.AuditID+"/coverage", nil)
	coverage.SetPathValue("auditId", draft.AuditID)
	coverageResponse := httptest.NewRecorder()
	h.listAuditCoverage(coverageResponse, coverage)
	var page auditCoveragePageResponse
	if coverageResponse.Code != http.StatusOK || json.Unmarshal(coverageResponse.Body.Bytes(), &page) != nil ||
		len(page.Items) != 1 || page.Items[0].Coverage.Status != auditstore.CoverageNotTested {
		t.Fatalf("coverage = %d body=%s", coverageResponse.Code, coverageResponse.Body.String())
	}
	if page.Items[0].Details == nil || page.Items[0].Details.Objective == "" || !json.Valid(page.Items[0].Details.TaskDocument) {
		t.Fatalf("coverage omitted the worker task: %s", coverageResponse.Body.String())
	}
	if current, err := audits.Get(ctx, project.OwnerID, draft.AuditID); err != nil || page.AuditRevision != current.Revision {
		t.Fatalf("coverage page revision = %d, Audit = (%+v, %v)", page.AuditRevision, current, err)
	}
	for _, pin := range []struct {
		revision uint64
		status   int
	}{{page.AuditRevision, http.StatusOK}, {page.AuditRevision - 1, http.StatusConflict}} {
		pinned := auditAuthenticatedRequest(http.MethodGet,
			"/v1/audits/"+draft.AuditID+"/coverage?auditRevision="+strconv.FormatUint(pin.revision, 10), nil)
		pinned.SetPathValue("auditId", draft.AuditID)
		pinnedResponse := httptest.NewRecorder()
		h.listAuditCoverage(pinnedResponse, pinned)
		if pinnedResponse.Code != pin.status {
			t.Fatalf("coverage pinned to revision %d = %d %s", pin.revision, pinnedResponse.Code, pinnedResponse.Body.String())
		}
	}

	t.Run("paused Resume and terminal rejection", func(t *testing.T) {
		mutate := func(action, key, revision string, body []byte, handler func(http.ResponseWriter, *http.Request)) *httptest.ResponseRecorder {
			t.Helper()
			request := auditAuthenticatedRequest(http.MethodPost, "/v1/audits/"+draft.AuditID+"/"+action, body)
			request.SetPathValue("auditId", draft.AuditID)
			request.Header.Set("Idempotency-Key", key)
			request.Header.Set("If-Match", revision)
			if body != nil {
				request.Header.Set("Content-Type", "application/json")
			}
			response := httptest.NewRecorder()
			handler(response, request)
			return response
		}
		paused := mutate("pause", "pause-public-audit", `"2"`, nil, h.pauseAudit)
		if paused.Code != http.StatusOK {
			t.Fatalf("pause = %d %s", paused.Code, paused.Body.String())
		}
		assertIdempotencyConflict(t, mutate("pause", "pause-public-audit", `"3"`, nil, h.pauseAudit))
		body := []byte(`{"deadlineSeconds":0}`)
		resumed := mutate("resume", "resume-public-audit", `"3"`, body, h.resumeAudit)
		var current auditResponse
		if resumed.Code != http.StatusOK || json.Unmarshal(resumed.Body.Bytes(), &current) != nil || current.State != auditstore.AuditActive || current.DeadlineAt != nil || current.Baseline == nil {
			t.Fatalf("paused Resume = %d %s", resumed.Code, resumed.Body.String())
		}
		replayed := mutate("resume", "resume-public-audit", `"3"`, body, h.resumeAudit)
		if replayed.Code != http.StatusOK || replayed.Header().Get("Idempotency-Replayed") != "true" || replayed.Body.String() != resumed.Body.String() {
			t.Fatalf("Resume replay = %d %s", replayed.Code, replayed.Body.String())
		}
		assertIdempotencyConflict(t, mutate("resume", "resume-public-audit", `"3"`, []byte(`{"deadlineSeconds":3600}`), h.resumeAudit))
		for _, state := range []auditstore.AuditState{auditstore.AuditCompleted, auditstore.AuditFailed} {
			if _, err := pool.Exec(ctx, `UPDATE audits SET state=$2,dispatch_state='closed',hold_state='released',finished_at=clock_timestamp(),stop_reason_code='deadline_exhausted',stop_reason_message='Legacy deadline' WHERE audit_id=$1`, draft.AuditID, state); err != nil {
				t.Fatal(err)
			}
			rejected := mutate("resume", "resume-terminal-"+string(state), `"4"`, body, h.resumeAudit)
			if rejected.Code != http.StatusPreconditionFailed {
				t.Fatalf("terminal %s Resume = %d %s", state, rejected.Code, rejected.Body.String())
			}
		}
	})

}

func loadPublicAuditConfiguration(t *testing.T) *config.Snapshot {
	t.Helper()
	targetRoot := filepath.Join(t.TempDir(), "catalog")
	if err := os.CopyFS(targetRoot, os.DirFS(publicCatalogFixture)); err != nil {
		t.Fatalf("copy configuration: %v", err)
	}
	profile := []byte(`apiVersion: contractor/v1alpha1
kind: AuditProfile
metadata: {name: public-checklist, version: "1"}
spec:
  mode: custom-checklist
  standards: []
  inputs:
    source: {required: true, mediaTypes: [application/zip]}
    checklist: {required: true, mediaTypes: [application/json]}
  inventory:
    implementation: checklist@1
    source: {source: audit-input, name: checklist}
    itemWorkflowRole: check
  workflows:
    check:
      kind: check
      ref: openapi-from-workspace@7
      inputs:
        source: {source: audit-input, name: source}
      parameters:
        objective: {source: item-field, name: subjectKey}
      outputs: {result: openapi}
  execution:
    roundMode: fixed-barrier
    maxRounds: 1
    batchSize: 1
    maxItemsPerRound: 10
    maxItemsTotal: 10
    maxSubmittedRuns: 20
    maxItemRunAttempts: 2
    deadlineSeconds: 3600
    maxEvidenceBytes: 1048576
    incompleteRound: assess-with-gaps
  interaction:
    activeChecks: prohibited
    findingConfirmation: disabled
    notApplicable: profile-rule
    reportAcceptance: automatic
`)
	if err := os.WriteFile(filepath.Join(targetRoot, "audit-profiles", "public_checklist.yaml"), profile, 0o600); err != nil {
		t.Fatal(err)
	}
	snapshot, err := configload.Load(targetRoot)
	if err != nil {
		t.Fatalf("load public Audit configuration: %v", err)
	}
	return snapshot
}
