package public

import (
	"context"
	"encoding/json"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
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
	audits, err := auditservice.New(auditservice.Options{Storage: artifactpgtest.NewStorage(),
		Pool: pool, Profiles: configuration,
		TransactionLLMCredentials: runtimeconfigtest.TransactionLLMCredentialLookupFactoryFunc(
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
	projectArtifacts, err := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Project(project.ProjectID)
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
	h.Projects = projects
	requestBody, err := json.Marshal(createAuditRequest{
		Labels:  map[string]string{"cloud": "aws", "team": "appsec"},
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
	serve(h.createAudit)(created, create)
	if created.Code != http.StatusCreated || created.Header().Get("ETag") != `"1"` {
		t.Fatalf("create Audit = %d headers=%v body=%s", created.Code, created.Header(), created.Body.String())
	}
	// The same key with another request is permanent; it never creates a
	// second draft or changes the first.
	changedBody, err := json.Marshal(createAuditRequest{
		Labels:  map[string]string{"local": ""},
		Profile: auditProfileSelectorRequest{Name: "public-checklist", Version: "1"},
		Inputs: map[string]contracts.ArtifactRef{
			"source": source.Ref, "checklist": checklist.Ref,
		},
		RuntimeLabels: []string{}, Scope: auditservice.Scope{Objective: "Review authentication"},
	})
	if err != nil {
		t.Fatal(err)
	}
	reusedCreate := auditAuthenticatedRequest(http.MethodPost, "/v1/projects/"+project.ProjectID+"/audits", changedBody)
	reusedCreate.SetPathValue("projectId", project.ProjectID)
	reusedCreate.Header.Set("Content-Type", "application/json")
	reusedCreate.Header.Set("Idempotency-Key", "create-public-audit")
	reusedCreated := httptest.NewRecorder()
	serve(h.createAudit)(reusedCreated, reusedCreate)
	assertIdempotencyConflict(t, reusedCreated)
	var draft auditResponse
	if err := json.Unmarshal(created.Body.Bytes(), &draft); err != nil || draft.State != auditstore.AuditDraft ||
		!slices.Equal(draft.RequiredPlacementLabels, []string{"cloud/aws"}) ||
		draft.Inputs["source"].Ref.Revision == nil || draft.Baseline != nil {
		t.Fatalf("draft response = (%+v, %v)", draft, err)
	}

	start := auditAuthenticatedRequest(http.MethodPost, "/v1/audits/"+draft.AuditID+"/start", nil)
	start.SetPathValue("auditId", draft.AuditID)
	start.Header.Set("Idempotency-Key", "start-public-audit")
	start.Header.Set("If-Match", `"1"`)
	startedResponse := httptest.NewRecorder()
	serve(h.startAudit)(startedResponse, start)
	if startedResponse.Code != http.StatusOK || startedResponse.Header().Get("ETag") != `"2"` {
		t.Fatalf("start Audit = %d headers=%v body=%s", startedResponse.Code, startedResponse.Header(), startedResponse.Body.String())
	}
	var started auditStartResponse
	if err := json.Unmarshal(startedResponse.Body.Bytes(), &started); err != nil ||
		started.Audit.State != auditstore.AuditActive || started.Audit.HoldState != auditstore.HoldHeld ||
		started.Audit.Baseline == nil || started.Audit.TaskCount != 0 {
		t.Fatalf("started response = (%+v, %v)", started, err)
	}
	if !slices.Equal(started.Audit.Baseline.RequiredPlacementLabels, draft.RequiredPlacementLabels) {
		t.Fatalf("start lost placement selection: %+v", started.Audit.Baseline)
	}
	if !maps.Equal(started.Audit.Labels, draft.Labels) || !maps.Equal(started.Audit.Baseline.Labels, draft.Labels) || started.Audit.Labels["team"] != "appsec" {
		t.Fatalf("start lost immutable labels: %+v", started.Audit)
	}

	replay := auditAuthenticatedRequest(http.MethodPost, "/v1/audits/"+draft.AuditID+"/start", nil)
	replay.SetPathValue("auditId", draft.AuditID)
	replay.Header.Set("Idempotency-Key", "start-public-audit")
	replay.Header.Set("If-Match", `"1"`)
	replayed := httptest.NewRecorder()
	serve(h.startAudit)(replayed, replay)
	if replayed.Code != http.StatusOK || replayed.Header().Get("Idempotency-Replayed") != "true" ||
		replayed.Body.String() != startedResponse.Body.String() {
		t.Fatalf("start replay = %d headers=%v body=%s", replayed.Code, replayed.Header(), replayed.Body.String())
	}
	reusedStart := auditAuthenticatedRequest(http.MethodPost, "/v1/audits/"+draft.AuditID+"/start", nil)
	reusedStart.SetPathValue("auditId", draft.AuditID)
	reusedStart.Header.Set("Idempotency-Key", "start-public-audit")
	reusedStart.Header.Set("If-Match", `"2"`)
	reusedStarted := httptest.NewRecorder()
	serve(h.startAudit)(reusedStarted, reusedStart)
	assertIdempotencyConflict(t, reusedStarted)

	tasksRequest := auditAuthenticatedRequest(http.MethodGet, "/v1/audits/"+draft.AuditID+"/tasks", nil)
	tasksRequest.SetPathValue("auditId", draft.AuditID)
	tasksResponse := httptest.NewRecorder()
	serve(h.listAuditTasks).ServeHTTP(tasksResponse, tasksRequest)
	var taskPage struct {
		Items []auditTaskResponse `json:"items"`
	}
	if tasksResponse.Code != http.StatusOK || json.Unmarshal(tasksResponse.Body.Bytes(), &taskPage) != nil || len(taskPage.Items) != 0 {
		t.Fatalf("start invented selected tasks before controller intake: %d %s", tasksResponse.Code, tasksResponse.Body.String())
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
		paused := mutate("pause", "pause-public-audit", `"2"`, nil, serve(h.pauseAudit))
		if paused.Code != http.StatusOK {
			t.Fatalf("pause = %d %s", paused.Code, paused.Body.String())
		}
		assertIdempotencyConflict(t, mutate("pause", "pause-public-audit", `"3"`, nil, serve(h.pauseAudit)))
		body := []byte(`{"deadlineSeconds":0}`)
		resumed := mutate("resume", "resume-public-audit", `"3"`, body, serve(h.resumeAudit))
		var current auditResponse
		if resumed.Code != http.StatusOK || json.Unmarshal(resumed.Body.Bytes(), &current) != nil || current.State != auditstore.AuditActive || current.DeadlineAt != nil || current.Baseline == nil {
			t.Fatalf("paused Resume = %d %s", resumed.Code, resumed.Body.String())
		}
		replayed := mutate("resume", "resume-public-audit", `"3"`, body, serve(h.resumeAudit))
		if replayed.Code != http.StatusOK || replayed.Header().Get("Idempotency-Replayed") != "true" || replayed.Body.String() != resumed.Body.String() {
			t.Fatalf("Resume replay = %d %s", replayed.Code, replayed.Body.String())
		}
		assertIdempotencyConflict(t, mutate("resume", "resume-public-audit", `"3"`, []byte(`{"deadlineSeconds":3600}`), serve(h.resumeAudit)))
		for _, state := range []auditstore.AuditState{auditstore.AuditCompleted, auditstore.AuditFailed} {
			if _, err := pool.Exec(ctx, `UPDATE audits SET state=$2,dispatch_state='closed',hold_state='released',finished_at=clock_timestamp(),stop_reason_code='deadline_exhausted',stop_reason_message='Legacy deadline' WHERE audit_id=$1`, draft.AuditID, state); err != nil {
				t.Fatal(err)
			}
			rejected := mutate("resume", "resume-terminal-"+string(state), `"4"`, body, serve(h.resumeAudit))
			if rejected.Code != http.StatusPreconditionFailed {
				t.Fatalf("terminal %s Resume = %d %s", state, rejected.Code, rejected.Body.String())
			}
		}
	})

}

func loadPublicAuditConfiguration(t *testing.T) *config.Snapshot {
	t.Helper()
	return loadPublicAuditConfigurationWith(t, func(string) {})
}

// loadPublicAuditConfigurationWith loads the public Audit catalog after edit
// changed its copy at root.
func loadPublicAuditConfigurationWith(t *testing.T, edit func(root string)) *config.Snapshot {
	t.Helper()
	targetRoot := filepath.Join(t.TempDir(), "catalog")
	if err := os.CopyFS(targetRoot, os.DirFS(publicCatalogFixture)); err != nil {
		t.Fatalf("copy configuration: %v", err)
	}
	edit(targetRoot)
	profile := []byte(`apiVersion: contractor/v1alpha1
kind: AuditProfile
metadata: {name: public-checklist, version: "1"}
spec:
  schema: contractor.audit.program.v1
  inputs:
    source: {required: true, media_types: [application/zip]}
    checklist: {required: true, media_types: [application/json]}
  workflows:
    inspect:
      ref: openapi-from-workspace@7
      inputs: {source: {source: audit-input, name: source}}
      parameters: {}
      outputs: {material: openapi}
  hypotheses: {}
  initial_work: [{key: inspect, workflow: inspect, objective: Inspect selected source}]
  limits: {max_tasks: 10, max_iterations: 1, max_tasks_per_iteration: 10, batch_size: 1, max_submitted_runs: 20, max_attempts: 2, deadline_seconds: 3600, max_evidence_bytes: 1048576, max_hypotheses: 128, max_hypothesis_revisions: 128, max_hypothesis_bytes: 32768}
  policy: {active_operations: prohibited}
`)
	if err := os.WriteFile(filepath.Join(targetRoot, "audit-profiles", "public_checklist.yaml"), profile, 0o600); err != nil {
		t.Fatal(err)
	}
	snapshot, _, err := configload.LoadUnionReadOnly(targetRoot, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatalf("load public Audit configuration: %v", err)
	}
	return snapshot
}
