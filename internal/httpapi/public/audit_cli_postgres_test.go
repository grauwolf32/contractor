//go:build integration

package public

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
)

// The real CLI process talks to ordinary public routes backed by PostgreSQL.
// The first task acknowledgement is deliberately lost after commit.
func TestAuditPostgresCLIResearchAndMaterialPublication(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPublicPool(t, ctx)
	config := loadPublicAuditConfiguration(t)
	gateway, err := config.LLMGateway("local-litellm@1")
	if err != nil {
		t.Fatal(err)
	}
	credentials := newFakeManagedCredentials()
	credentials.lookups["development-worker"] = resolved.CredentialMetadata{Ref: llmgateway.LLMCredentialRef{CredentialID: "development-worker"}, LLMGateway: gateway.Ref, Unrestricted: true}
	service, err := auditservice.New(auditservice.Options{Storage: artifactpgtest.NewStorage(), Pool: pool, Profiles: config, CredentialGuard: credentials, TransactionLLMCredentials: runtimeconfigtest.TransactionLLMCredentialLookupFactoryFunc(func(pgx.Tx) (resolved.CredentialLookup, error) { return credentials, nil })})
	if err != nil {
		t.Fatal(err)
	}
	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{ProjectID: "project-cli", OwnerID: "user-1", Kind: projectstore.KindProject, Name: "CLI research", IdempotencyKey: "project-cli", RequestDigest: auditHandlerDigest("project-cli")})
	if err != nil {
		t.Fatal(err)
	}
	art := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	store, _ := art.Project(project.ProjectID)
	inputs := map[string]contracts.ArtifactRef{}
	for name, payload := range map[string]artifacts.Payload{"source": {MediaType: "application/zip", Data: []byte("exact source")}, "checklist": {MediaType: "application/json", Data: []byte(`{"items":[]}`)}} {
		out, err := store.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: name}, payload, nil)
		if err != nil {
			t.Fatal(err)
		}
		inputs[name] = out.Ref
	}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(d *testServices) {
		d.Audits = service
		d.Projects = projects
		d.NewID = func(prefix string) (string, error) { return prefix + "cli", nil }
	})
	taskPosts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "POST" && strings.HasSuffix(r.URL.Path, "/tasks") {
			taskPosts++
			if taskPosts == 1 {
				response := httptest.NewRecorder()
				fixture.handler.ServeHTTP(response, r)
				if response.Code != 200 {
					t.Errorf("task commit: %d %s", response.Code, response.Body)
				}
				conn, _, err := w.(http.Hijacker).Hijack()
				if err != nil {
					t.Error(err)
					return
				}
				conn.Close()
				return
			}
		}
		fixture.handler.ServeHTTP(w, r)
	}))
	defer server.Close()
	binary := filepath.Join(t.TempDir(), "contractor")
	build := exec.CommandContext(ctx, "go", "build", "-o", binary, "../../../cmd/contractor")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build CLI: %v %s", err, out)
	}
	call := func(body any, args ...string) []byte {
		t.Helper()
		encoded, _ := json.Marshal(body)
		cmd := exec.CommandContext(ctx, binary, append([]string{"--server", server.URL, "--output", "json", "audit"}, args...)...)
		cmd.Env = append(os.Environ(), "CONTRACTOR_API_TOKEN="+testBearerToken)
		cmd.Stdin = bytes.NewReader(encoded)
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		out, err := cmd.Output()
		if err != nil {
			t.Fatalf("audit %v: %v %s", args, err, stderr.String())
		}
		return out
	}
	var audit auditResponse
	data := call(map[string]any{"profile": map[string]string{"name": "public-checklist", "version": "1"}, "inputs": inputs, "scope": map[string]string{}, "runtimeLabels": []string{}, "labels": map[string]string{}}, "create", project.ProjectID, "--idempotency-key", "draft", "--request", "-")
	if json.Unmarshal(data, &audit) != nil || audit.AuditID == "" {
		t.Fatalf("create: %s", data)
	}
	call(nil, "start", audit.AuditID, "--revision", "1", "--idempotency-key", "start")
	call(nil, "pause", audit.AuditID, "--revision", "2", "--idempotency-key", "pause")
	request := map[string]any{"work": []any{map[string]any{"key": "first", "workflow": "inspect", "objective": "Inspect source", "inputs": map[string]any{}}, map[string]any{"key": "second", "workflow": "inspect", "objective": "Inspect separately", "inputs": map[string]any{}}}}
	var accepted struct {
		TaskIDs []string `json:"task_ids"`
	}
	data = call(request, "tasks", "create", audit.AuditID, "--revision", "3", "--idempotency-key", "work", "--request", "-")
	if json.Unmarshal(data, &accepted) != nil || len(accepted.TaskIDs) != 2 || taskPosts != 2 {
		t.Fatalf("lost-reply receipt: posts=%d %s", taskPosts, data)
	}
	current, err := service.Get(ctx, "user-1", audit.AuditID)
	if err != nil || current.TaskCount != 2 || current.SubmittedRunCount != 0 {
		t.Fatalf("replayed work duplicated: %+v %v", current, err)
	}
	var page struct {
		Items []auditTaskResponse `json:"items"`
		Page  pageInfoResponse    `json:"page"`
	}
	if json.Unmarshal(call(nil, "tasks", "list", audit.AuditID, "--limit", "1"), &page) != nil || len(page.Items) != 1 || page.Page.NextCursor == nil {
		t.Fatalf("first task page: %+v", page)
	}
	first := page.Items[0].Document.TaskID
	if json.Unmarshal(call(nil, "tasks", "list", audit.AuditID, "--limit", "1", "--cursor", *page.Page.NextCursor), &page) != nil || len(page.Items) != 1 || page.Items[0].Document.TaskID == first || page.Page.HasMore {
		t.Fatalf("continuation: %+v", page)
	}
	call(nil, "tasks", "attempts", audit.AuditID, first)
	call(map[string]any{"key": "claim", "statement": "A scoped assertion", "scope": map[string]string{}, "grounds": "Selected source"}, "hypotheses", "create", audit.AuditID, "--revision", fmt.Sprint(current.Revision), "--idempotency-key", "claim", "--request", "-")
	call(nil, "hypotheses", "list", audit.AuditID)
	// Before a collection receipt commits, private input forks do not become
	// public materials merely because a Task was accepted.
	var empty struct {
		Items []auditstore.MaterialEntry `json:"items"`
	}
	if json.Unmarshal(call(nil, "materials", "list", audit.AuditID), &empty) != nil || len(empty.Items) != 0 {
		t.Fatal("staged inputs became publishable")
	}
	current, err = service.Get(ctx, "user-1", audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	call(nil, "resume", audit.AuditID, "--revision", fmt.Sprint(current.Revision), "--idempotency-key", "run-work")
	claims := auditstoretest.NewStore(pool, artifactpgtest.NewStorage())
	claimed, err := claims.Claim(ctx, auditstore.ClaimParams{HolderID: "cli-fixture", Lease: time.Minute, Limit: 1})
	if err != nil || len(claimed) != 1 {
		t.Fatalf("claim: %+v %v", claimed, err)
	}
	execution, _, err := claims.SelectTasks(ctx, auditstore.SelectTasksParams{Claim: claimed[0], TaskIDs: []string{first}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := claims.ObserveSubmissionFailure(ctx, auditstore.ObserveSubmissionFailureParams{Claim: claimed[0], ExecutionID: execution.ExecutionID}); err != nil {
		t.Fatal(err)
	}
	task, err := service.Task(ctx, "user-1", audit.AuditID, first)
	if err != nil {
		t.Fatal(err)
	}
	material := task.Document.Inputs["source"]
	gap := "submission-failed"
	if _, _, err := claims.Collect(ctx, auditstore.CollectParams{Claim: claimed[0], ExecutionID: execution.ExecutionID, ReceiptID: "cli-collection", RequestDigest: auditHandlerDigest("cli-collection"), Disposition: auditstore.CollectionExecutionFailed, Tasks: []auditstore.TaskCollection{{ExecutionItemID: auditdomain.DeterministicID("attempt", execution.ExecutionID, first), TaskID: first, GapCode: &gap}}, Retained: []auditstore.ArtifactLink{{Scope: "audit", LogicalKey: "collected/source", Artifact: material.Exact(), SourceProvenance: []byte(`{"kind":"selected-evidence"}`)}}}); err != nil {
		t.Fatal(err)
	}
	call(nil, "tasks", "attempts", audit.AuditID, first)
	var materials struct {
		Items []auditstore.MaterialEntry `json:"items"`
		Page  pageInfoResponse           `json:"page"`
	}
	if json.Unmarshal(call(nil, "materials", "list", audit.AuditID), &materials) != nil || len(materials.Items) != 1 {
		t.Fatalf("retained material list: %+v", materials)
	}
	selected := materials.Items[0]
	bytesRead := call(nil, "materials", "get", audit.AuditID, selected.Key, "--revision", *selected.Material.Ref.Revision)
	publication := map[string]any{"auditId": audit.AuditID, "source": selected.Material.Ref, "target": map[string]string{"namespace": "materials", "name": "published"}}
	var published auditservice.MaterialPublication
	data = call(publication, "materials", "publish", project.ProjectID, "--idempotency-key", "publish", "--request", "-")
	if json.Unmarshal(data, &published) != nil {
		t.Fatalf("publication: %s", data)
	}
	again := call(publication, "materials", "publish", project.ProjectID, "--idempotency-key", "publish", "--request", "-")
	if !bytes.Equal(data, again) {
		t.Fatalf("publication replay changed identity: %s %s", data, again)
	}
	read, err := store.Read(ctx, published.Artifact)
	if err != nil || !bytes.Equal(read.Payload.Data, bytesRead) {
		t.Fatalf("published bytes differ: %v", err)
	}
	current, err = service.Get(ctx, "user-1", audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	call(nil, "pause", audit.AuditID, "--revision", fmt.Sprint(current.Revision), "--idempotency-key", "finish-inspection")
	call(nil, "workspace", audit.AuditID)
	for _, path := range []string{"/items", "/coverage", "/reviews"} {
		request, _ := http.NewRequestWithContext(ctx, "GET", server.URL+"/v1/audits/"+audit.AuditID+path, nil)
		request.Header.Set("Authorization", "Bearer "+testBearerToken)
		response, err := server.Client().Do(request)
		if err != nil {
			t.Fatal(err)
		}
		response.Body.Close()
		if response.StatusCode != 404 {
			t.Fatalf("retired %s status=%d", path, response.StatusCode)
		}
	}
}
