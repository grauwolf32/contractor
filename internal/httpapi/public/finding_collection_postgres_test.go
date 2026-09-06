//go:build integration

package public

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func TestFindingCollectionHTTPPublicationAndOrdinaryRunInput(t *testing.T) {
	if os.Getenv("CONTRACTOR_TEST_DATABASE_URL") == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required for the selected findings collection HTTP integration test")
	}
	ctx := t.Context()
	pool := isolatedPublicPool(t, ctx)
	root := t.TempDir()
	if err := os.CopyFS(root, os.DirFS("../../config/testdata/valid")); err != nil {
		t.Fatal(err)
	}
	workflowPath := filepath.Join(root, "workflows", "artifact_copy.yaml")
	workflowBytes, err := os.ReadFile(workflowPath)
	if err != nil {
		t.Fatal(err)
	}
	workflowBytes = bytes.Replace(workflowBytes, []byte("source: {required: true, mediaTypes: [text/plain]}"), []byte("source: {required: true, mediaTypes: ["+auditdomain.FindingCollectionMediaType+"]}"), 1)
	if err := os.WriteFile(workflowPath, workflowBytes, 0o600); err != nil {
		t.Fatal(err)
	}
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	runs := runstore.NewPostgresStore(pool)
	publisher, err := findingintake.NewCollectionPublisher(pool, &auditservice.Service{})
	if err != nil {
		t.Fatal(err)
	}
	fixture := newHandlerFixtureWithAuth(t, root, newTestAuthentication(t), mustTestOrigins(t), false, nil, func(d *Dependencies) {
		d.Runs, d.Artifacts, d.Projects = runs, artifactService, projectstore.NewPostgresStore(pool)
		d.Transactions = integrationUnitOfWork{pool: pool, credentials: d.Credentials}
		d.FindingCollections = publisher
	})
	workflow, err := fixture.configs.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	snapshot, _ := json.Marshal(workflow)
	if _, err := runs.CreateRun(ctx, runstore.CreateRunParams{RunID: "source-run", OwnerID: "user-1", WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version,
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: snapshot, Parameters: map[string]string{}, RuntimeConfig: runtimeconfig.BuiltInRunSnapshot()}); err != nil {
		t.Fatal(err)
	}
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(`{"clientKey":"http-snapshot","sources":[{"kind":"run","id":"source-run","receiptIds":[],"findings":[]}]}`)
	publish := func(payload []byte, authenticated bool, validate bool) *httptest.ResponseRecorder {
		request := newPublicContractRequest(http.MethodPost, "/v1/finding-collections", payload)
		request.Header.Set("Content-Type", "application/json")
		if !authenticated {
			request.Header.Del("Authorization")
		}
		return serveAndValidatePublicContract(t, router, fixture.handler, request, validate)
	}
	if response := publish(body, false, true); response.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated publication = %d", response.Code)
	}
	for _, bad := range []string{
		strings.Replace(string(body), `{"clientKey"`, `{"ownerId":"foreign","clientKey"`, 1),
		strings.Replace(string(body), `"receiptIds":[]`, `"receiptIds":null`, 1),
	} {
		if response := publish([]byte(bad), true, false); response.Code != http.StatusBadRequest {
			t.Fatalf("invalid publication = %d %s", response.Code, response.Body.String())
		}
	}
	if response := publish(bytes.Replace(body, []byte("source-run"), []byte("foreign-run"), 1), true, true); response.Code != http.StatusNotFound {
		t.Fatalf("foreign source = %d %s", response.Code, response.Body.String())
	}
	created := publish(body, true, true)
	if created.Code != http.StatusCreated {
		t.Fatalf("publish = %d %s", created.Code, created.Body.String())
	}
	var publication findingintake.PublishedCollection
	if err := json.Unmarshal(created.Body.Bytes(), &publication); err != nil {
		t.Fatal(err)
	}
	if response := publish(body, true, true); response.Code != http.StatusOK {
		t.Fatalf("replay = %d %s", response.Code, response.Body.String())
	}
	createBody, _ := json.Marshal(map[string]any{"workflow": "artifact-copy@1", "parameters": map[string]string{}, "artifacts": map[string]contracts.ArtifactRef{"source": publication.Artifact.Ref}})
	request := newPublicContractRequest(http.MethodPost, "/v1/runs", createBody)
	request.Header.Set("Idempotency-Key", "collection-consumer")
	request.Header.Set("Content-Type", "application/json")
	createdRun := serveAndValidatePublicContract(t, router, fixture.handler, request, true)
	if createdRun.Code != http.StatusAccepted {
		t.Fatalf("ordinary Run creation = %d %s", createdRun.Code, createdRun.Body.String())
	}
	reader, _ := artifactService.Run("run_fixed")
	input, err := reader.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"})
	if err != nil {
		t.Fatal(err)
	}
	collection, pkg, err := auditdomain.DecodeFindingCollectionPackage(input.Payload.Data)
	if err != nil || len(collection.Entries) != 0 || pkg.Digest != publication.Artifact.Digest {
		t.Fatalf("ordinary fork changed collection: %v", err)
	}
}
