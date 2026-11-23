package public

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
)

func TestPostgresPublicArtifactUploadAvailability(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPublicPool(t, ctx)
	service := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	projects, runs := projectstore.NewPostgresStore(pool), runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	project, _, err := projects.Create(ctx, projectstore.CreateParams{ProjectID: "upload-project", OwnerID: "user-1", Kind: projectstore.KindProject, Name: "Upload", IdempotencyKey: "upload-project", RequestDigest: "sha256:" + strings.Repeat("a", 64)})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runstoretest.CreateRun(ctx, runs, runstore.CreateRunParams{LaneClass: resolved.LaneClassModel, RunID: "upload-run", OwnerID: "user-1", WorkflowName: "copy", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`), RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot()}); err != nil {
		t.Fatal(err)
	}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(dependencies *testServices) {
		dependencies.Artifacts, dependencies.Projects, dependencies.Runs = service, projects, runs
	})
	contract := loadPublicOpenAPI(t)
	get := func(path, schema string) map[string]any {
		t.Helper()
		rec := httptest.NewRecorder()
		fixture.handler.ServeHTTP(rec, authenticatedRequest(http.MethodGet, path, bytes.NewReader(nil)))
		if rec.Code != http.StatusOK {
			t.Fatalf("GET %s: %d %s", path, rec.Code, rec.Body.String())
		}
		var value map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &value); err != nil {
			t.Fatal(err)
		}
		if err := contract.Components.Schemas[schema].Value.VisitJSON(value, openapi3.EnableJSONSchema2020()); err != nil {
			t.Fatal(err)
		}
		return value
	}
	userScope, _ := artifacts.UserScope("user-1")
	projectScope, _ := artifacts.ProjectScope(project.ProjectID)
	runScope, _ := artifacts.RunScope("upload-run")
	for _, test := range []struct {
		name, base, namespace string
		scope                 artifacts.Scope
		frozen, want          bool
	}{
		{"user", "/v1/artifacts", "docs", userScope, false, true},
		{"project", "/v1/projects/upload-project/artifacts", "docs", projectScope, false, true},
		{"managed", "/v1/projects/upload-project/artifacts", "sitemap", projectScope, false, false},
		{"frozen", "/v1/projects/upload-project/artifacts", "locked", projectScope, true, false},
		{"run", "/v1/runs/upload-run/artifacts", "work", runScope, false, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			target := artifacts.ArtifactRef{Namespace: test.namespace, Name: "value"}
			first, err := service.Create(ctx, test.scope, target, artifacts.Payload{MediaType: "application/vnd.example.result+json", Data: []byte(`{"version":1}`)})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := service.Replace(ctx, test.scope, target, artifacts.Payload{MediaType: "application/vnd.example.result+json", Data: []byte(`{"version":2}`)}, *first.Ref.Revision); err != nil {
				t.Fatal(err)
			}
			if test.frozen {
				if err := service.Freeze(ctx, test.scope, test.namespace); err != nil {
					t.Fatal(err)
				}
			}
			path := test.base + "/" + test.namespace + "/value"
			for _, item := range []map[string]any{
				get(path+"/metadata", "ArtifactMetadata"),
				get(test.base+"?namespace="+test.namespace, "ArtifactPage")["items"].([]any)[0].(map[string]any),
			} {
				if item["canUploadNewVersion"] != test.want {
					t.Fatalf("current availability=%v want %t", item["canUploadNewVersion"], test.want)
				}
			}
			for _, item := range get(path+"/versions", "ArtifactPage")["items"].([]any) {
				row := item.(map[string]any)
				if row["canUploadNewVersion"] != (test.want && row["current"] == true) {
					t.Fatalf("history availability: %+v", row)
				}
			}
			if old := get(path+"/metadata?revision="+*first.Ref.Revision, "ArtifactMetadata"); old["canUploadNewVersion"] != false {
				t.Fatal("historical revision offered upload")
			}
		})
	}
	// A true projection cannot authorize a later write after state changes.
	path := "/v1/projects/upload-project/artifacts/docs/value"
	before := get(path+"/metadata", "ArtifactMetadata")
	if err := service.Freeze(ctx, projectScope, "docs"); err != nil {
		t.Fatal(err)
	}
	request := authenticatedRequest(http.MethodPut, path, bytes.NewReader([]byte(`{}`)))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("If-Match", fmt.Sprintf("%q", before["artifact"].(map[string]any)["revision"]))
	rec := httptest.NewRecorder()
	fixture.handler.ServeHTTP(rec, request)
	if rec.Code != http.StatusConflict || get(path+"/metadata", "ArtifactMetadata")["canUploadNewVersion"] != false {
		t.Fatalf("frozen write after availability read: %d %s", rec.Code, rec.Body.String())
	}
	if _, err := service.Create(ctx, projectScope, artifacts.ArtifactRef{Namespace: "docs", Name: "lifecycle"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("mutable")}); err != nil {
		t.Fatal(err)
	}
	path = "/v1/projects/upload-project/artifacts/docs/lifecycle"
	if current := get(path+"/metadata", "ArtifactMetadata"); current["canUploadNewVersion"] != true {
		t.Fatal("active Project did not offer ordinary upload")
	}
	if _, _, err := projects.BeginDeletion(ctx, projectstore.BeginDeletionParams{OwnerID: "user-1", ProjectID: project.ProjectID, ExpectedRevision: project.Revision}); err != nil {
		t.Fatal(err)
	}
	if current := get(path+"/metadata", "ArtifactMetadata"); current["canUploadNewVersion"] != false {
		t.Fatal("deleting Project offered upload")
	}
}

func TestPostgresRepeatArtifactUploadAvailabilityFollowsSourceProject(t *testing.T) {
	ctx := t.Context()
	catalog := newHandlerFixture(t)
	pool := isolatedPublicPool(t, ctx)
	service := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	projects, runs := projectstore.NewPostgresStore(pool), runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	project, _, err := projects.Create(ctx, projectstore.CreateParams{ProjectID: "repeat-upload-project", OwnerID: "user-1", Kind: projectstore.KindProject, Name: "Repeat upload", IdempotencyKey: "repeat-upload-project", RequestDigest: "sha256:" + strings.Repeat("a", 64)})
	if err != nil {
		t.Fatal(err)
	}
	sources, _ := service.Project(project.ProjectID)
	source, err := sources.Write(ctx, contracts.ArtifactRef{Namespace: "sources", Name: "original"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("source")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	const runID = "repeat-upload-run"
	if _, err := runstoretest.CreateRun(ctx, runs, runstore.CreateRunParams{LaneClass: resolved.LaneClassModel, RunID: runID, OwnerID: "user-1", ProjectID: &project.ProjectID, WorkflowName: "artifact-copy", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: repeatTestSnapshot(t, catalog), RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot()}); err != nil {
		t.Fatal(err)
	}
	if _, err := artifactstest.ForkProjectInput(ctx, service, project.ProjectID, source.Ref, runID, "source"); err != nil {
		t.Fatal(err)
	}

	if _, err := runs.TransitionRun(ctx, runID, runstore.RunInitializing, runstore.RunFailed, runstore.Reason{Code: "fixture-complete"}); err != nil {
		t.Fatal(err)
	}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(dependencies *testServices) {
		dependencies.Artifacts, dependencies.Projects, dependencies.Runs = service, projects, runs
	})
	for _, deleting := range []bool{false, true} {
		if deleting {
			if _, _, err := projects.BeginDeletion(ctx, projectstore.BeginDeletionParams{OwnerID: "user-1", ProjectID: project.ProjectID, ExpectedRevision: project.Revision}); err != nil {
				t.Fatal(err)
			}
		}
		response := serveQuery(t, fixture.handler, "/v1/runs/"+runID+"/repeat-draft")
		var repeat runRepeatDraftResponse
		decodeQueryResponse(t, response, &repeat)
		if response.Code != http.StatusOK || repeat.Draft == nil {
			t.Fatalf("repeat projection: %d %s", response.Code, response.Body.String())
		}
		input := repeat.Draft.Inputs["source"]
		if input.Metadata == nil || input.Metadata.CanUploadNewVersion != !deleting {
			t.Fatalf("repeat source availability for deleting=%t: %+v", deleting, input.Metadata)
		}
	}
}
