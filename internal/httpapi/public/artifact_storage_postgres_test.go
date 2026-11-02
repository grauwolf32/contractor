package public

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
)

func TestPostgresPublicArtifactStorageTimestamps(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPublicPool(t, ctx)
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	projects := projectstore.NewPostgresStore(pool)
	runs := runstore.NewPostgresStore(pool)
	if _, _, err := projects.Create(ctx, projectstore.CreateParams{ProjectID: "dates-project", OwnerID: "user-1", Kind: projectstore.KindProject, Name: "Dates", IdempotencyKey: "dates-project", RequestDigest: "sha256:" + strings.Repeat("a", 64)}); err != nil {
		t.Fatal(err)
	}
	if _, err := runstoretest.CreateRun(ctx, runs, runstore.CreateRunParams{LaneClass: resolved.LaneClassModel, RunID: "dates-run", OwnerID: "user-1", WorkflowName: "copy", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`), RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot()}); err != nil {
		t.Fatal(err)
	}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(dependencies *testServices) {
		dependencies.Artifacts = service
		dependencies.Projects = projects
		dependencies.Runs = runs
	})
	contract := loadPublicOpenAPI(t)
	get := func(path, schema string) map[string]any {
		t.Helper()
		rec := httptest.NewRecorder()
		fixture.handler.ServeHTTP(rec, authenticatedRequest(http.MethodGet, path, bytes.NewReader(nil)))
		if rec.Code != 200 {
			t.Fatalf("%s = %d %s", path, rec.Code, rec.Body.String())
		}
		var value map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &value); err != nil {
			t.Fatal(err)
		}
		if err := contract.Components.Schemas[schema].Value.VisitJSON(value, openapi3.EnableJSONSchema2020()); err != nil {
			t.Fatalf("%s schema: %v", path, err)
		}
		if strings.Contains(rec.Body.String(), "object_key") || strings.Contains(rec.Body.String(), "version_id") {
			t.Fatal("physical identity leaked")
		}
		return value
	}
	for _, scope := range []struct{ kind, id, base string }{{"user", "user-1", "/v1/artifacts"}, {"project", "dates-project", "/v1/projects/dates-project/artifacts"}, {"run", "dates-run", "/v1/runs/dates-run/artifacts"}} {
		t.Run(scope.kind, func(t *testing.T) {
			var store artifacts.ScopedStore
			switch scope.kind {
			case "user":
				store, _ = service.User(scope.id)
			case "project":
				store, _ = service.Project(scope.id)
			case "run":
				store, _ = service.Run(scope.id)
			}
			first, err := store.Write(ctx, artifacts.ArtifactRef{Namespace: "docs", Name: "source"}, artifacts.Payload{MediaType: "application/json", Data: []byte(`{"storageTimestamps":{"scopeCreatedAt":"invented"}}`)}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := store.Write(ctx, artifacts.ArtifactRef{Namespace: "docs", Name: "source"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("next")}, first.Ref.Revision); err != nil {
				t.Fatal(err)
			}
			expected, err := store.Metadata(ctx, first.Ref)
			if err != nil {
				t.Fatal(err)
			}
			old := get(scope.base+"/docs/source/metadata?revision="+*first.Ref.Revision, "ArtifactMetadata")
			dates, ok := old["storageTimestamps"].(map[string]any)
			if !ok || len(dates) != 4 {
				t.Fatal("missing dates")
			}
			expectedJSON, _ := json.Marshal(expected.StorageTimestamps)
			gotJSON, _ := json.Marshal(dates)
			if !bytes.Equal(expectedJSON, gotJSON) {
				var projected artifacts.StorageTimestamps
				if err := json.Unmarshal(gotJSON, &projected); err != nil || projected != expected.StorageTimestamps {
					t.Fatalf("dates mismatch: %s %s", expectedJSON, gotJSON)
				}
			}
			if old["createdAt"] != expected.CreatedAt.Format(time.RFC3339Nano) {
				t.Fatal("ordinary selected revision date changed")
			}
			get(scope.base+"/docs/source/metadata", "ArtifactMetadata")
			get(scope.base+"?limit=1", "ArtifactPage")
			firstPage := get(scope.base+"/docs/source/versions?limit=1", "ArtifactPage")
			cursor := firstPage["page"].(map[string]any)["nextCursor"].(string)
			get(scope.base+"/docs/source/versions?limit=1&cursor="+cursor, "ArtifactPage")
			rec := httptest.NewRecorder()
			fixture.handler.ServeHTTP(rec, authenticatedRequest(http.MethodGet, scope.base+"/docs/source/metadata?revision=missing", bytes.NewReader(nil)))
			assertErrorEnvelope(t, rec, 404, "not_found", false)
		})
	}
	for _, path := range []string{"/v1/projects/foreign/artifacts/docs/source/metadata", "/v1/runs/foreign/artifacts/docs/source/metadata"} {
		rec := httptest.NewRecorder()
		fixture.handler.ServeHTTP(rec, authenticatedRequest(http.MethodGet, path, bytes.NewReader(nil)))
		assertErrorEnvelope(t, rec, 404, "not_found", false)
	}
}
