package public

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPostgresNullRunLabelCannotCreateOrSelectPlacement(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPublicPool(t, ctx)
	storage := artifactpgtest.NewStorage()
	runs := runstore.NewPostgresStore(pool, storage)
	projects := projectstore.NewPostgresStore(pool)
	artifactService := artifacts.NewService(storage.Repository(pool))
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(d *testServices) {
		d.Runs, d.Artifacts, d.Projects = runs, artifactService, projects
		d.Transfers = storage
		d.RunQueue, d.RunLifecycle = runs, runs
		d.RunCreator = newTestRunCreator(t, *d, runs, integrationUnitOfWork{pool: pool, credentials: d.Credentials}, true)
	})
	_, _, err := projects.Create(ctx, projectstore.CreateParams{ProjectID: "project-labels", OwnerID: "user-1", Kind: projectstore.KindProject, Name: "Labels", IdempotencyKey: "labels-project", RequestDigest: "sha256:" + strings.Repeat("a", 64)})
	if err != nil {
		t.Fatal(err)
	}
	for _, selection := range []struct {
		open func(string) (artifacts.ScopedStore, error)
		id   string
	}{{artifactService.User, "user-1"}, {artifactService.Project, "project-labels"}} {
		scoped, err := selection.open(selection.id)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := scoped.Write(ctx, artifacts.ArtifactRef{Namespace: "sources", Name: "input"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("input")}, nil); err != nil {
			t.Fatal(err)
		}
	}
	for _, path := range []string{"/v1/runs", "/v1/projects/project-labels/runs"} {
		for _, label := range []string{`"cloud":null`, `"purpose":null`} {
			body := []byte(`{"workflow":"artifact-copy@1","parameters":{},"artifacts":{"source":{"namespace":"sources","name":"input"}},"labels":{` + label + `}}`)
			r := authenticatedRequest(http.MethodPost, path, bytes.NewReader(body))
			r.Header.Set(idempotencyKeyHeader, "null-label")
			w := httptest.NewRecorder()
			fixture.handler.ServeHTTP(w, r)
			if w.Code != http.StatusBadRequest {
				t.Fatalf("%s %s = %d %s", path, label, w.Code, w.Body.String())
			}
			assertErrorCode(t, w, "invalid_request")
		}
	}
	var count int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM workflow_runs").Scan(&count); err != nil || count != 0 {
		t.Fatalf("null labels stored %d Runs: %v", count, err)
	}
	body := []byte(`{"workflow":"artifact-copy@1","parameters":{},"artifacts":{"source":{"namespace":"sources","name":"input"}},"labels":{"cloud":""}}`)
	r := authenticatedRequest(http.MethodPost, "/v1/runs", bytes.NewReader(body))
	r.Header.Set(idempotencyKeyHeader, "null-label")
	w := httptest.NewRecorder()
	fixture.handler.ServeHTTP(w, r)
	if w.Code != http.StatusAccepted {
		t.Fatalf("valid empty label/reused refused key = %d %s", w.Code, w.Body.String())
	}
	run, err := runs.GetRun(ctx, "run_fixed")
	if err != nil || fmt.Sprint(run.RequiredPlacementLabels) != "[cloud]" {
		t.Fatalf("empty cloud placement = %v (%v)", run.RequiredPlacementLabels, err)
	}
}

func TestRunAndAuditMetadataLabelsKeepStrictJSONTypes(t *testing.T) {
	for _, labels := range []string{`{"cloud":null}`, `null`, `[]`, `{"purpose":1}`, `{"purpose":"a","purpose":"b"}`} {
		for _, target := range []any{&createRunRequest{}, &createAuditRequest{}} {
			if err := decodeRequestJSON([]byte(`{"labels":`+labels+`}`), target); err == nil {
				t.Fatalf("%T accepted labels %s", target, labels)
			}
		}
	}
}
