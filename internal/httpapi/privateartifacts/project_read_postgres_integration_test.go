//go:build integration

package privateartifacts

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type projectReadQueries struct {
	mu      sync.Mutex
	queries []string
}

func (q *projectReadQueries) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.queries = append(q.queries, data.SQL)
	return ctx
}
func (*projectReadQueries) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}
func (q *projectReadQueries) take() []string {
	q.mu.Lock()
	defer q.mu.Unlock()
	result := q.queries
	q.queries = nil
	return result
}

func TestPostgresPrivateProjectArtifactReadsUseConcreteTargetAndNoCacheHolds(t *testing.T) {
	ctx := t.Context()
	base := isolatedPrivateArtifactPool(t, ctx)
	trace := &projectReadQueries{}
	config := base.Config()
	config.MaxConns = 1
	config.ConnConfig.Tracer = trace
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	createPrivateSitemapProjectRun(t, ctx, pool, "project-a", "run-a")
	createPrivateSitemapProjectRun(t, ctx, pool, "project-b", "run-b")
	createPrivateArtifactRun(t, ctx, pool, "projectless")
	service := artifacts.NewService(artifactpostgres.NewRepository(pool))
	project, _ := service.Project("project-a")
	foreign, _ := service.Project("project-b")
	run, _ := service.Run("run-a")
	ref := contracts.ArtifactRef{Namespace: "notes", Name: "context"}
	history, _ := artifactpostgres.NewHistoryPolicy(2, artifactpostgres.DefaultHistoryMaxBytes)
	writeCtx := artifactpostgres.WithHistoryPolicy(ctx, history)
	original, err := project.Write(writeCtx, ref, artifacts.Payload{MediaType: "text/plain", Data: []byte("original Project")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	foreignWrite, err := foreign.Write(ctx, ref, artifacts.Payload{MediaType: "text/plain", Data: []byte("foreign Project")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := run.Write(ctx, ref, artifacts.Payload{MediaType: "text/plain", Data: []byte("ordinary Run")}, nil); err != nil {
		t.Fatal(err)
	}
	registry := &fakeRegistry{grant: testGrant("run-a")}
	registry.grant.ProjectArtifactReads = []controlplane.ProjectArtifactReadTarget{{Namespace: "notes", Name: "context", Operations: []string{"review_notes"}}}
	server, err := NewHandler(Dependencies{Registry: registry, Artifacts: service, RunProjects: runstore.NewPostgresStore(pool), Projects: projectstore.NewPostgresStore(pool)})
	if err != nil {
		t.Fatal(err)
	}
	const basePath = "/private/v1/allocations/allocation-1/artifacts/notes/context"
	const query = "?scope=project&operation=review_notes"
	request := func(suffix string) *httptest.ResponseRecorder {
		t.Helper()
		response := httptest.NewRecorder()
		server.ServeHTTP(response, trustedRequest(http.MethodGet, basePath+suffix, nil))
		return response
	}
	if response := request(""); response.Code != 200 || response.Body.String() != "ordinary Run" {
		t.Fatalf("default Run scope changed: %d %s", response.Code, response.Body.String())
	}
	if response := request(query); response.Code != 200 || response.Body.String() != "original Project" {
		t.Fatalf("concrete Project read: %d %s", response.Code, response.Body.String())
	}
	metadata := request("/metadata" + query)
	var described contracts.ArtifactMetadataResult
	if metadata.Code != 200 || json.Unmarshal(metadata.Body.Bytes(), &described) != nil || !described.Artifact.SameExact(original.Ref) {
		t.Fatalf("Project metadata: %d %s", metadata.Code, metadata.Body.String())
	}
	assertPrivateErrorEnvelope(t, request(query+"&revision="+*foreignWrite.Ref.Revision), 404, "artifact_not_found", false)
	for _, suffix := range []string{query, "/metadata" + query, "/access" + query} {
		wrong := strings.Replace(suffix, "review_notes", "another_operation", 1)
		assertPrivateErrorEnvelope(t, request(wrong), 403, "artifact_access_denied", false)
	}
	previous := original
	for _, value := range []string{"second Project", "latest Project"} {
		previous, err = project.Write(writeCtx, ref, artifacts.Payload{MediaType: "text/plain", Data: []byte(value)}, previous.Ref.Revision)
		if err != nil {
			t.Fatal(err)
		}
	}
	assertPrivateErrorEnvelope(t, request(query+"&revision="+*original.Ref.Revision), 404, "artifact_not_found", false)
	assertPrivateErrorEnvelope(t, request("/metadata"+query+"&revision="+*original.Ref.Revision), 404, "artifact_not_found", false)
	trace.take()
	access := request("/access" + query)
	queries := trace.take()
	if access.Code != 204 || access.Body.Len() != 0 {
		t.Fatalf("eviction revoked cached authority: %d %s", access.Code, access.Body.String())
	}
	if len(queries) != 2 {
		t.Fatalf("authority check used %d queries: %v", len(queries), queries)
	}
	for _, sql := range queries {
		if strings.Contains(sql, "artifact_") || strings.Contains(sql, "snapshot") {
			t.Fatalf("authority check read Artifact or Run payload: %s", sql)
		}
	}
	if response := request(query); response.Code != 200 || !bytes.Equal(response.Body.Bytes(), []byte("latest Project")) {
		t.Fatalf("current read: %d %s", response.Code, response.Body.String())
	}
	for _, id := range []string{"missing", "projectless"} {
		registry.grant.RunID = id
		for _, suffix := range []string{query, "/metadata" + query, "/access" + query} {
			assertPrivateErrorEnvelope(t, request(suffix), 403, "artifact_access_denied", false)
		}
	}
}
