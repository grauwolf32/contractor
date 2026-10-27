//go:build integration

package privateartifacts

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runstore/runstoretest"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5/pgxpool"
)

func createPrivateSitemapProjectRun(t *testing.T, ctx context.Context, pool *pgxpool.Pool, project, run string) {
	t.Helper()
	_, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: project, OwnerID: "user-1", Kind: projectstore.KindProject, Name: project, IdempotencyKey: project, RequestDigest: "sha256:" + strings.Repeat("a", 64)})
	if err != nil {
		t.Fatal(err)
	}
	_, err = runstoretest.CreateRun(ctx, runstore.NewPostgresStore(pool), runstore.CreateRunParams{RunID: run, OwnerID: "user-1", ProjectID: &project, LaneClass: resolved.LaneClassModel, WorkflowName: "sitemap", WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: []byte(`{"ref":{"name":"sitemap","version":"1"}}`), RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot()})
	if err != nil {
		t.Fatal(err)
	}
}

func TestPostgresPrivateSitemapRolesSafeBytesScopeAndFence(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPrivateArtifactPool(t, ctx)
	createPrivateSitemapProjectRun(t, ctx, pool, "project-a", "run-a")
	createPrivateSitemapProjectRun(t, ctx, pool, "project-b", "run-b")
	service := projectsitemap.NewService(pool)
	declare := func(project, source, mode, path, expected, key string) projectsitemap.Result {
		t.Helper()
		result, err := service.Mutate(ctx, projectsitemap.Authority{OwnerID: "user-1", ProjectID: project, SourceID: source, Kind: "manual", Mode: mode}, key, projectsitemap.Mutation{Operation: "declare", ExpectedRevision: expected, URL: "https://example.test" + path, Method: "GET"})
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	first := declare("project-a", "visible", "blackbox", "/visible", "absent", "first")
	mixed := declare("project-a", "restricted", "source-assisted", "/restricted-canary", *first.Ref.Revision, "mixed")
	foreign := declare("project-b", "foreign", "blackbox", "/foreign-project-canary", "absent", "foreign")
	registry := &fakeRegistry{grant: testGrant("run-a")}
	registry.grant.SitemapPolicy = &contracts.ProjectSitemapPolicy{Version: contracts.SitemapAccessVersion, SourceMode: "blackbox"}
	registry.grant.SitemapTools = []string{contracts.SitemapListTool}
	handler, err := NewHandler(Dependencies{Registry: registry, Artifacts: artifacts.NewService(artifacts.NewPostgresRepository(pool)), Sitemap: service, NewRequestID: func() (string, error) { return "sitemap-test", nil }})
	if err != nil {
		t.Fatal(err)
	}
	request := func(method, suffix, key string, body any) *httptest.ResponseRecorder {
		t.Helper()
		var encoded []byte
		if body != nil {
			encoded, err = json.Marshal(body)
			if err != nil {
				t.Fatal(err)
			}
		}
		r := trustedRequest(method, "/private/v1/allocations/allocation-1/sitemap"+suffix, bytes.NewReader(encoded))
		if body != nil {
			r.Header.Set("Content-Type", "application/json")
		}
		if key != "" {
			r.Header.Set("Idempotency-Key", key)
		}
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, r)
		return response
	}
	response := request(http.MethodGet, "", "", nil)
	if response.Code != 200 || bytes.Contains(response.Body.Bytes(), []byte("restricted-canary")) || bytes.Contains(response.Body.Bytes(), []byte("foreign-project-canary")) {
		t.Fatalf("safe map: %d %s", response.Code, response.Body.String())
	}
	var document projectsitemap.Document
	if err := json.Unmarshal(response.Body.Bytes(), &document); err != nil || len(document.Groups) != 1 {
		t.Fatalf("safe document: %v", err)
	}
	revision := response.Header().Get(sitemapRevisionHeader)
	if revision != *mixed.Ref.Revision {
		t.Fatal("view lost original revision")
	}
	for _, query := range []string{"?projectId=project-b", "?runId=run-b", "?sourceMode=source-assisted"} {
		assertPrivateErrorEnvelope(t, request(http.MethodGet, query, "", nil), 400, "invalid_request", false)
	}
	assertPrivateErrorEnvelope(t, request(http.MethodGet, "?revision="+*foreign.Ref.Revision, "", nil), 404, "artifact_not_found", false)
	mutation := sitemapMutationRequest{APIVersion: contracts.APIVersion, Mutation: projectsitemap.Mutation{Operation: "declare", ExpectedRevision: revision, URL: "https://example.test/manual", Method: "GET"}}
	assertPrivateErrorEnvelope(t, request(http.MethodPost, "/mutations", "denied", mutation), 403, "sitemap_access_denied", false)
	selection := sitemapSnapshotRequest{APIVersion: contracts.APIVersion, Selection: projectsitemap.SnapshotSelection{Revision: revision, ExpectedPolicyDigest: response.Header().Get(sitemapPolicyHeader), Groups: []string{document.Groups[0].ID}}}
	assertPrivateErrorEnvelope(t, request(http.MethodPost, "/snapshots", "denied-snapshot", selection), 403, "sitemap_access_denied", false)
	registry.grant.SitemapTools = []string{contracts.SitemapSnapshotTool}
	saved := request(http.MethodPost, "/snapshots", "save", selection)
	if saved.Code != 200 {
		t.Fatalf("snapshot: %d %s", saved.Code, saved.Body.String())
	}
	var snapshot struct {
		Result projectsitemap.SnapshotResult `json:"result"`
	}
	if err := json.Unmarshal(saved.Body.Bytes(), &snapshot); err != nil {
		t.Fatal(err)
	}
	run, _ := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Run("run-a")
	stored, err := run.Read(ctx, snapshot.Result.Artifact.Ref)
	if err != nil || bytes.Contains(stored.Payload.Data, []byte("restricted-canary")) {
		t.Fatalf("unsafe Run context: %v", err)
	}
	assertPrivateErrorEnvelope(t, request(http.MethodGet, "/access?policy=sha256:"+strings.Repeat("b", 64), "", nil), 409, "sitemap_cursor_expired", false)
	registry.grant.SitemapTools = []string{contracts.SitemapUpsertTool}
	assertPrivateErrorEnvelope(t, request(http.MethodGet, "", "", nil), 403, "sitemap_access_denied", false)
	if edited := request(http.MethodPost, "/mutations", "edit", mutation); edited.Code != 200 {
		t.Fatalf("editor: %d %s", edited.Code, edited.Body.String())
	}
	registry.grant.SitemapTools = nil
	registry.grant.SitemapPolicy.CollectObservations = true
	if access := request(http.MethodGet, "/access", "", nil); access.Code != 200 {
		t.Fatalf("collector metadata: %d", access.Code)
	}
	assertPrivateErrorEnvelope(t, request(http.MethodGet, "", "", nil), 403, "sitemap_access_denied", false)
	ingress := sitemapCollectionRequest{APIVersion: contracts.APIVersion, Input: projectsitemap.CollectionIngress{
		Observations: []projectsitemap.HTTPObservation{{Observation: projectsitemap.Observation{URL: "https://example.test/session/short-password?token=query-canary&password=query-canary", Method: "POST", Outcome: "response", Status: 201, At: "2026-10-26T00:00:00Z"}, SessionGeneration: "session-one"}},
	}}
	for range 2 {
		published := request(http.MethodPost, "/observations", "http-publish", ingress)
		if published.Code != 200 {
			t.Fatalf("collector publication: %d %s", published.Code, published.Body.String())
		}
	}
	registry.grant.SitemapTools = []string{contracts.SitemapListTool}
	retained := request(http.MethodGet, "", "", nil)
	if retained.Code != 200 || !bytes.Contains(retained.Body.Bytes(), []byte("/session/short-password")) || !bytes.Contains(retained.Body.Bytes(), []byte(`"token"`)) || !bytes.Contains(retained.Body.Bytes(), []byte(`"password"`)) || bytes.Contains(retained.Body.Bytes(), []byte("query-canary")) {
		t.Fatalf("private collection metadata: %d %s", retained.Code, retained.Body.String())
	}
	assertPrivateErrorEnvelope(t, request(http.MethodPost, "/observations", "obsolete-secrets", map[string]any{"apiVersion": contracts.APIVersion, "input": map[string]any{"observations": ingress.Input.Observations, "secrets": []string{}}}), 400, "invalid_request", false)
	registry.grant.SitemapPolicy.CollectObservations = false
	assertPrivateErrorEnvelope(t, request(http.MethodPost, "/observations", "denied-input", ingress), 403, "sitemap_access_denied", false)
	assertPrivateErrorEnvelope(t, request(http.MethodPost, "/observations", "denied-collect", ingress), 403, "sitemap_access_denied", false)
	registry.grant.SitemapPolicy.CollectObservations = true
	registry.grant.SitemapTools = []string{contracts.SitemapListTool, contracts.SitemapUpsertTool}
	registry.grant.WriteFenced = true
	assertPrivateErrorEnvelope(t, request(http.MethodPost, "/observations", "late-collect", ingress), 409, "allocation_write_fenced", false)
	assertPrivateErrorEnvelope(t, request(http.MethodPost, "/observations", "denied-input", ingress), 409, "allocation_write_fenced", false)
	assertPrivateErrorEnvelope(t, request(http.MethodGet, "/access", "", nil), 409, "allocation_write_fenced", false)
	assertPrivateErrorEnvelope(t, request(http.MethodGet, "", "", nil), 409, "allocation_write_fenced", false)
	assertPrivateErrorEnvelope(t, request(http.MethodPost, "/mutations", "late", mutation), 409, "allocation_write_fenced", false)
	registry.grant.WriteFenced = false
	registry.grant.Lost = true
	assertPrivateErrorEnvelope(t, request(http.MethodGet, "", "", nil), 409, "allocation_write_fenced", false)
	registry.grant = controlplane.AllocationGrant{}
	assertPrivateErrorEnvelope(t, request(http.MethodGet, "", "", nil), 404, "allocation_not_found", false)
}
