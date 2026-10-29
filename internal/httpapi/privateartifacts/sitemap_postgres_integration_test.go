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
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	registry := &fakeRegistry{grant: testGrant("run-a")}
	registry.grant.SitemapPolicy = &contracts.ProjectSitemapPolicy{Version: contracts.SitemapAccessVersion}
	setTools := func(tools ...string) {
		registry.grant.SitemapTools = tools
		registry.grant.ProjectArtifactReads = nil
		for _, operation := range tools {
			if operation == contracts.SitemapListTool || operation == contracts.SitemapReadTool || operation == contracts.SitemapSnapshotTool {
				registry.grant.ProjectArtifactReads = append(registry.grant.ProjectArtifactReads, controlplane.ProjectArtifactReadTarget{Namespace: "sitemap", Name: "map", Operations: []string{operation}})
			}
		}
	}
	setTools(contracts.SitemapListTool)
	handler, err := NewHandler(Dependencies{Registry: registry, Artifacts: artifactService, Sitemap: service, RunProjects: runstore.NewPostgresStore(pool), Projects: projectstore.NewPostgresStore(pool)})
	if err != nil {
		t.Fatal(err)
	}
	request := func(method, path, key string, body any) *httptest.ResponseRecorder {
		t.Helper()
		var encoded []byte
		if body != nil {
			var err error
			encoded, err = json.Marshal(body)
			if err != nil {
				t.Fatal(err)
			}
		}
		r := trustedRequest(method, "/private/v1/allocations/allocation-1"+path, bytes.NewReader(encoded))
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
	const readPath = "/artifacts/sitemap/map?scope=project&operation=list_sitemap_entries"
	const accessPath = "/artifacts/sitemap/map/access?scope=project&operation=list_sitemap_entries"
	if response := request(http.MethodGet, accessPath, "", nil); response.Code != 204 || response.Body.Len() != 0 {
		t.Fatalf("absent binding access: %d %s", response.Code, response.Body.String())
	}
	assertPrivateErrorEnvelope(t, request(http.MethodGet, readPath, "", nil), 404, "artifact_not_found", false)
	declare := func(project, source, path, expected, key string) projectsitemap.Result {
		t.Helper()
		result, err := service.Mutate(ctx, projectsitemap.Authority{OwnerID: "user-1", ProjectID: project, SourceID: source, Kind: "manual"}, key, projectsitemap.Mutation{Operation: "declare", ExpectedRevision: expected, URL: "https://example.test" + path, Method: "GET"})
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	first := declare("project-a", "visible", "/visible", "absent", "first")
	mixed := declare("project-a", "other-source", "/shared-canary", *first.Ref.Revision, "mixed")
	foreign := declare("project-b", "foreign", "/foreign-project-canary", "absent", "foreign")
	response := request(http.MethodGet, readPath, "", nil)
	if response.Code != 200 || !bytes.Contains(response.Body.Bytes(), []byte("shared-canary")) || bytes.Contains(response.Body.Bytes(), []byte("foreign-project-canary")) {
		t.Fatalf("shared map: %d %s", response.Code, response.Body.String())
	}
	var document projectsitemap.Document
	if err := json.Unmarshal(response.Body.Bytes(), &document); err != nil || len(document.Groups) != 2 {
		t.Fatalf("map document: %v", err)
	}
	revision := strings.Trim(response.Header().Get("ETag"), `"`)
	if revision != *mixed.Ref.Revision || response.Header().Get("X-Contractor-Sitemap-Revision") != "" {
		t.Fatal("read did not use ordinary Artifact ETag")
	}
	exact := request(http.MethodGet, readPath+"&revision="+*first.Ref.Revision, "", nil)
	if exact.Code != 200 || bytes.Contains(exact.Body.Bytes(), []byte("shared-canary")) {
		t.Fatalf("exact revision substituted current: %d %s", exact.Code, exact.Body.String())
	}
	assertPrivateErrorEnvelope(t, request(http.MethodGet, readPath+"&revision="+*foreign.Ref.Revision, "", nil), 404, "artifact_not_found", false)
	setTools(contracts.SitemapReadTool)
	assertPrivateErrorEnvelope(t, request(http.MethodGet, readPath, "", nil), 403, "artifact_access_denied", false)
	assertPrivateErrorEnvelope(t, request(http.MethodGet, accessPath, "", nil), 403, "artifact_access_denied", false)
	setTools(contracts.SitemapListTool)
	mutation := sitemapMutationRequest{APIVersion: contracts.APIVersion, Mutation: projectsitemap.EditorMutation{Operation: "declare", ExpectedRevision: revision, URL: "https://example.test/manual", Method: "GET"}}
	assertPrivateErrorEnvelope(t, request(http.MethodPost, "/sitemap/mutations", "denied", mutation), 403, "sitemap_access_denied", false)
	selection := sitemapSnapshotRequest{APIVersion: contracts.APIVersion, Selection: projectsitemap.SnapshotSelection{Revision: revision, Groups: []string{document.Groups[0].ID, document.Groups[1].ID}}}
	assertPrivateErrorEnvelope(t, request(http.MethodPost, "/sitemap/snapshots", "denied-snapshot", selection), 403, "sitemap_access_denied", false)
	setTools(contracts.SitemapSnapshotTool)
	registry.grant.WritePolicy = ""
	assertPrivateErrorEnvelope(t, request(http.MethodPost, "/sitemap/snapshots", "no-write", selection), 403, "sitemap_access_denied", false)
	registry.grant.WritePolicy = controlplane.WriteInputsAndIntermediates
	saved := request(http.MethodPost, "/sitemap/snapshots", "save", selection)
	if saved.Code != 200 {
		t.Fatalf("snapshot: %d %s", saved.Code, saved.Body.String())
	}
	var snapshot struct {
		Result projectsitemap.SnapshotResult `json:"result"`
	}
	if err := json.Unmarshal(saved.Body.Bytes(), &snapshot); err != nil {
		t.Fatal(err)
	}
	run, _ := artifactService.Run("run-a")
	if _, err := run.Read(ctx, snapshot.Result.Artifact.Ref); err != nil {
		t.Fatal(err)
	}
	setTools(contracts.SitemapUpsertTool)
	assertPrivateErrorEnvelope(t, request(http.MethodGet, readPath, "", nil), 403, "artifact_access_denied", false)
	edited := request(http.MethodPost, "/sitemap/mutations", "edit", mutation)
	if edited.Code != 200 {
		t.Fatalf("editor: %d %s", edited.Code, edited.Body.String())
	}
	var editResult struct {
		Result projectsitemap.Result `json:"result"`
	}
	if err := json.Unmarshal(edited.Body.Bytes(), &editResult); err != nil {
		t.Fatal(err)
	}
	setTools(contracts.SitemapGroupingTool, contracts.SitemapListTool)
	var editedMap projectsitemap.Document
	if response := request(http.MethodGet, readPath, "", nil); response.Code != 200 || json.Unmarshal(response.Body.Bytes(), &editedMap) != nil {
		t.Fatalf("edited map: %d %s", response.Code, response.Body.String())
	}
	var expectedSource *projectsitemap.Source
	manualSource := ""
	for _, group := range editedMap.Groups {
		for _, declaration := range group.Declarations {
			if declaration.Provenance.RunID == "run-a" {
				manualSource = declaration.Provenance.SourceID
			}
		}
	}
	for _, source := range editedMap.Sources {
		if source.Provenance.SourceID == manualSource {
			copy := source
			expectedSource = &copy
		}
	}
	if expectedSource == nil {
		t.Fatal("manual source precondition missing from Project map")
	}
	grouping := sitemapMutationRequest{APIVersion: contracts.APIVersion, Mutation: projectsitemap.EditorMutation{Operation: "replace-source", ExpectedRevision: *editResult.Result.Ref.Revision, Source: &projectsitemap.EditorSourceUpdate{Expected: expectedSource, Rules: []projectsitemap.EditorRule{{Origin: "https://example.test", Method: "GET", Path: "/items/{id}", Pointer: "", Parameters: []string{"id"}, Structural: []projectsitemap.StructuralParameter{}}}}}}
	if grouped := request(http.MethodPost, "/sitemap/mutations", "group", grouping); grouped.Code != 200 {
		t.Fatalf("group without submitted source identity: %d %s", grouped.Code, grouped.Body.String())
	}
	forged := map[string]any{"apiVersion": contracts.APIVersion, "mutation": map[string]any{"operation": "replace-source", "source": map[string]any{"sourceId": "forged", "rules": []any{}}}}
	assertPrivateErrorEnvelope(t, request(http.MethodPost, "/sitemap/mutations", "forged-source", forged), 400, "invalid_request", false)
	setTools()
	registry.grant.SitemapPolicy.CollectObservations = true
	ingress := sitemapCollectionRequest{APIVersion: contracts.APIVersion, Input: projectsitemap.CollectionIngress{Observations: []projectsitemap.HTTPObservation{{Observation: projectsitemap.Observation{URL: "https://example.test/session/short-password?token=query-canary&password=query-canary", Method: "POST", Outcome: "response", Status: 201, At: "2026-10-26T00:00:00Z"}, SessionGeneration: "session-one"}}}}
	for range 2 {
		if response := request(http.MethodPost, "/sitemap/observations", "http-publish", ingress); response.Code != 200 {
			t.Fatalf("collection: %d %s", response.Code, response.Body.String())
		}
	}
	assertPrivateErrorEnvelope(t, request(http.MethodGet, readPath, "", nil), 403, "artifact_access_denied", false)
	setTools(contracts.SitemapListTool)
	retained := request(http.MethodGet, readPath, "", nil)
	if retained.Code != 200 || !bytes.Contains(retained.Body.Bytes(), []byte("/session/short-password")) || bytes.Contains(retained.Body.Bytes(), []byte("query-canary")) {
		t.Fatalf("collection metadata: %d %s", retained.Code, retained.Body.String())
	}
	registry.grant.SitemapPolicy.CollectObservations = false
	assertPrivateErrorEnvelope(t, request(http.MethodPost, "/sitemap/observations", "denied", ingress), 403, "sitemap_access_denied", false)
	registry.grant.WriteFenced = true
	for _, path := range []string{readPath, accessPath} {
		assertPrivateErrorEnvelope(t, request(http.MethodGet, path, "", nil), 409, "allocation_write_fenced", false)
	}
	assertPrivateErrorEnvelope(t, request(http.MethodPost, "/sitemap/mutations", "late", mutation), 409, "allocation_write_fenced", false)
	registry.grant.WriteFenced = false
	registry.grant.Lost = true
	assertPrivateErrorEnvelope(t, request(http.MethodGet, readPath, "", nil), 409, "allocation_write_fenced", false)
	registry.grant.Lost = false
	if _, err := pool.Exec(ctx, `UPDATE projects SET lifecycle_state='deleting',deletion_phase='cancelling',deletion_requested_at=clock_timestamp(),revision=revision+1,updated_at=GREATEST(clock_timestamp(),updated_at+interval '1 microsecond') WHERE project_id='project-a'`); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{readPath, accessPath} {
		assertPrivateErrorEnvelope(t, request(http.MethodGet, path, "", nil), 409, "project_deleting", false)
	}
	registry.grant = controlplane.AllocationGrant{}
	assertPrivateErrorEnvelope(t, request(http.MethodGet, readPath, "", nil), 404, "allocation_not_found", false)
}
