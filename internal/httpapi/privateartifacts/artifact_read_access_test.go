package privateartifacts

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type readRunProject func(context.Context, string) (runstore.ProjectContext, error)

func (f readRunProject) ProjectContext(ctx context.Context, id string) (runstore.ProjectContext, error) {
	return f(ctx, id)
}

type readProject func(context.Context, string, string) (projectstore.Project, error)

func (f readProject) Get(ctx context.Context, owner, project string) (projectstore.Project, error) {
	return f(ctx, owner, project)
}

func TestPrivateProjectArtifactAuthorityDoesNotReadOrRequireContent(t *testing.T) {
	repository := newMemoryRepository()
	registry := &fakeRegistry{grant: testGrant("run-a")}
	registry.grant.ProjectArtifactReads = []controlplane.ProjectArtifactReadTarget{{Namespace: "notes", Name: "context", Operations: []string{"review_notes"}}}
	runReads, projectReads := 0, 0
	projectErr, runErr := error(nil), error(nil)
	lifecycle := projectstore.LifecycleActive
	handler, err := NewHandler(Dependencies{
		Registry: registry, Artifacts: artifacts.NewService(repository),
		RunProjects: readRunProject(func(_ context.Context, run string) (runstore.ProjectContext, error) {
			runReads++
			if run != "run-a" {
				t.Fatalf("requested foreign Run %q", run)
			}
			return runstore.ProjectContext{OwnerID: "owner-a", ProjectID: "project-a"}, runErr
		}),
		Projects: readProject(func(_ context.Context, owner, project string) (projectstore.Project, error) {
			projectReads++
			if owner != "owner-a" || project != "project-a" {
				t.Fatalf("requested foreign Project %s/%s", owner, project)
			}
			return projectstore.Project{Lifecycle: lifecycle}, projectErr
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	const base = "/private/v1/allocations/allocation-1/artifacts/notes/context"
	const query = "?scope=project&operation=review_notes"
	request := func(path string) *httptest.ResponseRecorder {
		t.Helper()
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, trustedRequest(http.MethodGet, path, nil))
		return response
	}
	for range 2 {
		response := request(base + "/access" + query)
		if response.Code != 204 || response.Body.Len() != 0 || response.Header().Get("Cache-Control") != "no-store" {
			t.Fatalf("authority-only check: %d %s", response.Code, response.Body.String())
		}
	}
	if repository.readCalls != 0 || repository.metadataCalls != 0 || runReads != 2 || projectReads != 2 {
		t.Fatalf("access touched content or reused authority: raw=%d metadata=%d Run=%d Project=%d", repository.readCalls, repository.metadataCalls, runReads, projectReads)
	}
	for _, suffix := range []string{"", "/metadata", "/access"} {
		assertPrivateErrorEnvelope(t, request(base+suffix+"?scope=project&operation=another_operation"), 403, "artifact_access_denied", false)
		assertPrivateErrorEnvelope(t, request(strings.Replace(base, "/context", "/other", 1)+suffix+query), 403, "artifact_access_denied", false)
	}
	if runReads != 2 {
		t.Fatal("ungranted target reached the Project resolver")
	}
	for _, suffix := range []string{"", "/metadata", "/access"} {
		for _, invalid := range []string{"?scope=run&operation=review_notes", "?scope=project", "?scope=project&operation=", "?operation=review_notes", "?scope=&operation=review_notes", query + "&projectId=other", query + "&scope=project", query + "&operation=other"} {
			assertPrivateErrorEnvelope(t, request(base+suffix+invalid), 400, "invalid_request", false)
		}
	}
	assertPrivateErrorEnvelope(t, request(base+"/access"+query+"&revision=revision-a"), 400, "invalid_request", false)
	for _, path := range []string{"/private/v1/allocations/allocation-1/sitemap", "/private/v1/allocations/allocation-1/sitemap/access?operation=list_sitemap_entries"} {
		assertPrivateErrorEnvelope(t, request(path), 404, "not_found", false)
	}
	projectErr = projectstore.ErrNotFound
	assertPrivateErrorEnvelope(t, request(base+"/access"+query), 403, "artifact_access_denied", false)
	projectErr, runErr = nil, runstore.ErrNotFound
	assertPrivateErrorEnvelope(t, request(base+"/access"+query), 403, "artifact_access_denied", false)
	runErr, lifecycle = nil, projectstore.LifecycleDeleting
	assertPrivateErrorEnvelope(t, request(base+"/access"+query), 409, "project_deleting", false)
	lifecycle = projectstore.LifecycleActive
	registry.grant.WriteFenced = true
	assertPrivateErrorEnvelope(t, request(base+"/access"+query), 409, "allocation_write_fenced", false)
	registry.grant.WriteFenced, registry.grant.Lost = false, true
	assertPrivateErrorEnvelope(t, request(base+"/access"+query), 409, "allocation_write_fenced", false)
	registry.grant = controlplane.AllocationGrant{}
	assertPrivateErrorEnvelope(t, request(base+"/access"+query), 404, "allocation_not_found", false)
}

func TestPrivateProjectReadSelectorsCannotListOrWrite(t *testing.T) {
	repository := newMemoryRepository()
	server := newTestHandler(t, &fakeRegistry{grant: testGrant("run-a")}, repository)
	for _, test := range []struct{ method, path string }{
		{http.MethodGet, "/artifacts?limit=1&scope=project&operation=read"},
		{http.MethodGet, "/artifacts?limit=1&operation=read"},
		{http.MethodPut, "/artifacts/notes/context?scope=project&operation=read"},
		{http.MethodPut, "/artifacts/notes/context?operation=read"},
	} {
		request := trustedRequest(test.method, "/private/v1/allocations/allocation-1"+test.path, strings.NewReader("changed"))
		request.Header.Set("Content-Type", "text/plain")
		request.Header.Set("If-None-Match", "*")
		response := httptest.NewRecorder()
		server.ServeHTTP(response, request)
		assertPrivateErrorEnvelope(t, response, 400, "invalid_request", false)
	}
	if len(repository.bindings) != 0 {
		t.Fatal("Project selector wrote an artifact")
	}
}

func TestSharedArtifactReadTargets(t *testing.T) {
	var cases struct{ Valid, Invalid []map[string]string }
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "artifact-read-target-cases.json"), &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases.Valid) != 5 || len(cases.Invalid) < 16 {
		t.Fatal("incomplete read-target cases")
	}
	for _, group := range []struct {
		valid  bool
		values []map[string]string
	}{{true, cases.Valid}, {false, cases.Invalid}} {
		for _, value := range group.values {
			registry := &fakeRegistry{grant: testGrant("run-a")}
			registry.grant.ProjectArtifactReads = []controlplane.ProjectArtifactReadTarget{{Namespace: "notes", Name: "context", Operations: []string{value["operation"]}}}
			server, err := NewHandler(Dependencies{Registry: registry, Artifacts: artifacts.NewService(newMemoryRepository()),
				RunProjects: readRunProject(func(context.Context, string) (runstore.ProjectContext, error) {
					return runstore.ProjectContext{OwnerID: "owner", ProjectID: "project"}, nil
				}),
				Projects: readProject(func(context.Context, string, string) (projectstore.Project, error) {
					return projectstore.Project{Lifecycle: projectstore.LifecycleActive}, nil
				}),
			})
			if err != nil {
				t.Fatal(err)
			}
			query := url.Values{}
			for key, item := range value {
				query.Set(key, item)
			}
			response := httptest.NewRecorder()
			server.ServeHTTP(response, trustedRequest(http.MethodGet, "/private/v1/allocations/allocation-1/artifacts/notes/context/access?"+query.Encode(), nil))
			if group.valid {
				if response.Code != 204 {
					t.Fatalf("valid selector %+v: %d %s", value, response.Code, response.Body.String())
				}
			} else {
				assertPrivateErrorEnvelope(t, response, 400, "invalid_request", false)
			}
		}
	}
}

func TestProjectReadTargetMatchingIsConcrete(t *testing.T) {
	ref := contracts.ArtifactRef{Namespace: "notes", Name: "context"}
	for _, target := range []controlplane.ProjectArtifactReadTarget{
		{Namespace: "*", Name: "context", Operations: []string{"read"}},
		{Namespace: "notes", Name: "*", Operations: []string{"read"}},
		{Namespace: "notes", Name: "context", Operations: []string{"*"}},
	} {
		if permitsProjectRead([]controlplane.ProjectArtifactReadTarget{target}, ref, "read") {
			t.Fatal("wildcard target granted access")
		}
	}
}
