package public

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

type fakeSitemap struct {
	err      error
	document projectsitemap.Document
	revision string
}

func (f *fakeSitemap) Mutate(_ context.Context, a projectsitemap.Authority, _ string, m projectsitemap.Mutation) (projectsitemap.Result, error) {
	if f.err != nil {
		return projectsitemap.Result{}, f.err
	}
	if f.document.Version == "" {
		f.document = projectsitemap.Empty()
	}
	expected := "absent"
	if f.revision != "" {
		expected = f.revision
	}
	if m.ExpectedRevision != expected {
		return projectsitemap.Result{}, projectsitemap.ErrConflict
	}
	document, _, err := projectsitemap.Reduce(f.document, m, a)
	if err != nil {
		return projectsitemap.Result{}, err
	}
	f.document, f.revision = document, "rev-sitemap"
	revision := f.revision
	return projectsitemap.Result{Ref: contracts.ArtifactRef{Namespace: "sitemap", Name: "map", Revision: &revision}, Changed: true}, nil
}

func TestPublicProjectSitemapContract(t *testing.T) {
	fixture := newHandlerFixture(t)
	if _, _, err := fixture.projects.Create(t.Context(), projectstore.CreateParams{ProjectID: "project-one", OwnerID: "user-1", Kind: projectstore.KindProject, Name: "Sitemap", IdempotencyKey: "create-sitemap", RequestDigest: contentdigest.Bytes([]byte("sitemap"))}); err != nil {
		t.Fatal(err)
	}
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ method, path, body string }{
		{"POST", "/v1/projects/project-one/sitemap/mutations", `{"operation":"replace-source","expectedRevision":"absent","source":{"rules":[{"origin":"https://example.test","method":"GET","path":"/users/{id}","pointer":"","parameters":["id"],"structural":[]}]}}`},
		{"POST", "/v1/projects/project-one/sitemap/mutations", `{"operation":"declare","expectedRevision":"rev-sitemap","url":"https://example.test/users","method":"GET"}`},
	} {
		request := newPublicContractRequest(tc.method, tc.path, []byte(tc.body))
		if tc.body != "" {
			request.Header.Set("Content-Type", "application/json")
			request.Header.Set("Idempotency-Key", "sitemap-create")
		}
		response := serveAndValidatePublicContract(t, router, fixture.handler, request, true)
		if response.Code != 200 {
			t.Fatalf("%s: %d %s", tc.path, response.Code, response.Body.String())
		}
	}
	removed := httptest.NewRecorder()
	fixture.handler.ServeHTTP(removed, newPublicContractRequest(http.MethodGet, "/v1/projects/project-one/sitemap", nil))
	if removed.Code != http.StatusNotFound {
		t.Fatalf("removed sitemap read route: %d %s", removed.Code, removed.Body.String())
	}
}
