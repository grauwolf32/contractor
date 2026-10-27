package public

import (
	"context"
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

func (f *fakeSitemap) Read(_ context.Context, _, _, _ string) (projectsitemap.View, error) {
	if f.document.Version == "" {
		f.document = projectsitemap.Empty()
	}
	return projectsitemap.View{Revision: f.revision, Document: f.document}, nil
}
func (f *fakeSitemap) Mutate(ctx context.Context, a projectsitemap.Authority, _ string, m projectsitemap.Mutation) (projectsitemap.Result, error) {
	if f.err != nil {
		return projectsitemap.Result{}, f.err
	}
	view, _ := f.Read(ctx, "", "", "")
	expected := "absent"
	if f.revision != "" {
		expected = f.revision
	}
	if m.ExpectedRevision != expected {
		return projectsitemap.Result{}, projectsitemap.ErrConflict
	}
	document, _, err := projectsitemap.Reduce(view.Document, m, a)
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
		{"GET", "/v1/projects/project-one/sitemap", ""},
		{"POST", "/v1/projects/project-one/sitemap/mutations", `{"operation":"declare","expectedRevision":"absent","url":"https://example.test/users","method":"GET"}`},
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
}
