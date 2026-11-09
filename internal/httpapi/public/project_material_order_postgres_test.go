package public

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

func TestPostgresPublicProjectMaterialOrderBindsCursorAndFilters(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPublicPool(t, ctx)
	projects := projectstore.NewPostgresStore(pool)
	service := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	for _, id := range []string{"materials-one", "materials-two"} {
		if _, _, err := projects.Create(ctx, projectstore.CreateParams{ProjectID: id, OwnerID: "user-1", Kind: projectstore.KindProject, Name: id, IdempotencyKey: id, RequestDigest: "sha256:" + strings.Repeat("a", 64)}); err != nil {
			t.Fatal(err)
		}
	}
	store, _ := service.Project("materials-one")
	for _, entry := range []struct{ ns, name, media string }{
		{"a-generated", "first", "application/json"},
		{"sources", "code", "application/zip"},
		{"z-output", "api", "application/vnd.oai.openapi+json"},
		{"docs", "guide", "text/markdown"},
		{"historical-results", "old-api", "application/vnd.oai.openapi+json"},
	} {
		if _, err := store.Write(ctx, artifacts.ArtifactRef{Namespace: entry.ns, Name: entry.name}, artifacts.Payload{MediaType: entry.media, Data: []byte(entry.name)}, nil); err != nil {
			t.Fatal(err)
		}
	}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(d *testServices) { d.Artifacts = service; d.Projects = projects })
	const base = "/v1/projects/materials-one/artifacts"
	get := func(path string, status int) artifactPageResponse {
		t.Helper()
		r := httptest.NewRecorder()
		fixture.handler.ServeHTTP(r, authenticatedRequest(http.MethodGet, path, bytes.NewReader(nil)))
		if r.Code != status {
			t.Fatalf("%s: %d %s", path, r.Code, r.Body.String())
		}
		if status != 200 {
			return artifactPageResponse{}
		}
		if strings.Contains(r.Body.String(), "material_rank") || strings.Contains(r.Body.String(), "materialRank") {
			t.Fatal("private order key leaked")
		}
		var page artifactPageResponse
		if err := json.Unmarshal(r.Body.Bytes(), &page); err != nil {
			t.Fatal(err)
		}
		return page
	}
	first := get(base+"?order=material&limit=2", 200)
	if first.Items[0].Ref.Namespace != "sources" || first.Items[1].Ref.Name != "api" || first.Page.NextCursor == nil {
		t.Fatalf("first page %+v", first)
	}
	cursor := url.QueryEscape(*first.Page.NextCursor)
	next := get(base+"?order=material&limit=2&cursor="+cursor, 200)
	if len(next.Items) != 2 || next.Items[0].Ref.Namespace != "docs" || next.Items[1].Ref.Namespace != "a-generated" || next.Page.HasMore {
		t.Fatalf("next page %+v", next)
	}
	get(base+"?limit=2&cursor="+cursor, 400)
	get(base+"?order=material&namespace=docs&cursor="+cursor, 400)
	get("/v1/projects/materials-two/artifacts?order=material&cursor="+cursor, 400)
	get(base+"?order=bogus", 400)
	get(base+"?order=", 400)
	generic := get(base+"?limit=2", 200)
	if generic.Items[0].Ref.Namespace != "a-generated" {
		t.Fatal("generic order changed")
	}
	get(base+"?order=material&cursor="+url.QueryEscape(*generic.Page.NextCursor), 400)
	filtered := get(base+"?order=material&namespace=z-output", 200)
	if len(filtered.Items) != 1 || filtered.Items[0].Ref.Name != "api" {
		t.Fatal("namespace filter changed")
	}
	archived := get(base+"?order=material&namespace=historical-results", 200)
	if len(archived.Items) != 1 || archived.Items[0].Ref.Name != "old-api" {
		t.Fatalf("explicit archive material order: %+v", archived)
	}
}
