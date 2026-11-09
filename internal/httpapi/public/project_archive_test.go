package public

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

func TestProjectArchiveListingRequiresExplicitNamespace(t *testing.T) {
	fixture := newHandlerFixture(t)
	create := authenticatedRequest(http.MethodPost, "/v1/projects", bytes.NewReader([]byte(`{"kind":"project","name":"Archive"}`)))
	create.Header.Set("Content-Type", "application/json")
	create.Header.Set("Idempotency-Key", "archive-project")
	created := httptest.NewRecorder()
	fixture.handler.ServeHTTP(created, create)
	if created.Code != http.StatusCreated {
		t.Fatal(created.Body.String())
	}
	scope, _ := artifacts.ProjectScope("project_fixed")
	for _, ns := range []string{"notes", "historical-audits", "historical-materials", "historical-results"} {
		for _, name := range []string{"one", "two"} {
			if _, err := fixture.repository.Write(t.Context(), scope, artifacts.ArtifactRef{Namespace: ns, Name: name}, artifacts.Payload{MediaType: "text/plain", Data: []byte("kept")}, nil); err != nil {
				t.Fatal(err)
			}
		}
	}
	get := func(path string, want int) *httptest.ResponseRecorder {
		t.Helper()
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, authenticatedRequest(http.MethodGet, path, bytes.NewReader(nil)))
		if response.Code != want {
			t.Fatalf("%s = %d %s", path, response.Code, response.Body.String())
		}
		return response
	}
	const path = "/v1/projects/project_fixed/artifacts"
	var first artifactPageResponse
	if err := json.Unmarshal(get(path+"?limit=1", http.StatusOK).Body.Bytes(), &first); err != nil {
		t.Fatal(err)
	}
	if len(first.Items) != 1 || first.Items[0].Ref.Namespace != "notes" || first.Page.NextCursor == nil {
		t.Fatalf("default page=%+v", first)
	}
	for _, ns := range []string{"historical-audits", "historical-materials", "historical-results"} {
		var page artifactPageResponse
		if err := json.Unmarshal(get(path+"?namespace="+ns, http.StatusOK).Body.Bytes(), &page); err != nil {
			t.Fatal(err)
		}
		if len(page.Items) != 2 || page.Items[0].Ref.Namespace != ns {
			t.Fatalf("explicit %s=%+v", ns, page)
		}
		if body := get(path+"/"+ns+"/one", http.StatusOK).Body.String(); body != "kept" {
			t.Fatal(body)
		}
	}
	get(path+"?namespace=historical-audits&cursor="+url.QueryEscape(*first.Page.NextCursor), http.StatusBadRequest)
	get(path+"?namespace=hide-archives&cursor="+url.QueryEscape(*first.Page.NextCursor), http.StatusBadRequest)
}
