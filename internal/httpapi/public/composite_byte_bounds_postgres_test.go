package public

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/runlabels"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// Exercise the actual public handlers and domain/store validators. In
// particular, the selector envelope cannot substitute for its value bound,
// and separate route-part envelopes cannot substitute for their sum bound.
func TestSharedPostgresCompositeStringByteBounds(t *testing.T) {
	characters, fields := contractstest.ReadUTF8ByteBoundCases(t)
	bounds := map[string]contractstest.UTF8ByteBound{}
	for _, field := range fields {
		if field.Schema == "RunMetadataLabels" || field.Schema == "RunMetadataLabelSelector" || field.Schema == "SitemapMutation" && field.Rule == "LIM-SMT-2" {
			bounds[field.Schema] = field
		}
	}
	if len(bounds) != 3 {
		t.Fatalf("missing composite boundary cases: %v", bounds)
	}
	pool := isolatedPublicPool(t, t.Context())
	projects := projectstore.NewPostgresStore(pool)
	runs := runstore.NewPostgresStore(pool)
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(d *testServices) {
		d.Projects, d.Runs = projects, runs
		d.RunQueue, d.RunLifecycle = runs, runs
		d.Sitemap = projectsitemap.NewService(pool)
		d.Artifacts = artifacts.NewService(artifacts.NewPostgresRepository(pool))
	})
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	createProject := func(id string) {
		t.Helper()
		if _, _, err := projects.Create(t.Context(), projectstore.CreateParams{
			ProjectID: id, OwnerID: "user-1", Kind: projectstore.KindProject, Name: id,
			IdempotencyKey: id, RequestDigest: contentdigest.Bytes([]byte(id)),
		}); err != nil {
			t.Fatal(err)
		}
	}
	createProject("project-label-bounds")
	for _, value := range bounds["RunMetadataLabels"].Values(characters) {
		for _, key := range []string{"a", strings.Repeat("a", runlabels.MaxRunMetadataLabelKey)} {
			for _, path := range []string{"/v1/runs", "/v1/projects/project-label-bounds/runs"} {
				request := newPublicContractRequest(http.MethodGet, path+"?label="+url.QueryEscape(key+"="+value.Value), nil)
				response := serveAndValidatePublicContract(t, router, fixture.handler, request, value.Within)
				want := http.StatusBadRequest
				if value.Within {
					want = http.StatusOK
				}
				if response.Code != want {
					t.Fatalf("%s key=%d value=%d bytes of %q: %d %s", path, len(key), len(value.Value), value.Character, response.Code, response.Body.String())
				}
				if !value.Within {
					assertErrorCode(t, response, "invalid_request")
				}
			}
		}
	}
	for _, path := range []string{"/v1/runs", "/v1/projects/project-label-bounds/runs"} {
		response := serveAndValidatePublicContract(t, router, fixture.handler,
			newPublicContractRequest(http.MethodGet, path+"?label="+strings.Repeat("a", runlabels.MaxRunMetadataLabelKey+1)+"=", nil), false)
		if response.Code != http.StatusBadRequest {
			t.Fatalf("over-bound key: %d %s", response.Code, response.Body.String())
		}
		assertErrorCode(t, response, "invalid_request")
	}
	selector := bounds["RunMetadataLabelSelector"]
	if len(selector.Prefix) != runlabels.MaxRunMetadataLabelKey+1 || selector.MaxBytes-len(selector.Prefix) != runlabels.MaxRunMetadataLabelValue {
		t.Fatal("selector fixture envelope does not derive from the part bounds")
	}
	for index, character := range characters {
		t.Run("sitemap/"+character, func(t *testing.T) {
			projectID := fmt.Sprintf("project-rule-bounds-%d", index)
			createProject(projectID)
			origin := "https://example.test"
			room := bounds["SitemapMutation"].MaxBytes - len(origin) - len("/")
			escaped := url.PathEscape(character)
			path := "/" + strings.Repeat(escaped, room/len(escaped)) + strings.Repeat("a", room%len(escaped))
			// Each part of the first request is below 8 KiB, but their sum
			// exceeds it. Reusing its key for the accepted write proves the
			// rejection committed neither a publication nor a receipt.
			for _, extra := range []string{"a", ""} {
				command := projectsitemap.EditorMutation{
					Operation: "replace-source", ExpectedRevision: "absent",
					Source: &projectsitemap.EditorSourceUpdate{Rules: []projectsitemap.EditorRule{{
						Origin: origin, Method: "GET", Path: path + extra,
						Parameters: []string{}, Structural: []projectsitemap.StructuralParameter{},
					}}},
				}
				body, err := json.Marshal(command)
				if err != nil {
					t.Fatal(err)
				}
				request := newPublicContractRequest(http.MethodPost, "/v1/projects/"+projectID+"/sitemap/mutations", body)
				request.Header.Set("Content-Type", "application/json")
				request.Header.Set("Idempotency-Key", projectID)
				response := serveAndValidatePublicContract(t, router, fixture.handler, request, extra == "")
				want := http.StatusOK
				if extra != "" {
					want = http.StatusBadRequest
				}
				if response.Code != want {
					t.Fatalf("route of %d bytes: %d %s", len(origin)+len(path+extra), response.Code, response.Body.String())
				}
				if extra != "" {
					assertErrorCode(t, response, "invalid_request")
				}
			}
		})
	}
}
