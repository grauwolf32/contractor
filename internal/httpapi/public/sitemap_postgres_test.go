package public

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

func TestPostgresPublicProjectSitemap(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPublicPool(t, ctx)
	projects := projectstore.NewPostgresStore(pool)
	for _, p := range []struct{ id, owner string }{{"project-one", "user-1"}, {"project-foreign", "user-2"}} {
		if _, _, err := projects.Create(ctx, projectstore.CreateParams{ProjectID: p.id, OwnerID: p.owner, Kind: projectstore.KindProject, Name: p.id, IdempotencyKey: p.id, RequestDigest: contentdigest.Bytes([]byte(p.id))}); err != nil {
			t.Fatal(err)
		}
	}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(d *testServices) {
		d.Projects = projects
		d.Sitemap = projectsitemap.NewService(pool)
		d.Artifacts = artifacts.NewService(artifacts.NewPostgresRepository(pool))
	})
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	policy, _ := artifacts.NewHistoryPolicy(2, artifacts.DefaultHistoryMaxBytes)
	ctx = artifacts.WithHistoryPolicy(ctx, policy)
	send := func(method, path, key, body string, status int) []byte {
		t.Helper()
		request := newPublicContractRequest(method, path, []byte(body)).WithContext(ctx)
		if body != "" {
			request.Header.Set("Content-Type", "application/json")
		}
		if key != "" {
			request.Header.Set("Idempotency-Key", key)
		}
		response := serveAndValidatePublicContract(t, router, fixture.handler, request, status < 400)
		if response.Code != status {
			t.Fatalf("%s %s: %d %s", method, path, response.Code, response.Body.String())
		}
		return response.Body.Bytes()
	}
	endpoint := "/v1/projects/project-one/sitemap"
	create := `{"operation":"declare","expectedRevision":"absent","url":"https://example.test/users?token=query-canary","method":"GET"}`
	var first projectsitemap.Result
	if err := json.Unmarshal(send("POST", endpoint+"/mutations", "create", create, 200), &first); err != nil {
		t.Fatal(err)
	}
	read := func() struct {
		Revision string
		Document projectsitemap.Document
	} {
		t.Helper()
		var metadata artifactMetadataResponse
		if err := json.Unmarshal(send("GET", "/v1/projects/project-one/artifacts/sitemap/map/metadata", "", "", 200), &metadata); err != nil {
			t.Fatal(err)
		}
		data := send("GET", "/v1/projects/project-one/artifacts/sitemap/map?revision="+*metadata.Ref.Revision, "", "", 200)
		if strings.Contains(string(data), "query-canary") {
			t.Fatal("unsafe URL value reached public map")
		}
		document, err := projectsitemap.Decode(data)
		if err != nil {
			t.Fatal(err)
		}
		view := struct {
			Revision string
			Document projectsitemap.Document
		}{*metadata.Ref.Revision, document}
		return view
	}
	view := read()
	member := view.Document.Groups[0].Members[0]
	if len(member.Summaries) != 0 {
		t.Fatal("manual declaration became observed traffic")
	}
	annotate := fmt.Sprintf(`{"operation":"annotate","expectedRevision":%q,"memberIds":[%q],"note":"owner note"}`, view.Revision, member.ID)
	send("POST", endpoint+"/mutations", "annotate", annotate, 200)
	var stale struct {
		Code      string `json:"code"`
		Retryable bool   `json:"retryable"`
	}
	if err := json.Unmarshal(send("POST", endpoint+"/mutations", "stale", annotate, 409), &stale); err != nil || stale.Code != "sitemap_conflict" || stale.Retryable {
		t.Fatalf("stale intent must be permanent: %+v %v", stale, err)
	}
	for _, operation := range []string{"remove", "restore"} {
		view = read()
		selection := ""
		if operation == "remove" {
			selection = fmt.Sprintf(`,"memberIds":[%q]`, member.ID)
		}
		send("POST", endpoint+"/mutations", operation, fmt.Sprintf(`{"operation":%q,"expectedRevision":%q,"markerId":"hidden"%s}`, operation, view.Revision, selection), 200)
	}
	var replay projectsitemap.Result
	if err := json.Unmarshal(send("POST", endpoint+"/mutations", "create", create, 200), &replay); err != nil {
		t.Fatal(err)
	}
	if !replay.Replayed || !replay.Ref.SameExact(first.Ref) {
		t.Fatal("public replay followed latest")
	}
	send("GET", "/v1/projects/project-one/artifacts/sitemap/map?revision="+*first.Ref.Revision, "", "", 404)
	send("GET", "/v1/projects/project-foreign/artifacts/sitemap/map", "", "", 404)
	send("POST", endpoint+"/mutations", "forged-source", `{"operation":"replace-source","source":{"sourceId":"http-transport","rules":[]}}`, 400)
	send("POST", endpoint+"/mutations", "forged-owner-source", `{"operation":"replace-source","source":{"sourceId":"owner","rules":[]}}`, 400)
	send("POST", endpoint+"/mutations", "forged-rule-source", `{"operation":"replace-source","source":{"rules":[{"sourceId":"owner","origin":"https://example.test","method":"GET","path":"/users/{id}","pointer":"","parameters":["id"],"structural":[]}]}}`, 400)
	send("POST", endpoint+"/mutations", "forged", `{"operation":"observe","observations":[{"url":"https://example.test/forged","method":"GET","outcome":"response","status":200,"at":"2026-10-26T00:00:00Z"}]}`, 400)
	view = read()
	expectedSource, err := json.Marshal(view.Document.Sources[0])
	if err != nil {
		t.Fatal(err)
	}
	grouping := fmt.Sprintf(`{"operation":"replace-source","expectedRevision":%q,"source":{"expected":%s,"rules":[{"origin":"https://example.test","method":"GET","path":"/users/{id}","pointer":"","parameters":["id"],"structural":[]}]}}`, view.Revision, expectedSource)
	send("POST", endpoint+"/mutations", "grouping", grouping, 200)
	view = read()
	if len(view.Document.Rules) != 1 || view.Document.Rules[0].SourceID != projectsitemap.OwnerSourceID || view.Document.Sources[0].Provenance.SourceID != projectsitemap.OwnerSourceID {
		t.Fatal("owner grouping did not use trusted source identity")
	}
	// The ordinary public Artifact writer must not replace domain bytes.
	view = read()
	request := newPublicContractRequest(http.MethodPut, "/v1/projects/project-one/artifacts/sitemap/map", []byte(`{}`)).WithContext(ctx)
	request.Header.Set("Content-Type", projectsitemap.MediaType)
	request.Header.Set("If-Match", fmt.Sprintf("%q", view.Revision))
	response := serveAndValidatePublicContract(t, router, fixture.handler, request, false)
	if response.Code < 400 {
		t.Fatal("generic Artifact API bypassed the reducer")
	}
	if read().Revision != view.Revision {
		t.Fatal("rejected public mutations changed the binding")
	}
}
