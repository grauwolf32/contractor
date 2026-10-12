package public

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

// ART-R25 reserves exactly audit-standards. A UserScope Namespace that only
// starts with that name is an ordinary one: writable, readable and listed,
// while the catalog stays hidden even together with a caller's
// excludeNamespace.
func TestPostgresUserArtifactsReserveOnlyTheExactCatalogNamespace(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t),
		mustTestOrigins(t), false, nil, func(dependencies *testServices) { dependencies.Artifacts = service })
	if _, err := service.WriteAuditStandardPackage(ctx, "user-1",
		contracts.ArtifactRef{Namespace: artifactpolicy.AuditStandardCatalogNamespace, Name: "catalog"},
		artifacts.Payload{MediaType: "application/zip", Data: []byte("catalog")}, nil); err != nil {
		t.Fatal(err)
	}
	for _, ref := range [][2]string{{"audit-standards-notes", "draft"}, {"docs", "readme"}, {"skills", "package"}} {
		if created := putPublicArtifact(fixture.handler, ref[0], ref[1], "If-None-Match", "*", ref[1]); created.Code != http.StatusCreated {
			t.Fatalf("create %s/%s = %d %s", ref[0], ref[1], created.Code, created.Body.String())
		}
	}
	reserved := putPublicArtifact(fixture.handler, artifactpolicy.AuditStandardCatalogNamespace, "forged", "If-None-Match", "*", "forged")
	assertErrorEnvelope(t, reserved, http.StatusBadRequest, "invalid_request", false)

	for _, suffix := range []string{"", "/metadata", "/versions", "/lineage"} {
		read := serveQuery(t, fixture.handler, "/v1/artifacts/audit-standards-notes/draft"+suffix)
		if read.Code != http.StatusOK {
			t.Errorf("ordinary similar Namespace read %q = %d %s", suffix, read.Code, read.Body.String())
		}
		hidden := serveQuery(t, fixture.handler, "/v1/artifacts/"+artifactpolicy.AuditStandardCatalogNamespace+"/catalog"+suffix)
		if hidden.Code != http.StatusNotFound {
			t.Errorf("catalog read %q = %d %s", suffix, hidden.Code, hidden.Body.String())
		}
	}

	var listed []string
	cursor := ""
	for page := 0; ; page++ {
		if page > 5 {
			t.Fatal("listing did not terminate")
		}
		query := url.Values{"limit": {"1"}, "excludeNamespace": {"skills"}}
		if cursor != "" {
			query.Set("cursor", cursor)
		}
		response := serveQuery(t, fixture.handler, "/v1/artifacts?"+query.Encode())
		var result struct {
			Items []artifacts.Metadata `json:"items"`
			Page  pageInfoResponse     `json:"page"`
		}
		if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &result) != nil {
			t.Fatalf("list page = %d %s", response.Code, response.Body.String())
		}
		for _, item := range result.Items {
			listed = append(listed, item.Ref.Namespace+"/"+item.Ref.Name)
		}
		if !result.Page.HasMore {
			break
		}
		cursor = *result.Page.NextCursor
	}
	if len(listed) != 2 || listed[0] != "audit-standards-notes/draft" || listed[1] != "docs/readme" {
		t.Fatalf("listed = %v, want the similar Namespace and docs only", listed)
	}
}

// If-Match carries a Revision with the same grammar as ?revision, so a
// malformed one is a bad request, never a CAS conflict (ART-R70, API-R70).
func TestPublicArtifactIfMatchUsesTheRevisionGrammar(t *testing.T) {
	fixture := newHandlerFixture(t)
	if _, _, err := fixture.projects.Create(t.Context(), projectstore.CreateParams{
		ProjectID: "project-one", OwnerID: "user-1", Kind: projectstore.KindProject, Name: "Revisions",
		IdempotencyKey: "revisions-project", RequestDigest: auditHandlerDigest("revisions-project"),
	}); err != nil {
		t.Fatal(err)
	}
	created := putPublicArtifact(fixture.handler, "notes", "report", "If-None-Match", "*", "first")
	if created.Code != http.StatusCreated {
		t.Fatalf("create = %d %s", created.Code, created.Body.String())
	}
	writes := fixture.repository.writes
	for _, revision := range []string{
		`"has space"`, `"bad/revision"`, `"-leading"`, `"tab	inside"`, `"` + string(bytes.Repeat([]byte("r"), 257)) + `"`,
		// Escaped or differently quoted spellings of the valid revision r.
		`'r'`, "`r`", `"\x72"`,
	} {
		for _, path := range []string{
			"/v1/artifacts/notes/report",
			"/v1/projects/project-one/artifacts/notes/report",
		} {
			request := authenticatedRequest(http.MethodPut, path, bytes.NewReader([]byte("second")))
			request.Header.Set("Content-Type", "text/plain")
			request.Header.Set("If-Match", revision)
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, request)
			assertErrorEnvelope(t, response, http.StatusBadRequest, "invalid_request", false)
		}
		for _, path := range []string{
			"/v1/artifacts/notes/report/git-import",
			"/v1/projects/project-one/artifacts/notes/report/git-import",
		} {
			request := authenticatedRequest(http.MethodPost, path,
				bytes.NewReader([]byte(`{"repositoryUrl":"https://git.example.test/repo.git"}`)))
			request.Header.Set("Content-Type", "application/json")
			request.Header.Set("If-Match", revision)
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, request)
			assertErrorEnvelope(t, response, http.StatusBadRequest, "invalid_request", false)
		}
	}
	if fixture.repository.writes != writes {
		t.Fatal("a malformed If-Match reached the store")
	}
	// A well-formed revision, including one of the maximum 256 characters,
	// reaches the store and is an ordinary stale CAS conflict.
	for _, revision := range []string{`"stale-revision"`, `"` + string(bytes.Repeat([]byte("r"), 256)) + `"`} {
		stale := putPublicArtifact(fixture.handler, "notes", "report", "If-Match", revision, "second")
		assertErrorEnvelope(t, stale, http.StatusConflict, "conflict", true)
	}
	current := putPublicArtifact(fixture.handler, "notes", "report", "If-Match", created.Header().Get("ETag"), "second")
	if current.Code != http.StatusOK {
		t.Fatalf("current If-Match update = %d %s", current.Code, current.Body.String())
	}
}

// The OpenAPI ArtifactIfMatch header accepts exactly what the Server accepts,
// so the contract cannot drift from the Server's one Revision grammar,
// contracts.ValidResourceID (API-R70).
func TestArtifactIfMatchContractMatchesTheServer(t *testing.T) {
	t.Parallel()
	parameter := loadPublicOpenAPI(t).Components.Parameters["ArtifactIfMatch"]
	if parameter == nil || parameter.Value == nil || parameter.Value.Schema == nil ||
		parameter.Value.Schema.Value == nil {
		t.Fatal("the ArtifactIfMatch parameter has no schema")
	}
	schema := parameter.Value.Schema.Value
	probes := map[string]bool{
		`""`: true, `"`: true, "r": true, `W/"r"`: true, `"r", "s"`: true, `'r'`: true, "`r`": true,
		`"\x72"`: true, `"r"`: true, `"r\"s"`: true, ` "r"`: true, `"r" `: true, `"отчет"`: true,
	}
	for character := rune(0x20); character < 0x7f; character++ {
		for _, value := range []string{string(character), "r" + string(character), string(character) + "r"} {
			probes[`"`+value+`"`] = true
		}
	}
	for _, length := range []int{255, 256, 257} {
		probes[`"`+strings.Repeat("r", length)+`"`] = true
	}
	accepted := 0
	for header := range probes {
		request := httptest.NewRequest(http.MethodPut, "/v1/artifacts/notes/report", nil)
		request.Header.Set("If-Match", header)
		revision, err := artifactWritePrecondition(request)
		server := err == nil && revision != nil
		contract := schema.VisitJSON(header, openapi3.EnableJSONSchema2020()) == nil
		if server != contract {
			t.Errorf("If-Match %q: Server accepts %t, OpenAPI accepts %t", header, server, contract)
		}
		if server {
			accepted++
		}
	}
	if accepted < 60 {
		t.Fatalf("only %d probes were accepted; the probe set no longer covers the grammar", accepted)
	}
}
