//go:build integration

package privateartifacts

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPostgresPrivateSitemapImportAuthorityAndExactSources(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPrivateArtifactPool(t, ctx)
	createPrivateSitemapProjectRun(t, ctx, pool, "project-import", "run-import")
	createPrivateSitemapProjectRun(t, ctx, pool, "project-foreign", "run-foreign")
	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	payload := []byte("opaque evidence from an arbitrary source")
	create := func(runID, namespace string) contracts.ArtifactRef {
		t.Helper()
		scope, err := artifacts.RunScope(runID)
		if err != nil {
			t.Fatal(err)
		}
		written, err := artifactService.Create(ctx, scope, contracts.ArtifactRef{Namespace: namespace, Name: "report"}, artifacts.Payload{MediaType: "application/octet-stream", Data: payload})
		if err != nil {
			t.Fatal(err)
		}
		return written.Ref
	}
	own := create("run-import", "builder")
	otherNamespace := create("run-import", "other-worker")
	foreign := create("run-foreign", "builder")
	registry := &fakeRegistry{grant: testGrant("run-import")}
	registry.grant.SitemapPolicy = &contracts.ProjectSitemapPolicy{Version: contracts.SitemapAccessVersion}
	handler, err := NewHandler(Dependencies{Registry: registry, Artifacts: artifactService, Sitemap: projectsitemap.NewService(pool), RunProjects: runstore.NewPostgresStore(pool), Projects: projectstore.NewPostgresStore(pool)})
	if err != nil {
		t.Fatal(err)
	}
	request := func(key string, body any) *httptest.ResponseRecorder {
		t.Helper()
		data, err := json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
		r := trustedRequest(http.MethodPost, "/private/v1/allocations/allocation-1/sitemap/observations", bytes.NewReader(data))
		r.Header.Set("Content-Type", "application/json")
		r.Header.Set("Idempotency-Key", key)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, r)
		return response
	}
	input := sitemapCollectionRequest{APIVersion: contracts.APIVersion, Input: projectsitemap.CollectionIngress{Kind: "custom-scanner.v2", Artifact: &own, Observations: []projectsitemap.Observation{{URL: "https://example.test/selected", Method: "GET", Outcome: "response", Status: 200, At: "2026-10-29T00:00:00Z"}}}}
	assertPrivateErrorEnvelope(t, request("denied", input), 403, "sitemap_access_denied", false)
	registry.grant.SitemapTools = []string{contracts.SitemapImportTool}
	first := request("explicit", input)
	if first.Code != http.StatusOK {
		t.Fatalf("explicit import with collection disabled: %d %s", first.Code, first.Body.String())
	}
	var accepted, replayed struct {
		Result projectsitemap.Result `json:"result"`
	}
	if err := json.Unmarshal(first.Body.Bytes(), &accepted); err != nil {
		t.Fatal(err)
	}
	replay := request("explicit", input)
	if err := json.Unmarshal(replay.Body.Bytes(), &replayed); err != nil || replay.Code != http.StatusOK || !replayed.Result.Replayed || !accepted.Result.Ref.SameExact(replayed.Result.Ref) {
		t.Fatalf("exact import replay: %d %s %v", replay.Code, replay.Body.String(), err)
	}
	input.Input.Artifact = &foreign
	assertPrivateErrorEnvelope(t, request("foreign", input), 404, "artifact_not_found", false)
	input.Input.Artifact = &own
	input.Input.Observations[0].Status = 700
	assertPrivateErrorEnvelope(t, request("mismatch", input), 400, "invalid_request", false)
	input.Input.Observations[0].Status = 200
	input.Input.Artifact = &otherNamespace
	if response := request("other-explicit", input); response.Code != http.StatusOK {
		t.Fatalf("ordinary current-Run report access: %d %s", response.Code, response.Body.String())
	}
	input.Automatic = true
	assertPrivateErrorEnvelope(t, request("auto-disabled", input), 403, "sitemap_access_denied", false)
	registry.grant.SitemapTools = nil
	registry.grant.SitemapPolicy.CollectObservations = true
	assertPrivateErrorEnvelope(t, request("auto-other-namespace", input), 403, "sitemap_access_denied", false)
	input.Input.Artifact = &own
	if response := request("automatic", input); response.Code != http.StatusOK {
		t.Fatalf("automatic report without explicit tool grant: %d %s", response.Code, response.Body.String())
	}
	registry.grant.WritePolicy = ""
	assertPrivateErrorEnvelope(t, request("no-write", input), 403, "sitemap_access_denied", false)
	registry.grant.WritePolicy = controlplane.WriteInputsAndIntermediates
	registry.grant.WriteFenced = true
	assertPrivateErrorEnvelope(t, request("fenced", input), 409, "allocation_write_fenced", false)
}
