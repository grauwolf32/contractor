//go:build e2e

package e2e

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Seed declarations through the public owner and trusted domain boundaries.
// Only the real HTTP Runtime supplies observed target outcomes.
func seedUnifiedSitemap(t *testing.T, ctx context.Context, pool *pgxpool.Pool, operations *runtimeOperations, ownerID, projectID, targetURL string) {
	t.Helper()
	parsed, err := url.Parse(targetURL)
	if err != nil {
		t.Fatal(err)
	}
	origin := parsed.Scheme + "://" + parsed.Host
	operations.request(http.MethodPost, "/v1/projects/"+projectID+"/sitemap/mutations", map[string]any{
		"operation": "declare", "expectedRevision": "absent", "url": origin + "/manual-canary", "method": "GET",
	}, http.StatusOK, map[string]string{"Idempotency-Key": "unified-manual"})
	payload := []byte(`{"openapi":"3.1.0","info":{"title":"Sitemap fixture","version":"1"},"paths":{"/{route}":{"post":{}},"/openapi-only":{"get":{}}}}`)
	scope, err := artifacts.ProjectScope(projectID)
	if err != nil {
		t.Fatal(err)
	}
	stored, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Create(ctx, scope, contracts.ArtifactRef{Namespace: "schemas", Name: "unified-api"}, artifacts.Payload{MediaType: "application/json", Data: payload})
	if err != nil {
		t.Fatal(err)
	}
	exact := contracts.ExactContent(stored.Ref, "application/json", payload)
	_, err = projectsitemap.NewService(pool).Mutate(ctx, projectsitemap.Authority{
		OwnerID: ownerID, ProjectID: projectID, SourceID: "unified-openapi", Kind: "openapi",
	}, "unified-openapi", projectsitemap.Mutation{Operation: "replace-source", Source: &projectsitemap.SourceUpdate{
		SourceID: "unified-openapi", Artifact: &exact,
		Rules: []projectsitemap.Rule{
			{SourceID: "unified-openapi", Origin: origin, Method: "POST", Path: "/{route}", Pointer: "/paths/~1{route}/post", Parameters: []string{"route"}, Structural: []projectsitemap.StructuralParameter{}},
			{SourceID: "unified-openapi", Origin: origin, Method: "GET", Path: "/openapi-only", Pointer: "/paths/~1openapi-only/get", Parameters: []string{}, Structural: []projectsitemap.StructuralParameter{}},
		},
	}})
	if err != nil {
		t.Fatal(err)
	}
}

// Inspect actual model-visible tool responses, excluding the scripted calls.
func sitemapToolResponses(request map[string]any, name string) []map[string]any {
	var results []map[string]any
	messages, _ := request["messages"].([]any)
	for _, raw := range messages {
		message, _ := raw.(map[string]any)
		calls, _ := message["tool_calls"].([]any)
		for _, rawCall := range calls {
			call, _ := rawCall.(map[string]any)
			function, _ := call["function"].(map[string]any)
			if function["name"] != name {
				continue
			}
			id, _ := call["id"].(string)
			value, ok := toolResponse(request, id)
			if !ok {
				continue
			}
			object, ok := value.(map[string]any)
			if !ok {
				continue
			}
			if inner, wrapped := object["result"].(map[string]any); wrapped {
				object = inner
			}
			results = append(results, object)
		}
	}
	return results
}

func unifiedSitemapSelection(request map[string]any) ([]string, error) {
	for _, result := range sitemapToolResponses(request, "list_sitemap_entries") {
		entries, _ := result["entries"].([]any)
		var ids, routes []string
		for _, raw := range entries {
			entry, _ := raw.(map[string]any)
			id, hasID := entry["id"].(string)
			route, hasRoute := entry["route"].(string)
			if hasID && hasRoute {
				ids, routes = append(ids, id), append(routes, route)
			}
		}
		if len(ids) == 3 && slices.Contains(routes, "/manual-canary") && slices.Contains(routes, "/openapi-only") && slices.Contains(routes, "/{route}") && result["policyDigest"] == nil {
			return ids, nil
		}
	}
	return nil, fmt.Errorf("shared sitemap reader lost manual/OpenAPI metadata or changed grouping")
}

func requireUnifiedSitemapMetadata(request map[string]any) error {
	for _, result := range sitemapToolResponses(request, "list_sitemap_entries") {
		entries, _ := result["entries"].([]any)
		kinds := map[string]bool{}
		for _, raw := range entries {
			entry, _ := raw.(map[string]any)
			value, _ := entry["value"].(map[string]any)
			provenance, _ := value["provenance"].(map[string]any)
			kind, _ := provenance["kind"].(string)
			if entry["kind"] == "source" {
				kinds[kind] = true
				if _, obsolete := provenance["mode"]; obsolete {
					return fmt.Errorf("shared provenance retained obsolete source mode")
				}
			}
		}
		if kinds["manual"] && kinds["openapi"] && kinds["http"] {
			return nil
		}
	}
	return fmt.Errorf("metadata reader omitted a manual, OpenAPI or actual HTTP source")
}

func assertUnifiedSitemapSnapshot(t *testing.T, operations *runtimeOperations, runID string) {
	t.Helper()
	var page struct {
		Items []struct {
			Ref       artifactRef `json:"artifact"`
			MediaType string      `json:"mediaType"`
		} `json:"items"`
	}
	data := operations.request(http.MethodGet, "/v1/runs/"+runID+"/artifacts?namespace=http&limit=100", nil, http.StatusOK, nil)
	if err := json.Unmarshal(data, &page); err != nil {
		t.Fatal(err)
	}
	found := 0
	for _, item := range page.Items {
		if item.MediaType != projectsitemap.SnapshotMediaType {
			continue
		}
		found++
		var snapshot struct {
			Document projectsitemap.Document   `json:"document"`
			Inputs   []contracts.ExactArtifact `json:"inputs"`
		}
		data = operations.request(http.MethodGet, "/v1/runs/"+runID+"/artifacts/http/"+item.Ref.Name+"?revision="+*item.Ref.Revision, nil, http.StatusOK, nil)
		if err := json.Unmarshal(data, &snapshot); err != nil {
			t.Fatal(err)
		}
		var raw map[string]json.RawMessage
		if err := json.Unmarshal(data, &raw); err != nil {
			t.Fatal(err)
		}
		if raw["policy"] != nil || raw["policyDigest"] != nil || len(snapshot.Inputs) != 0 {
			t.Fatalf("snapshot retained obsolete policy or ungranted input bytes: %s", data)
		}
		kinds := map[string]bool{}
		for _, source := range snapshot.Document.Sources {
			kinds[source.Provenance.Kind] = true
			if source.Artifact != nil {
				t.Fatal("provenance granted original artifact file access")
			}
		}
		if !kinds["manual"] || !kinds["openapi"] || !kinds["http"] || !slices.ContainsFunc(snapshot.Document.Gaps, func(gap projectsitemap.Gap) bool { return gap.Reason == "source-unavailable" }) {
			t.Fatalf("snapshot hid source metadata or lost missing-input gap: %s", data)
		}
		assertUnifiedSitemapGroups(t, snapshot.Document)
	}
	if found != 1 {
		t.Fatalf("retained sitemap snapshots = %d, want one", found)
	}
}

func assertUnifiedSitemapGroups(t *testing.T, document projectsitemap.Document) {
	t.Helper()
	var routes []string
	observed := false
	for _, group := range document.Groups {
		routes = append(routes, group.Route)
		for _, member := range group.Members {
			for _, summary := range member.Summaries {
				if member.Path == "/failure" && group.Route == "/{route}" && summary.Provenance.Kind == "http" && summary.Status == 500 {
					observed = true
				}
			}
		}
	}
	if !observed || !slices.Contains(routes, "/manual-canary") || !slices.Contains(routes, "/openapi-only") {
		t.Fatalf("unified sitemap lost actual outcome, declarations or source grouping: %+v", document.Groups)
	}
}
