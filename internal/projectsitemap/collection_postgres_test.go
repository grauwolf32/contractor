package projectsitemap

import (
	"bytes"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
)

func TestPostgresSitemapHTTPCollectionProjectionAndReceipt(t *testing.T) {
	artifactRuntime := artifactpgtest.NewRuntime()
	artifactHistory := artifactpostgres.DefaultHistoryPolicy()
	artifactStorage := artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)

	pool := sitemapPool(t)
	ctx := t.Context()
	createSitemapRun(t, ctx, pool, "http-consumer")
	s := NewService(pool, artifactStorage)
	policy := readerPolicy()
	policy.CollectObservations = true
	access, err := s.AllocationAccess(ctx, "http-consumer", "research", policy)
	if err != nil {
		t.Fatal(err)
	}
	input := CollectionIngress{Kind: "http", Observations: []Observation{}}
	for _, row := range []struct {
		path, outcome string
		status        int
	}{
		{"/safe?q=query-canary&token=query-canary&password=query-canary", "response", 200},
		{"/safe?q=other-query-value", "timeout", 0},
		{"/session/short-password", "response", 200},
		{"/users/short%252Dpassword", "response", 200},
		{"/opaque/ABCDEFGHIJKLMNOPQRSTUVWXYZ123456", "response", 200},
		{"/redirect", "response", 302},
	} {
		o := observation(row.path)
		o.URL = "https://api.example.test" + row.path
		o.Outcome, o.Status = row.outcome, row.status
		input.Observations = append(input.Observations, o)
	}
	committed, err := s.CollectAllocation(ctx, access, "http-first", input)
	if err != nil {
		t.Fatal(err)
	}
	view, err := readSitemap(s, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	safe, _ := json.Marshal(view.Document)
	for _, excluded := range []string{"query-canary", "other-query-value"} {
		if bytes.Contains(safe, []byte(excluded)) {
			t.Fatalf("non-structural query value retained %q", excluded)
		}
	}
	for _, retained := range []string{"/session/short-password", "/users/short%252Dpassword", "/opaque/ABCDEFGHIJKLMNOPQRSTUVWXYZ123456", `"token"`, `"password"`} {
		if !bytes.Contains(safe, []byte(retained)) {
			t.Fatalf("HTTP surface metadata was discarded: %q", retained)
		}
	}
	if len(view.Document.Groups) != 5 {
		t.Fatalf("unexpected visited routes: %+v", view.Document.Groups)
	}
	for _, group := range view.Document.Groups {
		for _, member := range group.Members {
			for _, summary := range member.Summaries {
				if summary.Provenance.SourceID != key("http", "") || summary.Provenance.Kind != "http" {
					t.Fatal("forged provenance")
				}
				if summary.Outcome == "timeout" && summary.Status != 0 {
					t.Fatal("invented response")
				}
			}
		}
	}
	history, _ := artifactpostgres.NewHistoryPolicy(2, artifactpostgres.DefaultHistoryMaxBytes)
	artifactHistory = history
	artifactStorage = artifactpgtest.NewConfigured(artifactRuntime, artifactHistory)
	s = NewService(pool, artifactStorage)
	for n := range 4 {
		next := input
		next.Observations = []Observation{observation(fmt.Sprintf("/next/%d", n))}
		if _, err = s.CollectAllocation(ctx, access, fmt.Sprintf("next-%d", n), next); err != nil {
			t.Fatal(err)
		}
	}
	replayed, err := s.CollectAllocation(ctx, access, "http-first", input)
	if err != nil || !replayed.Replayed || *replayed.Ref.Revision != *committed.Ref.Revision {
		t.Fatalf("lost receipt after eviction: %+v %v", replayed, err)
	}
	if _, err = readSitemap(s, ctx, "project", *committed.Ref.Revision); err == nil {
		t.Fatal("evicted bytes were revived")
	}
	access.Policy.CollectObservations = false
	if _, err = s.CollectAllocation(ctx, access, "explicit", input); err != nil {
		t.Fatal("domain incorrectly required automatic collection policy for an authorized explicit batch")
	}
}

func TestPostgresSitemapRetainsCredentialNamedMetadata(t *testing.T) {
	pool := sitemapPool(t)
	ctx := t.Context()
	createSitemapRun(t, ctx, pool, "metadata-consumer")
	service := NewService(pool, artifactpgtest.NewStorage())
	policy := readerPolicy()
	policy.CollectObservations = true
	access, err := service.AllocationAccess(ctx, "metadata-consumer", "research", policy)
	if err != nil {
		t.Fatal(err)
	}
	rule := EditorRule{Origin: "https://example.test", Method: "GET", Path: "/dispatch", Pointer: "#/paths/~1session~1short-password", Parameters: []string{"password", "token"}, Structural: []StructuralParameter{{Name: "token", Allowed: []string{"short-password", "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"}}}}
	_, err = service.MutateAllocation(ctx, access, "configure", EditorMutation{Operation: "replace-source", ExpectedRevision: "absent", Source: &EditorSourceUpdate{Rules: []EditorRule{rule}}})
	if err != nil {
		t.Fatal(err)
	}
	_, err = service.CollectAllocation(ctx, access, "observe", CollectionIngress{Kind: "http", Observations: []Observation{observation("/dispatch?token=short-password&password=excluded-query-value")}})
	if err != nil {
		t.Fatal(err)
	}
	view, err := readSitemap(service, ctx, access.ProjectID, "")
	if err != nil {
		t.Fatal(err)
	}
	var memberID string
	for _, group := range view.Document.Groups {
		if len(group.Structural) == 1 && group.Structural[0].Name == "token" && len(group.Structural[0].Values) == 1 && group.Structural[0].Values[0] == "short-password" && len(group.Members) == 1 {
			memberID = group.Members[0].ID
		}
	}
	if memberID == "" {
		t.Fatal("credential-named structural selection was discarded")
	}
	_, err = service.MutateAllocation(ctx, access, "annotate", EditorMutation{Operation: "annotate", ExpectedRevision: view.Revision, MemberIDs: []string{memberID}, Note: "password token short-password ABCDEFGHIJKLMNOPQRSTUVWXYZ123456", Tags: []string{"password", "token", "short-password", "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"}})
	if err != nil {
		t.Fatal(err)
	}
	view, err = readSitemap(service, ctx, access.ProjectID, "")
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(view.Document)
	if err != nil {
		t.Fatal(err)
	}
	for _, retained := range []string{"password token short-password ABCDEFGHIJKLMNOPQRSTUVWXYZ123456", `"password"`, `"token"`, `"short-password"`, `"ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"`, "#/paths/~1session~1short-password"} {
		if !bytes.Contains(data, []byte(retained)) {
			t.Fatalf("metadata lost after real-store read: %q", retained)
		}
	}
	if bytes.Contains(data, []byte("excluded-query-value")) {
		t.Fatal("non-structural query value persisted")
	}
}
