package projectsitemap

import (
	"bytes"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestPostgresSitemapHTTPCollectionProjectionSessionsAndReceipt(t *testing.T) {
	pool := sitemapPool(t)
	ctx := t.Context()
	createSitemapRun(t, ctx, pool, "http-consumer")
	s := NewService(pool)
	policy := readerPolicy("blackbox")
	policy.CollectObservations = true
	access, err := s.AllocationAccess(ctx, "http-consumer", "research", policy)
	if err != nil {
		t.Fatal(err)
	}
	input := CollectionIngress{Secrets: []contracts.SecretString{contracts.NewSecretString("private-canary")}, Observations: []HTTPObservation{}}
	for _, row := range []struct {
		path, outcome, generation string
		status                    int
	}{
		{"/safe?q=private-canary&token=private-canary", "response", "session-one", 200},
		{"/safe?q=other", "timeout", "session-two", 0},
		{"/private-canary", "response", "session-one", 200},
		{"/redirect", "response", "session-one", 302},
	} {
		o := observation(row.path)
		o.URL = "https://api.example.test" + row.path
		o.Outcome, o.Status = row.outcome, row.status
		input.Observations = append(input.Observations, HTTPObservation{Observation: o, SessionGeneration: row.generation})
	}
	committed, err := s.CollectHTTPAllocation(ctx, access, "http-first", input)
	if err != nil {
		t.Fatal(err)
	}
	view, err := s.Read(ctx, "owner", "project", "")
	if err != nil {
		t.Fatal(err)
	}
	safe, _ := json.Marshal(view.Document)
	for _, secret := range []string{"private-canary", "other", "token"} {
		if bytes.Contains(safe, []byte(secret)) {
			t.Fatalf("unsafe projection retained %q", secret)
		}
	}
	if len(view.Document.Groups) != 2 {
		t.Fatalf("unexpected visited routes: %+v", view.Document.Groups)
	}
	contexts := map[string]bool{}
	for _, group := range view.Document.Groups {
		for _, member := range group.Members {
			for _, summary := range member.Summaries {
				if summary.Provenance.RunID != "http-consumer" || summary.Provenance.Kind != "http" {
					t.Fatal("forged provenance")
				}
				if summary.Outcome == "timeout" && summary.Status != 0 {
					t.Fatal("invented response")
				}
				contexts[summary.Context] = true
			}
		}
	}
	if len(contexts) != 2 {
		t.Fatal("unknown sessions were merged")
	}
	history, _ := artifacts.NewHistoryPolicy(2, artifacts.DefaultHistoryMaxBytes)
	ctx = artifacts.WithHistoryPolicy(ctx, history)
	for n := range 4 {
		next := input
		next.Observations = []HTTPObservation{{Observation: observation(fmt.Sprintf("/next/%d", n)), SessionGeneration: "session-one"}}
		if _, err = s.CollectHTTPAllocation(ctx, access, fmt.Sprintf("next-%d", n), next); err != nil {
			t.Fatal(err)
		}
	}
	replayed, err := s.CollectHTTPAllocation(ctx, access, "http-first", input)
	if err != nil || !replayed.Replayed || *replayed.Ref.Revision != *committed.Ref.Revision {
		t.Fatalf("lost receipt after eviction: %+v %v", replayed, err)
	}
	if _, err = s.Read(ctx, "owner", "project", *committed.Ref.Revision); err == nil {
		t.Fatal("evicted bytes were revived")
	}
	access.Policy.CollectObservations = false
	if _, err = s.CollectHTTPAllocation(ctx, access, "denied", input); err == nil {
		t.Fatal("reader gained collection authority")
	}
}
