package projectsitemap

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func receiptProjectionRule(path string, selectors ...StructuralParameter) Rule {
	return Rule{SourceID: "rules", Origin: "https://example.test", Method: "GET", Path: path,
		Parameters: []string{}, Structural: selectors}
}

func TestPostgresSitemapReceiptPreservesStructuralOmissions(t *testing.T) {
	action := StructuralParameter{Name: "action", Allowed: []string{"read", "write"}}
	format := StructuralParameter{Name: "format", Allowed: []string{"json", "xml"}}
	stable := receiptProjectionRule("/stable", action)
	for _, tc := range []struct {
		name  string
		rules []Rule
		paths []string
	}{
		{
			name: "conflicting-absent-present-and-repeated-paths",
			rules: []Rule{
				receiptProjectionRule("/dispatch/{id}", action),
				receiptProjectionRule("/dispatch/{id}", StructuralParameter{Name: "action", Allowed: []string{"other"}}),
			},
			paths: []string{"/dispatch/one", "/dispatch/one?action=read", "/dispatch/one?action=unknown", "/dispatch/two?other=value"},
		},
		{
			name: "equal-policies-with-absent-and-rejected-values",
			rules: []Rule{
				receiptProjectionRule("/dispatch/{id}", action),
				receiptProjectionRule("/dispatch/{id}", StructuralParameter{Name: "action", Allowed: []string{"write", "read"}}),
			},
			paths: []string{"/dispatch/one", "/dispatch/one?action=unknown", "/dispatch/one?action=read", "/dispatch/two?action=write&action=read"},
		},
		{
			name:  "partial-selection-and-multiplicity-overflow",
			rules: []Rule{receiptProjectionRule("/dispatch/{id}", action, format)},
			paths: []string{"/dispatch/one?action=read&format=unknown", "/dispatch/one?action=read", "/dispatch/one?action=write&format=json", "/dispatch/two?" + strings.Repeat("action=read&", MaxTags+1)},
		},
		{
			name: "ambiguous-and-concrete-routes-in-one-batch",
			rules: []Rule{
				receiptProjectionRule("/dispatch/{id}", action),
				receiptProjectionRule("/{entity}/one", format),
			},
			paths: []string{"/dispatch/one?action=unknown", "/dispatch/two?action=read", "/dispatch/two?action=unknown"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pool := sitemapPool(t)
			history, err := artifacts.NewHistoryPolicy(2, artifacts.DefaultHistoryMaxBytes)
			if err != nil {
				t.Fatal(err)
			}
			ctx := artifacts.WithHistoryPolicy(t.Context(), history)
			service := NewService(pool)
			owner := Authority{OwnerID: "owner", ProjectID: "project", SourceID: "rules", Kind: "manual"}
			rules := append(slices.Clone(tc.rules), stable)
			if _, err := service.Mutate(ctx, owner, "configure", Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: owner.SourceID, Rules: rules}}); err != nil {
				t.Fatal(err)
			}
			command := Mutation{Operation: "observe", Observations: []Observation{observation("/stable?action=read&action=write")}}
			for _, path := range tc.paths {
				command.Observations = append(command.Observations, observation(path))
			}
			producer := httpAuthority("producer")
			first, err := service.Mutate(ctx, producer, "observations", command)
			if err != nil {
				t.Fatal(err)
			}
			// Retry the identical operation as if its successful reply was lost.
			replayed, err := service.Mutate(ctx, producer, "observations", command)
			if err != nil || !replayed.Replayed || !replayed.Ref.SameExact(first.Ref) {
				t.Fatalf("identical retry: %+v %v", replayed, err)
			}
			changed := command
			changed.Observations = slices.Clone(command.Observations)
			changed.Observations[0].URL = "https://example.test/stable?action=write&action=read"
			if _, err := service.Mutate(ctx, producer, "observations", changed); !errors.Is(err, contracts.ErrIdempotencyConflict) {
				t.Fatalf("changed accepted selector order: %v", err)
			}
			view, err := readSitemap(service, ctx, "project", "")
			if err != nil {
				t.Fatal(err)
			}
			index := slices.IndexFunc(view.Document.Sources, func(s Source) bool { return s.Provenance.SourceID == owner.SourceID })
			if index < 0 {
				t.Fatal("missing configured source")
			}
			expected := view.Document.Sources[index]
			if _, err := service.Mutate(ctx, owner, "replace-policy", Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: owner.SourceID, Expected: &expected, Rules: []Rule{}}}); err != nil {
				t.Fatal(err)
			}
			current, err := service.Mutate(ctx, producer, "advance", Mutation{Operation: "observe", Observations: []Observation{observation("/after-replacement")}})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := readSitemap(service, ctx, "project", *first.Ref.Revision); !errors.Is(err, artifacts.ErrArtifactNotFound) {
				t.Fatalf("original result was not evicted: %v", err)
			}
			replayed, err = service.Mutate(ctx, producer, "observations", command)
			if err != nil || !replayed.Replayed || !replayed.Ref.SameExact(first.Ref) {
				t.Fatalf("retry after policy replacement and eviction: %+v %v", replayed, err)
			}
			if _, err := service.Mutate(ctx, producer, "observations", changed); !errors.Is(err, contracts.ErrIdempotencyConflict) {
				t.Fatalf("replacement policy changed accepted-intent conflict: %v", err)
			}
			view, err = readSitemap(service, ctx, "project", "")
			if err != nil || view.Revision != *current.Ref.Revision {
				t.Fatalf("replay changed current revision: %s %v", view.Revision, err)
			}
		})
	}
}

func TestPostgresSitemapDeclarationReceiptPreservesStructuralValues(t *testing.T) {
	pool := sitemapPool(t)
	service := NewService(pool)
	ctx := t.Context()
	owner := Authority{OwnerID: "owner", ProjectID: "project", SourceID: "rules", Kind: "manual"}
	rule := receiptProjectionRule("/dispatch", StructuralParameter{Name: "action", Allowed: []string{"read", "write"}})
	configured, err := service.Mutate(ctx, owner, "configure", Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: owner.SourceID, Rules: []Rule{rule}}})
	if err != nil {
		t.Fatal(err)
	}
	command := Mutation{Operation: "declare", ExpectedRevision: *configured.Ref.Revision, URL: "https://example.test/dispatch?action=read", Method: "GET"}
	first, err := service.Mutate(ctx, owner, "declare", command)
	if err != nil {
		t.Fatal(err)
	}
	changed := command
	changed.URL = "https://example.test/dispatch?action=write"
	if _, err := service.Mutate(ctx, owner, "declare", changed); !errors.Is(err, contracts.ErrIdempotencyConflict) {
		t.Fatalf("changed declaration structural value: %v", err)
	}
	view, err := readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	expected := view.Document.Sources[0]
	if _, err := service.Mutate(ctx, owner, "replace-policy", Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: owner.SourceID, Expected: &expected, Rules: []Rule{}}}); err != nil {
		t.Fatal(err)
	}
	replayed, err := service.Mutate(ctx, owner, "declare", command)
	if err != nil || !replayed.Replayed || !replayed.Ref.SameExact(first.Ref) {
		t.Fatalf("declaration replay after policy replacement: %+v %v", replayed, err)
	}
}

func TestSitemapReceiptConflictsStayWithinBatchAndReceiptBounds(t *testing.T) {
	allowed := make([]string, MaxTags)
	for i := range allowed {
		allowed[i] = fmt.Sprintf("%02d", i) + strings.Repeat("v", MaxTagBytes-2)
	}
	other := slices.Clone(allowed)
	other[0] = strings.Repeat("w", MaxTagBytes)
	rules := []Rule{
		receiptProjectionRule("/items/{id}", StructuralParameter{Name: "action", Allowed: allowed}),
		receiptProjectionRule("/items/{id}", StructuralParameter{Name: "action", Allowed: other}),
	}
	command := Mutation{Operation: "observe", Observations: make([]Observation, MaxBatch)}
	for i := range command.Observations {
		command.Observations[i] = observation(fmt.Sprintf("/items/%03d", i) + strings.Repeat("界", 760))
	}
	if len(jsonBytes(command)) > MaxBatchBytes {
		t.Fatal("fixture exceeds the ingress bound")
	}
	producer := httpAuthority("producer")
	intent, projected, policy, err := projectIntent(command, producer, rules)
	if err != nil {
		t.Fatal(err)
	}
	if len(policy) != MaxBatch {
		t.Fatalf("expected one conflicting pair per path, got %d replay rules", len(policy))
	}
	encoded, err := json.Marshal(receiptRecord{Policy: policy})
	if err != nil || len(encoded) > 1<<20 {
		t.Fatalf("receipt exceeds existing bound: bytes=%d err=%v", len(encoded), err)
	}
	replayIntent, replayProjected, _, err := projectIntent(command, producer, policy)
	if err != nil || !reflect.DeepEqual(intent, replayIntent) || !reflect.DeepEqual(projected, replayProjected) {
		t.Fatalf("maximum batch changed intent on replay: %v", err)
	}
	t.Logf("batch_bytes=%d receipt_bytes=%d paths=%d replay_rules=%d", len(jsonBytes(command)), len(encoded), len(command.Observations), len(policy))
}

func TestPostgresSitemapDeclarationReceiptPreservesParameterNames(t *testing.T) {
	pool := sitemapPool(t)
	service := NewService(pool)
	ctx := t.Context()
	owner := Authority{OwnerID: "owner", ProjectID: "project", SourceID: "rules", Kind: "manual"}
	command := Mutation{Operation: "declare", ExpectedRevision: "absent", URL: "https://example.test/search?q=one", Method: "GET"}
	first, err := service.Mutate(ctx, owner, "declare", command)
	if err != nil {
		t.Fatal(err)
	}
	checkReplay := func() {
		t.Helper()
		changed := command
		changed.URL = "https://example.test/search?token=one"
		if _, err := service.Mutate(ctx, owner, "declare", changed); !errors.Is(err, contracts.ErrIdempotencyConflict) {
			t.Fatalf("changed declaration parameter name: %v", err)
		}
		for _, raw := range []string{"https://example.test/search?q=two", "https://example.test/search?q=two&q=three"} {
			same := command
			same.URL = raw
			replayed, err := service.Mutate(ctx, owner, "declare", same)
			if err != nil || !replayed.Replayed || !replayed.Ref.SameExact(first.Ref) {
				t.Fatalf("ordinary parameter values changed receipt identity: %+v %v", replayed, err)
			}
		}
	}
	checkReplay()
	view, err := readSitemap(service, ctx, "project", "")
	if err != nil {
		t.Fatal(err)
	}
	expected := view.Document.Sources[0]
	rule := receiptProjectionRule("/search", StructuralParameter{Name: "q", Allowed: []string{"one"}})
	if _, err := service.Mutate(ctx, owner, "configure-selector", Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: owner.SourceID, Expected: &expected, Rules: []Rule{rule}}}); err != nil {
		t.Fatal(err)
	}
	checkReplay()
}
