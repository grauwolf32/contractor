package projectsitemap

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func joinContract(t *testing.T, document Document, source string) joinedContract {
	t.Helper()
	revision := "schema-revision"
	input := contracts.ExactContent(contracts.ArtifactRef{Namespace: "inputs", Name: "openapi", Revision: &revision}, "application/json", []byte(source))
	joined, err := joinOpenAPI(t.Context(), document, []byte(source), "application/json", input, JoinOptions{Origin: "https://example.test", Contract: "public-api"})
	if err != nil {
		t.Fatal(err)
	}
	return joined
}

func TestOpenAPIJoinConcretePrecedenceAndHonestComparison(t *testing.T) {
	document := applyObservations(t, Empty(), httpAuthority("capture"), observation("/users/me?extra=1"), observation("/users/42"), observation("/undocumented"))
	source := `{"openapi":"3.1.0","paths":{"/users/{id}":{"get":{"parameters":[{"in":"path","name":"id"}],"responses":{"2XX":{}}}},"/users/me":{"get":{"parameters":[{"in":"query","name":"expected"}],"responses":{"201":{}}}},"/declared":{"post":{"responses":{"204":{}}}}}}`
	joined := joinContract(t, document, source)
	if got := joined.summary; got.Declared != 3 || got.Observed != 3 || got.Matched != 2 || got.DeclaredOnly != 1 || got.ObservedOnly != 1 || got.Ambiguous != 0 {
		t.Fatalf("comparison: %+v", got)
	}
	var names, statuses bool
	for _, difference := range joined.summary.Differences {
		if difference.Kind == "parameters" && difference.Path == "/users/me" {
			names = reflect.DeepEqual(difference.Missing, []string{"expected"}) && reflect.DeepEqual(difference.Extra, []string{"extra"})
		}
		if difference.Kind == "statuses" && difference.Path == "/users/me" {
			statuses = reflect.DeepEqual(difference.Statuses, []int{200})
		}
	}
	if !names || !statuses {
		t.Fatalf("lost parameter/status differences: %+v", joined.summary)
	}
	for _, group := range joined.document.Groups {
		for _, member := range group.Members {
			if member.Path == "/users/me" && group.Route != "/users/me" {
				t.Fatal("template swallowed concrete declaration")
			}
		}
		if group.Route == "/declared" && len(group.Members) != 0 {
			t.Fatal("schema manufactured an observation")
		}
	}
	if again := joinContract(t, document, source); !bytes.Equal(again.payload, joined.payload) || !reflect.DeepEqual(again.summary, joined.summary) {
		t.Fatal("join is not deterministic")
	}
}

func TestOpenAPIJoinAmbiguityAndOtherContractIsolation(t *testing.T) {
	document := applyObservations(t, Empty(), httpAuthority("capture"), observation("/users/me"))
	ambiguous := joinContract(t, document, `{"openapi":"3.0.3","paths":{"/users/{id}":{"get":{}},"/users/{name}":{"get":{}}}}`)
	if ambiguous.summary.Ambiguous != 1 || ambiguous.summary.Matched != 0 || ambiguous.summary.Unresolved != 2 || ambiguous.summary.DeclaredOnly != 0 {
		t.Fatalf("guessed ambiguous route: %+v", ambiguous.summary)
	}
	other := Authority{SourceID: "other-contract", Kind: "openapi"}
	var err error
	document, _, err = Reduce(document, Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: other.SourceID, Rules: []Rule{{SourceID: other.SourceID, Origin: "https://example.test", Method: "GET", Path: "/users/me", Parameters: []string{}, Structural: []StructuralParameter{}}}}}, other)
	if err != nil {
		t.Fatal(err)
	}
	selected := joinContract(t, document, `{"openapi":"3.1.0","paths":{"/users/{id}":{"get":{}}}}`)
	if selected.summary.Matched != 1 || selected.summary.DeclaredOnly != 0 {
		t.Fatalf("other contract changed selected comparison: %+v", selected.summary)
	}
}

func TestOpenAPIJoinRetainsEditsSuppressionAndRejectsStaleSource(t *testing.T) {
	document := applyObservations(t, Empty(), httpAuthority("capture"), observation("/users/42"))
	id := document.Groups[0].Members[0].ID
	owner := Authority{SourceID: OwnerSourceID, Kind: "manual"}
	for _, command := range []Mutation{{Operation: "annotate", MemberIDs: []string{id}, Note: "owner note", Tags: []string{"review"}}, {Operation: "remove", MemberIDs: []string{id}, MarkerID: "removed"}} {
		var err error
		document, _, err = Reduce(document, command, owner)
		if err != nil {
			t.Fatal(err)
		}
	}
	source := `{"openapi":"3.1.0","paths":{"/users/{id}":{"get":{}}}}`
	first := joinContract(t, document, source)
	if len(first.document.Markers) != 1 || first.document.Markers[0].Members[0] != id {
		t.Fatal("suppression selection changed")
	}
	found := false
	for _, group := range first.document.Groups {
		for _, member := range group.Members {
			if member.ID == id {
				found = len(member.Summaries) == 1 && len(member.Annotations) == 1 && member.Annotations[0].Note == "owner note"
			}
		}
	}
	if !found {
		t.Fatal("retained member metadata lost")
	}
	older := joinContract(t, first.document, source)
	newer := joinContract(t, first.document, `{"openapi":"3.1.0","paths":{"/new":{"get":{}}}}`)
	if _, _, err := Reduce(newer.document, older.mutation, older.authority); !errors.Is(err, ErrConflict) {
		t.Fatalf("stale source rollback: %v", err)
	}
	appended := applyObservations(t, first.document, httpAuthority("capture"), observation("/concurrent"))
	merged, _, err := Reduce(appended, older.mutation, older.authority)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.ContainsFunc(merged.Groups, func(g Group) bool { return g.Route == "/concurrent" }) {
		t.Fatal("append lost during source replacement")
	}
	if manualIntent(appended) != manualIntent(first.document) {
		t.Fatal("append changed manual intent")
	}
}

func TestOpenAPIJoinUsesSharedLocalReferencesAndExplicitBase(t *testing.T) {
	source := `{"openapi":"3.1.0","paths":{"/items":{"$ref":"#/components/pathItems/Items"}},"components":{"pathItems":{"Items":{"parameters":[{"$ref":"#/components/parameters/Page"}],"get":{"parameters":[{"in":"query","name":"page"}],"responses":{"200":{"$ref":"#/components/responses/OK"}}}}},"parameters":{"Page":{"in":"query","name":"page"}},"responses":{"OK":{"description":"ok"}}}}`
	joined := joinContract(t, Empty(), source)
	if len(joined.document.Rules) != 1 || !reflect.DeepEqual(joined.document.Rules[0].Parameters, []string{"page"}) {
		t.Fatal("local reference/parameter override changed")
	}
	rev := "r"
	input := contracts.ExactContent(contracts.ArtifactRef{Namespace: "inputs", Name: "openapi", Revision: &rev}, "application/json", []byte(source))
	base, err := joinOpenAPI(t.Context(), Empty(), []byte(source), "application/json", input, JoinOptions{Origin: "https://example.test/api/v1/", Contract: "api"})
	if err != nil || base.document.Rules[0].Path != "/api/v1/items" {
		t.Fatalf("base: %+v %v", base, err)
	}
	for _, bad := range []string{strings.Replace(source, "#/components/pathItems/Items", "https://external.invalid/spec.json", 1), `{"openapi":"3.1.0","paths":{"/items":{"$ref":"#/paths/~1items"}}}`, `{"openapi":"2.0","paths":{}}`, `{"openapi":"3.1.0","paths":{},"paths":{}}`} {
		if _, err := joinOpenAPI(t.Context(), Empty(), []byte(bad), "application/json", input, JoinOptions{Origin: "https://example.test", Contract: "api"}); err == nil {
			t.Fatal("accepted invalid or external schema")
		}
	}
	for _, origin := range []string{"", "https://user:password@example.test", "https://{host}", "https://example.test?x=1"} {
		if _, err := joinOpenAPI(t.Context(), Empty(), []byte(source), "application/json", input, JoinOptions{Origin: origin, Contract: "api"}); err == nil {
			t.Fatalf("accepted origin %q", origin)
		}
	}
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := joinOpenAPI(cancelled, Empty(), []byte(source), "application/json", input, JoinOptions{Origin: "https://example.test", Contract: "api"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
}

func TestOpenAPIJoinBoundsRulesAndComparison(t *testing.T) {
	paths := map[string]any{}
	for i := range MaxRules {
		paths[fmt.Sprintf("/items/%03d", i)] = map[string]any{"get": map[string]any{}}
	}
	source, _ := json.Marshal(map[string]any{"openapi": "3.1.0", "paths": paths})
	joined := joinContract(t, Empty(), string(source))
	if joined.summary.DeclaredOnly != MaxRules || len(joined.summary.Differences) != 32 || joined.summary.OmittedDifferences != MaxRules-32 {
		t.Fatalf("unbounded/incomplete report: %+v", joined.summary)
	}
	paths["/overflow"] = map[string]any{"get": map[string]any{}}
	source, _ = json.Marshal(map[string]any{"openapi": "3.1.0", "paths": paths})
	rev := "r"
	input := contracts.ExactContent(contracts.ArtifactRef{Namespace: "inputs", Name: "openapi", Revision: &rev}, "application/json", source)
	if _, err := joinOpenAPI(t.Context(), Empty(), source, "application/json", input, JoinOptions{Origin: "https://example.test", Contract: "api"}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("accepted over-bound source: %v", err)
	}
}
