package projectsitemap

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func httpAuthority(run string) Authority {
	return Authority{OwnerID: "owner", ProjectID: "project", RunID: run, SourceID: run, Kind: "http", Mode: "blackbox", CanObserve: true}
}
func observation(path string) Observation {
	return Observation{URL: "https://example.test" + path, Method: "GET", Outcome: "response", Status: 200, At: "2026-10-26T00:00:00Z"}
}
func applyObservations(t *testing.T, d Document, a Authority, observations ...Observation) Document {
	t.Helper()
	result, _, err := Reduce(d, Mutation{Operation: "observe", Observations: observations}, a)
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func TestSharedProjectSitemapCases(t *testing.T) {
	data := contractstest.ReadFile(t, "api", "testdata", "project-sitemap-cases.json")
	var fixture struct {
		Version    string          `json:"version"`
		Empty      json.RawMessage `json:"empty"`
		Invalid    []string        `json:"invalid"`
		Projection []struct {
			URL        string   `json:"url"`
			Path       string   `json:"path"`
			Omitted    bool     `json:"omitted"`
			Parameters []string `json:"parameters"`
		} `json:"projection"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Version != Version || len(fixture.Invalid) < 4 || len(fixture.Projection) < 4 {
		t.Fatal("incomplete shared fixture")
	}
	document, err := Decode(fixture.Empty)
	if err != nil || !reflect.DeepEqual(document, Empty()) {
		t.Fatalf("empty: %v", err)
	}
	for _, invalid := range fixture.Invalid {
		if _, err := Decode([]byte(invalid)); err == nil {
			t.Fatalf("accepted %s", invalid)
		}
	}
	for _, tc := range fixture.Projection {
		a := httpAuthority("run-1")
		o := observation("/")
		o.URL = tc.URL
		d := applyObservations(t, Empty(), a, o)
		if tc.Omitted {
			if len(d.Groups) != 0 || len(d.Gaps) == 0 {
				t.Fatalf("omission: %+v", d)
			}
		} else if len(d.Groups) != 1 || d.Groups[0].Members[0].Path != tc.Path {
			t.Fatalf("projection: %+v", d)
		}
		if len(d.Groups) > 0 && !reflect.DeepEqual(d.Groups[0].Members[0].Summaries[0].Parameters, tc.Parameters) {
			t.Fatalf("parameter names were changed: %+v", d.Groups[0].Members[0].Summaries[0].Parameters)
		}
		encoded, err := encode(d)
		if err != nil {
			t.Fatal(err)
		}
		for _, forbidden := range []string{"query-canary", "header-canary"} {
			if bytes.Contains(encoded, []byte(forbidden)) || bytes.Contains(encoded, []byte(key(forbidden))) {
				t.Fatalf("retained excluded query value or hash: %s", forbidden)
			}
		}
	}
}

func TestSharedSitemapTextBoundsCountUTF8Bytes(t *testing.T) {
	characters, fields := contractstest.ReadUTF8ByteBoundCases(t)
	checked := map[string]bool{}
	for _, field := range fields {
		if !strings.HasPrefix(field.Schema, "Sitemap") {
			continue
		}
		bound := MaxTagBytes
		if field.Rule == "LIM-SMT-6" {
			bound = MaxNoteBytes
		}
		if bound != field.MaxBytes {
			t.Fatalf("shared bound drift: %+v", field)
		}
		checked[field.Rule] = true
		for _, value := range field.Values(characters) {
			var err error
			if field.Rule == "LIM-SMT-6" {
				_, err = projectAnnotation(value.Value, nil, Authority{})
			} else {
				_, err = projectAnnotation("", []string{value.Value}, Authority{})
			}
			// Metadata is retained within its UTF-8 byte limit; over-bound
			// metadata remains a permanent error.
			if (err == nil) != value.Within {
				t.Fatalf("%s %s of %d bytes: %v", field.Schema, field.Rule, len(value.Value), err)
			}
		}
	}
	if len(checked) != 2 {
		t.Fatal("shared sitemap text rules are incomplete")
	}
}

func TestReducerKeepsIdentityContextsAndSetSummaries(t *testing.T) {
	d := Empty()
	a := httpAuthority("first-run")
	d = applyObservations(t, d, a, observation("/Users/%2F/"), observation("/Users/%2F/?x=one"))
	if len(d.Groups) != 1 || d.Groups[0].Route != "/Users/%2F/" {
		t.Fatalf("path changed: %+v", d.Groups)
	}
	first, err := encode(d)
	if err != nil {
		t.Fatal(err)
	}
	d = applyObservations(t, d, a, observation("/Users/%2F/?x=two"))
	second, _ := encode(d)
	if !bytes.Equal(first, second) {
		t.Fatal("repeated set observation changed map")
	}
	a.RunID, a.SourceID = "second-run", "second-run"
	d = applyObservations(t, d, a, observation("/Users/%2F/"))
	if len(d.Groups[0].Members[0].Summaries) != 3 {
		t.Fatalf("summary diversity: %+v", d.Groups[0])
	}
	contexts := map[string]bool{}
	for _, s := range d.Groups[0].Members[0].Summaries {
		contexts[s.Context] = true
	}
	if len(contexts) != 2 {
		t.Fatal("ordinary Runs conflated")
	}
	a.Engagement = "engagement-one"
	a.IdentityID = "user-a"
	firstContext := a.contextKey()
	a.Engagement = "engagement-two"
	if firstContext == a.contextKey() {
		t.Fatal("engagement-local labels conflated")
	}
	a.IdentityID = ""
	firstContext = a.contextKey()
	a.RunID = "another-unknown-run"
	if firstContext == a.contextKey() {
		t.Fatal("unknown engagement actors conflated across Runs")
	}
}

func TestReducerRegroupsRetainedMembersAndKeepsSuppression(t *testing.T) {
	a := httpAuthority("traffic")
	d := applyObservations(t, Empty(), a, observation("/users/123"), observation("/users/me"))
	selected := memberID("https://example.test", "GET", "/users/123")
	owner := Authority{SourceID: "owner", Kind: "manual", Mode: "source-assisted"}
	var err error
	d, _, err = Reduce(d, Mutation{Operation: "remove", MemberIDs: []string{selected}, MarkerID: "hidden"}, owner)
	if err != nil {
		t.Fatal(err)
	}
	schema := Authority{SourceID: "schema", Kind: "openapi", Mode: "blackbox", Deployment: "prod"}
	update := SourceUpdate{SourceID: "schema", Rules: []Rule{
		{SourceID: "schema", Origin: "https://example.test", Method: "GET", Path: "/users/{id}", Parameters: []string{}},
		{SourceID: "schema", Origin: "https://example.test", Method: "GET", Path: "/users/me", Parameters: []string{}},
	}}
	d, _, err = Reduce(d, Mutation{Operation: "replace-source", Source: &update}, schema)
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Markers) != 1 || !reflect.DeepEqual(d.Markers[0].Members, []string{selected}) {
		t.Fatal("suppression selection changed")
	}
	for _, g := range d.Groups {
		for _, m := range g.Members {
			if m.Path == "/users/me" && g.Route != "/users/me" {
				t.Fatal("fixed declaration lost precedence")
			}
			if m.ID == selected && g.Route != "/users/{id}" {
				t.Fatal("member not regrouped")
			}
		}
	}
	before, _ := encode(d)
	if _, _, err := Reduce(d, Mutation{Operation: "replace-source", Source: &update}, schema); !errors.Is(err, ErrConflict) {
		t.Fatalf("stale source: %v", err)
	}
	after, _ := encode(d)
	if !bytes.Equal(before, after) {
		t.Fatal("failed mutation changed original")
	}
}

func TestReducerOverflowNeverInventsMembership(t *testing.T) {
	a := httpAuthority("traffic")
	d := Empty()
	for i := 0; i < MaxContexts+2; i++ {
		a.IdentityID = fmt.Sprint("principal-", i)
		d = applyObservations(t, d, a, observation("/users/me"))
	}
	if len(d.Groups[0].Members[0].Summaries) != MaxContexts || d.Groups[0].Members[0].Omitted != 2 {
		t.Fatal("context overflow was attributed to a shared principal")
	}
	// A template retains only the concrete inventory, independent of examples.
	rules := []Rule{{SourceID: "owner", Origin: "https://example.test", Method: "GET", Path: "/items/{id}", Parameters: []string{}}}
	owner := Authority{SourceID: "owner", Kind: "manual", Mode: "blackbox"}
	var err error
	d, _, err = Reduce(d, Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: "owner", Rules: rules}}, owner)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < MaxGroupMembers+3; i++ {
		d = applyObservations(t, d, a, observation(fmt.Sprintf("/items/%d", i)))
	}
	for _, g := range d.Groups {
		if g.Route == "/items/{id}" && (len(g.Members) != MaxGroupMembers || g.Omitted != 3) {
			t.Fatalf("member overflow: %+v", g)
		}
	}
	data, _ := encode(d)
	if strings.Contains(string(data), "/items/66") {
		t.Fatal("discarded concrete identity retained as example")
	}
}

func TestReducerStructuralSelectionAndSuppressionConflict(t *testing.T) {
	owner := Authority{SourceID: "owner", Kind: "manual", Mode: "blackbox"}
	rule := Rule{SourceID: "owner", Origin: "https://example.test", Method: "GET", Path: "/dispatch", Parameters: []string{}, Structural: []StructuralParameter{{Name: "action", Allowed: []string{"read", "short-password"}}}}
	d, _, err := Reduce(Empty(), Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: "owner", Rules: []Rule{rule}}}, owner)
	if err != nil {
		t.Fatal(err)
	}
	a := httpAuthority("traffic")
	d = applyObservations(t, d, a, observation("/dispatch?action=read"), observation("/dispatch?action=short-password"))
	if len(d.Groups) != 3 {
		t.Fatalf("structural functions conflated: %+v", d.Groups)
	}
	var member string
	for _, g := range d.Groups {
		if len(g.Structural) > 0 && g.Structural[0].Values[0] == "read" {
			member = g.Members[0].ID
		}
	}
	d, _, err = Reduce(d, Mutation{Operation: "remove", MarkerID: "hidden", MemberIDs: []string{member}}, owner)
	if err != nil {
		t.Fatal(err)
	}
	index := slices.IndexFunc(d.Sources, func(s Source) bool { return s.Provenance.SourceID == "owner" })
	expected := d.Sources[index]
	if _, _, err := Reduce(d, Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: "owner", Expected: &expected, Rules: []Rule{}}}, owner); !errors.Is(err, ErrConflict) {
		t.Fatalf("structural suppression widened or resurrected: %v", err)
	}
}

func TestReducerSequentialRunsUseStableSourceAndDistinctContexts(t *testing.T) {
	d := Empty()
	for i := range 200 {
		a := httpAuthority(fmt.Sprintf("run-%d", i))
		a.SourceID = "http-transport"
		d = applyObservations(t, d, a, observation("/shared"))
	}
	if len(d.Sources) != 1 || len(d.Groups[0].Members[0].Summaries) != MaxContexts || d.Groups[0].Members[0].Omitted != 200-MaxContexts {
		t.Fatalf("sequential Runs consumed a per-Run source registry or merged contexts: %+v", d)
	}
}

func TestReducerSourceCanonicalizationDoesNotMutateIntent(t *testing.T) {
	a := Authority{SourceID: "owner", Kind: "manual", Mode: "source-assisted"}
	m := Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: "owner", Rules: []Rule{{SourceID: "owner", Origin: "https://example.test", Method: "GET", Path: "/users/{id}", Parameters: []string{"beta", "alpha"}}}}}
	before := jsonBytes(m)
	d, _, err := Reduce(Empty(), m, a)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, jsonBytes(m)) {
		t.Fatal("canonicalization changed caller-owned intent")
	}
	data, err := encode(d)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Decode(data); err != nil {
		t.Fatalf("reducer published an unreadable strict document: %v", err)
	}
}
