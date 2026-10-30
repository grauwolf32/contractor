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
	return Authority{OwnerID: "owner", ProjectID: "project", RunID: run, SourceID: run, Kind: "http", CanObserve: true}
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
		Observed   json.RawMessage `json:"observed"`
		Reported   json.RawMessage `json:"reported"`
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
	if observed, err := Decode(fixture.Observed); err != nil || len(observed.Groups) != 1 || len(observed.Groups[0].Members[0].Summaries) != 1 {
		t.Fatalf("observed: %v", err)
	}
	if reported, err := Decode(fixture.Reported); err != nil || len(reported.Gaps) != 3 {
		t.Fatalf("reported gaps: %v", err)
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

// TestSharedSitemapTextBoundsCountUTF8Bytes checks every shared sitemap byte
// bound with each shared character at and one byte over it. A URL's run sits
// in its fragment, which projection drops before the canonical URL is
// checked, so only the raw 8 KiB bound (LIM-SMT-2) applies to it.
func TestSharedSitemapTextBoundsCountUTF8Bytes(t *testing.T) {
	characters, fields := contractstest.ReadUTF8ByteBoundCases(t)
	bounds := map[string]int{"LIM-SMT-2": MaxMemberKeyBytes, "LIM-SMT-6": MaxNoteBytes, "LIM-SMT-11": MaxTagBytes, "LIM-SMT-16": MaxPointerBytes}
	checked := map[string]bool{}
	for _, field := range fields {
		bound, owned := bounds[field.Rule]
		if !owned {
			continue
		}
		if bound != field.MaxBytes {
			t.Fatalf("shared bound drift: %+v", field)
		}
		checked[field.Rule] = true
		values := field.Values(characters)
		if field.Rule == "LIM-SMT-2" {
			values = field.EmbeddedValues(characters, "https://example.test/#", "")
		}
		for _, value := range values {
			var err error
			switch field.Rule {
			case "LIM-SMT-2":
				_, _, _, err = projectURL(value.Value)
			case "LIM-SMT-6":
				_, err = projectAnnotation(value.Value, nil, Authority{})
			case "LIM-SMT-11":
				_, err = projectAnnotation("", []string{value.Value}, Authority{})
			case "LIM-SMT-16":
				err = validateRule(Rule{
					SourceID: "owner", Origin: "https://example.test", Method: "GET", Path: "/items",
					Pointer: value.Value, Parameters: []string{}, Structural: []StructuralParameter{},
				})
			}
			// Metadata is retained within its UTF-8 byte limit; over-bound
			// metadata remains a permanent error.
			if (err == nil) != value.Within {
				t.Fatalf("%s %s of %d bytes of %q: %v", field.Schema, field.Rule, len(value.Value), value.Character, err)
			}
		}
	}
	if len(checked) != len(bounds) {
		t.Fatalf("shared sitemap byte bounds checked: %v", checked)
	}
}

func TestReducerKeepsSourceAndBehaviorSummaries(t *testing.T) {
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
	a.RunID = "second-run"
	d = applyObservations(t, d, a, observation("/Users/%2F/"))
	if len(d.Groups[0].Members[0].Summaries) != 2 {
		t.Fatalf("summary diversity: %+v", d.Groups[0])
	}
	a.SourceID = "second-source"
	d = applyObservations(t, d, a, observation("/Users/%2F/"))
	if len(d.Groups[0].Members[0].Summaries) != 3 {
		t.Fatal("different sources were conflated")
	}
}

func TestReducerRegroupsRetainedMembersAndKeepsSuppression(t *testing.T) {
	a := httpAuthority("traffic")
	d := applyObservations(t, Empty(), a, observation("/users/123"), observation("/users/me"))
	selected := memberID("https://example.test", "GET", "/users/123")
	owner := Authority{SourceID: "owner", Kind: "manual"}
	var err error
	d, _, err = Reduce(d, Mutation{Operation: "remove", MemberIDs: []string{selected}, MarkerID: "hidden"}, owner)
	if err != nil {
		t.Fatal(err)
	}
	schema := Authority{SourceID: "schema", Kind: "openapi", Deployment: "prod"}
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
	for i := 0; i < MaxSummaries+2; i++ {
		o := observation("/users/me")
		o.Status += i
		d = applyObservations(t, d, a, o)
	}
	if len(d.Groups[0].Members[0].Summaries) != MaxSummaries || d.Groups[0].Members[0].Omitted != 2 {
		t.Fatal("summary overflow did not preserve accepted behavior")
	}
	o := observation("/users/me")
	o.At = "2026-10-27T00:00:00Z"
	a.RunID = "later-run"
	d = applyObservations(t, d, a, o)
	if d.Groups[0].Members[0].Omitted != 2 || !slices.ContainsFunc(d.Groups[0].Members[0].Summaries, func(s Summary) bool { return s.Status == 200 && s.Last == o.At }) {
		t.Fatal("known behavior stopped updating at capacity")
	}
	// A template retains only the concrete inventory, independent of examples.
	rules := []Rule{{SourceID: "owner", Origin: "https://example.test", Method: "GET", Path: "/items/{id}", Parameters: []string{}}}
	owner := Authority{SourceID: "owner", Kind: "manual"}
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
	owner := Authority{SourceID: "owner", Kind: "manual"}
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

func TestReducerSequentialRunsShareBehaviorSummaries(t *testing.T) {
	d := Empty()
	for i := range 200 {
		a := httpAuthority(fmt.Sprintf("run-%d", i))
		a.SourceID = "http-transport"
		d = applyObservations(t, d, a, observation("/shared"))
	}
	if len(d.Sources) != 1 || len(d.Groups[0].Members[0].Summaries) != 1 || d.Groups[0].Members[0].Omitted != 0 || len(d.Gaps) != 0 {
		t.Fatalf("sequential Runs consumed map capacity: %+v", d)
	}
}

func TestReducerSourceCanonicalizationDoesNotMutateIntent(t *testing.T) {
	a := Authority{SourceID: "owner", Kind: "manual"}
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
