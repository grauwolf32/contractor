package projectsitemap

import (
	"bytes"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func readerPolicy(mode string) contracts.ProjectSitemapPolicy {
	return contracts.ProjectSitemapPolicy{Version: contracts.SitemapAccessVersion, SourceMode: mode}
}

func TestSafeViewRemovesRestrictedPathsNotesAndGroupingBeforeEncoding(t *testing.T) {
	bb := httpAuthority("bb-run")
	bb.SourceID = "bb-http"
	d := applyObservations(t, Empty(), bb, observation("/users/123"))
	sa := httpAuthority("sa-run")
	sa.SourceID, sa.Mode = "sa-http", "source-assisted"
	d = applyObservations(t, d, sa, observation("/source-canary"))
	owner := Authority{SourceID: "sa-note", Kind: "manual", Mode: "source-assisted"}
	var err error
	d, _, err = Reduce(d, Mutation{Operation: "annotate", MemberIDs: []string{memberID("https://example.test", "GET", "/users/123")}, Note: "note-canary"}, owner)
	if err != nil {
		t.Fatal(err)
	}
	schema := Authority{SourceID: "sa-schema", Kind: "openapi", Mode: "source-assisted"}
	d, _, err = Reduce(d, Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: schema.SourceID, Rules: []Rule{{
		SourceID: schema.SourceID, Origin: "https://example.test", Method: "GET", Path: "/users/{route_canary}", Pointer: "pointer-canary", Parameters: []string{"parameter_canary"},
	}}}}, schema)
	if err != nil {
		t.Fatal(err)
	}
	before, _ := encode(d)
	view, err := SafeProjection(d, readerPolicy("blackbox"), "reader", nil)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := encode(view)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"source-canary", "note-canary", "route_canary", "pointer-canary", "parameter_canary", "sa-http", "sa-schema", "sa-note"} {
		if bytes.Contains(encoded, []byte(forbidden)) {
			t.Fatalf("restricted value in view: %s", forbidden)
		}
	}
	if len(view.Groups) != 1 || view.Groups[0].Route != "/users/123" || len(view.Groups[0].Members[0].Annotations) != 0 {
		t.Fatalf("unsafe reconstruction: %+v", view.Groups)
	}
	after, _ := encode(d)
	if !bytes.Equal(before, after) {
		t.Fatal("projection modified shared map")
	}
	full, err := SafeProjection(d, readerPolicy("source-assisted"), "reader", nil)
	if err != nil {
		t.Fatal(err)
	}
	fullBytes, _ := encode(full)
	if !bytes.Equal(before, fullBytes) {
		t.Fatal("explicit source-assisted view lost authorized content")
	}
}

func TestSafeViewPermitsOnlyExplicitExactOpenAPIAndRebindsItsAlias(t *testing.T) {
	bb := httpAuthority("bb-run")
	d := applyObservations(t, Empty(), bb, observation("/users/123"))
	sourceRevision := "schema-source-revision"
	source := contracts.ExactArtifact{
		Ref:    contracts.ArtifactRef{Namespace: "sources", Name: "private-alias", Revision: &sourceRevision},
		Digest: "sha256:" + strings.Repeat("a", 64), MediaType: "application/json", SizeBytes: 100,
	}
	schema := Authority{SourceID: "openapi", Kind: "openapi", Mode: "source-assisted"}
	var err error
	d, _, err = Reduce(d, Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: schema.SourceID, Artifact: &source, Rules: []Rule{{SourceID: schema.SourceID, Origin: "https://example.test", Method: "GET", Path: "/users/{id}", Parameters: []string{"declared"}}}}}, schema)
	if err != nil {
		t.Fatal(err)
	}
	input := source
	revision := "consumer-input-revision"
	input.Ref = contracts.ArtifactRef{Namespace: "inputs", Name: "schema", Revision: &revision}
	view, err := SafeProjection(d, readerPolicy("blackbox"), "reader", []contracts.ExactArtifact{input})
	if err != nil {
		t.Fatal(err)
	}
	if len(view.Rules) != 1 || len(view.Sources) != 2 {
		t.Fatal("selected schema was excluded")
	}
	encoded, _ := encode(view)
	if bytes.Contains(encoded, []byte("private-alias")) || bytes.Contains(encoded, []byte(sourceRevision)) {
		t.Fatal("view exposed ungranted source alias")
	}
	if !bytes.Contains(encoded, []byte(revision)) {
		t.Fatal("view omitted authorized consumer ref")
	}
	input.Digest = "sha256:" + strings.Repeat("b", 64)
	view, err = SafeProjection(d, readerPolicy("blackbox"), "reader", []contracts.ExactArtifact{input})
	if err != nil || len(view.Rules) != 0 {
		t.Fatal("different exact input authorized an unselected schema")
	}
}

func TestSafeViewOmitsUnreconstructableStructuralMembers(t *testing.T) {
	schema := Authority{SourceID: "schema", Kind: "openapi", Mode: "source-assisted"}
	d, _, err := Reduce(Empty(), Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: schema.SourceID, Rules: []Rule{{SourceID: schema.SourceID, Origin: "https://example.test", Method: "GET", Path: "/dispatch", Structural: []StructuralParameter{{Name: "action", Allowed: []string{"private"}}}}}}}, schema)
	if err != nil {
		t.Fatal(err)
	}
	d = applyObservations(t, d, httpAuthority("traffic"), observation("/dispatch?action=private"))
	view, err := SafeProjection(d, readerPolicy("blackbox"), "reader", nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(view.Groups) != 0 || len(view.Gaps) == 0 {
		t.Fatal("restricted structural policy became a fabricated concrete member")
	}
}

func TestReducerWorkerManualSourcesDoNotMultiplyAcrossRuns(t *testing.T) {
	a := Authority{SourceID: "manual-editor", Kind: "manual", Mode: "blackbox", RunID: "first-run"}
	d, _, err := Reduce(Empty(), Mutation{Operation: "declare", URL: "https://example.test/one", Method: "GET"}, a)
	if err != nil {
		t.Fatal(err)
	}
	a.RunID = "second-run"
	d, _, err = Reduce(d, Mutation{Operation: "declare", URL: "https://example.test/two", Method: "GET"}, a)
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Sources) != 1 || d.Sources[0].Provenance.RunID != "" {
		t.Fatal("manual Worker Run consumed another logical source")
	}
	seen := map[string]bool{}
	for _, group := range d.Groups {
		for _, declaration := range group.Declarations {
			seen[declaration.Provenance.RunID] = true
		}
	}
	if !seen["first-run"] || !seen["second-run"] {
		t.Fatal("declaration lost real producer identity")
	}
}

func TestSafeViewRetainsConcreteManualMembersAfterRestrictedRegrouping(t *testing.T) {
	manual := Authority{SourceID: "manual-bb", Kind: "manual", Mode: "blackbox", RunID: "declaring-run"}
	document, _, err := Reduce(Empty(), Mutation{Operation: "declare", URL: "https://example.test/users/123", Method: "GET"}, manual)
	if err != nil {
		t.Fatal(err)
	}
	schema := Authority{SourceID: "schema-sa", Kind: "openapi", Mode: "source-assisted"}
	document, _, err = Reduce(document, Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: schema.SourceID, Rules: []Rule{{SourceID: schema.SourceID, Origin: "https://example.test", Method: "GET", Path: "/users/{restricted_parameter}"}}}}, schema)
	if err != nil {
		t.Fatal(err)
	}
	view, err := SafeProjection(document, readerPolicy("blackbox"), "consumer", nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(view.Groups) != 1 || view.Groups[0].Route != "/users/123" || len(view.Groups[0].Members) != 1 || view.Groups[0].Members[0].ID != memberID("https://example.test", "GET", "/users/123") || len(view.Groups[0].Declarations) != 1 {
		t.Fatalf("manual concrete selection lost after regrouping: %+v", view.Groups)
	}
	data, _ := encode(view)
	if bytes.Contains(data, []byte("restricted_parameter")) {
		t.Fatal("restricted template affected the authorized view")
	}
}
