package projectsitemap

import (
	"fmt"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
)

func TestSitemapDeclarationObservationSharePlacement(t *testing.T) {
	for _, structural := range []bool{false, true} {
		for _, declarationFirst := range []bool{false, true} {
			t.Run(fmt.Sprintf("structural=%t/declaration-first=%t", structural, declarationFirst), func(t *testing.T) {
				document := Empty()
				exerciseSitemapMemberPlacement(t, structural, declarationFirst, func(authority Authority, mutation Mutation) Document {
					var err error
					document, _, err = Reduce(document, mutation, authority)
					if err != nil {
						t.Fatalf("%s: %v", mutation.Operation, err)
					}
					return document
				})
			})
		}
	}
}

func TestPostgresSitemapDeclarationObservationSharePlacement(t *testing.T) {
	for _, structural := range []bool{false, true} {
		for _, declarationFirst := range []bool{false, true} {
			t.Run(fmt.Sprintf("structural=%t/declaration-first=%t", structural, declarationFirst), func(t *testing.T) {
				pool := sitemapPool(t)
				ctx := t.Context()
				createSitemapRun(t, ctx, pool, "grouping-run")
				service := NewService(pool, artifactpgtest.NewStorage())
				revision, sequence := "absent", 0
				exerciseSitemapMemberPlacement(t, structural, declarationFirst, func(authority Authority, mutation Mutation) Document {
					authority.OwnerID, authority.ProjectID = "owner", "project"
					mutation.ExpectedRevision = revision
					sequence++
					result, err := service.Mutate(ctx, authority, fmt.Sprintf("grouping-%d", sequence), mutation)
					if err != nil {
						t.Fatalf("%s: %v", mutation.Operation, err)
					}
					revision = *result.Ref.Revision
					view, err := readSitemap(service, ctx, "project", revision)
					if err != nil {
						t.Fatalf("read committed %s: %v", mutation.Operation, err)
					}
					return view.Document
				})
			})
		}
	}
}

// Both the pure reducer and production PostgreSQL path must keep exactly one
// member when declarations arrive before or after observed traffic. The first
// contribution is annotated and suppressed before the second arrives.
func exerciseSitemapMemberPlacement(t *testing.T, structural, declarationFirst bool, apply func(Authority, Mutation) Document) {
	t.Helper()
	owner := Authority{SourceID: "manual", Kind: "manual"}
	rules := Authority{SourceID: "rules", Kind: "manual"}
	traffic := httpAuthority("grouping-run")
	rule := Rule{SourceID: rules.SourceID, Origin: "https://example.test", Method: "GET", Path: "/users/{id}", Parameters: []string{}, Structural: []StructuralParameter{}}
	url := "https://example.test/users/me"
	selection := []StructuralValue{}
	if structural {
		rule.Structural = []StructuralParameter{{Name: "action", Allowed: []string{"read", "write"}}}
		selection = []StructuralValue{{Name: "action", Values: []string{"read"}}}
		url += "?action=read"
	}
	apply(rules, Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: rules.SourceID, Rules: []Rule{rule}}})
	declaration := Mutation{Operation: "declare", URL: url, Method: "GET"}
	observation := observation("/users/me")
	observation.URL = url
	observed := Mutation{Operation: "observe", Observations: []Observation{observation}}
	first, second := declaration, observed
	firstAuthority, secondAuthority := owner, traffic
	if !declarationFirst {
		first, second = second, first
		firstAuthority, secondAuthority = secondAuthority, firstAuthority
	}
	document := apply(firstAuthority, first)
	id := memberID(rule.Origin, rule.Method, "/users/me", selection...)
	requireSingleSitemapMember(t, document, id, rule.Path)
	apply(owner, Mutation{Operation: "annotate", MemberIDs: []string{id}, Note: "keep concrete selection", Tags: []string{"retained"}})
	apply(owner, Mutation{Operation: "remove", MarkerID: "suppressed", MemberIDs: []string{id}})
	document = apply(secondAuthority, second)
	requireSingleSitemapMember(t, document, id, rule.Path)
	member := findSitemapMember(t, document, id)
	if len(member.Summaries) != 1 || len(member.Annotations) != 1 || member.Annotations[0].Note != "keep concrete selection" || len(document.Markers) != 1 || !slices.Equal(document.Markers[0].Members, []string{id}) {
		t.Fatal("second contribution lost observation, annotation or suppression")
	}
	expected := document.Sources[slices.IndexFunc(document.Sources, func(source Source) bool { return source.Provenance.SourceID == rules.SourceID })]
	rule.Path = "/users/{user}"
	document = apply(rules, Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: rules.SourceID, Expected: &expected, Rules: []Rule{rule}}})
	requireSingleSitemapMember(t, document, id, rule.Path)
	member = findSitemapMember(t, document, id)
	if len(member.Annotations) != 1 || len(document.Markers) != 1 || document.Markers[0].Members[0] != id {
		t.Fatal("regrouping lost metadata or suppression")
	}
	declarationFound := false
	for _, group := range document.Groups {
		for _, declaration := range group.Declarations {
			if !declaration.Rule && declaration.Provenance.SourceID == owner.SourceID {
				declarationFound = group.Route == "/users/me" && slices.EqualFunc(group.Structural, selection, func(a, b StructuralValue) bool { return a.Name == b.Name && slices.Equal(a.Values, b.Values) })
			}
		}
	}
	if !declarationFound {
		t.Fatal("manual declaration lost its concrete path or structural selection")
	}
}

func requireSingleSitemapMember(t *testing.T, document Document, id, route string) {
	t.Helper()
	count := 0
	for _, group := range document.Groups {
		for _, member := range group.Members {
			count++
			if member.ID != id || group.Route != route {
				t.Fatalf("unexpected member placement: %s in %s", member.ID, group.Route)
			}
		}
	}
	if count != 1 {
		t.Fatalf("expected one globally unique concrete member, got %d", count)
	}
}

func findSitemapMember(t *testing.T, document Document, id string) Member {
	t.Helper()
	for _, group := range document.Groups {
		for _, member := range group.Members {
			if member.ID == id {
				return member
			}
		}
	}
	t.Fatalf("member %s missing", id)
	return Member{}
}

func TestSitemapRegroupReportsAmbiguousTemplates(t *testing.T) {
	for _, rulesFirst := range []bool{false, true} {
		t.Run(fmt.Sprintf("rules-first=%t", rulesFirst), func(t *testing.T) {
			document := Empty()
			authority := Authority{SourceID: "rules", Kind: "manual"}
			rules := Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: authority.SourceID, Rules: []Rule{
				{SourceID: authority.SourceID, Origin: "https://example.test", Method: "GET", Path: "/users/{id}", Parameters: []string{}},
				{SourceID: authority.SourceID, Origin: "https://example.test", Method: "GET", Path: "/{kind}/me", Parameters: []string{}},
			}}}
			applyRules := func() {
				var err error
				document, _, err = Reduce(document, rules, authority)
				if err != nil {
					t.Fatal(err)
				}
			}
			if rulesFirst {
				applyRules()
			}
			document = applyObservations(t, document, httpAuthority("traffic"), observation("/users/me"))
			if !rulesFirst {
				applyRules()
			}
			requireSingleSitemapMember(t, document, memberID("https://example.test", "GET", "/users/me"), "/users/me")
			if !slices.ContainsFunc(document.Gaps, func(gap Gap) bool { return gap.Reason == "ambiguous" && gap.Count > 0 }) {
				t.Fatal("ambiguous retained member was silently regrouped")
			}
		})
	}
}

func TestSitemapManualDeclarationsPreserveConcretePathsAlongsideTemplates(t *testing.T) {
	rules := Authority{SourceID: "rules", Kind: "manual"}
	owner := Authority{SourceID: "manual", Kind: "manual"}
	document, _, err := Reduce(Empty(), Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: rules.SourceID, Rules: []Rule{{SourceID: rules.SourceID, Origin: "https://example.test", Method: "GET", Path: "/users/{id}", Parameters: []string{}}}}}, rules)
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"/users/alice", "/users/bob"} {
		document, _, err = Reduce(document, Mutation{Operation: "declare", URL: "https://example.test" + path, Method: "GET"}, owner)
		if err != nil {
			t.Fatal(err)
		}
	}
	alice := memberID("https://example.test", "GET", "/users/alice")
	document, _, err = Reduce(document, Mutation{Operation: "remove", MarkerID: "only-alice", MemberIDs: []string{alice}}, owner)
	if err != nil {
		t.Fatal(err)
	}
	if len(document.Groups) != 3 || len(document.Markers) != 1 || !slices.Equal(document.Markers[0].Members, []string{alice}) {
		t.Fatalf("concrete declarations or suppression changed: %+v", document)
	}
	for _, group := range document.Groups {
		if group.Route == "/users/{id}" {
			if len(group.Members) != 2 {
				t.Fatalf("template lost concrete members: %+v", group)
			}
			continue
		}
		if len(group.Members) != 0 || len(group.Declarations) != 1 || group.Declarations[0].Rule || group.Declarations[0].Provenance.SourceID != owner.SourceID {
			t.Fatalf("manual concrete declaration did not survive grouping: %+v", group)
		}
	}
}
