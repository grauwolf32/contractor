package projectsitemap

import (
	"bytes"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func TestSitemapSnapshotKeepsSelectedSourceCaveats(t *testing.T) {
	selected := httpAuthority("selected-run")
	selected.SourceID = "selected-source"
	other := httpAuthority("other-run")
	other.SourceID = "other-source"
	document := applyObservations(t, Empty(), selected, observation("/selected"))
	document = applyObservations(t, document, other, observation("/other"))
	want := []Gap{{SourceID: selected.SourceID, Reason: "members", Count: 20}, {Reason: "groups", Count: 3}}
	document.Gaps = append(slices.Clone(want), Gap{SourceID: other.SourceID, Reason: "incomplete", Count: 2})
	view, _, err := selectSnapshot(document, SnapshotSelection{Members: []string{memberID("https://example.test", "GET", "/selected")}}, "reader", nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(view.Gaps) != len(want) {
		t.Fatalf("snapshot caveats: %+v", view.Gaps)
	}
	for _, gap := range want {
		if !slices.Contains(view.Gaps, gap) {
			t.Fatalf("snapshot discarded a selected source or whole-map caveat: %+v", gap)
		}
	}
}

func TestReducerWorkerManualSourcesDoNotMultiplyAcrossRuns(t *testing.T) {
	a := Authority{SourceID: "manual-editor", Kind: "manual", RunID: "first-run"}
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

func TestSitemapTypedBoundariesRejectRequiredNullCollections(t *testing.T) {
	owner := Authority{SourceID: "owner", Kind: "manual"}
	document, _, err := Reduce(Empty(), Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: "owner", Rules: []Rule{{
		SourceID: "owner", Origin: "https://example.test", Method: "GET", Path: "/dispatch",
		Structural: []StructuralParameter{{Name: "action", Allowed: []string{"read"}}},
	}}}}, owner)
	if err != nil {
		t.Fatal(err)
	}
	document = applyObservations(t, document, httpAuthority("traffic"), observation("/dispatch?action=read"))
	member := memberID("https://example.test", "GET", "/dispatch", StructuralValue{Name: "action", Values: []string{"read"}})
	document, _, err = Reduce(document, Mutation{Operation: "annotate", MemberIDs: []string{member}, Note: "Reviewed"}, owner)
	if err != nil {
		t.Fatal(err)
	}
	command := Mutation{Operation: "annotate", MemberIDs: []string{member}, Note: "Another note"}
	var paths [][]int
	var collect func(reflect.Value, []int)
	collect = func(value reflect.Value, path []int) {
		switch value.Kind() {
		case reflect.Struct:
			for i := range value.NumField() {
				field := value.Type().Field(i)
				next := append(append([]int{}, path...), i)
				if value.Field(i).Kind() == reflect.Slice && strings.Contains(field.Tag.Get("strictjson"), "required") {
					paths = append(paths, next)
				}
				collect(value.Field(i), next)
			}
		case reflect.Slice:
			for i := range value.Len() {
				collect(value.Index(i), append(append([]int{}, path...), i))
			}
		}
	}
	collect(reflect.ValueOf(document), nil)
	if len(paths) < 18 {
		t.Fatalf("incomplete nested collection fixture: %d", len(paths))
	}
	for _, path := range paths {
		malformed := cloneDocument(document)
		field := reflect.ValueOf(&malformed).Elem()
		for _, i := range path {
			if field.Kind() == reflect.Struct {
				field = field.Field(i)
			} else {
				field = field.Index(i)
			}
		}
		field.SetZero()
		if _, err := Decode(jsonBytes(malformed)); err == nil {
			t.Fatalf("stored null accepted at %v", path)
		}
		if _, _, err := Reduce(malformed, command, owner); err == nil {
			t.Fatalf("typed reduction accepted null at %v", path)
		}
	}
}

func TestSitemapReductionPublishesCanonicalBytesWithoutChangingInput(t *testing.T) {
	current := Empty()
	authority := httpAuthority("traffic")
	before := jsonBytes(current)
	next, data, err := Reduce(current, Mutation{Operation: "observe", Observations: []Observation{
		{URL: "https://example.test/z?token=value", Method: "GET", Outcome: "response", Status: 200, At: "2026-10-26T00:00:00Z"},
		{URL: "https://example.test/a?password=value", Method: "POST", Outcome: "response", Status: 201, At: "2026-10-26T00:00:00Z"},
	}}, authority)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, jsonBytes(current)) {
		t.Fatal("successful reduction changed the caller's document")
	}
	decoded, err := Decode(data)
	if err != nil || !reflect.DeepEqual(next, decoded) {
		t.Fatalf("publication bytes differ from reduced document: %v", err)
	}
	want, err := encode(cloneDocument(next))
	if err != nil || !bytes.Equal(want, data) {
		t.Fatalf("publication bytes are not canonical: %v", err)
	}
	before = jsonBytes(next)
	owner := Authority{SourceID: "owner", Kind: "manual"}
	changed, _, err := Reduce(next, Mutation{Operation: "annotate", MemberIDs: []string{next.Groups[0].Members[0].ID}, Note: "Reviewed", Tags: []string{"token"}}, owner)
	if err != nil {
		t.Fatal(err)
	}
	changed.Groups[0].Members[0].Summaries[0].Parameters[0] = "changed"
	if !bytes.Equal(before, jsonBytes(next)) {
		t.Fatal("mutable reduction output aliases the caller's document")
	}
}

// Populate every field, including future mutable fields, to keep the private
// copy contract independent of which optional metadata a fixture happens to use.
func populatedDocumentValue(kind reflect.Type) reflect.Value {
	value := reflect.New(kind).Elem()
	switch kind.Kind() {
	case reflect.Struct:
		for i := range kind.NumField() {
			value.Field(i).Set(populatedDocumentValue(kind.Field(i).Type))
		}
	case reflect.Pointer:
		value.Set(reflect.New(kind.Elem()))
		value.Elem().Set(populatedDocumentValue(kind.Elem()))
	case reflect.Slice:
		value.Set(reflect.MakeSlice(kind, 1, 1))
		value.Index(0).Set(populatedDocumentValue(kind.Elem()))
	case reflect.String:
		value.SetString("metadata")
	case reflect.Bool:
		value.SetBool(true)
	case reflect.Int, reflect.Int64:
		value.SetInt(1)
	case reflect.Uint64:
		value.SetUint(1)
	default:
		panic("unsupported sitemap field: " + kind.String())
	}
	return value
}

func TestSitemapDocumentCopyOwnsEveryMutableField(t *testing.T) {
	original := populatedDocumentValue(reflect.TypeFor[Document]()).Interface().(Document)
	copy := cloneDocument(original)
	if !reflect.DeepEqual(copy, original) {
		t.Fatal("copy changed metadata")
	}
	var inspect func(reflect.Value, reflect.Value, string)
	inspect = func(left, right reflect.Value, path string) {
		switch left.Kind() {
		case reflect.Struct:
			for i := range left.NumField() {
				inspect(left.Field(i), right.Field(i), path+"."+left.Type().Field(i).Name)
			}
		case reflect.Pointer:
			if left.Pointer() == right.Pointer() {
				t.Fatalf("aliased pointer: %s", path)
			}
			inspect(left.Elem(), right.Elem(), path)
		case reflect.Slice:
			if left.Pointer() == right.Pointer() {
				t.Fatalf("aliased collection: %s", path)
			}
			for i := range left.Len() {
				inspect(left.Index(i), right.Index(i), path)
			}
		}
	}
	inspect(reflect.ValueOf(original), reflect.ValueOf(copy), "document")
}

func BenchmarkSitemapDocumentMutation(b *testing.B) {
	for _, count := range []int{10000, 50000} {
		b.Run(fmt.Sprint(count), func(b *testing.B) {
			a := httpAuthority("traffic")
			document := Empty()
			document.Sources = append(document.Sources, Source{Provenance: a.provenance(), Generation: 1, PolicyDigest: key(Projection)})
			for i := range 10000 {
				route := fmt.Sprintf("/items/%d/{id}", i)
				group := Group{ID: groupID("https://example.test", "GET", route), Origin: "https://example.test", Method: "GET", Route: route, Structural: []StructuralValue{}, Members: []Member{}, Declarations: []Declaration{}}
				for j := range count / 10000 {
					path := fmt.Sprintf("/items/%d/%d", i, j)
					group.Members = append(group.Members, Member{ID: memberID(group.Origin, group.Method, path), Path: path, Summaries: []Summary{{Provenance: a.provenance(), Context: a.contextKey(), Outcome: "response", Status: 200, Parameters: []string{}, First: "2026-10-26T00:00:00Z", Last: "2026-10-26T00:00:00Z"}}, Examples: []string{}, Evidence: []Evidence{}, Annotations: []Annotation{}})
				}
				document.Groups = append(document.Groups, group)
			}
			data, err := encode(document)
			if err != nil {
				b.Fatal(err)
			}
			command := Mutation{Operation: "annotate", MemberIDs: []string{document.Groups[0].Members[0].ID}, Note: "Reviewed"}
			owner := Authority{SourceID: "owner", Kind: "manual"}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				current, err := Decode(data)
				if err != nil {
					b.Fatal(err)
				}
				_, _, err = Reduce(current, command, owner)
				if err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(len(data)), "document-bytes")
		})
	}
}
