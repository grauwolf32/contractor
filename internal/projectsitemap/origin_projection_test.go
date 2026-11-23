package projectsitemap

import "testing"

func TestSitemapOriginProjectionUsesURLLabelsAndConcreteExamples(t *testing.T) {
	for _, origin := range []string{"https://api.example.test", "http://127.0.0.1:8080", "https://one.two.three.test", "https://abcdefghijklmnopqrstuvwxyz123456.example.test", "https://password.example.test"} {
		t.Run(origin, func(t *testing.T) {
			a := httpAuthority("origin-producer")
			o := observation("/path")
			o.URL = origin + "/path"
			document, _, err := Reduce(Empty(), Mutation{Operation: "observe", Observations: []Observation{o}}, a)
			if err != nil || len(document.Groups) != 1 || document.Groups[0].Origin != origin {
				t.Fatalf("valid HTTP origin omitted: %+v %v", document, err)
			}
			encoded, err := encode(document)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := Decode(encoded); err != nil {
				t.Fatal(err)
			}
			a.Kind, a.CanObserve = "manual", false
			document, _, err = Reduce(Empty(), Mutation{Operation: "declare", ExpectedRevision: "absent", URL: o.URL, Method: "GET"}, a)
			if err != nil || len(document.Groups) != 1 {
				t.Fatalf("manual sibling rejected URL origin: %v", err)
			}
			rule := Rule{SourceID: a.SourceID, Origin: origin, Method: "GET", Path: "/items/{id}", Pointer: "manual", Parameters: []string{}, Structural: []StructuralParameter{}}
			if _, _, err := Reduce(Empty(), Mutation{Operation: "replace-source", Source: &SourceUpdate{SourceID: a.SourceID, Rules: []Rule{rule}}}, a); err != nil {
				t.Fatalf("grouping sibling rejected URL origin: %v", err)
			}
		})
	}
}
