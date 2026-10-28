package projectsitemap

import (
	"encoding/json"
	"fmt"
	"testing"
)

// Benchmark the document work at the allocation read boundary, excluding SQL
// and transport. The fixture reaches both group and total-member limits.
func BenchmarkSitemapAllocationDocumentRead(b *testing.B) {
	a := httpAuthority("traffic")
	document := Empty()
	document.Sources = append(document.Sources, Source{Provenance: a.provenance(), Generation: 1, PolicyDigest: key(Projection)})
	for i := range MaxGroups {
		route := fmt.Sprintf("/items/%d/{id}", i)
		group := Group{ID: groupID("https://example.test", "GET", route), Origin: "https://example.test", Method: "GET", Route: route, Structural: []StructuralValue{}, Members: []Member{}, Declarations: []Declaration{}}
		for j := range MaxMembers / MaxGroups {
			path := fmt.Sprintf("/items/%d/%d", i, j)
			group.Members = append(group.Members, Member{ID: memberID(group.Origin, group.Method, path), Path: path, Summaries: []Summary{{Provenance: a.provenance(), Context: a.contextKey(), Outcome: "response", Status: 200, Parameters: []string{}, First: "2026-10-26T00:00:00Z", Last: "2026-10-26T00:00:00Z"}}, Examples: []string{}, Evidence: []Evidence{}, Annotations: []Annotation{}})
		}
		document.Groups = append(document.Groups, group)
	}
	data, err := encode(document)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		current, err := Decode(data)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := json.Marshal(current); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(len(data)), "document-bytes")
}
