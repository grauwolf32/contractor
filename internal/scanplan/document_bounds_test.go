package scanplan

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedOpenAPIDocumentBoundsMatchRuntimeAuthoring holds the source
// document bounds of scan preparation (LIM-SCN-39, LIM-SCN-41, LIM-SCN-42) to
// the verdicts the Runtime OpenAPI authoring Toolset gives on the same
// generated JSON and YAML documents.
func TestSharedOpenAPIDocumentBoundsMatchRuntimeAuthoring(t *testing.T) {
	fixture := contractstest.ReadOpenAPIDocumentBounds(t)
	if fixture.Bounds.SourceBytes != MaxSourceBytes || fixture.Bounds.Depth != MaxDepth || fixture.Bounds.Nodes != MaxNodes {
		t.Fatalf("shared bounds = %+v, want %d bytes, depth %d and %d nodes", fixture.Bounds, MaxSourceBytes, MaxDepth, MaxNodes)
	}
	if len(fixture.Cases) < 12 {
		t.Fatalf("shared document bound cases missing: %d", len(fixture.Cases))
	}
	for _, tc := range fixture.Cases {
		for _, format := range []string{"json", "yaml"} {
			t.Run(tc.Name+"/"+format, func(t *testing.T) {
				document, err := tc.Generate(fixture.Prefix[format], format == "json")
				if err != nil {
					t.Fatal(err)
				}
				if sum := sha256.Sum256(document); hex.EncodeToString(sum[:]) != tc.SHA256[format] {
					t.Fatalf("generated document digest differs from the fixture; the generators disagree")
				}
				parsed, err := parseDocument(document, "application/"+format)
				if tc.Violation == nil {
					if err != nil {
						t.Fatalf("parseDocument rejected a document within the bounds: %v", err)
					}
					if nodes, depth := documentExtent(parsed, 1); nodes != tc.Nodes || depth != tc.Depth {
						t.Fatalf("document has %d nodes at depth %d, fixture says %d at %d", nodes, depth, tc.Nodes, tc.Depth)
					}
					return
				}
				var failure *PreparationError
				if !errors.As(err, &failure) || failure.Code != "source_limit_exceeded" {
					t.Fatalf("parseDocument = %v, want source_limit_exceeded for the %s bound", err, *tc.Violation)
				}
			})
		}
	}
}

// documentExtent counts the nodes of a parsed value, member names included,
// and the depth of its deepest node, with value at depth.
func documentExtent(value any, depth int) (nodes, deepest int) {
	nodes, deepest = 1, depth
	visit := func(child any) {
		childNodes, childDepth := documentExtent(child, depth+1)
		nodes += childNodes
		deepest = max(deepest, childDepth)
	}
	switch typed := value.(type) {
	case map[string]any:
		for _, member := range typed {
			nodes++
			visit(member)
		}
	case []any:
		for _, element := range typed {
			visit(element)
		}
	}
	return nodes, deepest
}
