package scanplan

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// documentBoundsCase is one generated document of
// api/testdata/openapi-document-bounds.json; the fixture's description states
// how the document is generated.
type documentBoundsCase struct {
	Name string `json:"name"`
	Deep *struct {
		Kind  string `json:"kind"`
		Depth int    `json:"depth"`
	} `json:"deep"`
	Elements int `json:"elements"`
	Members  int `json:"members"`
	Pad      *struct {
		Bytes int    `json:"bytes"`
		Fill  string `json:"fill"`
	} `json:"pad"`
	Nodes     int               `json:"nodes"`
	Depth     int               `json:"depth"`
	Violation *string           `json:"violation"`
	SHA256    map[string]string `json:"sha256"`
}

// TestSharedOpenAPIDocumentBoundsMatchRuntimeAuthoring holds the source
// document bounds of scan preparation (LIM-SCN-39, LIM-SCN-41, LIM-SCN-42) to
// the verdicts the Runtime OpenAPI authoring Toolset gives on the same
// generated JSON and YAML documents.
func TestSharedOpenAPIDocumentBoundsMatchRuntimeAuthoring(t *testing.T) {
	var fixture struct {
		Bounds struct {
			SourceBytes int `json:"sourceBytes"`
			Depth       int `json:"depth"`
			Nodes       int `json:"nodes"`
		} `json:"bounds"`
		Prefix map[string]string    `json:"prefix"`
		Cases  []documentBoundsCase `json:"cases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "openapi-document-bounds.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Bounds.SourceBytes != MaxSourceBytes || fixture.Bounds.Depth != MaxDepth || fixture.Bounds.Nodes != MaxNodes {
		t.Fatalf("shared bounds = %+v, want %d bytes, depth %d and %d nodes", fixture.Bounds, MaxSourceBytes, MaxDepth, MaxNodes)
	}
	if len(fixture.Cases) < 12 {
		t.Fatalf("shared document bound cases missing: %d", len(fixture.Cases))
	}
	for _, tc := range fixture.Cases {
		for _, format := range []string{"json", "yaml"} {
			t.Run(tc.Name+"/"+format, func(t *testing.T) {
				document, err := tc.generate(fixture.Prefix[format], format == "json")
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

// generate renders the case after prefix in compact JSON or block YAML.
func (c documentBoundsCase) generate(prefix string, compactJSON bool) ([]byte, error) {
	var document strings.Builder
	document.WriteString(prefix)
	if c.Deep != nil {
		levels := c.Deep.Depth - 2
		switch c.Deep.Kind {
		case "mapping":
			if compactJSON {
				document.WriteString(`,"x-deep":` + strings.Repeat(`{"a":`, levels) + "0" + strings.Repeat("}", levels))
				break
			}
			document.WriteString("x-deep:\n")
			for level := 1; level < levels; level++ {
				document.WriteString(strings.Repeat("  ", level) + "a:\n")
			}
			document.WriteString(strings.Repeat("  ", levels) + "a: 0\n")
		case "sequence":
			nested := strings.Repeat("[", levels) + "0" + strings.Repeat("]", levels)
			if compactJSON {
				document.WriteString(`,"x-deep":` + nested)
			} else {
				document.WriteString("x-deep: " + nested + "\n")
			}
		default:
			return nil, fmt.Errorf("unknown deep kind %q", c.Deep.Kind)
		}
	}
	if c.Elements > 0 {
		if compactJSON {
			document.WriteString(`,"x-items":[` + strings.Repeat("0,", c.Elements-1) + "0]")
		} else {
			document.WriteString("x-items:\n" + strings.Repeat("- 0\n", c.Elements))
		}
	}
	if c.Members > 0 {
		if compactJSON {
			document.WriteString(`,"x-keys":{`)
			for member := range c.Members {
				if member > 0 {
					document.WriteString(",")
				}
				fmt.Fprintf(&document, `"k%06d":0`, member)
			}
			document.WriteString("}")
		} else {
			document.WriteString("x-keys:\n")
			for member := range c.Members {
				fmt.Fprintf(&document, "  k%06d: 0\n", member)
			}
		}
	}
	if c.Pad == nil {
		if compactJSON {
			document.WriteString("}")
		}
		return []byte(document.String()), nil
	}
	head, tail := document.String()+"x-pad: ", "\n"
	if compactJSON {
		head, tail = document.String()+`,"x-pad":"`, `"}`
	}
	room := c.Pad.Bytes - len(head) - len(tail)
	if room < 0 || c.Pad.Fill == "" {
		return nil, fmt.Errorf("pad of %d bytes cannot hold the document", c.Pad.Bytes)
	}
	pad := strings.Repeat(c.Pad.Fill, room/len(c.Pad.Fill)) + strings.Repeat("a", room%len(c.Pad.Fill))
	return []byte(head + pad + tail), nil
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
