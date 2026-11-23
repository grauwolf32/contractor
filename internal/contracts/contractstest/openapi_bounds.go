package contractstest

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// OpenAPIDocumentBoundsCase is one generated document of
// api/testdata/openapi-document-bounds.json; the fixture's description states
// how the document is generated.
type OpenAPIDocumentBoundsCase struct {
	Name   string `json:"name"`
	String *struct {
		Kind  string `json:"kind"`
		Bytes int    `json:"bytes"`
	} `json:"string"`
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

// OpenAPIDocumentBoundsFixture pins the source bounds and generated documents.
type OpenAPIDocumentBoundsFixture struct {
	Bounds struct {
		SourceBytes int `json:"sourceBytes"`
		Depth       int `json:"depth"`
		Nodes       int `json:"nodes"`
	} `json:"bounds"`
	Prefix map[string]string           `json:"prefix"`
	Cases  []OpenAPIDocumentBoundsCase `json:"cases"`
}

// ReadOpenAPIDocumentBounds loads the common fixture through the shared-reader gate.
func ReadOpenAPIDocumentBounds(t *testing.T) OpenAPIDocumentBoundsFixture {
	t.Helper()
	var fixture OpenAPIDocumentBoundsFixture
	if err := json.Unmarshal(ReadFile(t, "api", "testdata", "openapi-document-bounds.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	return fixture
}

// Generate renders the case after prefix in compact JSON or block YAML.
func (c OpenAPIDocumentBoundsCase) Generate(prefix string, compactJSON bool) ([]byte, error) {
	if c.String != nil {
		return c.stringDocument(compactJSON)
	}
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

func (c OpenAPIDocumentBoundsCase) stringDocument(compactJSON bool) ([]byte, error) {
	text := strings.Repeat("a", c.String.Bytes)
	path, extraJSON, extraYAML := "/bounds", "", ""
	switch c.String.Kind {
	case "value":
		extraJSON = `,"description":"` + text + `"`
		extraYAML = "      description: \"" + text + "\"\n"
	case "name":
		name := "x-" + text[2:]
		extraJSON = `,"` + name + `":0`
		extraYAML = "      ? \"" + name + "\"\n      : 0\n"
	case "path":
		path = "/" + text[1:]
	case "operationId":
		extraJSON = `,"operationId":"` + text + `"`
		extraYAML = "      operationId: \"" + text + "\"\n"
	default:
		return nil, fmt.Errorf("unknown string kind %q", c.String.Kind)
	}
	if compactJSON {
		return []byte(`{"openapi":"3.0.3","info":{"title":"Bounds","version":"1.0.0"},"paths":{"` + path + `":{"get":{"responses":{"200":{"description":"ok"}}` + extraJSON + `}}}}`), nil
	}
	return []byte("openapi: 3.0.3\ninfo:\n  title: Bounds\n  version: 1.0.0\npaths:\n  ? \"" + path + "\"\n  :\n    get:\n      responses:\n        '200':\n          description: ok\n" + extraYAML), nil
}
