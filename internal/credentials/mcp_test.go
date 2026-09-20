package credentials

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMCPHeadersUseSharedPrivateHeaderPolicy(t *testing.T) {
	root := filepath.Join("..", "..", "api", "testdata", "toolset-runtime-configuration")
	index, err := os.ReadFile(filepath.Join(root, "cases.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		File  string
		Valid bool
	}
	if err := json.Unmarshal(index, &cases); err != nil {
		t.Fatal(err)
	}
	for _, test := range cases {
		if !strings.HasPrefix(test.File, "header-") && test.File != "settings-mcp.json" {
			continue
		}
		t.Run(test.File, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join(root, test.File))
			if err != nil {
				t.Fatal(err)
			}
			var wire struct {
				Toolsets map[string]struct{ Headers map[string]string }
			}
			if err := json.Unmarshal(raw, &wire); err != nil {
				t.Fatal(err)
			}
			material, err := NewMCPHeadersCredential(wire.Toolsets["mcp@1"].Headers)
			if (err == nil) != test.Valid {
				t.Fatalf("header policy differs, valid=%v: %v", test.Valid, err)
			}
			if strings.Contains(fmt.Sprintf("%v %+v %#v", material, material, err), "mcp-secret-canary") {
				t.Fatal("secret escaped formatting")
			}
			if test.Valid {
				decoded, err := runtimeCredentialMaterialFromCanonical(material.Kind(), material.canonical)
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(decoded.canonical, material.canonical) {
					t.Fatal("private material changed")
				}
				if _, err := json.Marshal(material); err == nil {
					t.Fatal("public material serialization allowed")
				}
			}
		})
	}
	if _, err := NewMCPHeadersCredential(nil); err == nil {
		t.Fatal("empty credential accepted")
	}
}
