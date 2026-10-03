package e2e

import (
	"bytes"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// checkArtifactPlaneRouteInventory scans package sources rather than one router
// file, so splitting route registration cannot silently weaken these guards.
func checkArtifactPlaneRouteInventory(repositoryRoot, segment string) error {
	sources := []string{filepath.Join(repositoryRoot, "api", "openapi", "contractor-public-v1.yaml")}
	for _, directory := range []string{"internal/httpapi/public", "internal/httpapi/privateartifacts"} {
		err := filepath.WalkDir(filepath.Join(repositoryRoot, directory), func(path string, entry fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if !entry.IsDir() && filepath.Ext(path) == ".go" && !strings.HasSuffix(entry.Name(), "_test.go") {
				sources = append(sources, path)
			}
			return nil
		})
		if err != nil {
			return err
		}
	}
	for _, path := range sources {
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if containsRouteSegment(data, segment) {
			relative, err := filepath.Rel(repositoryRoot, path)
			if err != nil {
				return err
			}
			return fmt.Errorf("forbidden HTTP route segment %q appears in %s", segment, relative)
		}
	}
	return nil
}

func containsRouteSegment(data []byte, segment string) bool {
	lower := bytes.ToLower(data)
	needle := []byte(strings.ToLower(segment))
	for offset := 0; offset < len(lower); {
		index := bytes.Index(lower[offset:], needle)
		if index < 0 {
			return false
		}
		end := offset + index + len(needle)
		if end == len(lower) || !routeSegmentByte(lower[end]) {
			return true
		}
		offset = end
	}
	return false
}

func routeSegmentByte(value byte) bool {
	return value >= 'a' && value <= 'z' || value >= '0' && value <= '9' ||
		value == '_' || value == '-' || value == '.' || value >= 0x80
}

func TestArtifactPlaneInventoryScansNewProductionFiles(t *testing.T) {
	for _, test := range []struct {
		name, relative, content, segment string
		forbidden                        bool
	}{
		{"new public route", "internal/httpapi/public/routes_new.go", `mux.HandleFunc("GET /v1/skills", handler)`, "/skills", true},
		{"Operations Skill route", "internal/httpapi/public/routes_new.go", `mux.HandleFunc("GET /v1/operations/skills", handler)`, "/skills", true},
		{"new Memory route", "internal/httpapi/public/routes_new.go", `mux.HandleFunc("GET /v1/operations/memory", handler)`, "/memory", true},
		{"private route", "internal/httpapi/privateartifacts/routes_new.go", `mux.HandleFunc("GET /private/v1/memory", handler)`, "/memory", true},
		{"OpenAPI route", "api/openapi/contractor-public-v1.yaml", `/v1/operations/skills:`, "/skills", true},
		{"test-only example", "internal/httpapi/public/routes_new_test.go", `mux.HandleFunc("GET /v1/skills", handler)`, "/skills", false},
		{"longer segment", "internal/httpapi/public/routes_new.go", `mux.HandleFunc("GET /v1/skillset", handler)`, "/skills", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			root := t.TempDir()
			for _, directory := range []string{
				"api/openapi", "internal/httpapi/public", "internal/httpapi/privateartifacts",
			} {
				if err := os.MkdirAll(filepath.Join(root, directory), 0o700); err != nil {
					t.Fatal(err)
				}
			}
			openAPI := filepath.Join(root, "api/openapi/contractor-public-v1.yaml")
			if err := os.WriteFile(openAPI, []byte("paths: {}\n"), 0o600); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(root, test.relative), []byte(test.content), 0o600); err != nil {
				t.Fatal(err)
			}
			err := checkArtifactPlaneRouteInventory(root, test.segment)
			if (err != nil) != test.forbidden {
				t.Fatalf("inventory result = %v, forbidden = %t", err, test.forbidden)
			}
		})
	}
}
