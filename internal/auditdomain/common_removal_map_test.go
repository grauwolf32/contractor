package auditdomain

import (
	"encoding/json"
	"os"
	"regexp"
	"strings"
	"testing"

	auditschema "github.com/grauwolf32/contractor/api/audit/v1"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"go.yaml.in/yaml/v4"
)

func TestAuditCommonRemovalMapCoversRelationsAndOperations(t *testing.T) {
	type disposition struct {
		Name   string `yaml:"name"`
		Path   string `yaml:"path"`
		Method string `yaml:"method"`
		Action string `yaml:"action"`
		Owner  string `yaml:"owner"`
		Facts  string `yaml:"facts"`
	}
	var inventory struct {
		Relations  []disposition `yaml:"relations"`
		Profiles   []disposition `yaml:"profiles"`
		Operations []disposition `yaml:"operations"`
		Wire       []disposition `yaml:"wire"`
	}
	read := func(parts ...string) []byte {
		data, err := os.ReadFile(contractstest.Path(t, parts...))
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	if err := yaml.Unmarshal(read("docs", "proposals", "0050-audit-removal-map.yml"), &inventory); err != nil {
		t.Fatal(err)
	}
	relations := map[string]bool{}
	for _, entry := range inventory.Relations {
		relations[entry.Name] = true
	}
	registry := string(read("internal", "persistence", "table_ownership.txt"))
	registry = strings.Split(registry, "[reads]")[0]
	for _, match := range regexp.MustCompile(`(?m)^([a-z_]+)\s+internal/auditstore/`).FindAllStringSubmatch(registry, -1) {
		if !relations[match[1]] {
			t.Fatalf("unmapped Audit relation %s", match[1])
		}
	}
	var api struct {
		Paths map[string]map[string]any `yaml:"paths"`
	}
	if err := yaml.Unmarshal(read("api", "openapi", "contractor-public-v1.yaml"), &api); err != nil {
		t.Fatal(err)
	}
	operations := map[string]bool{}
	for _, entry := range inventory.Operations {
		operations[entry.Method+" "+entry.Path] = true
	}
	for path, methods := range api.Paths {
		if !strings.Contains(path, "audit") {
			continue
		}
		for method := range methods {
			if method != "get" && method != "post" && method != "put" && method != "patch" && method != "delete" {
				continue
			}
			if !operations[method+" "+path] {
				t.Fatalf("unmapped Audit operation %s %s", method, path)
			}
		}
	}
	for _, entries := range [][]disposition{inventory.Relations, inventory.Profiles, inventory.Operations, inventory.Wire} {
		if len(entries) == 0 {
			t.Fatal("empty removal inventory")
		}
		for _, entry := range entries {
			if entry.Owner == "" || entry.Facts == "" || (entry.Action != "preserve" && entry.Action != "replace" && entry.Action != "delete") {
				t.Fatalf("incomplete disposition %+v", entry)
			}
		}
	}
	// Profiles are frozen path identities in this reviewed map. Tests deliberately
	// do not open the live configs tree; shipped validation owns that tree.
}

func TestSharedAuditCommonBoundsMatchSchemaAndOwners(t *testing.T) {
	var schema struct {
		Maximum int `json:"x-maxEncodedBytes"`
		Defs    map[string]struct {
			Maximum      int                       `json:"x-maxEncodedBytes"`
			ValueMaximum float64                   `json:"maximum"`
			Properties   map[string]map[string]any `json:"properties"`
		} `json:"$defs"`
	}
	if err := json.Unmarshal(auditschema.Document, &schema); err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Limits          map[string]int `json:"limits"`
		DocumentBytes   int            `json:"document_bytes"`
		TextBytes       int            `json:"text_bytes"`
		Evidence        int            `json:"evidence_per_item"`
		Coverage        int            `json:"coverage_values"`
		MutationBytes   int            `json:"mutation_bytes"`
		HypothesisBytes int            `json:"hypothesis_bytes"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-common", "boundaries.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if schema.Maximum != MaximumDocumentBytes || fixture.DocumentBytes != MaximumDocumentBytes || fixture.TextBytes != MaximumStringBytes || fixture.Evidence != MaximumEvidencePerItem || fixture.Coverage != MaximumCoverageValues {
		t.Fatal("common contract changed an existing Audit owner ceiling")
	}
	if schema.Defs["hypothesis"].Maximum != fixture.HypothesisBytes || schema.Defs["mutation"].Maximum != fixture.MutationBytes {
		t.Fatal("hypothesis/mutation byte-bound drift")
	}
	for name, want := range fixture.Limits {
		got := schema.Defs["limits"].Properties[name]["maximum"]
		if ref, ok := schema.Defs["limits"].Properties[name]["$ref"].(string); ok {
			got = schema.Defs[strings.TrimPrefix(ref, "#/$defs/")].ValueMaximum
		}
		if got != float64(want) {
			t.Fatalf("schema maximum %s differs from frozen fixture", name)
		}
	}
}
