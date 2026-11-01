package privateartifacts

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	jsonschema "github.com/santhosh-tekuri/jsonschema/v6"
)

// This is a direction-aware wire fixture: Go decodes commands and encodes
// replies; Python does the inverse. The owning domain still validates mutation
// intent and source selections, rather than a second wire reducer.
func TestSharedSitemapPrivateWire(t *testing.T) {
	var cases struct {
		CollectorLimits map[string]int `json:"collectorLimits"`
		Valid, Invalid  []struct {
			Type  string          `json:"type"`
			Value json.RawMessage `json:"value"`
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "sitemap-wire-cases.json"), &cases); err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]int{"items": projectsitemap.MaxBatch, "batchBytes": projectsitemap.MaxBatchBytes, "attempts": projectsitemap.MaxAttempts, "flushBudgetSeconds": int(projectsitemap.OperationTimeout.Seconds())} {
		if cases.CollectorLimits[name] != want {
			t.Fatalf("collector limit %s differs: %d want %d", name, cases.CollectorLimits[name], want)
		}
	}
	if len(cases.Valid) < 9 || len(cases.Invalid) < 26 {
		t.Fatal("incomplete private sitemap cases")
	}
	compiler := jsonschema.NewCompiler()
	compiler.AssertFormat()
	paths, err := filepath.Glob(contractstest.Path(t, "api", "v1alpha1", "*.schema.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range paths {
		document, err := jsonschema.UnmarshalJSON(bytes.NewReader(contractstest.ReadFile(t, "api", "v1alpha1", filepath.Base(path))))
		if err != nil {
			t.Fatal(err)
		}
		id := document.(map[string]any)["$id"].(string)
		if err := compiler.AddResource(id, document); err != nil {
			t.Fatal(err)
		}
	}
	rootSchema, err := compiler.Compile("https://contractor.local/schema/v1alpha1/project-sitemap.schema.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, group := range []struct {
		valid bool
		cases []struct {
			Type  string          `json:"type"`
			Value json.RawMessage `json:"value"`
		}
	}{{true, cases.Valid}, {false, cases.Invalid}} {
		for _, c := range group.cases {
			schema, err := compiler.Compile("https://contractor.local/schema/v1alpha1/project-sitemap.schema.json#/$defs/" + c.Type)
			if err != nil {
				t.Fatal(err)
			}
			value, err := jsonschema.UnmarshalJSON(bytes.NewReader(c.Value))
			if err != nil {
				t.Fatal(err)
			}
			if err := schema.Validate(value); (err == nil) != group.valid {
				t.Fatalf("%s schema valid=%t: %s: %v", c.Type, group.valid, c.Value, err)
			}
			if group.valid {
				if err := rootSchema.Validate(value); err != nil {
					t.Fatalf("%s is missing from the root schema: %v", c.Type, err)
				}
			}
			// Only requests are consumed by the Server. Replies are schema-checked
			// above and exercised as actual production bytes by the PostgreSQL cases.
			switch c.Type {
			case "SitemapMutationRequest":
				var input sitemapMutationRequest
				err := strictjson.DecodeNonNull(c.Value, &input)
				valid := err == nil && input.APIVersion == contracts.APIVersion
				if valid != group.valid {
					t.Fatalf("mutation command valid=%t: %s: %v", group.valid, c.Value, err)
				}
			case "SitemapCollectionRequest":
				var input sitemapCollectionRequest
				err := strictjson.DecodeNonNull(c.Value, &input)
				valid := err == nil && input.APIVersion == contracts.APIVersion && input.Input.Validate() == nil
				if valid != group.valid {
					t.Fatalf("collection command valid=%t: %s: %v", group.valid, c.Value, err)
				}
			case "SitemapSnapshotRequest":
				var input sitemapSnapshotRequest
				err := strictjson.DecodeNonNull(c.Value, &input)
				valid := err == nil && input.APIVersion == contracts.APIVersion && input.Selection.Validate() == nil
				if valid != group.valid {
					t.Fatalf("snapshot command valid=%t: %s: %v", group.valid, c.Value, err)
				}
			}
		}
	}
}
