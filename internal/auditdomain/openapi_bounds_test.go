package auditdomain

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// TestSharedOpenAPIDocumentBoundsMatchAuthoringAndPreparation exercises both
// public inventory entry points and the retained task codecs, not just parsing.
func TestSharedOpenAPIDocumentBoundsMatchAuthoringAndPreparation(t *testing.T) {
	fixture := contractstest.ReadOpenAPIDocumentBounds(t)
	for _, tc := range fixture.Cases {
		for _, format := range []string{"json", "yaml"} {
			t.Run(tc.Name+"/"+format, func(t *testing.T) {
				source, err := tc.Generate(fixture.Prefix[format], format == "json")
				if err != nil {
					t.Fatal(err)
				}
				if sum := sha256.Sum256(source); hex.EncodeToString(sum[:]) != tc.SHA256[format] {
					t.Fatal("fixture generators disagree")
				}
				mediaType := "application/" + format
				payload, _, err := BuildPackage("bounds", PackageKindOpenAPISource, "openapi."+format, []PackageInput{{ID: "source", Path: "openapi." + format, MediaType: mediaType, Data: source}})
				if err != nil {
					t.Fatal(err)
				}
				for _, packaged := range []bool{false, true} {
					options := testInventoryOptions("trace")
					var inventory Inventory
					if packaged {
						inventory, err = BuildOpenAPIInventoryFromPackage(payload, options)
					} else {
						inventory, err = BuildOpenAPIInventory(source, mediaType, options)
					}
					if tc.Violation != nil {
						if err == nil {
							t.Fatalf("packaged=%v accepted %s violation", packaged, *tc.Violation)
						}
						continue
					}
					if err != nil {
						t.Fatalf("packaged=%v rejected accepted source: %v", packaged, err)
					}
					if len(inventory.CanonicalInventory) == 0 {
						t.Fatal("missing canonical inventory")
					}
					for _, task := range inventory.Tasks {
						encoded, err := EncodeItemTask(task.Document)
						if err != nil {
							t.Fatal(err)
						}
						decoded, err := DecodeItemTask(encoded)
						if err != nil || decoded.Operation == nil {
							t.Fatalf("retained operation task: %v", err)
						}
					}
				}
				// Scan inventory copies source identities, not source strings, into tasks.
				// The shared path-template cases exercise source parsing; URL and operation
				// pointer bounds of actual scan preparation remain separate contracts.
				if tc.String != nil && tc.String.Kind != "path" {
					settings := []byte(`{"schema":"contractor.audit.openapi-scan-settings.v1","scanner":"nuclei","server":"https://target.test","operations":{"#/paths/~1bounds/get":{}}}`)
					options := testInventoryOptions("scan")
					options.ApprovalRequirement = ApprovalActiveCheck
					input := ExactInput{Name: "settings", Ref: options.SourceRef.Clone(), Digest: DigestBytes(settings)}
					input.Ref.Name = "settings"
					if _, err := BuildOpenAPIScanInventory(source, mediaType, settings, input, options); err != nil {
						t.Fatalf("scan inventory rejected shared string case: %v", err)
					}
				}
			})
		}
	}
}

func TestOpenAPIInventoryDoesNotRelaxChecklistStringBounds(t *testing.T) {
	for _, format := range []string{"json", "yaml"} {
		text := strings.Repeat("a", MaximumStringBytes+1)
		source := `{"description":"` + text + `"}`
		if format == "yaml" {
			source = "description: " + text + "\n"
		}
		if _, err := parseJSONOrYAML([]byte(source), "application/"+format); err == nil {
			t.Fatal("checklist parser accepted oversized string")
		}
	}
}

func TestOpenAPIInventoryResolvesLongLocalPointersAndKeepsInternalBounds(t *testing.T) {
	name := "x-" + strings.Repeat("a", MaximumStringBytes)
	description := strings.Repeat("d", MaximumStringBytes+1)
	source := []byte(`{"openapi":"3.1.0","paths":{"/bounds":{"get":{"responses":{"200":{"$ref":"#/components/responses/` + name + `"}}}}},"components":{"responses":{"` + name + `":{"description":"` + description + `"}}}}`)
	inventory, err := BuildOpenAPIInventory(source, "application/json", testInventoryOptions("trace"))
	if err != nil {
		t.Fatal(err)
	}
	operation := inventory.Tasks[0].Document.Operation.Resolved["operation"].(map[string]any)
	response := operation["responses"].(map[string]any)["200"].(map[string]any)
	if response["description"] != description {
		t.Fatal("local reference lost source description")
	}
	task := inventory.Tasks[0].Document
	task.Scope = map[string]string{"objective": description}
	if _, err := EncodeItemTask(task); err == nil {
		t.Fatal("operation source allowance relaxed internal scope limit")
	}
}

func TestOpenAPIInventoryCanonicalEncodingDoesNotInflateHTML(t *testing.T) {
	description := strings.Repeat("<", 1500000)
	source := []byte(`{"openapi":"3.1.0","paths":{"/bounds":{"get":{"description":"` + description + `","responses":{"200":{"description":"ok"}}}}}}`)
	inventory, err := BuildOpenAPIInventory(source, "application/json", testInventoryOptions("trace"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(inventory.CanonicalInventory), description) {
		t.Fatal("canonical inventory changed source text")
	}
}

func TestOpenAPIInventoryBoundsExpandedValuesBeforePublishing(t *testing.T) {
	// One source has fewer than 100,000 nodes. Expanding the same 90,000-element
	// schema twice remains below LIM-AUD-30; three copies exceed it.
	values := strings.TrimSuffix(strings.Repeat("0,", 90000), ",")
	ref := `{"$ref":"#/components/schemas/Shared"}`
	for _, copies := range []int{2, 3} {
		source := []byte(`{"openapi":"3.1.0","paths":{"/bounds":{"get":{"x-repeated":[` + strings.TrimSuffix(strings.Repeat(ref+",", copies), ",") + `]}}},"components":{"schemas":{"Shared":{"enum":[` + values + `]}}}}`)
		inventory, err := BuildOpenAPIInventory(source, "application/json", testInventoryOptions("trace"))
		if copies == 2 && (err != nil || len(inventory.Tasks) != 1) {
			t.Fatalf("bounded expansion: %v", err)
		}
		if copies == 3 && (ErrorCode(err) != CodeLimitExceeded || len(inventory.Tasks) != 0) {
			t.Fatalf("over-budget expansion: %v, tasks=%d", err, len(inventory.Tasks))
		}
	}
}

// encodingProbe records whether encoding starts at all. The over-budget tree
// precedes it, so a streaming preflight must reject before observing this value.
type encodingProbe struct{ called *bool }

func (probe encodingProbe) MarshalJSON() ([]byte, error) {
	*probe.called = true
	return []byte("null"), nil
}

func TestOpenAPICanonicalBytePreflightRunsBeforeEncoding(t *testing.T) {
	called := false
	values := make([]any, 7)
	shared := map[string]any{"description": strings.Repeat("d", 1500000)}
	for i := range values {
		values[i] = shared
	}
	_, err := canonicalOpenAPIJSON([]any{values, encodingProbe{called: &called}})
	if !errors.Is(err, strictjson.ErrBounds) || called {
		t.Fatalf("byte bound must reject before encoding: err=%v, encoded=%v", err, called)
	}
}

func TestOpenAPIEncodingPreflightCountsRealJSONBytes(t *testing.T) {
	values := []any{
		nil, true, false, 1, int64(-15), uint64(42), 1e-10, 1e25,
		"<>&\u2028\u2029\n\t\x01\"\\é" + string([]byte{0xff}),
		map[string]any{"a\"": []any{nil, true, "<"}},
		[]string(nil), map[string]string(nil), []string{},
		struct {
			Required string  `json:"required"`
			Optional *string `json:"optional,omitempty"`
		}{Required: "ok"},
	}
	for _, value := range values {
		budget := openAPIEncodingBudget{remaining: MaximumDocumentBytes}
		if err := budget.value(reflect.ValueOf(value), 1); err != nil {
			t.Fatal(err)
		}
		var encoded bytes.Buffer
		encoder := json.NewEncoder(&encoded)
		encoder.SetEscapeHTML(false)
		if err := encoder.Encode(value); err != nil {
			t.Fatal(err)
		}
		if got, want := MaximumDocumentBytes-budget.remaining, encoded.Len()-1; got != want {
			t.Fatalf("%T bytes=%d, encoder=%d", value, got, want)
		}
	}
	for _, extra := range []int{0, 1} {
		value := strings.Repeat("a", MaximumDocumentBytes-2+extra)
		err := checkOpenAPIEncoding(value)
		if extra == 0 && err != nil || extra == 1 && !errors.Is(err, strictjson.ErrBounds) {
			t.Fatalf("byte boundary +%d: %v", extra, err)
		}
	}
}

func TestOpenAPIInventoryRejectsReferenceByteAmplification(t *testing.T) {
	description := strings.Repeat("d", 1500000)
	ref := `{"$ref":"#/components/schemas/Shared"}`
	source := []byte(`{"openapi":"3.1.0","paths":{"/bounds":{"get":{"x-repeated":[` + strings.TrimSuffix(strings.Repeat(ref+",", 1000), ",") + `]}}},"components":{"schemas":{"Shared":{"description":"` + description + `"}}}}`)
	if len(source) > 2<<20 {
		t.Fatal("regression source must fit the shared source byte bound")
	}
	payload, _, err := BuildPackage("amplification", PackageKindOpenAPISource, "openapi.json", []PackageInput{{ID: "source", Path: "openapi.json", MediaType: "application/json", Data: source}})
	if err != nil {
		t.Fatal(err)
	}
	for _, packaged := range []bool{false, true} {
		var inventory Inventory
		if packaged {
			inventory, err = BuildOpenAPIInventoryFromPackage(payload, testInventoryOptions("trace"))
		} else {
			inventory, err = BuildOpenAPIInventory(source, "application/json", testInventoryOptions("trace"))
		}
		if ErrorCode(err) != CodeInventoryInvalid || len(inventory.Tasks) != 0 {
			t.Fatalf("packaged=%v: err=%v tasks=%d", packaged, err, len(inventory.Tasks))
		}
	}
}

type pointerEncodingProbe struct{ called *bool }

func (probe *pointerEncodingProbe) MarshalJSON() ([]byte, error) {
	*probe.called = true
	return []byte("null"), nil
}

func TestOpenAPIEncodingPreflightRejectsPointerMarshalers(t *testing.T) {
	called := false
	_, err := canonicalOpenAPIJSON([]pointerEncodingProbe{{called: &called}})
	if err == nil || called {
		t.Fatalf("pointer marshaler must be rejected before encoding: err=%v encoded=%v", err, called)
	}
}
