package auditdomain

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestOpenAPIInventoryIdentitySeparatesSourceAndCanonicalContent(t *testing.T) {
	jsonSource := []byte(`{
  "openapi":"3.1.0",
  "info":{"title":"Example","version":"1"},
  "paths":{
    "/z":{"post":{"operationId":"duplicate","responses":{"200":{"description":"ok"}}},"get":{"operationId":"duplicate","parameters":[{"$ref":"#/components/parameters/ID"}],"responses":{"200":{"description":"ok"}}}},
    "/a":{"get":{"responses":{"200":{"description":"ok","content":{"application/json":{"schema":{"$ref":"#/components/schemas/Pet"}}}}}}}
  },
  "components":{"parameters":{"ID":{"name":"id","in":"query","schema":{"type":"string"}}},"schemas":{"Pet":{"properties":{"id":{"type":"string"}},"type":"object"}}}
}`)
	yamlSource := []byte(`
components:
  schemas:
    Pet:
      type: object
      properties: {id: {type: string}}
  parameters:
    ID: {in: query, schema: {type: string}, name: id}
paths:
  /a:
    get:
      operationId: display-name-does-not-affect-identity
      responses:
        "200":
          content:
            application/json:
              schema: {$ref: "#/components/schemas/Pet"}
          description: ok
  /z:
    get:
      parameters: [{$ref: "#/components/parameters/ID"}]
      responses: {"200": {description: ok}}
    post:
      operationId: another-display-name
      responses: {"200": {description: ok}}
info: {version: "1", title: Example}
openapi: 3.1.0
`)
	jsonInventory, err := prepareTestWork(t, jsonSource, "application/json")
	if err != nil {
		t.Fatal(err)
	}
	yamlInventory, err := prepareTestWork(t, yamlSource, "application/yaml")
	if err != nil {
		t.Fatal(err)
	}
	if decodePreparedTestItem(t, jsonInventory.Items[0]).Source.Digest == decodePreparedTestItem(t, yamlInventory.Items[0]).Source.Digest {
		t.Fatal("exact source digests unexpectedly match")
	}
	if jsonInventory.Items[0].Key != yamlInventory.Items[0].Key {
		t.Fatalf("canonical identity differs:\n%s\n%s", jsonInventory.Items[0].Data, yamlInventory.Items[0].Data)
	}
	if len(jsonInventory.Items) != 3 || len(yamlInventory.Items) != 3 {
		t.Fatalf("operations = %d/%d", len(jsonInventory.Items), len(yamlInventory.Items))
	}
	wantOrder := []struct{ path, method string }{{"/a", "get"}, {"/z", "get"}, {"/z", "post"}}
	for index, want := range wantOrder {
		left := decodePreparedTestItem(t, jsonInventory.Items[index]).Operation
		right := decodePreparedTestItem(t, yamlInventory.Items[index]).Operation
		if left.Path != want.path || left.Method != want.method || right.Path != want.path || right.Method != want.method || jsonInventory.Items[index].Key != yamlInventory.Items[index].Key {
			t.Fatalf("operation %d = %+v / %+v", index, left, right)
		}
		if decodePreparedTestItem(t, jsonInventory.Items[index]).Source.Digest == decodePreparedTestItem(t, yamlInventory.Items[index]).Source.Digest {
			t.Fatalf("task %d lost exact source provenance", index)
		}
	}
}

func TestOpenAPIInventoryPreservesUnquotedYAMLDate(t *testing.T) {
	jsonSource := []byte(`{"openapi":"3.0.3","info":{"title":"Example","version":"2024-01-01"},"paths":{"/a":{"get":{"responses":{"200":{"description":"ok"}}}}}}`)
	yamlSource := []byte("openapi: 3.0.3\ninfo:\n  title: Example\n  version: 2024-01-01\npaths:\n  /a:\n    get:\n      responses:\n        '200': {description: ok}\n")
	root, err := parseJSONOrYAML(yamlSource, "application/yaml")
	if err != nil {
		t.Fatal(err)
	}
	info, ok := root["info"].(map[string]any)
	if !ok || info["version"] != "2024-01-01" {
		t.Fatalf("version = %#v, want original string", root["info"])
	}
	jsonInventory, err := prepareTestWork(t, jsonSource, "application/json")
	if err != nil {
		t.Fatal(err)
	}
	yamlInventory, err := prepareTestWork(t, yamlSource, "application/yaml")
	if err != nil {
		t.Fatal(err)
	}
	if jsonInventory.Items[0].Key != yamlInventory.Items[0].Key {
		t.Fatalf("canonical identity differs:\n%s\n%s", jsonInventory.Items[0].Data, yamlInventory.Items[0].Data)
	}
	if decodePreparedTestItem(t, jsonInventory.Items[0]).Source.Digest == decodePreparedTestItem(t, yamlInventory.Items[0]).Source.Digest {
		t.Fatal("distinct source digests unexpectedly match")
	}
	bad := []byte("openapi: 3.0.3\ninfo:\n  version: !!binary c2VjcmV0\npaths: {}\n")
	if _, err := prepareTestWork(t, bad, "application/yaml"); testDomainErrorCode(err) != CodeInvalid {
		t.Fatalf("binary scalar error = %v", err)
	}
}

func TestOpenAPIYAMLPlainScalarNumberBoundsMatchScanplan(t *testing.T) {
	for _, test := range []struct {
		scalar string
		want   float64
	}{
		{"0x10", 16}, {"1_000", 1000}, {"1.25", 1.25}, {"9007199254740991", 9007199254740991},
	} {
		root, err := parseJSONOrYAML([]byte("value: "+test.scalar+"\n"), "application/yaml")
		if err != nil || root["value"] != test.want {
			t.Fatalf("scalar %s = %#v, %v; want %v", test.scalar, root, err, test.want)
		}
	}
	for _, scalar := range []string{
		"9007199254740992", "9007199254740991.1", "1e9999", ".nan", "!!binary c2VjcmV0",
	} {
		if _, err := parseJSONOrYAML([]byte("value: "+scalar+"\n"), "application/yaml"); testDomainErrorCode(err) != CodeInvalid {
			t.Fatalf("scalar %s error = %v", scalar, err)
		}
	}
}

func TestOpenAPIJSONAndYAMLNumberBoundsMatch(t *testing.T) {
	for _, scalar := range []string{"9007199254740991", "-9007199254740991", "1.25", "1e3"} {
		jsonRoot, jsonErr := parseJSONOrYAML([]byte(`{"value":`+scalar+`}`), "application/json")
		yamlRoot, yamlErr := parseJSONOrYAML([]byte("value: "+scalar+"\n"), "application/yaml")
		if jsonErr != nil || yamlErr != nil || !reflect.DeepEqual(jsonRoot, yamlRoot) {
			t.Fatalf("number %s: JSON %#v, %v; YAML %#v, %v", scalar, jsonRoot, jsonErr, yamlRoot, yamlErr)
		}
	}
	for _, scalar := range []string{"9007199254740992", "9223372036854775807", "9007199254740991.1", "1e300"} {
		if _, err := parseJSONOrYAML([]byte(`{"value":`+scalar+`}`), "application/json"); testDomainErrorCode(err) != CodeInvalid {
			t.Fatalf("JSON number %s error = %v", scalar, err)
		}
		if _, err := parseJSONOrYAML([]byte("value: "+scalar+"\n"), "application/yaml"); testDomainErrorCode(err) != CodeInvalid {
			t.Fatalf("YAML number %s error = %v", scalar, err)
		}
	}
	if _, err := canonicalJSON(map[string]any{"value": int64(1) << 60}); err != nil {
		t.Fatalf("canonical encoding of a large Server value: %v", err)
	}
}

func TestOpenAPISelectedRemoteRefRejectsButUnsupportedSurfaceIsGap(t *testing.T) {
	selected := []byte(`{"openapi":"3.1.0","paths":{"/x":{"get":{"responses":{"200":{"description":"ok","content":{"application/json":{"schema":{"$ref":"https://example.invalid/schema.json"}}}}}}}}}`)
	if _, err := prepareTestWork(t, selected, "application/json"); testDomainErrorCode(err) != CodeRemoteReference {
		t.Fatalf("selected remote ref error = %v", err)
	}
	unsupported := []byte(`{
  "openapi":"3.1.0",
  "paths":{"/x":{"post":{"callbacks":{"done":{"{$request.body#/url}":{"post":{"requestBody":{"$ref":"https://example.invalid/body"}}}}},"responses":{"200":{"description":"ok"}}}}},
  "webhooks":{"event":{"post":{"requestBody":{"$ref":"https://example.invalid/event"}}}},
  "components":{"schemas":{"Unused":{"$ref":"https://example.invalid/unused"}}}
}`)
	inventory, err := prepareTestWork(t, unsupported, "application/json")
	if err != nil {
		t.Fatal(err)
	}
	if len(inventory.Gaps) < 3 || len(inventory.Items) != 1 || len(decodePreparedTestItem(t, inventory.Items[0]).Operation.Gaps) != 1 {
		t.Fatalf("unsupported surface gaps were lost: global=%v operation=%v", inventory.Gaps, decodePreparedTestItem(t, inventory.Items[0]).Operation.Gaps)
	}
}

func TestOpenAPISelectedSecuritySchemesAreResolvedAndRemoteSchemesReject(t *testing.T) {
	local := []byte(`{
  "openapi":"3.1.0",
  "security":[{"bearer":[]}],
  "paths":{"/x":{"get":{"responses":{"200":{"description":"ok"}}}}},
  "components":{"securitySchemes":{"bearer":{"$ref":"#/components/securitySchemes/base"},"base":{"type":"http","scheme":"bearer"}}}
}`)
	inventory, err := prepareTestWork(t, local, "application/json")
	if err != nil {
		t.Fatal(err)
	}
	schemes, ok := decodePreparedTestItem(t, inventory.Items[0]).Operation.Resolved["security_schemes"].(map[string]any)
	if !ok || schemes["bearer"] == nil {
		t.Fatalf("selected security closure is missing: %+v", decodePreparedTestItem(t, inventory.Items[0]).Operation.Resolved)
	}
	remote := []byte(`{
  "openapi":"3.1.0",
  "security":[{"bearer":[]}],
  "paths":{"/x":{"get":{"responses":{"200":{"description":"ok"}}}}},
  "components":{"securitySchemes":{"bearer":{"$ref":"https://example.invalid/security"}}}
}`)
	if _, err := prepareTestWork(t, remote, "application/json"); testDomainErrorCode(err) != CodeRemoteReference {
		t.Fatalf("selected remote security error = %v", err)
	}
}

func TestOpenAPIRejectsCyclesAndDepthBeforeReturningInventory(t *testing.T) {
	cyclic := []byte(`{
  "openapi":"3.1.0",
  "paths":{"/x":{"get":{"responses":{"200":{"description":"ok","content":{"application/json":{"schema":{"$ref":"#/components/schemas/A"}}}}}}}},
  "components":{"schemas":{"A":{"$ref":"#/components/schemas/B"},"B":{"$ref":"#/components/schemas/A"}}}
}`)
	if inventory, err := prepareTestWork(t, cyclic, "application/json"); testDomainErrorCode(err) != CodeReferenceInvalid || len(inventory.Items) != 0 {
		t.Fatalf("cycle = %+v, %v", inventory, err)
	}

	components := make(map[string]any)
	for index := 0; index < MaximumReferenceDepth+2; index++ {
		name := "S" + strings.Repeat("x", index)
		next := "S" + strings.Repeat("x", index+1)
		components[name] = map[string]any{"$ref": "#/components/schemas/" + next}
	}
	components["S"+strings.Repeat("x", MaximumReferenceDepth+2)] = map[string]any{"type": "string"}
	document := map[string]any{
		"openapi":    "3.1.0",
		"paths":      map[string]any{"/x": map[string]any{"get": map[string]any{"responses": map[string]any{"200": map[string]any{"description": "ok", "content": map[string]any{"application/json": map[string]any{"schema": map[string]any{"$ref": "#/components/schemas/S"}}}}}}}},
		"components": map[string]any{"schemas": components},
	}
	encoded, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := prepareTestWork(t, encoded, "application/json"); testDomainErrorCode(err) != CodeLimitExceeded {
		t.Fatalf("deep refs error = %v", err)
	}
}

func TestOpenAPIPackageResolvesOnlyPackageLocalDocuments(t *testing.T) {
	root := []byte(`
openapi: 3.1.0
paths:
  /pets:
    get:
      responses:
        "200":
          description: ok
          content:
            application/json:
              schema: {$ref: "schemas.json#/PetList"}
`)
	schemas := []byte(`{"Pet":{"type":"object"},"PetList":{"type":"array","items":{"$ref":"#/Pet"}}}`)
	payload, _, err := BuildPackage("openapi-fixture", PackageKindOpenAPISource, "root.yaml", []PackageInput{
		{ID: "root", Path: "root.yaml", MediaType: "application/yaml", Data: root},
		{ID: "schemas", Path: "schemas.json", MediaType: "application/json", Data: schemas},
	})
	if err != nil {
		t.Fatal(err)
	}
	inventory, err := prepareTestWork(t, payload, PackageMediaType)
	if err != nil {
		t.Fatal(err)
	}
	if len(inventory.Items) != 1 || decodePreparedTestItem(t, inventory.Items[0]).Source.Digest != DigestBytes(payload) {
		t.Fatalf("package inventory = %+v", inventory)
	}
	material := decodePreparedTestItem(t, inventory.Items[0])
	if material.Operation == nil || material.Source.Digest != DigestBytes(payload) || material.Source.Ref.Revision == nil {
		t.Fatal("exact package provenance missing")
	}

}

func TestOpenAPIPackagePreservesReferencedDocumentOrigin(t *testing.T) {
	root := []byte(`
openapi: 3.1.0
paths:
  /pets: {$ref: "paths/pets.yaml#/PetPath"}
`)
	pathItem := []byte(`
PetPath:
  get:
    responses:
      "200":
        description: ok
        content:
          application/json:
            schema: {$ref: "../schemas.json#/Pet"}
`)
	schemas := []byte(`{"Pet":{"type":"object"}}`)
	payload, _, err := BuildPackage("referenced-path", PackageKindOpenAPISource, "root.yaml", []PackageInput{
		{ID: "root", Path: "root.yaml", MediaType: "application/yaml", Data: root},
		{ID: "path", Path: "paths/pets.yaml", MediaType: "application/yaml", Data: pathItem},
		{ID: "schemas", Path: "schemas.json", MediaType: "application/json", Data: schemas},
	})
	if err != nil {
		t.Fatal(err)
	}
	inventory, err := prepareTestWork(t, payload, PackageMediaType)
	if err != nil || len(inventory.Items) != 1 {
		t.Fatalf("external Path Item origin was lost: %+v, %v", inventory, err)
	}
}

func TestOpenAPIDuplicateYAMLKeyFailsAtomically(t *testing.T) {
	source := []byte("openapi: 3.1.0\npaths: {}\npaths: {}\n")
	inventory, err := prepareTestWork(t, source, "application/yaml")
	if err == nil || len(inventory.Items) != 0 {
		t.Fatalf("duplicate YAML key accepted: %+v, %v", inventory, err)
	}
}

func TestOpenAPIPathsExtensionsAreNotEnumerated(t *testing.T) {
	source := []byte(`{"openapi":"3.1.0","paths":{"x-inventory-note":{"$ref":"https://example.invalid/note"},"/ok":{"get":{"responses":{"200":{"description":"ok"}}}}}}`)
	inventory, err := prepareTestWork(t, source, "application/json")
	if err != nil {
		t.Fatal(err)
	}
	if len(inventory.Items) != 1 || len(inventory.Gaps) != 1 || decodePreparedTestItem(t, inventory.Items[0]).Operation.Path != "/ok" {
		t.Fatalf("Paths extension handling = %+v", inventory)
	}
}

func FuzzOpenAPIInventory(f *testing.F) {
	f.Add([]byte(`{"openapi":"3.1.0","paths":{}}`), "application/json")
	f.Add([]byte("openapi: 3.1.0\npaths: {}\n"), "application/yaml")
	f.Add([]byte("not openapi"), "application/yaml")
	f.Fuzz(func(t *testing.T, source []byte, mediaType string) {
		inventory, err := prepareTestWork(t, source, mediaType)
		if err != nil {
			if testDomainErrorCode(err) == "" {
				t.Fatalf("unstable error type: %T: %v", err, err)
			}
			return
		}
		if len(inventory.Items) > MaximumItems || len(inventory.Items) > 256 {
			t.Fatalf("accepted out-of-bounds inventory: %+v", inventory)
		}
	})
}
