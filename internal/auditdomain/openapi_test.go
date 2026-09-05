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
	options := testInventoryOptions("trace")
	jsonInventory, err := BuildOpenAPIInventory(jsonSource, "application/json", options)
	if err != nil {
		t.Fatal(err)
	}
	yamlInventory, err := BuildOpenAPIInventory(yamlSource, "application/yaml", options)
	if err != nil {
		t.Fatal(err)
	}
	if jsonInventory.SourceContentDigest == yamlInventory.SourceContentDigest {
		t.Fatal("exact source digests unexpectedly match")
	}
	if jsonInventory.CanonicalInventoryDigest != yamlInventory.CanonicalInventoryDigest || !reflect.DeepEqual(jsonInventory.CanonicalInventory, yamlInventory.CanonicalInventory) {
		t.Fatalf("canonical identity differs:\n%s\n%s", jsonInventory.CanonicalInventory, yamlInventory.CanonicalInventory)
	}
	if len(jsonInventory.Worklist.Items) != 3 || len(yamlInventory.Worklist.Items) != 3 {
		t.Fatalf("operations = %d/%d", len(jsonInventory.Worklist.Items), len(yamlInventory.Worklist.Items))
	}
	wantOrder := []struct{ path, method string }{{"/a", "get"}, {"/z", "get"}, {"/z", "post"}}
	for index, want := range wantOrder {
		left := jsonInventory.Tasks[index].Document.Operation
		right := yamlInventory.Tasks[index].Document.Operation
		if left.Path != want.path || left.Method != want.method || right.Path != want.path || right.Method != want.method || jsonInventory.Worklist.Items[index].ItemKey != yamlInventory.Worklist.Items[index].ItemKey {
			t.Fatalf("operation %d = %+v / %+v", index, left, right)
		}
		if jsonInventory.Tasks[index].PackageDigest == yamlInventory.Tasks[index].PackageDigest {
			t.Fatalf("task %d lost exact source provenance", index)
		}
	}
}

func TestOpenAPISelectedRemoteRefRejectsButUnsupportedSurfaceIsGap(t *testing.T) {
	options := testInventoryOptions("trace")
	selected := []byte(`{"openapi":"3.1.0","paths":{"/x":{"get":{"responses":{"200":{"description":"ok","content":{"application/json":{"schema":{"$ref":"https://example.invalid/schema.json"}}}}}}}}}`)
	if _, err := BuildOpenAPIInventory(selected, "application/json", options); ErrorCode(err) != CodeRemoteReference {
		t.Fatalf("selected remote ref error = %v", err)
	}
	unsupported := []byte(`{
  "openapi":"3.1.0",
  "paths":{"/x":{"post":{"callbacks":{"done":{"{$request.body#/url}":{"post":{"requestBody":{"$ref":"https://example.invalid/body"}}}}},"responses":{"200":{"description":"ok"}}}}},
  "webhooks":{"event":{"post":{"requestBody":{"$ref":"https://example.invalid/event"}}}},
  "components":{"schemas":{"Unused":{"$ref":"https://example.invalid/unused"}}}
}`)
	inventory, err := BuildOpenAPIInventory(unsupported, "application/json", options)
	if err != nil {
		t.Fatal(err)
	}
	if len(inventory.Gaps) < 3 || len(inventory.Tasks) != 1 || len(inventory.Tasks[0].Document.Operation.Gaps) != 1 {
		t.Fatalf("unsupported surface gaps were lost: global=%v operation=%v", inventory.Gaps, inventory.Tasks[0].Document.Operation.Gaps)
	}
}

func TestOpenAPISelectedSecuritySchemesAreResolvedAndRemoteSchemesReject(t *testing.T) {
	options := testInventoryOptions("trace")
	local := []byte(`{
  "openapi":"3.1.0",
  "security":[{"bearer":[]}],
  "paths":{"/x":{"get":{"responses":{"200":{"description":"ok"}}}}},
  "components":{"securitySchemes":{"bearer":{"$ref":"#/components/securitySchemes/base"},"base":{"type":"http","scheme":"bearer"}}}
}`)
	inventory, err := BuildOpenAPIInventory(local, "application/json", options)
	if err != nil {
		t.Fatal(err)
	}
	schemes, ok := inventory.Tasks[0].Document.Operation.Resolved["security_schemes"].(map[string]any)
	if !ok || schemes["bearer"] == nil {
		t.Fatalf("selected security closure is missing: %+v", inventory.Tasks[0].Document.Operation.Resolved)
	}
	remote := []byte(`{
  "openapi":"3.1.0",
  "security":[{"bearer":[]}],
  "paths":{"/x":{"get":{"responses":{"200":{"description":"ok"}}}}},
  "components":{"securitySchemes":{"bearer":{"$ref":"https://example.invalid/security"}}}
}`)
	if _, err := BuildOpenAPIInventory(remote, "application/json", options); ErrorCode(err) != CodeRemoteReference {
		t.Fatalf("selected remote security error = %v", err)
	}
}

func TestOpenAPIRejectsCyclesAndDepthBeforeReturningInventory(t *testing.T) {
	options := testInventoryOptions("trace")
	cyclic := []byte(`{
  "openapi":"3.1.0",
  "paths":{"/x":{"get":{"responses":{"200":{"description":"ok","content":{"application/json":{"schema":{"$ref":"#/components/schemas/A"}}}}}}}},
  "components":{"schemas":{"A":{"$ref":"#/components/schemas/B"},"B":{"$ref":"#/components/schemas/A"}}}
}`)
	if inventory, err := BuildOpenAPIInventory(cyclic, "application/json", options); ErrorCode(err) != CodeReferenceInvalid || len(inventory.Tasks) != 0 {
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
	if _, err := BuildOpenAPIInventory(encoded, "application/json", options); ErrorCode(err) != CodeLimitExceeded {
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
	inventory, err := BuildOpenAPIInventoryFromPackage(payload, testInventoryOptions("trace"))
	if err != nil {
		t.Fatal(err)
	}
	if len(inventory.Tasks) != 1 || inventory.SourceContentDigest != digestBytes(payload) {
		t.Fatalf("package inventory = %+v", inventory)
	}
	validated, err := ValidatePackage(inventory.Tasks[0].Package)
	if err != nil {
		t.Fatal(err)
	}
	if len(validated.Members()) != 1 || inventory.Tasks[0].Document.SourceContentDigest != digestBytes(payload) ||
		len(inventory.ExecutionManifest.Items[0].Inputs) != 1 || inventory.ExecutionManifest.Items[0].Inputs[0].Ref.Revision == nil {
		t.Fatal("task package does not retain exact source provenance")
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
	inventory, err := BuildOpenAPIInventoryFromPackage(payload, testInventoryOptions("trace"))
	if err != nil || len(inventory.Tasks) != 1 {
		t.Fatalf("external Path Item origin was lost: %+v, %v", inventory, err)
	}
}

func TestOpenAPIDuplicateYAMLKeyFailsAtomically(t *testing.T) {
	source := []byte("openapi: 3.1.0\npaths: {}\npaths: {}\n")
	inventory, err := BuildOpenAPIInventory(source, "application/yaml", testInventoryOptions("trace"))
	if err == nil || len(inventory.Tasks) != 0 {
		t.Fatalf("duplicate YAML key accepted: %+v, %v", inventory, err)
	}
}

func TestOpenAPIPathsExtensionsAreNotEnumerated(t *testing.T) {
	source := []byte(`{"openapi":"3.1.0","paths":{"x-inventory-note":{"$ref":"https://example.invalid/note"},"/ok":{"get":{"responses":{"200":{"description":"ok"}}}}}}`)
	inventory, err := BuildOpenAPIInventory(source, "application/json", testInventoryOptions("trace"))
	if err != nil {
		t.Fatal(err)
	}
	if len(inventory.Tasks) != 1 || len(inventory.Gaps) != 1 || inventory.Tasks[0].Document.Operation.Path != "/ok" {
		t.Fatalf("Paths extension handling = %+v", inventory)
	}
}

func FuzzOpenAPIInventory(f *testing.F) {
	f.Add([]byte(`{"openapi":"3.1.0","paths":{}}`), "application/json")
	f.Add([]byte("openapi: 3.1.0\npaths: {}\n"), "application/yaml")
	f.Add([]byte("not openapi"), "application/yaml")
	f.Fuzz(func(t *testing.T, source []byte, mediaType string) {
		inventory, err := BuildOpenAPIInventory(source, mediaType, testInventoryOptions("trace"))
		if err != nil {
			if ErrorCode(err) == "" {
				t.Fatalf("unstable error type: %T: %v", err, err)
			}
			return
		}
		if len(inventory.Tasks) > MaximumItems || len(inventory.CanonicalInventory) > MaximumDocumentBytes || !validDigest(inventory.SourceContentDigest) || !validDigest(inventory.CanonicalInventoryDigest) {
			t.Fatalf("accepted out-of-bounds inventory: %+v", inventory)
		}
	})
}
