package public

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/getkin/kin-openapi/openapi3"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/credentials"
)

// TestSharedTargetCredentialMaterialCasesGateCreation sends every material
// case of api/testdata/v1alpha1/target-credential-cases.json in a public
// create request of its target kind (RCF-R184). A valid material is stored
// and answered with metadata only; an invalid one is 400 invalid_request,
// stores nothing and is not repeated in the answer. The material schema that
// the public CreateRuntimeCredentialRequest names for the case's kind gives
// each case the fixture's schema verdict; it is checked on its own because
// kin-openapi validates a $ref inside oneOf without propertyNames. Every Basic
// kind shares the target-basic@1 rule (RCF-R114), so its cases are sent as
// http-origin-basic@1 and http-proxy-basic@1 too.
func TestSharedTargetCredentialMaterialCasesGateCreation(t *testing.T) {
	cases := contractstest.ReadTargetCredentialCases(t)
	document := loadPublicOpenAPI(t)
	reference := document.Components.Schemas["CreateRuntimeCredentialRequest"]
	if reference == nil || reference.Value == nil {
		t.Fatal("components.schemas.CreateRuntimeCredentialRequest is missing")
	}
	materialSchemas := map[string]*openapi3.Schema{}
	for _, branch := range reference.Value.OneOf {
		kind, material := branch.Value.Properties["kind"], branch.Value.Properties["material"]
		if kind == nil || material == nil {
			t.Fatal("a CreateRuntimeCredentialRequest branch lacks kind or material")
		}
		name := strings.TrimPrefix(material.Ref, "#/components/schemas/")
		component := document.Components.Schemas[name]
		value, ok := kind.Value.Const.(string)
		if !ok || component == nil || component.Value == nil {
			t.Fatalf("CreateRuntimeCredentialRequest branch %v names no material component", kind.Value.Const)
		}
		materialSchemas[value] = component.Value
	}
	fixture := newHandlerFixture(t)
	type sent struct {
		contractstest.TargetCredentialCase
		kind string
	}
	var requests []sent
	for _, test := range cases.Materials {
		requests = append(requests, sent{test, test.Kind})
		if test.Kind == string(credentials.RuntimeCredentialTargetBasic) {
			requests = append(requests,
				sent{test, string(credentials.RuntimeCredentialOriginBasic)},
				sent{test, string(credentials.RuntimeCredentialProxyBasic)})
		}
	}
	for index, request := range requests {
		test := request.TargetCredentialCase
		test.Name = request.kind + " " + test.Name
		credentialID := fmt.Sprintf("target-case-%d", index)
		var body bytes.Buffer
		fmt.Fprintf(&body, `{"credentialId":%q,"kind":%q,"material":`, credentialID, request.kind)
		body.Write(test.Data)
		body.WriteString("}")

		schema := materialSchemas[request.kind]
		if schema == nil {
			t.Fatalf("%s: no public material schema for %s", test.Name, request.kind)
		}
		accepted := false
		if test.SchemaDocument() {
			var material any
			if err := json.Unmarshal(test.Data, &material); err != nil {
				t.Fatalf("%s: %v", test.Name, err)
			}
			accepted = schema.VisitJSON(material, openapi3.EnableJSONSchema2020(), openapi3.EnableFormatValidation()) == nil
		}
		if want := test.Valid || test.SchemaAccepts; accepted != want {
			t.Errorf("%s: OpenAPI accepted = %t, want %t", test.Name, accepted, want)
		}

		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, runtimeMutationRequest(
			t, http.MethodPost, "/v1/operations/runtime-credentials", body.String(), credentialID, "", "",
		))
		_, stored := fixture.runtimeCredentials.Get(t.Context(), credentialID)
		switch {
		case test.Valid && (response.Code != http.StatusCreated || stored != nil):
			t.Errorf("%s: create = %d (stored error %v): %s", test.Name, response.Code, stored, response.Body.String())
		case !test.Valid && response.Code != http.StatusBadRequest:
			t.Errorf("%s: create = %d, want 400: %s", test.Name, response.Code, response.Body.String())
		case !test.Valid:
			assertErrorCode(t, response, "invalid_request")
			if !errors.Is(stored, credentials.ErrRuntimeCredentialNotFound) {
				t.Errorf("%s: rejected create stored a credential: %v", test.Name, stored)
			}
		}
		var values []string
		var walk func(any)
		walk = func(value any) {
			switch typed := value.(type) {
			case map[string]any:
				for _, child := range typed {
					walk(child)
				}
			case string:
				if len(typed) >= 3 {
					values = append(values, typed)
				}
			}
		}
		var material any
		if json.Unmarshal(test.Data, &material) == nil {
			walk(material)
		}
		for _, value := range values {
			if strings.Contains(response.Body.String(), value) {
				t.Errorf("%s: the answer repeats a material value: %s", test.Name, response.Body.String())
			}
		}
	}
}

// TestTargetCredentialCreateReturnsOnlySafeMetadata creates each engagement
// target credential kind (RCF-R184) through the public API: success returns
// the identity and kind without any secret byte, and an out-of-bounds login
// material is 400 invalid_request.
func TestTargetCredentialCreateReturnsOnlySafeMetadata(t *testing.T) {
	const canary = "canary-target-secret-6f1d"
	fixture := newHandlerFixture(t)
	for _, tc := range []struct {
		id, kind, material string
	}{
		{"target-basic", "target-basic@1", `{"username":"alice","password":"` + canary + `"}`},
		{"target-bearer", "target-bearer@1", `{"token":"` + canary + `"}`},
		{"target-cookies", "target-cookies@1", `{"cookies":{"session":"` + canary + `"}}`},
	} {
		body := fmt.Sprintf(`{"credentialId":%q,"kind":%q,"material":%s}`, tc.id, tc.kind, tc.material)
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, runtimeMutationRequest(
			t, http.MethodPost, "/v1/operations/runtime-credentials", body, tc.id, "", "",
		))
		if response.Code != http.StatusCreated {
			t.Fatalf("%s: create = %d: %s", tc.kind, response.Code, response.Body.String())
		}
		if strings.Contains(response.Body.String(), canary) {
			t.Fatalf("%s: response exposes the secret: %s", tc.kind, response.Body.String())
		}
		var created struct {
			CredentialID string `json:"credentialId"`
			Kind         string `json:"kind"`
		}
		if err := json.Unmarshal(response.Body.Bytes(), &created); err != nil ||
			created.CredentialID != tc.id || created.Kind != tc.kind {
			t.Fatalf("%s: created = %+v, %v: %s", tc.kind, created, err, response.Body.String())
		}
	}
	fields := make([]string, 0, 129)
	for i := range 129 {
		fields = append(fields, fmt.Sprintf(`"f%d":"%s"`, i, canary))
	}
	body := `{"credentialId":"target-cookies-oversized","kind":"target-cookies@1","material":{"cookies":{` + strings.Join(fields, ",") + `}}}`
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, runtimeMutationRequest(
		t, http.MethodPost, "/v1/operations/runtime-credentials", body, "target-cookies-oversized", "", "",
	))
	if response.Code != http.StatusBadRequest || strings.Contains(response.Body.String(), canary) {
		t.Fatalf("oversized login create = %d: %s", response.Code, response.Body.String())
	}
	assertErrorCode(t, response, "invalid_request")
}
