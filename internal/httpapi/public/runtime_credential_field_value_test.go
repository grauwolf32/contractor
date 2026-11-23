package public

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/getkin/kin-openapi/openapi3"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/credentials"
)

// TestSharedCredentialFieldValueCasesGateRuntimeCredentialCreation sends
// every shared credential field-value case of
// api/testdata/v1alpha1/credential-field-value-cases.json in the creation
// material of its Runtime credential kind. A value outside the grammar or its
// byte bound is 400 invalid_request and stores nothing; every other value is
// stored. The public CreateRuntimeCredentialRequest schema gives each case
// the same verdict.
func TestSharedCredentialFieldValueCasesGateRuntimeCredentialCreation(t *testing.T) {
	cases := contractstest.ReadCredentialFieldValueCases(t)
	document := loadPublicOpenAPI(t)
	reference := document.Components.Schemas["CreateRuntimeCredentialRequest"]
	if reference == nil || reference.Value == nil {
		t.Fatal("components.schemas.CreateRuntimeCredentialRequest is missing")
	}
	schema := reference.Value
	fixture := newHandlerFixture(t)
	serial := 0
	checked := 0
	for _, field := range cases.Fields {
		if field.Credential == nil {
			continue
		}
		checked++
		for _, test := range field.Cases(cases.Grammar) {
			serial++
			credentialID := fmt.Sprintf("field-value-%d", serial)
			material := map[string]any{}
			contractstest.SetJSONPointer(t, material, field.Credential.Member, test.Value)
			body, err := json.Marshal(map[string]any{
				"credentialId": credentialID, "kind": field.Credential.Kind, "material": material,
			})
			if err != nil {
				t.Fatal(err)
			}
			var request any
			if err := json.Unmarshal(body, &request); err != nil {
				t.Fatal(err)
			}
			if err := schema.VisitJSON(request, openapi3.EnableJSONSchema2020(), openapi3.EnableFormatValidation()); (err == nil) != test.Valid {
				t.Errorf("%s %s: OpenAPI accepted = %t, want %t: %v", field.Name, test.Name, err == nil, test.Valid, err)
			}

			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, runtimeMutationRequest(
				t, http.MethodPost, "/v1/operations/runtime-credentials", string(body), credentialID, "", "",
			))
			_, stored := fixture.runtimeCredentials.Get(t.Context(), credentialID)
			switch {
			case test.Valid && (response.Code != http.StatusCreated || stored != nil):
				t.Errorf("%s %s: create = %d (stored error %v): %s", field.Name, test.Name, response.Code, stored, response.Body.String())
			case !test.Valid && response.Code != http.StatusBadRequest:
				t.Errorf("%s %s: create = %d, want 400: %s", field.Name, test.Name, response.Code, response.Body.String())
			case !test.Valid:
				assertErrorCode(t, response, "invalid_request")
				if !errors.Is(stored, credentials.ErrRuntimeCredentialNotFound) {
					t.Errorf("%s %s: rejected create stored a credential: %v", field.Name, test.Name, stored)
				}
			}
		}
	}
	if checked == 0 {
		t.Fatal("no shared credential field value names a Runtime credential kind")
	}
}
