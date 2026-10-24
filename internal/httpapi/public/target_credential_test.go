package public

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestTargetCredentialCreateReturnsOnlySafeMetadata creates each pentest
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
		{"target-login", "target-login@1", `{"fields":{"username":"alice","password":"` + canary + `"}}`},
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
	fields := make([]string, 0, 17)
	for i := range 17 {
		fields = append(fields, fmt.Sprintf(`"f%d":"%s"`, i, canary))
	}
	body := `{"credentialId":"target-login-oversized","kind":"target-login@1","material":{"fields":{` + strings.Join(fields, ",") + `}}}`
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, runtimeMutationRequest(
		t, http.MethodPost, "/v1/operations/runtime-credentials", body, "target-login-oversized", "", "",
	))
	if response.Code != http.StatusBadRequest || strings.Contains(response.Body.String(), canary) {
		t.Fatalf("oversized login create = %d: %s", response.Code, response.Body.String())
	}
	assertErrorCode(t, response, "invalid_request")
}
