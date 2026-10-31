package public

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/contracts/control"
)

func TestPublicSharedConfigContract(t *testing.T) {
	document := loadPublicOpenAPI(t)
	router, err := gorillamux.NewRouter(document)
	if err != nil {
		t.Fatal(err)
	}
	fixture := newHandlerFixture(t)
	request := newPublicContractRequest(http.MethodGet, "/v1/config", nil)
	response := serveAndValidatePublicContract(t, router, fixture.handler, request, true)
	if response.Code != http.StatusOK {
		t.Fatalf("config = %d %s", response.Code, response.Body)
	}
	expected, err := control.DefaultSharedConfig()
	if err != nil {
		t.Fatal(err)
	}
	var value control.SharedConfig
	if err := json.Unmarshal(response.Body.Bytes(), &value); err != nil || value.Validate() != nil || value != expected {
		t.Fatalf("public config differs from private policy: %+v %v", value, err)
	}
	if response.Header().Get("Cache-Control") != "no-store" {
		t.Fatal("config may be cached")
	}
	for _, tc := range []struct {
		path      string
		body      []byte
		anonymous bool
		status    int
	}{
		{"/v1/config", nil, true, http.StatusUnauthorized},
		{"/v1/config?unexpected=1", nil, false, http.StatusBadRequest},
		{"/v1/config", []byte(`{}`), false, http.StatusBadRequest},
	} {
		request := newPublicContractRequest(http.MethodGet, tc.path, tc.body)
		if tc.anonymous {
			request.Header.Del("Authorization")
		}
		response := serveAndValidatePublicContract(t, router, fixture.handler, request, false)
		if response.Code != tc.status {
			t.Fatalf("%s status = %d, body %s", tc.path, response.Code, response.Body)
		}
	}
	head := httptest.NewRecorder()
	fixture.handler.ServeHTTP(head, newPublicContractRequest(http.MethodHead, "/v1/config", nil))
	if head.Code != http.StatusMethodNotAllowed {
		t.Fatalf("HEAD config = %d", head.Code)
	}
}
