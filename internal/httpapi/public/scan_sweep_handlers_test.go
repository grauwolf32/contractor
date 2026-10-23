package public

import (
	"encoding/json"
	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"net/http"
	"testing"
)

func TestScanSweepPublicContract(t *testing.T) {
	fixture := newHandlerFixture(t)
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		method, path, body string
		status             int
	}{
		{"POST", "/v1/scan-sweeps", `{"projectIds":["project-one"],"workflow":"nuclei-web@1","parameters":{}}`, http.StatusCreated},
		{"GET", "/v1/scan-sweeps", "", http.StatusOK},
		{"GET", "/v1/scan-sweeps/sweep-one", "", http.StatusOK},
		{"GET", "/v1/scan-sweeps/sweep-one/members", "", http.StatusOK},
		{"POST", "/v1/scan-sweeps/sweep-one/cancel", "", http.StatusOK},
		{"DELETE", "/v1/scan-sweeps/sweep-one", "", http.StatusNoContent},
	} {
		t.Run(tc.method+tc.path, func(t *testing.T) {
			request := newPublicContractRequest(tc.method, tc.path, []byte(tc.body))
			if tc.body != "" {
				request.Header.Set("Content-Type", "application/json")
				request.Header.Set("Idempotency-Key", "sweep-create")
			}
			got := serveAndValidatePublicContract(t, router, fixture.handler, request, true)
			if got.Code != tc.status {
				t.Fatalf("status=%d: %s", got.Code, got.Body.String())
			}
		})
	}
}

func TestScanSweepSnapshotsStayOutOfOwnerLibrary(t *testing.T) {
	fixture := newHandlerFixture(t)
	scope, _ := artifacts.UserScope("user-1")
	hidden := contracts.ArtifactRef{Namespace: artifactpolicy.RunSystemNamespace, Name: "scan-sweep.hidden.rulepack"}
	if _, err := fixture.artifacts.CreateFrozen(t.Context(), scope, hidden, artifacts.Payload{MediaType: "application/yaml", Data: []byte("retained")}); err != nil {
		t.Fatal(err)
	}
	user, _ := fixture.artifacts.User("user-1")
	if _, err := user.Write(t.Context(), contracts.ArtifactRef{Namespace: "zeta", Name: "owner"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("visible")}, nil); err != nil {
		t.Fatal(err)
	}
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	response := serveAndValidatePublicContract(t, router, fixture.handler, newPublicContractRequest(http.MethodGet, "/v1/artifacts?limit=1", nil), true)
	var listed struct {
		Items []artifacts.Metadata `json:"items"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &listed); err != nil || len(listed.Items) != 1 || listed.Items[0].Ref.Namespace != "zeta" {
		t.Fatalf("library=%s %v", response.Body.String(), err)
	}
	for _, suffix := range []string{"", "/metadata", "/versions", "/lineage"} {
		response := serveAndValidatePublicContract(t, router, fixture.handler, newPublicContractRequest(http.MethodGet, "/v1/artifacts/"+hidden.Namespace+"/"+hidden.Name+suffix, nil), true)
		if response.Code != http.StatusNotFound {
			t.Fatalf("hidden read=%d %s", response.Code, response.Body.String())
		}
	}
	request := newPublicContractRequest(http.MethodPut, "/v1/artifacts/"+hidden.Namespace+"/new", []byte("other"))
	request.Header.Set("Content-Type", "application/yaml")
	request.Header.Set("If-None-Match", "*")
	response = serveAndValidatePublicContract(t, router, fixture.handler, request, true)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("system upload=%d %s", response.Code, response.Body.String())
	}
}
