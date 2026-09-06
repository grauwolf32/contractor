package public

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/config"
)

func TestAgentInstructionsAuthenticatedExactCatalogRead(t *testing.T) {
	fixture := newHandlerFixtureWithConfig(t, "../../../configs")
	page := serveQuery(t, fixture.handler, "/v1/configurations/agent-templates?limit=1")
	var listed configurationPageResponse
	decodeQueryResponse(t, page, &listed)
	if len(listed.Items) != 1 {
		t.Fatalf("missing fixture template: %s", page.Body.String())
	}
	ref := listed.Items[0].Ref
	target := "/v1/configurations/agent-templates/" + ref.Name + "/versions/" + ref.Version + "/instructions"
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	response := serveAndValidatePublicContract(t, router, fixture.handler, newPublicContractRequest(http.MethodGet, target, nil), true)
	if response.Code != http.StatusOK {
		t.Fatalf("instructions = %d: %s", response.Code, response.Body.String())
	}
	var resource config.AgentInstructions
	decodeQueryResponse(t, response, &resource)
	if resource.Template.TemplateID != ref.Name || resource.Template.Version != ref.Version || resource.Template.Digest != ref.Digest || resource.Instructions.Text == "" {
		t.Fatalf("wrong instruction identity: %+v", resource)
	}
	if response.Header().Get("ETag") != strconv.Quote(ref.Digest) || response.Header().Get("Cache-Control") != "private, no-cache" {
		t.Fatalf("cache headers = %v", response.Header())
	}
	encoded, _ := json.Marshal(resource.Instructions.Text)
	if strings.Contains(page.Body.String(), string(encoded)) {
		t.Fatal("configuration list expanded prompt text")
	}
	for _, test := range []struct {
		name, method, path string
		auth               bool
		want               int
	}{
		{"unauthenticated", http.MethodGet, target, false, http.StatusUnauthorized},
		{"unknown version", http.MethodGet, strings.Replace(target, "/versions/"+ref.Version+"/", "/versions/missing/", 1), true, http.StatusNotFound},
		{"invalid selector", http.MethodGet, strings.Replace(target, "/"+ref.Name+"/", "/bad!name/", 1), true, http.StatusBadRequest},
		{"arbitrary path query", http.MethodGet, target + "?path=/etc/passwd", true, http.StatusBadRequest},
		{"head", http.MethodHead, target, true, http.StatusMethodNotAllowed},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(test.method, test.path, nil)
			if test.auth {
				request.Header.Set("Authorization", "Bearer "+testBearerToken)
			}
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, request)
			if response.Code != test.want {
				t.Fatalf("status = %d, want %d: %s", response.Code, test.want, response.Body.String())
			}
		})
	}
}
