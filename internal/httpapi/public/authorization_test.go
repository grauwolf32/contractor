package public

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/grauwolf32/contractor/internal/auth"
)

func TestOperationsCapabilityMiddlewareCoversManagementSubtree(t *testing.T) {
	current := &handler{}
	called := 0
	protected := current.requireOperationsRoutes(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		called++
		w.WriteHeader(http.StatusNoContent)
	}))
	user := auth.Principal{UserID: "user", Username: "user", Capabilities: []string{auth.CapabilityUser}}
	operator := auth.Principal{UserID: "operator", Username: "operator", Capabilities: []string{auth.CapabilityOperations}}
	for _, route := range []struct{ method, path string }{
		{http.MethodPost, "/v1/operations/credentials"},
		{http.MethodDelete, "/v1/operations/credentials/key"},
		{http.MethodPost, "/v1/operations/runtime-configs"},
		{http.MethodPost, "/v1/operations/runtime-credentials"},
		{http.MethodPut, "/v1/operations/runtime-labels/debug"},
		{http.MethodGet, "/v1/operations/performance"},
		{http.MethodGet, "/v1/operations/unknown"},
		{http.MethodGet, "/v1/operations"},
	} {
		t.Run(route.method+" "+route.path, func(t *testing.T) {
			for _, test := range []struct {
				name      string
				principal *auth.Principal
				status    int
			}{
				{"unauthenticated", nil, http.StatusUnauthorized},
				{"user", &user, http.StatusForbidden},
				{"operator", &operator, http.StatusNoContent},
			} {
				t.Run(test.name, func(t *testing.T) {
					request := httptest.NewRequest(route.method, route.path, nil)
					if test.principal != nil {
						request = request.WithContext(auth.WithPrincipal(request.Context(), *test.principal))
					}
					before := called
					response := httptest.NewRecorder()
					protected.ServeHTTP(response, request)
					if response.Code != test.status {
						t.Fatalf("status = %d, want %d: %s", response.Code, test.status, response.Body.String())
					}
					wantCalls := before
					if test.status == http.StatusNoContent {
						wantCalls++
					}
					if called != wantCalls {
						t.Fatal("denied request reached management handler")
					}
				})
			}
		})
	}
	request := httptest.NewRequest(http.MethodGet, "/v1/projects", nil)
	request = request.WithContext(auth.WithPrincipal(request.Context(), user))
	response := httptest.NewRecorder()
	protected.ServeHTTP(response, request)
	if response.Code != http.StatusNoContent {
		t.Fatalf("non-Operations route status = %d", response.Code)
	}
}

func TestConfigurationPublicationRequiresOperationsCapability(t *testing.T) {
	current := &handler{}
	mux := http.NewServeMux()
	current.registerCatalogRoutes(mux)
	for _, test := range []struct {
		name         string
		capabilities []string
		status       int
	}{
		{"user", []string{auth.CapabilityUser}, http.StatusForbidden},
		{"operator", []string{auth.CapabilityOperations}, http.StatusBadRequest},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodPost, "/v1/configurations/agent-templates", nil)
			request = request.WithContext(auth.WithPrincipal(request.Context(), auth.Principal{
				UserID: "person", Username: "person", Capabilities: test.capabilities,
			}))
			response := httptest.NewRecorder()
			mux.ServeHTTP(response, request)
			if response.Code != test.status {
				t.Fatalf("publication status = %d, want %d: %s", response.Code, test.status, response.Body.String())
			}
		})
	}
}
