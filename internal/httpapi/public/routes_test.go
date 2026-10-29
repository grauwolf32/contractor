package public

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/getkin/kin-openapi/openapi3filter"
	"github.com/getkin/kin-openapi/routers"
	"github.com/getkin/kin-openapi/routers/gorillamux"

	"github.com/grauwolf32/contractor/internal/auth"
)

// routePathValues gives every path parameter of the route table a value
// that its OpenAPI schema accepts (routeTarget checks it), so a request
// addresses the route's operation instead of failing its path grammar.
var routePathValues = map[string]string{
	"exportId": "export-" + strings.Repeat("a", 64), "auditId": "audit-1", "chart": "quality", "commandId": "command-1",
	"credentialId": "credential-1", "datasetId": "dataset-1", "findingId": "finding-1",
	"groupKey": strings.Repeat("a", 64), "id": "experiment-1", "kind": "agent-templates",
	"label": "debug", "memberId": strings.Repeat("a", 64), "name": "source",
	"namespace": "inputs", "pairId": strings.Repeat("b", 64), "projectId": "project-1",
	"requestId": "review-1", "revision": "revision-1", "runId": "run-1",
	"runtimeAgentId": strings.Repeat("c", 64), "scheme": "example", "slot": "result",
	"sweepId": "sweep-1", "version": "1", "stageExecutionId": "stage-1",
}

var routePathParameter = regexp.MustCompile(`\{[^/{}]+\}`)

// malformedRouteQuery is a query that every operation rejects as invalid.
const malformedRouteQuery = "?limit=not-a-number&unexpected=1"

// TestHeadNeverReachesAPublicOperationHandler registers the whole route
// table with handlers that record their entry, behind the Operations
// authorization that NewHandler puts in front of the mux. Each route's own
// method reaches its own handler; HEAD on every route's path is 405
// method_not_allowed, with or without a malformed query, and enters no
// handler. Under /v1/operations a principal without the Operations
// capability gets 403 for HEAD as for the declared method (API-R111).
func TestHeadNeverReachesAPublicOperationHandler(t *testing.T) {
	router := publicContractRouter(t)
	entered := ""
	table := (contextHandlers{}).routes()
	for index := range table {
		operationID := table[index].operationID
		table[index].handler = func(w http.ResponseWriter, _ *http.Request) {
			entered = operationID
			w.WriteHeader(http.StatusNoContent)
		}
	}
	mux := http.NewServeMux()
	register(mux, table)
	handler := requireOperationsRoutes(mux)
	operator := auth.Principal{
		UserID: "operator", Username: "operator", Capabilities: []string{auth.CapabilityUser, auth.CapabilityOperations},
	}
	user := auth.Principal{UserID: "user", Username: "user", Capabilities: []string{auth.CapabilityUser}}
	serve := func(principal auth.Principal, method, target string) *httptest.ResponseRecorder {
		entered = ""
		request := httptest.NewRequest(method, target, nil)
		request = request.WithContext(auth.WithPrincipal(request.Context(), principal))
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		return response
	}
	gets := 0
	for _, route := range table {
		target := routeTarget(t, router, route)
		if response := serve(operator, route.method, target); response.Code != http.StatusNoContent || entered != route.operationID {
			t.Errorf("%s %s = %d and entered %q, want %s", route.method, target, response.Code, entered, route.operationID)
		}
		for _, query := range []string{"", malformedRouteQuery} {
			response := serve(operator, http.MethodHead, target+query)
			checkMethodNotAllowed(t, response, route, target+query)
			if entered != "" {
				t.Errorf("HEAD %s entered %s", target+query, entered)
			}
		}
		if strings.HasPrefix(route.path, "/v1/operations/") {
			for _, method := range []string{route.method, http.MethodHead} {
				if response := serve(user, method, target); response.Code != http.StatusForbidden || entered != "" {
					t.Errorf("%s %s without Operations = %d and entered %q, want 403", method, target, response.Code, entered)
				}
			}
		}
		if route.method == http.MethodGet {
			gets++
		}
	}
	if gets == 0 {
		t.Fatal("the route table has no GET route")
	}
}

// TestHeadOnEveryPublicGetRouteIsRejectedAfterAuthentication sends HEAD
// through the public handler and its real operation handlers to every GET
// route of the table: none of the addressed resources exists. Without
// credentials HEAD is 401 like GET; authorized, it is 405 method_not_allowed
// with or without a malformed query, so no handler answers it with its own
// validation, not-found or success response (API-R73, API-R111).
func TestHeadOnEveryPublicGetRouteIsRejectedAfterAuthentication(t *testing.T) {
	fixture := newHandlerFixture(t)
	router := publicContractRouter(t)
	gets := 0
	for _, route := range (contextHandlers{}).routes() {
		if route.method != http.MethodGet {
			continue
		}
		gets++
		target := routeTarget(t, router, route)
		for _, method := range []string{http.MethodGet, http.MethodHead} {
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, httptest.NewRequest(method, target, nil))
			if response.Code != http.StatusUnauthorized {
				t.Errorf("unauthenticated %s %s (%s) = %d, want 401: %s",
					method, target, route.operationID, response.Code, response.Body.String())
			}
		}
		for _, query := range []string{"", malformedRouteQuery} {
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, authenticatedRequest(http.MethodHead, target+query, bytes.NewReader(nil)))
			checkMethodNotAllowed(t, response, route, target+query)
		}
	}
	if gets == 0 {
		t.Fatal("the route table has no GET route")
	}
}

// routeTarget returns the path of route with every parameter replaced by
// its routePathValues value, after checking that the public contract
// resolves that path to the route's operation and accepts each value.
func routeTarget(t *testing.T, router routers.Router, route route) string {
	t.Helper()
	target := routePathParameter.ReplaceAllStringFunc(route.path, func(parameter string) string {
		value, ok := routePathValues[strings.Trim(parameter, "{}")]
		if !ok {
			t.Fatalf("%s %s: no test value for path parameter %s", route.method, route.path, parameter)
		}
		return value
	})
	request := httptest.NewRequest(route.method, "http://127.0.0.1:8080"+target, nil)
	found, pathParameters, err := router.FindRoute(request)
	if err != nil {
		t.Fatalf("%s %s: find contract route: %v", route.method, target, err)
	}
	if found.Operation.OperationID != route.operationID {
		t.Fatalf("%s %s resolves to %s, want %s", route.method, target, found.Operation.OperationID, route.operationID)
	}
	input := &openapi3filter.RequestValidationInput{Request: request, PathParams: pathParameters, Route: found}
	for _, parameters := range []openapi3.Parameters{found.PathItem.Parameters, found.Operation.Parameters} {
		for _, parameter := range parameters {
			if parameter.Value.In != openapi3.ParameterInPath {
				continue
			}
			if err := openapi3filter.ValidateParameter(t.Context(), input, parameter.Value); err != nil {
				t.Fatalf("%s %s: path parameter %s: %v", route.method, target, parameter.Value.Name, err)
			}
		}
	}
	return target
}

func publicContractRouter(t *testing.T) routers.Router {
	t.Helper()
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatalf("build contract router: %v", err)
	}
	return router
}

// checkMethodNotAllowed reports a HEAD response that is not the
// method_not_allowed error envelope.
func checkMethodNotAllowed(t *testing.T, response *httptest.ResponseRecorder, route route, target string) {
	t.Helper()
	var envelope errorResponse
	if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil ||
		response.Code != http.StatusMethodNotAllowed || envelope.Code != "method_not_allowed" || envelope.Retryable {
		t.Errorf("HEAD %s (%s) = %d %s, want 405 method_not_allowed",
			target, route.operationID, response.Code, response.Body.String())
	}
}
