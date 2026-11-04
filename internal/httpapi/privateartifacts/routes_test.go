package privateartifacts

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/httpapi/httpx"
)

// privateRoutePathValues gives every path parameter of the private route
// table the identity of the seeded test allocation and Artifact.
var privateRoutePathValues = map[string]string{
	"findingID": "finding-1", "allocationID": "allocation-1", "namespace": "inputs", "name": "source",
}

var privateRoutePathParameter = regexp.MustCompile(`\{[^/{}]+\}`)

// privateRouteQueries are a valid query of each operation and a malformed one.
var privateRouteQueries = []string{"?limit=1", "?revision=revision-a", "?limit=not-a-number&unexpected=1"}

// TestPrivateHeadNeverReachesARouteHandler serves the private route table
// with handlers that record their entry: each route's own method reaches its
// own handler, and HEAD, with any query, is 405 method_not_allowed on every
// route's path and enters no handler (ART-R120).
func TestPrivateHeadNeverReachesARouteHandler(t *testing.T) {
	entered := ""
	current := &handler{}
	table := current.routes()
	for index := range table {
		name := table[index].Method + " " + table[index].Path
		table[index].Handler = func(w http.ResponseWriter, _ *http.Request) {
			entered = name
			w.WriteHeader(http.StatusNoContent)
		}
	}
	mux := http.NewServeMux()
	httpx.ServeRoutes(mux, table, current.methodNotAllowed, current.notFound)
	gets := 0
	for _, route := range table {
		target := privateRouteTarget(t, route)
		entered = ""
		response := httptest.NewRecorder()
		mux.ServeHTTP(response, httptest.NewRequest(route.Method, target, nil))
		if want := route.Method + " " + route.Path; response.Code != http.StatusNoContent || entered != want {
			t.Errorf("%s %s = %d and entered %q, want %s", route.Method, target, response.Code, entered, want)
		}
		for _, query := range append([]string{""}, privateRouteQueries...) {
			entered = ""
			response := httptest.NewRecorder()
			mux.ServeHTTP(response, httptest.NewRequest(http.MethodHead, target+query, nil))
			checkPrivateMethodNotAllowed(t, response, target+query)
			if entered != "" {
				t.Errorf("HEAD %s entered %s", target+query, entered)
			}
		}
		if route.Method == http.MethodGet {
			gets++
		}
	}
	if gets == 0 {
		t.Fatal("the private route table has no GET route")
	}
}

// TestPrivateHeadIsRejectedAfterMTLSAndBeforeTheStore sends HEAD through
// NewHandler to every private GET route of a granted allocation whose
// Artifact exists. Without a Runtime certificate HEAD is 401 like GET; with
// it, HEAD is 405 for valid and malformed queries and reads no listing,
// metadata or payload, so it takes no transfer slot either (ART-R120).
func TestPrivateHeadIsRejectedAfterMTLSAndBeforeTheStore(t *testing.T) {
	repository := newMemoryRepository()
	repository.seed("run-a", "inputs", "source", "revision-a", []byte("run A"))
	server := newTestHandler(t, &fakeRegistry{grant: testGrant("run-a")}, repository)
	for _, route := range (&handler{}).routes() {
		if route.Method != http.MethodGet {
			continue
		}
		target := privateRouteTarget(t, route)
		for _, method := range []string{http.MethodGet, http.MethodHead} {
			response := httptest.NewRecorder()
			server.ServeHTTP(response, httptest.NewRequest(method, target, nil))
			if response.Code != http.StatusUnauthorized || !strings.Contains(response.Body.String(), `"mtls_required"`) {
				t.Errorf("%s %s without mTLS = %d %s, want 401 mtls_required", method, target, response.Code, response.Body.String())
			}
		}
		for _, query := range append([]string{""}, privateRouteQueries...) {
			response := httptest.NewRecorder()
			server.ServeHTTP(response, trustedRequest(http.MethodHead, target+query, nil))
			checkPrivateMethodNotAllowed(t, response, target+query)
		}
	}
	if repository.listCalls != 0 || repository.metadataCalls != 0 || repository.readCalls != 0 {
		t.Fatalf("HEAD reached the store: %d list, %d metadata and %d payload reads",
			repository.listCalls, repository.metadataCalls, repository.readCalls)
	}
	response := httptest.NewRecorder()
	server.ServeHTTP(response, trustedRequest(
		http.MethodGet, "/private/v1/allocations/allocation-1/artifacts/inputs/source", nil,
	))
	if response.Code != http.StatusOK || response.Body.String() != "run A" || repository.readCalls != 1 {
		t.Fatalf("GET after HEAD = %d %q with %d payload reads", response.Code, response.Body.String(), repository.readCalls)
	}
}

// privateRouteTarget returns the path of route with every parameter
// replaced by its privateRoutePathValues value.
func privateRouteTarget(t *testing.T, route httpx.Route) string {
	t.Helper()
	return privateRoutePathParameter.ReplaceAllStringFunc(route.Path, func(parameter string) string {
		value, ok := privateRoutePathValues[strings.Trim(parameter, "{}")]
		if !ok {
			t.Fatalf("%s %s: no test value for path parameter %s", route.Method, route.Path, parameter)
		}
		return value
	})
}

func checkPrivateMethodNotAllowed(t *testing.T, response *httptest.ResponseRecorder, target string) {
	t.Helper()
	var envelope errorResponse
	if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil ||
		response.Code != http.StatusMethodNotAllowed || envelope.Code != "method_not_allowed" || envelope.Retryable {
		t.Errorf("HEAD %s = %d %s, want 405 method_not_allowed", target, response.Code, response.Body.String())
	}
}
