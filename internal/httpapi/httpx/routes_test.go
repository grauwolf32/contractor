package httpx

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestServeRoutesAnswersOnlyDeclaredMethods pins the routing rule both
// Server HTTP adapters share: a declared method reaches its handler, HEAD on
// a GET path and any other method on a route's path reach methodNotAllowed,
// and any other path reaches notFound.
func TestServeRoutesAnswersOnlyDeclaredMethods(t *testing.T) {
	entered, pattern := "", ""
	record := func(name string) http.HandlerFunc {
		return func(_ http.ResponseWriter, r *http.Request) { entered = name; pattern = r.Pattern }
	}
	mux := http.NewServeMux()
	allowed := ServeRoutes(mux, []Route{
		{Method: http.MethodGet, Path: "/v1/items/{id}", Handler: record("get item")},
		{Method: http.MethodPut, Path: "/v1/items/{id}", Handler: record("put item")},
		{Method: http.MethodPost, Path: "/v1/items", Handler: record("create item")},
		{Method: http.MethodPost, Path: "/v1/items/exports", Handler: record("create export")},
		{Method: http.MethodGet, Path: "/v1/items/exports/{exportId}/content", Handler: record("download export")},
	}, record("method not allowed"), record("not found"))
	for _, test := range []struct{ method, target, want string }{
		{http.MethodGet, "/v1/items/one", "get item"},
		{http.MethodPut, "/v1/items/one", "put item"},
		{http.MethodPost, "/v1/items", "create item"},
		{http.MethodHead, "/v1/items/one", "method not allowed"},
		{http.MethodHead, "/v1/items/one?limit=x", "method not allowed"},
		{http.MethodDelete, "/v1/items/one", "method not allowed"},
		{http.MethodHead, "/v1/items", "method not allowed"},
		{http.MethodGet, "/v1/items", "method not allowed"},
		{http.MethodPost, "/v1/items/exports", "create export"},
		{http.MethodGet, "/v1/items/exports", "method not allowed"},
		{http.MethodHead, "/v1/items/exports", "method not allowed"},
		{http.MethodGet, "/v1/items/exports/one/content", "download export"},
		{http.MethodPost, "/v1/items/exports/one/content", "method not allowed"},
		{http.MethodGet, "/v1/other", "not found"},
		{http.MethodHead, "/v1/other", "not found"},
	} {
		entered = ""
		probe := httptest.NewRequest(test.method, test.target, nil)
		if allowed(probe) != (test.want != "not found" && test.want != "method not allowed") {
			t.Errorf("method matcher disagrees for %s %s", test.method, test.target)
		}
		mux.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(test.method, test.target, nil))
		if allowed(probe) && !strings.HasPrefix(pattern, test.method+" ") {
			t.Errorf("operation pattern lost its method: %q", pattern)
		}
		if entered != test.want {
			t.Errorf("%s %s reached %q, want %q", test.method, test.target, entered, test.want)
		}
	}
}
