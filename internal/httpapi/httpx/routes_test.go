package httpx

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestServeRoutesAnswersOnlyDeclaredMethods pins the routing rule both
// Server HTTP adapters share: a declared method reaches its handler, HEAD on
// a GET path and any other method on a route's path reach methodNotAllowed,
// and any other path reaches notFound.
func TestServeRoutesAnswersOnlyDeclaredMethods(t *testing.T) {
	entered := ""
	record := func(name string) http.HandlerFunc {
		return func(http.ResponseWriter, *http.Request) { entered = name }
	}
	mux := http.NewServeMux()
	ServeRoutes(mux, []Route{
		{Method: http.MethodGet, Path: "/v1/items/{id}", Handler: record("get item")},
		{Method: http.MethodPut, Path: "/v1/items/{id}", Handler: record("put item")},
		{Method: http.MethodPost, Path: "/v1/items", Handler: record("create item")},
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
		{http.MethodGet, "/v1/other", "not found"},
		{http.MethodHead, "/v1/other", "not found"},
	} {
		entered = ""
		mux.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(test.method, test.target, nil))
		if entered != test.want {
			t.Errorf("%s %s reached %q, want %q", test.method, test.target, entered, test.want)
		}
	}
}
