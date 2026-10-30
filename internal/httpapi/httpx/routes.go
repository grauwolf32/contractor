package httpx

import "net/http"

// Route is one method and ServeMux path pattern of an HTTP adapter and the
// handler that serves it.
type Route struct {
	Method, Path string
	Handler      http.HandlerFunc
}

// ServeRoutes serves exactly routes on mux: another method of a route's path
// goes to methodNotAllowed and any other path to notFound. ServeMux lets a GET
// pattern serve HEAD too; dispatching methods after paths prevents HEAD
// from reaching a GET handler (API-R111, ART-R120). The caller wraps mux
// in its authentication, which therefore still answers first.
func ServeRoutes(mux *http.ServeMux, routes []Route, methodNotAllowed, notFound http.HandlerFunc) func(*http.Request) bool {
	paths := map[string]map[string]http.HandlerFunc{}
	for _, route := range routes {
		methods := paths[route.Path]
		if methods == nil {
			methods = map[string]http.HandlerFunc{}
			paths[route.Path] = methods
		}
		if _, exists := methods[route.Method]; exists {
			panic("duplicate HTTP route: " + route.Method + " " + route.Path)
		}
		methods[route.Method] = route.Handler
	}
	// Select the most specific resource path before its method. Registering
	// methodless 405 fallbacks beside method-specific wildcard routes creates
	// ambiguous ServeMux patterns for a literal child collection such as
	// /items/exports beside GET /items/{id}.
	for path, methods := range paths {
		mux.HandleFunc(path, func(w http.ResponseWriter, r *http.Request) {
			if handler := methods[r.Method]; handler != nil {
				// Preserve the method-qualified operation pattern for diagnostics.
				r.Pattern = r.Method + " " + path
				handler(w, r)
				return
			}
			methodNotAllowed(w, r)
		})
	}
	mux.HandleFunc("/", notFound)
	// CORS probes the same resolved resource and method without invoking a
	// handler or interpreting ServeMux's implementation-specific Pattern text.
	return func(r *http.Request) bool {
		_, pattern := mux.Handler(r)
		return paths[pattern][r.Method] != nil
	}
}
