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
// pattern serve HEAD too, and no route of the public or private API defines
// HEAD, so every path with a GET route also gets an explicit HEAD pattern and
// HEAD never reaches a GET handler (API-R111, ART-R120). The caller wraps mux
// in its authentication, which therefore still answers first.
func ServeRoutes(mux *http.ServeMux, routes []Route, methodNotAllowed, notFound http.HandlerFunc) {
	paths, gets := map[string]bool{}, map[string]bool{}
	for _, route := range routes {
		mux.HandleFunc(route.Method+" "+route.Path, route.Handler)
		paths[route.Path] = true
		gets[route.Path] = gets[route.Path] || route.Method == http.MethodGet
	}
	for path := range paths {
		if gets[path] {
			mux.HandleFunc(http.MethodHead+" "+path, methodNotAllowed)
		}
		mux.HandleFunc(path, methodNotAllowed)
	}
	mux.HandleFunc("/", notFound)
}
