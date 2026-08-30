package public

import "net/http"

const (
	APIVersionHeader = "X-Contractor-API-Version"
	APIVersion       = "contractor.public.v1"
)

// withAPIVersion is the outer public boundary. It advertises compatibility on
// every response, including authentication errors, CORS rejections/preflights,
// method failures, and WebSocket upgrades.
func withAPIVersion(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(APIVersionHeader, APIVersion)
		next.ServeHTTP(w, r)
	})
}
