package public

import (
	"net/http"
	"strings"

	"github.com/grauwolf32/contractor/internal/auth"
)

func (h *handler) requireOperationsRoutes(next http.Handler) http.Handler {
	protected := h.withOperationsCapability(next)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/operations" || strings.HasPrefix(r.URL.Path, "/v1/operations/") {
			protected.ServeHTTP(w, r)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (h *handler) withOperationsCapability(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !h.requireOperationsCapability(w, r) {
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (h *handler) requireOperationsCapability(w http.ResponseWriter, r *http.Request) bool {
	principal, ok := auth.PrincipalFromContext(r.Context())
	if !ok {
		h.writeBearerUnauthorized(w)
		return false
	}
	for _, capability := range principal.Capabilities {
		if capability == auth.CapabilityOperations {
			return true
		}
	}
	h.writeError(w, http.StatusForbidden, "forbidden", "Operations capability is required", false)
	return false
}
