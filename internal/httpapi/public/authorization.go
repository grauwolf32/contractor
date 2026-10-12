package public

import (
	"context"
	"net/http"
	"slices"
	"strings"

	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func requireOperationsRoutes(next http.Handler) http.Handler {
	protected := withOperationsCapability(next)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/operations" || strings.HasPrefix(r.URL.Path, "/v1/operations/") {
			protected.ServeHTTP(w, r)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func withOperationsCapability(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !requireOperationsCapability(w, r) {
			return
		}
		next.ServeHTTP(w, r)
	})
}

func requireOperationsCapability(w http.ResponseWriter, r *http.Request) bool {
	principal, ok := auth.PrincipalFromContext(r.Context())
	if !ok {
		writeBearerUnauthorized(w)
		return false
	}
	for _, capability := range principal.Capabilities {
		if capability == auth.CapabilityOperations {
			return true
		}
	}
	writeError(w, http.StatusForbidden, "forbidden", "Operations capability is required", false)
	return false
}

func writeBearerUnauthorized(w http.ResponseWriter) {
	w.Header().Set("WWW-Authenticate", "Bearer")
	writeError(w, http.StatusUnauthorized, "unauthorized", "valid authentication is required", false)
}

func principalUserID(ctx context.Context) string {
	principal, ok := auth.PrincipalFromContext(ctx)
	if !ok {
		return ""
	}
	return principal.UserID
}

// runtimeCredentialUser describes the authenticated principal's authority to
// reference Runtime credentials from its owner-scoped configuration.
func runtimeCredentialUser(ctx context.Context) credentials.RuntimeCredentialUser {
	principal, ok := auth.PrincipalFromContext(ctx)
	if !ok {
		return credentials.RuntimeCredentialUser{}
	}
	return credentials.RuntimeCredentialUser{
		UserID:     principal.UserID,
		Operations: slices.Contains(principal.Capabilities, auth.CapabilityOperations),
	}
}

// ownedRun reads the Run of the route and hides another owner's Run as
// missing (API-R25).
func ownedRun(r *http.Request, runs RunGetter) (runstore.WorkflowRun, error) {
	run, err := runs.GetRun(r.Context(), r.PathValue("runID"))
	if err != nil {
		return runstore.WorkflowRun{}, err
	}
	if run.OwnerID != principalUserID(r.Context()) {
		return runstore.WorkflowRun{}, runstore.ErrNotFound
	}
	return run, nil
}
