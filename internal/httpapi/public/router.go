package public

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/requestid"
)

type handler struct {
	dependencies Dependencies
	tokenDigest  [sha256.Size]byte
	cookieName   string
	secureCookie bool
}

func NewHandler(dependencies Dependencies) (http.Handler, error) {
	if dependencies.ConfigurationPublisher == nil {
		if publisher, ok := dependencies.Config.(ConfigurationPublisher); ok {
			dependencies.ConfigurationPublisher = publisher
		}
	}
	if dependencies.Config == nil || dependencies.ConfigurationPublisher == nil ||
		dependencies.Credentials == nil || dependencies.ManagedCredentials == nil ||
		dependencies.Runs == nil || dependencies.RunQueue == nil ||
		dependencies.RunLifecycle == nil || dependencies.RunCreator == nil ||
		dependencies.RuntimeConfigs == nil || dependencies.RuntimeCredentials == nil ||
		dependencies.RuntimeAgentPrincipals == nil ||
		dependencies.Projects == nil || dependencies.Audits == nil ||
		dependencies.Artifacts == nil || dependencies.Operations == nil ||
		dependencies.Performance == nil || dependencies.AllocationResources == nil ||
		dependencies.OperationsInvalidator == nil || dependencies.SchedulerSettings == nil || dependencies.Events == nil ||
		dependencies.Authentication == nil || len(dependencies.BrowserOrigins.Values()) == 0 {
		return nil, fmt.Errorf("public API dependencies are incomplete")
	}
	bearerToken := dependencies.BearerToken.Reveal()
	if bearerToken == "" || len(bearerToken) > 4096 {
		return nil, fmt.Errorf("public API bearer token must contain 1 through 4096 bytes")
	}
	if dependencies.NewID == nil {
		dependencies.NewID = randomID
	}
	if dependencies.NewRequestID == nil {
		dependencies.NewRequestID = func() (string, error) { return randomID("request_") }
	}
	if dependencies.Now == nil {
		dependencies.Now = time.Now
	}
	if dependencies.Logger == nil {
		dependencies.Logger = slog.Default()
	}
	tokenDigest := sha256.Sum256([]byte(bearerToken))
	bearerToken = ""
	dependencies.BearerToken = contracts.NewSecretString("")
	current := &handler{
		dependencies: dependencies,
		tokenDigest:  tokenDigest,
		cookieName:   auth.SecureCookieName,
		secureCookie: true,
	}
	if dependencies.InsecureLoopbackCookie {
		current.cookieName = auth.LoopbackCookieName
		current.secureCookie = false
	}

	mux := http.NewServeMux()
	current.registerSettingsRoutes(mux)
	current.registerSessionRoutes(mux)
	current.registerCatalogRoutes(mux)
	current.registerProjectRoutes(mux)
	current.registerAuditRoutes(mux)
	current.registerOperationsRoutes(mux)
	current.registerArtifactRoutes(mux)
	current.registerRunRoutes(mux)
	current.registerQueueRoutes(mux)
	mux.HandleFunc("/", current.notFound)
	return current.withMiddleware(mux), nil
}

// Incoming requests pass API version, request ID, origin policy,
// authentication and error diagnostics in that order.
func (h *handler) withMiddleware(mux *http.ServeMux) http.Handler {
	diagnostics := withErrorDiagnostics(mux)
	authenticated := h.authenticate(diagnostics)
	originChecked := h.cors(authenticated, mux)
	return withAPIVersion(h.withRequestID(originChecked))
}

func (h *handler) withRequestID(next http.Handler) http.Handler {
	return requestid.Middleware(next, requestid.Options{
		Generator: h.dependencies.NewRequestID,
		Logger:    h.dependencies.Logger, Boundary: "public-api",
	})
}

func randomID(prefix string) (string, error) {
	buffer := make([]byte, 16)
	if _, err := rand.Read(buffer); err != nil {
		return "", fmt.Errorf("generate identifier: %w", err)
	}
	return prefix + hex.EncodeToString(buffer), nil
}

func (h *handler) methodNotAllowed(w http.ResponseWriter, _ *http.Request) {
	h.writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "method is not allowed", false)
}

func (h *handler) rejectHead(w http.ResponseWriter, r *http.Request) bool {
	if r.Method != http.MethodHead {
		return false
	}
	h.methodNotAllowed(w, r)
	return true
}

func (h *handler) notFound(w http.ResponseWriter, _ *http.Request) {
	h.writeError(w, http.StatusNotFound, "not_found", "resource was not found", false)
}
