package public

import (
	"net/http"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/requestid"
)

// NewStandbyHandler serves the public API of a standby Server, one that waits
// while another Server holds the database's single-active lease (PER-R3). It
// answers every request, the event upgrade included, with the retryable 503
// server_standby in the error envelope, so a client backs off instead of
// reading a missing resource. A standby has no services to authenticate or
// route with, and the answer discloses only what /readyz already reports. It
// logs no request: the standby logged once that it stands by, and a polling
// client would otherwise flood the log until the takeover.
func NewStandbyHandler() http.Handler {
	standby := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		writeError(w, apierror.ServerStandby, "")
	})
	return withAPIVersion(requestid.Middleware(standby, requestid.Options{Boundary: "public-api"}))
}
